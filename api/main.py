#!/usr/bin/env python3
"""FastAPI service: ECG ingestion, simulation control, and live predictions.

Responsibilities (per the system architecture, the API sits in front of
Kafka — it never performs streaming-path inference itself):

  * POST /api/ecg/send           -> publish ECG event to Kafka
  * POST /api/simulation/*       -> start/stop event stream simulations
  * background consumer          -> ecg-predictions -> PostgreSQL + WebSocket
  * WS  /ws/predictions          -> live predictions to the dashboard
  * REST read endpoints          -> predictions, statistics, metrics,
                                    explanations, system status

Run locally:
  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from api.routes import ecg, predictions, system
from api.services import broker
from api.services.kafka_service import latency_tracker
from api.services.simulation import get_simulation_engine
from api.services.websocket_manager import ws_manager
from ecg_core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("api")

prediction_consumer = None


def _on_prediction(prediction: dict) -> None:
    """Called by the Kafka consumer thread for every prediction event."""
    ws_manager.broadcast_sync({"type": "prediction", "data": prediction})
    latency = prediction.get("latency_ms") or prediction.get("inference_latency_ms")
    if latency is not None:
        latency_tracker.record(float(latency), prediction.get("predicted_class", "?"))
    logger.info(
        "Prediction %s -> %s (err=%.5f, latency=%.2fms, source=%s)",
        prediction.get("event_id"),
        prediction.get("predicted_class"),
        prediction.get("reconstruction_error", float("nan")),
        latency if latency is not None else float("nan"),
        prediction.get("source", "kafka"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_consumer

    ws_manager.set_loop(asyncio.get_running_loop())
    settings.ensure_dirs()

    # --- Database (best effort: the API degrades gracefully without it) ----
    try:
        from database.models import init_db

        init_db()
    except Exception as exc:  # noqa: BLE001
        logger.warning("PostgreSQL not reachable (%s); prediction persistence disabled.", exc)

    # --- Model warm-up (best effort) ---------------------------------------
    try:
        from inference.model_loader import get_model_bundle

        get_model_bundle()
        logger.info("Model bundle warmed up.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Models not ready (%s); train before inferring.", exc)

    # --- Background prediction consumer -------------------------------------
    # Broker is selected via MESSAGE_BROKER (kafka | redis).
    prediction_consumer = broker.make_prediction_consumer(on_prediction=_on_prediction)
    prediction_consumer.start()

    logger.info(
        "API ready on %s:%s (broker=%s)",
        settings.api_host, settings.api_port, broker.broker_name(),
    )
    try:
        yield
    finally:
        if prediction_consumer:
            prediction_consumer.stop()
        broker.stop_publishers()
        get_simulation_engine().stop()
        logger.info("API shut down.")


app = FastAPI(
    title="ECG Real-Time Analytics API",
    description=(
        "Autoencoder-assisted real-time ECG arrhythmia classification "
        "with Kafka + Spark Structured Streaming."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev dashboard runs on a different port; tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(ecg.router)
app.include_router(predictions.router)


@app.get("/health", tags=["system"])
def health() -> dict:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket) -> None:
    """Live prediction feed for the dashboard."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Ping/pong keeps the socket alive; clients may also send
            # {"type":"ping"} which we echo.
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:  # noqa: BLE001
        ws_manager.disconnect(websocket)


# Serve the built dashboard from /dashboard/dist when present (Docker path).
_DIST = Path(__file__).resolve().parent.parent / "dashboard" / "dist"
if _DIST.is_dir():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
