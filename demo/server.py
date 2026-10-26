#!/usr/bin/env python3
"""Demo server: a single-process backend for smooth, dependency-free demos.

Serves the built dashboard (dashboard/dist) plus every API the dashboard
calls — using the real trained models for inference — with **no Kafka,
no Spark, and no PostgreSQL**. Everything runs in one Python process:

    python demo/server.py          # → http://localhost:8000

What it does differently from the full api/ service:
  * /api/ecg/send always works: validation → scaler → AE gate → XGBoost.
  * Predictions are kept in memory (plus seeded at startup, so the
    Analytics / History pages have content the moment you open them).
  * The simulation engine runs as an asyncio task and pushes predictions
    to the dashboard over the same /ws/predictions socket.
  * /api/system/status reports every component green — there is no
    broker or database to be "down" in a demo.

The npm dev server also works against it (vite proxies /api and /ws to
localhost:8000), but the simplest demo path is this one command and a
browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ecg_core.config import settings
from ecg_core.labels import CLASS_DESCRIPTIONS, CLASS_NAMES
from inference.model_loader import bundle_status, get_model_bundle
from streaming.feature_pipeline import validate_event
from streaming.kafka_producer import DatasetSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("demo")

# ---------------------------------------------------------------------------
# Small in-memory state
# ---------------------------------------------------------------------------


class DemoStore:
    """Newest-first prediction list + feature cache + latency window."""

    def __init__(self, capacity: int = 2000):
        self._items: list[dict] = []
        self._capacity = capacity
        self._features: "OrderedDict[str, object]" = OrderedDict()
        self._latencies: list[float] = []
        self._event_times: list[float] = []
        self._class_counts: dict[str, int] = {}

    def add(self, prediction: dict, features=None) -> None:
        self._items.insert(0, prediction)
        del self._items[self._capacity:]
        if features is not None:
            self._features[prediction["event_id"]] = features
            while len(self._features) > 500:
                self._features.popitem(last=False)
        latency = prediction.get("latency_ms")
        if latency is not None:
            self._latencies = (self._latencies + [float(latency)])[-1000:]
            self._event_times = (self._event_times + [time.time()])[-1000:]
        cls = prediction.get("predicted_class", "?")
        self._class_counts[cls] = self._class_counts.get(cls, 0) + 1

    def features_for(self, event_id: str):
        return self._features.get(event_id)

    def latest(self, limit: int = 50, predicted_class: Optional[str] = None, abnormal_only: bool = False):
        rows = self._items
        if predicted_class:
            rows = [r for r in rows if r.get("predicted_class") == predicted_class.upper()]
        if abnormal_only:
            rows = [r for r in rows if r.get("is_abnormal")]
        return rows[:limit]

    def statistics(self) -> dict:
        total = len(self._items)
        abnormal = sum(1 for r in self._items if r.get("is_abnormal"))
        lat = [r.get("latency_ms") for r in self._items if r.get("latency_ms") is not None]
        return {
            "total_events": total,
            "normal_events": total - abnormal,
            "abnormal_events": abnormal,
            "abnormal_rate": round(abnormal / total, 4) if total else 0.0,
            "average_latency_ms": round(sum(lat) / len(lat), 3) if lat else None,
            "class_distribution": dict(self._class_counts),
        }

    def metrics(self, window_seconds: int = 60) -> dict:
        import numpy as np

        if self._latencies:
            arr = np.asarray(self._latencies)
            lat = {
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
            }
        else:
            lat = {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}

        cutoff = time.time() - window_seconds
        recent = [t for t in self._event_times if t >= cutoff]
        span = max((recent[-1] - recent[0]), 1e-9) if len(recent) > 1 else window_seconds
        eps = len(recent) / span if span > 0 else 0.0
        return {
            "total_events": len(self._latencies),
            "latency_ms": lat,
            "throughput_events_per_second": round(eps, 3),
            "class_distribution": dict(self._class_counts),
            "window_seconds": window_seconds,
        }


store = DemoStore()
sampler = DatasetSampler()

# ---------------------------------------------------------------------------
# Inference helpers (the real trained models)
# ---------------------------------------------------------------------------


def predict_event(event: dict, explain: bool = False) -> dict:
    """Validate + two-stage inference; returns the prediction dict."""
    from inference.predictor import predict_features

    validated = validate_event(event)
    result = predict_features(
        validated.features,
        event_id=validated.event_id,
        record_id=validated.record_id,
        true_label=validated.true_label,
        explain=explain,
    )
    payload = dict(vars(result))
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    payload["event_timestamp"] = event.get("timestamp")
    payload["source"] = "demo"
    return payload


def _handle_prediction(prediction: dict, features=None) -> None:
    store.add(prediction, features)
    ws_manager.broadcast({"type": "prediction", "data": prediction})
    logger.info(
        "Prediction %s -> %s (err=%.5f, latency=%.2fms)",
        prediction.get("event_id"), prediction.get("predicted_class"),
        prediction.get("reconstruction_error", float("nan")),
        prediction.get("latency_ms", float("nan")),
    )


# ---------------------------------------------------------------------------
# WebSocket broadcast (single event loop — plain asyncio, no threads)
# ---------------------------------------------------------------------------


class DemoSockets:
    def __init__(self):
        self.clients: set[WebSocket] = set()
        # asyncio primitives bind to an event loop when they first wait.
        # Construct the queue during application startup rather than at module
        # import time, otherwise uvicorn can create it on a different loop.
        self._queue: Optional[asyncio.Queue[str]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def start(self) -> None:
        """Initialize the broadcaster on FastAPI's running event loop."""
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        asyncio.create_task(self.pump())

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)

    def disconnect(self, ws: WebSocket):
        self.clients.discard(ws)

    def broadcast(self, payload: dict) -> None:
        """Queue a broadcast safely from async routes or sync worker threads."""
        if self._loop is None or self._queue is None:
            logger.warning("Dropping WebSocket broadcast before startup")
            return
        message = json.dumps(payload)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, message)

    async def pump(self):
        """Drain the broadcast queue to all connected clients."""
        if self._queue is None:
            raise RuntimeError("WebSocket broadcaster was not started")
        while True:
            message = await self._queue.get()
            dead = []
            for ws in self.clients:
                try:
                    await ws.send_text(message)
                except Exception:  # noqa: BLE001 - client vanished
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)


ws_manager = DemoSockets()

# ---------------------------------------------------------------------------
# Simulation engine (asyncio task, wall-clock paced)
# ---------------------------------------------------------------------------

simulation_state = {
    "running": False,
    "mode": None,
    "events_sent": 0,
    "events_per_second": None,
    "duration_seconds": None,
    "started_at": None,
}
simulation_task: Optional[asyncio.Task] = None


async def _run_simulation(mode: str, eps: float, duration: float, n_events: Optional[int]) -> None:
    interval = 1.0 / max(eps, 0.1)
    sent = 0
    start = time.time()
    logger.info("Demo simulation started: mode=%s rate=%.2f/s duration=%ss", mode, eps, duration)
    try:
        while time.time() - start < duration:
            if n_events and sent >= n_events:
                break
            try:
                event = sampler.sample_by_mode(mode)
                validated = validate_event(event)
                prediction = predict_event(event, explain=True)
                _handle_prediction(prediction, features=validated.features)
                sent += 1
                simulation_state["events_sent"] = sent
            except Exception as exc:  # noqa: BLE001 - keep simulating
                logger.warning("Simulation tick failed: %s", exc)
            next_tick = start + (sent + 1) * interval
            await asyncio.sleep(max(0.0, next_tick - time.time()))
    finally:
        simulation_state["running"] = False
        logger.info("Demo simulation finished: %d events sent", sent)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

app = FastAPI(title="ECG Demo Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DIST = Path(__file__).resolve().parent.parent / "dashboard" / "dist"


@app.on_event("startup")
async def startup() -> None:
    await ws_manager.start()
    try:
        get_model_bundle()
        logger.info("Model bundle loaded (AE threshold + Stage-2 XGBoost ready).")
    except Exception as exc:  # noqa: BLE001
        logger.error("Models failed to load — run the training scripts first: %s", exc)
        return
    # Seed history so every dashboard page has content immediately.
    try:
        for mode in (["normal"] * 8 + ["abnormal"] * 8 + ["random"] * 14):
            event = sampler.sample_by_mode(mode)
            prediction = predict_event(event, explain=True)
            store.add(prediction, features=validate_event(event).features)
        logger.info("Seeded %d demo predictions.", 30)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Seeding failed (dataset missing?): %s", exc)


@app.get("/health", include_in_schema=False)
def health() -> dict:
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/system/status")
def system_status() -> dict:
    return {
        "api": {"status": "up", "detail": "demo server"},
        "kafka": {"status": "up", "detail": "not required in demo mode"},
        "spark": {"status": "up", "detail": "in-process inference"},
        "database": {"status": "up", "detail": "in-memory demo store"},
        "models": {"status": "up", "detail": "autoencoder + xgboost + preprocessing ready"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/ecg/classes")
def ecg_classes() -> dict:
    try:
        counts = sampler.available_classes()
    except FileNotFoundError:
        counts = {c: 0 for c in CLASS_NAMES}
    return {
        "classes": [
            {"code": c, "description": CLASS_DESCRIPTIONS[c], "samples": counts.get(c, 0)}
            for c in CLASS_NAMES
        ]
    }


@app.get("/api/ecg/sample")
def ecg_sample(mode: str = "random") -> dict:
    try:
        return sampler.sample_by_mode(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {exc}") from exc


@app.post("/api/ecg/send")
def send_ecg_event(payload: dict) -> dict:
    try:
        validated = validate_event(payload)
        prediction = predict_event(payload, explain=True)
        _handle_prediction(prediction, features=validated.features)
        return {
            "event_id": prediction["event_id"],
            "published_to_kafka": True,
            "topic": "demo",
            "message": "Event processed by the two-stage inference pipeline.",
            "prediction": prediction,
        }
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Invalid event: {exc}") from exc


@app.post("/api/simulation/start")
async def start_simulation(request: dict) -> dict:
    global simulation_task
    if simulation_state["running"]:
        raise HTTPException(status_code=409, detail="A simulation is already running; stop it first.")
    simulation_state.update(
        running=True,
        mode=request.get("mode", "random"),
        events_sent=0,
        events_per_second=request.get("events_per_second", 2.0),
        duration_seconds=request.get("duration_seconds", 30.0),
        started_at=time.time(),
    )
    simulation_task = asyncio.create_task(
        _run_simulation(
            request.get("mode", "random"),
            float(request.get("events_per_second", 2.0)),
            float(request.get("duration_seconds", 30.0)),
            request.get("n_events"),
        )
    )
    return dict(simulation_state)


@app.post("/api/simulation/stop")
def stop_simulation() -> dict:
    global simulation_task
    if simulation_task and not simulation_task.done():
        simulation_task.cancel()
    simulation_state["running"] = False
    return dict(simulation_state)


@app.get("/api/simulation/status")
def simulation_status() -> dict:
    return dict(simulation_state)


@app.get("/api/predictions/latest")
def latest_prediction() -> dict:
    rows = store.latest(limit=1)
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions stored yet.")
    return rows[0]


@app.get("/api/predictions")
def list_predictions(
    limit: int = Query(50, ge=1, le=1000),
    predicted_class: Optional[str] = None,
    abnormal_only: bool = False,
) -> list:
    return store.latest(limit=limit, predicted_class=predicted_class, abnormal_only=abnormal_only)


@app.get("/api/statistics")
def statistics() -> dict:
    stats = store.statistics()
    live = store.metrics(window_seconds=60)
    return {
        **stats,
        "events_per_second": live["throughput_events_per_second"],
        "latency": live["latency_ms"],
    }


@app.get("/api/metrics")
def metrics(window_seconds: int = 60) -> dict:
    return store.metrics(window_seconds=window_seconds)


@app.get("/api/models/status")
def models_status() -> dict:
    return bundle_status()


@app.get("/api/threshold")
def threshold() -> dict:
    from ecg_core.models.autoencoder import load_threshold

    try:
        value = load_threshold()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    percentile = None
    meta_path = settings.autoencoder_dir / "autoencoder_meta.json"
    if meta_path.exists():
        percentile = json.loads(meta_path.read_text()).get("threshold_percentile")
    return {"anomaly_threshold": value, "threshold_percentile": percentile}


@app.get("/api/models/experiments")
def model_experiments() -> dict:
    """Offline experiment results from logs/training/experiments/."""
    labels = {
        "baseline_ae": "Baseline AE (Adam Default)",
        "pso_ae": "PSO-Tuned AE",
        "gwo_ae": "GWO-Tuned AE",
        "coa_ae": "COA-Tuned AE",
        "xgboost_only": "A · XGBoost only (5-class)",
        "autoencoder_only": "B · Autoencoder anomaly detection (Legacy)",
        "hybrid_ae_xgb": "C · Hybrid AE → XGBoost (production)",
        "xgboost_augmented": "D · XGBoost + AE latent/error features",
        "mlp_baseline": "E · MLP baseline",
        "xgboost_baseline": "XGBoost baseline (full data)",
        "xgboost_abnormal": "XGBoost Stage-2 (abnormal classes)",
    }
    exp_dir = settings.training_logs_dir / "experiments"
    experiments = []
    if exp_dir.exists():
        for path in sorted(exp_dir.glob("*.json")):
            if path.name == "comparison_summary.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            name = payload.get("experiment", path.stem)
            experiments.append(
                {"name": name, "label": labels.get(name, name),
                 "extra": payload.get("extra", {}), "metrics": payload.get("metrics", {})}
            )
    order = {name: i for i, name in enumerate(labels)}
    experiments.sort(key=lambda e: (order.get(e["name"], 99), e["name"]))
    return {"experiments": experiments, "directory": str(exp_dir)}


@app.get("/api/models/metaheuristics")
def metaheuristics_summary() -> dict:
    """Summary and complete metrics for Baseline, PSO, GWO, and COA Autoencoders."""
    summary_path = Path(__file__).resolve().parent.parent / "reports" / "tables" / "all_experiments_summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {}


@app.get("/api/explain/{event_id}")
def explain_prediction(event_id: str) -> dict:
    features = store.features_for(event_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found in the explanation cache (only recent events "
                   "sent through this server can be explained).",
        )
    row = next((r for r in store._items if r.get("event_id") == event_id), None)
    predicted_class = row.get("predicted_class") if row else None
    top = []
    try:
        import numpy as np

        bundle = get_model_bundle()
        X_scaled = bundle.transform(np.asarray(features).reshape(1, -1))
        from inference.predictor import explain_prediction as _explain

        class_index = None
        if predicted_class and getattr(bundle, "xgboost_classes", None):
            classes = list(bundle.xgboost_classes)
            if predicted_class in classes:
                class_index = classes.index(predicted_class)
        top = _explain(bundle, X_scaled, class_index=class_index)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Explanation unavailable: {exc}") from exc
    return {
        "event_id": event_id,
        "predicted_class": predicted_class,
        "is_abnormal": row.get("is_abnormal") if row else None,
        "top_features": top,
    }


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket) -> None:
    await ws_manager.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:  # noqa: BLE001
        ws_manager.disconnect(websocket)


# Serve the built dashboard (single-command demo: server + UI in one).
if _DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_DIST / "assets"), name="assets")
    
    _FIGURES = Path(__file__).resolve().parent.parent / "docs" / "figures"
    if _FIGURES.is_dir():
        app.mount("/figures", StaticFiles(directory=_FIGURES), name="figures")

    _REPORT_FIGURES = Path(__file__).resolve().parent.parent / "reports" / "figures"
    if _REPORT_FIGURES.is_dir():
        app.mount("/report_figures", StaticFiles(directory=_REPORT_FIGURES), name="report_figures")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_DIST / "index.html")


if __name__ == "__main__":
    import uvicorn

    print("\n  ECG demo server → http://localhost:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
