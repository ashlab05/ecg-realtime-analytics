"""ECG event ingestion and simulation endpoints."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ECGSendResponse,
    ECGEventPayload,
    SimulationStartRequest,
    SimulationStatus,
)
from api.services import broker
from api.services.simulation import (
    feature_cache,
    get_simulation_engine,
    send_direct_prediction,
)
from ecg_core.config import settings
from ecg_core.labels import CLASS_DESCRIPTIONS, CLASS_NAMES
from streaming.feature_pipeline import EventValidationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ecg"])


def _payload_to_event(payload: ECGEventPayload) -> dict:
    """Pydantic payload -> the dict shape published to Kafka."""
    return {
        "event_id": payload.event_id or str(uuid4()),
        "timestamp": (payload.timestamp or datetime.now(timezone.utc)).isoformat(),
        "record_id": payload.record_id,
        "true_label": payload.true_label,
        "lead_ii": payload.lead_ii.model_dump(),
        "lead_v5": payload.lead_v5.model_dump(),
    }


@router.get("/ecg/classes")
def ecg_classes() -> dict:
    """Classes available in the dataset (for dashboard dropdowns)."""
    engine = get_simulation_engine()
    try:
        counts = engine.available_classes()
    except FileNotFoundError:
        counts = {c: 0 for c in CLASS_NAMES}
    return {
        "classes": [
            {"code": c, "description": CLASS_DESCRIPTIONS[c], "samples": counts.get(c, 0)}
            for c in CLASS_NAMES
        ]
    }


@router.get("/ecg/sample")
def ecg_sample(mode: str = "random") -> dict:
    """Sample one real heartbeat from the dataset as a JSON event payload.

    mode: normal | abnormal | random | N | SVEB | VEB | F | Q
    """
    engine = get_simulation_engine()
    try:
        return engine.sample_event(mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Dataset unavailable: {exc}") from exc


@router.post("/ecg/send", response_model=ECGSendResponse)
def send_ecg_event(payload: ECGEventPayload) -> ECGSendResponse:
    """Publish one ECG event to the message broker (Kafka or Redis Streams).

    Falls back to direct in-process inference when the broker is
    unavailable, so local development works without the full stack.
    The full streaming path (broker -> streaming worker -> predictions)
    is used whenever the broker is reachable.
    """
    event = _payload_to_event(payload)
    event_id = event["event_id"]

    if broker.broker_available():
        try:
            _cache_features(event)
            broker.publish_event(event)
            return ECGSendResponse(
                event_id=event_id,
                published_to_kafka=True,
                topic=settings.kafka_topic_events,
                message=(
                    f"Event published to {broker.broker_name()}; "
                    "prediction will arrive via /ws/predictions."
                ),
            )
        except EventValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - broker errors fall back below
            logger.warning("%s publish failed (%s); using direct inference fallback",
                           broker.broker_name(), exc)

    # --- Direct fallback (no broker) ---------------------------------------
    try:
        prediction = send_direct_prediction(event)
        _persist_and_broadcast(prediction)
        return ECGSendResponse(
            event_id=prediction["event_id"],
            published_to_kafka=False,
            topic=settings.kafka_topic_events,
            message=(
                f"{broker.broker_name().capitalize()} unavailable — prediction computed "
                "in-process (direct mode) and broadcast to WebSocket clients."
            ),
            prediction=prediction,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Inference failed: {exc}") from exc


def _persist_and_broadcast(prediction: dict) -> None:
    """Best-effort persistence + WebSocket broadcast for direct-mode predictions."""
    from api.services.kafka_service import latency_tracker
    from api.services.websocket_manager import ws_manager

    ws_manager.broadcast_sync({"type": "prediction", "data": prediction})
    if prediction.get("latency_ms") is not None:
        latency_tracker.record(float(prediction["latency_ms"]), prediction.get("predicted_class", "?"))
    try:
        from database.models import insert_prediction, session_scope

        with session_scope() as session:
            insert_prediction(session, prediction)
    except Exception:  # noqa: BLE001 - DB may be down; the broadcast already happened
        logger.warning("Could not persist direct-mode prediction to PostgreSQL")


def _cache_features(event: dict):
    from streaming.feature_pipeline import validate_event

    validated = validate_event(event)
    feature_cache.put(validated.event_id, validated.features)
    return validated


@router.post("/simulation/start", response_model=SimulationStatus)
async def start_simulation(request: SimulationStartRequest) -> SimulationStatus:
    engine = get_simulation_engine()
    if not broker.broker_available():
        raise HTTPException(
            status_code=503,
            detail=(
                f"{broker.broker_name().capitalize()} is not reachable — the streaming "
                "simulation requires the full stack. Start the broker "
                "(docker compose up) or send single events (direct mode)."
            ),
        )
    try:
        await engine.start(
            mode=request.mode,
            events_per_second=request.events_per_second,
            duration_seconds=request.duration_seconds,
            n_events=request.n_events,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return SimulationStatus(**engine.status)


@router.post("/simulation/stop", response_model=SimulationStatus)
def stop_simulation() -> SimulationStatus:
    engine = get_simulation_engine()
    engine.stop()
    return SimulationStatus(**engine.status)


@router.get("/simulation/status", response_model=SimulationStatus)
def simulation_status() -> SimulationStatus:
    return SimulationStatus(**get_simulation_engine().status)
