"""Prediction history, statistics, and explanation endpoints."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from api.schemas import ExplanationOut, PredictionOut, StatisticsOut
from api.services.kafka_service import latency_tracker
from api.services.simulation import feature_cache

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["predictions"])


def _row_to_out(row) -> PredictionOut:
    return PredictionOut(
        event_id=row.event_id,
        timestamp=row.timestamp.isoformat() if row.timestamp else None,
        record_id=row.record_id,
        predicted_class=row.predicted_class,
        is_abnormal=row.is_abnormal,
        confidence=row.confidence,
        reconstruction_error=row.reconstruction_error,
        anomaly_threshold=row.anomaly_threshold,
        inference_latency_ms=row.inference_latency_ms,
        processing_time_ms=row.processing_time_ms,
        source=row.source,
        true_label=row.true_label,
        stage=row.stage,
    )


@router.get("/predictions/latest", response_model=PredictionOut)
def latest_prediction() -> PredictionOut:
    from database.models import latest_predictions, session_scope

    try:
        with session_scope() as session:
            rows = latest_predictions(session, limit=1)
    except Exception as exc:  # noqa: BLE001 - DB may be down
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions stored yet.")
    return _row_to_out(rows[0])


@router.get("/predictions", response_model=list[PredictionOut])
def list_predictions(
    limit: int = Query(50, ge=1, le=1000),
    predicted_class: Optional[str] = None,
    abnormal_only: bool = False,
) -> list[PredictionOut]:
    from sqlalchemy import select

    from database.models import ECGPrediction, session_scope

    try:
        with session_scope() as session:
            stmt = select(ECGPrediction).order_by(ECGPrediction.timestamp.desc()).limit(limit)
            if predicted_class:
                stmt = stmt.where(ECGPrediction.predicted_class == predicted_class.upper())
            if abnormal_only:
                stmt = stmt.where(ECGPrediction.is_abnormal)
            rows = list(session.execute(stmt).scalars())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc
    return [_row_to_out(r) for r in rows]


@router.get("/statistics", response_model=StatisticsOut)
def statistics() -> StatisticsOut:
    from database.models import compute_statistics, session_scope

    try:
        with session_scope() as session:
            stats = compute_statistics(session)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Database unavailable: {exc}") from exc

    live = latency_tracker.snapshot()
    return StatisticsOut(
        total_events=stats["total_events"],
        normal_events=stats["normal_events"],
        abnormal_events=stats["abnormal_events"],
        abnormal_rate=round(stats["abnormal_rate"], 4),
        average_latency_ms=stats["average_latency_ms"],
        class_distribution=stats["class_distribution"],
        events_per_second=live["events_per_second"],
        latency=live["latency_ms"],
    )


@router.get("/explain/{event_id}", response_model=ExplanationOut)
def explain_prediction(event_id: str) -> ExplanationOut:
    """SHAP feature-contribution explanation for a recent event.

    Feature vectors are cached in memory when the API publishes the
    event, so explanations are available for events sent through this
    service (single sends and simulations), not for external producers.
    """
    features = feature_cache.get(event_id)
    if features is None:
        raise HTTPException(
            status_code=404,
            detail="Event not found in the explanation cache "
            "(only recent events sent through this API can be explained).",
        )

    from database.models import get_prediction_by_event_id, session_scope

    predicted_class = None
    is_abnormal = None
    try:
        with session_scope() as session:
            row = get_prediction_by_event_id(session, event_id)
            if row:
                predicted_class, is_abnormal = row.predicted_class, row.is_abnormal
    except Exception:  # noqa: BLE001 - DB down should not block explanations
        logger.warning("Could not look up prediction %s in DB for explanation", event_id)

    from inference.model_loader import get_model_bundle
    from inference.predictor import explain_prediction as _explain

    try:
        bundle = get_model_bundle()
        X_scaled = bundle.transform(features.reshape(1, -1))
        # Explain for the class actually predicted for this event, when known.
        class_index = None
        if predicted_class and getattr(bundle, "xgboost_classes", None):
            classes = list(bundle.xgboost_classes)
            if predicted_class in classes:
                class_index = classes.index(predicted_class)
        top = _explain(bundle, X_scaled, class_index=class_index)
    except Exception as exc:  # noqa: BLE001 - model not ready, etc.
        raise HTTPException(status_code=503, detail=f"Explanation unavailable: {exc}") from exc

    return ExplanationOut(
        event_id=event_id,
        predicted_class=predicted_class,
        is_abnormal=is_abnormal,
        top_features=top,
    )
