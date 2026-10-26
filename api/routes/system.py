"""System / health / model status endpoints."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.schemas import (
    ComponentStatus,
    MetricsOut,
    SystemStatusOut,
    ThresholdOut,
)
from api.services import broker
from api.services.kafka_service import latency_tracker
from ecg_core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["system"])

_START_TIME = time.time()


def _spark_status() -> ComponentStatus:
    """Spark is 'active' if its batch metrics log was written recently."""
    try:
        metrics_path = settings.spark_stream_batch_log
        if not metrics_path.exists():
            return ComponentStatus(status="unknown", detail="No batch metrics yet (Spark job not started?)")
        age = time.time() - metrics_path.stat().st_mtime
        if age < 120:
            return ComponentStatus(status="up", detail=f"Last micro-batch {age:.0f}s ago")
        return ComponentStatus(status="down", detail=f"Last batch metrics {age:.0f}s ago")
    except OSError:
        return ComponentStatus(status="unknown")


def _database_status() -> ComponentStatus:
    try:
        from database.models import get_engine
        from sqlalchemy import text

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return ComponentStatus(status="up")
    except Exception as exc:  # noqa: BLE001
        return ComponentStatus(status="down", detail=str(exc)[:200])


def _models_status() -> ComponentStatus:
    from inference.model_loader import bundle_status

    status = bundle_status()
    parts = {k: v["ready"] for k, v in status.items() if isinstance(v, dict) and "ready" in v}
    if all(parts.values()):
        return ComponentStatus(status="up", detail="autoencoder + xgboost + preprocessing ready")
    missing = [k for k, ready in parts.items() if not ready]
    return ComponentStatus(status="degraded", detail=f"Not trained yet: {missing}")


@router.get("/system/status", response_model=SystemStatusOut)
def system_status() -> SystemStatusOut:
    # The 'kafka' slot reports the *active* broker (MESSAGE_BROKER env),
    # so the dashboard stays unchanged when running on Redis Streams.
    if settings.use_redis:
        broker_detail = f"{settings.redis_url} (Redis Streams)"
    else:
        broker_detail = settings.kafka_bootstrap_servers
    return SystemStatusOut(
        api=ComponentStatus(status="up", detail=f"uptime {time.time() - _START_TIME:.0f}s"),
        kafka=ComponentStatus(
            status="up" if broker.broker_available() else "down",
            detail=broker_detail,
        ),
        spark=_spark_status(),
        database=_database_status(),
        models=_models_status(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/models/status")
def models_status() -> dict:
    from inference.model_loader import bundle_status

    return bundle_status()


@router.get("/models/experiments")
def model_experiments() -> dict:
    """Offline experiment results (A-E ablations) from logs/training/experiments/.

    Returns each experiment's metrics plus a canonical ordering and
    human-readable labels so the dashboard can render the comparison
    without hardcoding anything.
    """
    import json

    LABELS = {
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
    experiments: list[dict] = []
    if exp_dir.exists():
        for path in sorted(exp_dir.glob("*.json")):
            if path.name == "comparison_summary.json":
                continue
            try:
                payload = json.loads(path.read_text())
            except (OSError, ValueError):
                logger.warning("Could not parse experiment file %s", path)
                continue
            name = payload.get("experiment", path.stem)
            experiments.append(
                {
                    "name": name,
                    "label": LABELS.get(name, name),
                    "extra": payload.get("extra", {}),
                    "metrics": payload.get("metrics", {}),
                }
            )

    order = {name: i for i, name in enumerate(LABELS)}
    experiments.sort(key=lambda e: (order.get(e["name"], 99), e["name"]))
    return {"experiments": experiments, "directory": str(exp_dir)}


@router.get("/models/metaheuristics")
def metaheuristics_summary() -> dict:
    """Summary and complete metrics for Baseline, PSO, GWO, and COA Autoencoders."""
    import json
    from pathlib import Path
    summary_path = Path(__file__).resolve().parent.parent.parent / "reports" / "tables" / "all_experiments_summary.json"
    if summary_path.exists():
        try:
            return json.loads(summary_path.read_text())
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {}


@router.get("/threshold", response_model=ThresholdOut)
def get_threshold() -> ThresholdOut:
    try:
        from ecg_core.models.autoencoder import load_threshold

        threshold = load_threshold()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    percentile = None
    meta_path = settings.autoencoder_dir / "autoencoder_meta.json"
    if meta_path.exists():
        import json

        percentile = json.loads(meta_path.read_text()).get("threshold_percentile")
    return ThresholdOut(anomaly_threshold=threshold, threshold_percentile=percentile)


@router.get("/metrics", response_model=MetricsOut)
def metrics(window_seconds: int = 60) -> MetricsOut:
    """Live inference metrics over an in-memory rolling window."""
    snapshot = latency_tracker.snapshot(window_seconds=window_seconds)
    return MetricsOut(
        total_events=snapshot["total_events"],
        latency_ms=snapshot["latency_ms"],
        throughput_events_per_second=snapshot["events_per_second"],
        class_distribution=snapshot["class_distribution"],
        window_seconds=window_seconds,
    )
