"""Two-stage predictor: autoencoder anomaly gate -> XGBoost arrhythmia classifier.

Stage 1  — the autoencoder scores each heartbeat's reconstruction error
           against the calibrated threshold; beats below the threshold
           are classified Normal.
Stage 2  — beats above the threshold are passed to the XGBoost model
           trained on abnormal classes (SVEB / VEB / F / Q).

Also provides SHAP-based feature-contribution explanations for Stage-2
decisions ("model feature contribution explanation" — not clinical
interpretability).
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ecg_core.config import settings
from ecg_core.features import FEATURE_DISPLAY_NAMES
from ecg_core.labels import NORMAL_CLASS
from inference.anomaly_detector import detect_anomalies
from inference.model_loader import ModelBundle, get_model_bundle

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "UNKNOWN"


@dataclass
class PredictionResult:
    """Serializable prediction for one ECG event."""

    event_id: str
    predicted_class: str
    is_abnormal: bool
    confidence: float
    reconstruction_error: float
    anomaly_threshold: float
    latency_ms: float
    timestamp: str
    record_id: Optional[str] = None
    true_label: Optional[str] = None          # provided only for simulated/replay events
    stage: str = "stage1_normal"              # stage1_normal | stage2_xgboost | unknown
    stage2_probability: Optional[float] = None
    top_features: Optional[List[Dict]] = field(default=None)  # SHAP explanation (optional)

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def predict_features(
    features: np.ndarray,
    bundle: Optional[ModelBundle] = None,
    event_id: Optional[str] = None,
    record_id: Optional[str] = None,
    true_label: Optional[str] = None,
    explain: bool = False,
) -> PredictionResult:
    """Run two-stage inference on a raw (unscaled) 32-feature vector."""
    bundle = bundle or get_model_bundle()
    event_id = event_id or str(uuid.uuid4())
    started = time.perf_counter()

    x = np.asarray(features, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != len(bundle.feature_columns):
        raise ValueError(
            f"Feature vector has {x.shape[1]} features, expected {len(bundle.feature_columns)} "
            f"in canonical order (see ecg_core.features.FEATURE_COLUMNS)."
        )

    x_scaled = bundle.transform(x)

    # ---- Stage 1: anomaly detection --------------------------------------
    anomaly = detect_anomalies(bundle, x_scaled)
    error = float(anomaly.reconstruction_error[0])

    if anomaly.is_normal[0]:
        predicted_class = NORMAL_CLASS
        confidence = float(anomaly.confidence[0])
        stage = "stage1_normal"
        stage2_prob = None
        top_features = None
    else:
        # ---- Stage 2: XGBoost abnormal classification ---------------------
        stage = "stage2_xgboost"
        if bundle.xgboost is not None:
            pred_idx = bundle.xgboost.predict(x_scaled)[0]
            probs = bundle.xgboost.predict_proba(x_scaled)[0]
            predicted_class = str(bundle.xgboost_classes[int(pred_idx)])
            stage2_prob = float(probs[int(pred_idx)])
            # Overall confidence blends the AE margin confidence and the
            # Stage-2 class probability.
            confidence = float(0.5 * (anomaly.confidence[0] + stage2_prob))
            top_features = (
                explain_prediction(bundle, x_scaled, class_index=int(pred_idx)) if explain else None
            )
        else:
            predicted_class = UNKNOWN_LABEL
            confidence = float(anomaly.confidence[0])
            stage2_prob = None
            top_features = None

    latency_ms = (time.perf_counter() - started) * 1000.0

    return PredictionResult(
        event_id=event_id,
        predicted_class=predicted_class,
        is_abnormal=predicted_class not in (NORMAL_CLASS, UNKNOWN_LABEL),
        confidence=round(confidence, 4),
        reconstruction_error=round(error, 6),
        anomaly_threshold=round(bundle.anomaly_threshold, 6),
        latency_ms=round(latency_ms, 3),
        timestamp=datetime.now(timezone.utc).isoformat(),
        record_id=record_id,
        true_label=true_label,
        stage=stage,
        stage2_probability=round(stage2_prob, 4) if stage2_prob is not None else None,
        top_features=top_features,
    )


def explain_prediction(
    bundle: ModelBundle,
    x_scaled: np.ndarray,
    top_k: int = 5,
    class_index: Optional[int] = None,
) -> List[Dict]:
    """SHAP feature-contribution explanation for a Stage-2 decision.

    Uses the XGBoost native SHAP (pred_contribs) implementation — no
    separate shap package required at inference time. This is a *model
    feature contribution explanation*, not a clinical interpretation.

    ``class_index`` selects which class of the multiclass Stage-2 model the
    contributions are computed for (the predicted class). When None, the
    per-class contributions are summed.
    """
    if bundle.xgboost is None:
        return []
    booster = bundle.xgboost.get_booster()
    import xgboost as xgb

    dmatrix = xgb.DMatrix(x_scaled)
    contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
    # Binary/regression: (n, n_features + 1); multiclass: (n, n_classes, n_features + 1).
    if contribs.ndim == 3:
        if class_index is not None:
            contribs = contribs[0, int(class_index)]
        else:
            contribs = contribs[0].sum(axis=0)
    elif contribs.ndim == 2:
        contribs = contribs[0]
    contribs = contribs[:-1]  # drop expected-value bias term

    order = np.argsort(-np.abs(contribs))[:top_k]
    return [
        {
            "feature": FEATURE_DISPLAY_NAMES.get(bundle.feature_columns[i], bundle.feature_columns[i]),
            "column": bundle.feature_columns[i],
            "shap_value": round(float(contribs[i]), 5),
        }
        for i in order
    ]


class InferenceLogger:
    """Append-only JSONL inference log (logs/inference/inference_log.jsonl)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else (settings.inference_logs_dir / "inference_log.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, result: PredictionResult) -> None:
        try:
            with self.path.open("a") as f:
                f.write(result.to_json() + "\n")
        except OSError:
            logger.exception("Failed to append to inference log %s", self.path)
