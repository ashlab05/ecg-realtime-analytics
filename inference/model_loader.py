"""Model loading: loads every artifact once and keeps it in a singleton.

All components (FastAPI, Spark executors, batch scripts) obtain models
through :func:`get_model_bundle` so that weights are deserialized a
single time per process — never per prediction.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import xgboost as xgb

from ecg_core.config import settings
from ecg_core.features import FEATURE_COLUMNS
from ecg_core.models.autoencoder import ECGAutoencoder, load_autoencoder, load_threshold
from ecg_core.preprocessing import PreprocessingArtifacts, load_artifacts

logger = logging.getLogger(__name__)


@dataclass
class ModelBundle:
    """All trained artifacts required for two-stage inference."""

    autoencoder: ECGAutoencoder
    scaler: object                      # sklearn StandardScaler
    feature_columns: list
    anomaly_threshold: float
    threshold_percentile: float
    xgboost: Optional[xgb.XGBClassifier]
    xgboost_classes: Optional[list]     # classes of the Stage-2 model

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale raw feature vectors into the model's standardized space."""
        return self.scaler.transform(np.asarray(X, dtype=np.float64))


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before models are trained/loaded."""


_bundle: Optional[ModelBundle] = None
_lock = threading.Lock()


def load_bundle(models_dir: Optional[Path] = None) -> ModelBundle:
    """Load all artifacts from disk (explicit, re-readable for tests)."""
    mdir = Path(models_dir) if models_dir else settings.models_dir

    # --- Preprocessing (scaler + metadata) --------------------------------
    artifacts: PreprocessingArtifacts = load_artifacts(mdir / "preprocessing")

    # --- Stage 1: autoencoder + threshold ----------------------------------
    ae_dir = mdir / "autoencoder"
    autoencoder = load_autoencoder(ae_dir)
    ae_meta = json.loads((ae_dir / "autoencoder_meta.json").read_text())
    threshold = float(ae_meta["threshold"])
    percentile = float(ae_meta.get("threshold_percentile", settings.ae_threshold_percentile))

    # --- Stage 2: XGBoost abnormal classifier (optional) -------------------
    xgb_path = mdir / "xgboost" / "xgboost_model.json"
    xgb_meta_path = mdir / "xgboost" / "xgboost_meta.json"
    xgb_model, xgb_classes = None, None
    if xgb_path.exists() and xgb_meta_path.exists():
        meta = json.loads(xgb_meta_path.read_text())
        if meta.get("mode") == "abnormal":
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(xgb_path)
            xgb_classes = meta["classes"]
            logger.info("Loaded Stage-2 XGBoost (classes=%s)", xgb_classes)
        else:
            logger.warning(
                "XGBoost model at %s was trained with mode=%s (expected 'abnormal'); "
                "Stage 2 unavailable — abnormal samples will be labeled 'UNKNOWN'.",
                xgb_path, meta.get("mode"),
            )
    else:
        logger.warning(
            "Stage-2 XGBoost model not found at %s — abnormal samples will be "
            "labeled 'UNKNOWN'. Run `python training/train_xgboost.py --mode abnormal`.",
            xgb_path,
        )

    logger.info(
        "Model bundle loaded: %d features, threshold=%.6f (p%g)",
        len(artifacts.feature_columns), threshold, percentile,
    )
    return ModelBundle(
        autoencoder=autoencoder,
        scaler=artifacts.scaler,
        feature_columns=artifacts.feature_columns,
        anomaly_threshold=threshold,
        threshold_percentile=percentile,
        xgboost=xgb_model,
        xgboost_classes=xgb_classes,
    )


def get_model_bundle() -> ModelBundle:
    """Thread-safe singleton accessor — loads once per process."""
    global _bundle
    with _lock:
        if _bundle is None:
            try:
                _bundle = load_bundle()
            except FileNotFoundError as exc:
                raise ModelNotReadyError(
                    f"{exc} Models must be trained before inference. "
                    "Run the training pipeline (see README) or `make train`."
                ) from exc
        return _bundle


def reset_bundle() -> None:
    """Force re-loading on next access (used by tests and model refresh)."""
    global _bundle
    with _lock:
        _bundle = None


def bundle_status() -> dict:
    """Human-readable status of each artifact for the /api/models/status endpoint."""
    def _exists(rel: str) -> bool:
        return (settings.models_dir / rel).exists()

    status = {
        "preprocessing": {
            "ready": _exists("preprocessing/scaler.joblib"),
            "path": str(settings.models_dir / "preprocessing"),
        },
        "autoencoder": {
            "ready": _exists("autoencoder/autoencoder.pt"),
            "threshold": load_threshold() if _exists("autoencoder/autoencoder_meta.json") else None,
        },
        "xgboost_abnormal": {
            "ready": _exists("xgboost/xgboost_model.json"),
        },
        "feature_columns": FEATURE_COLUMNS,
    }
    return status
