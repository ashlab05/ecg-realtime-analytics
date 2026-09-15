"""Stage 1: anomaly detection via autoencoder reconstruction error."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from inference.model_loader import ModelBundle


@dataclass
class AnomalyResult:
    """Outcome of Stage 1 for one or more samples."""

    reconstruction_error: np.ndarray   # per-sample MSE
    threshold: float
    is_normal: np.ndarray              # boolean mask
    confidence: np.ndarray             # distance-normalized confidence in the decision


def detect_anomalies(bundle: ModelBundle, X_scaled: np.ndarray) -> AnomalyResult:
    """Run Stage-1 anomaly detection on already-scaled feature vectors."""
    X = np.asarray(X_scaled, dtype=np.float64)
    recon = bundle.autoencoder.reconstruct(X)
    error = np.mean((X - recon.astype(np.float64)) ** 2, axis=1)

    threshold = bundle.anomaly_threshold
    is_normal = error <= threshold

    # Decision confidence: how far the error is from the threshold,
    # normalized by the threshold itself. 0.5 = exactly at the boundary.
    margin = np.abs(error - threshold) / max(threshold, 1e-12)
    confidence = np.clip(0.5 + 0.5 * np.tanh(margin), 0.0, 1.0)

    return AnomalyResult(
        reconstruction_error=error,
        threshold=float(threshold),
        is_normal=is_normal,
        confidence=confidence,
    )


def single_anomaly(bundle: ModelBundle, x_scaled: np.ndarray) -> Tuple[float, bool, float]:
    """Convenience wrapper for one sample: (error, is_normal, confidence)."""
    result = detect_anomalies(bundle, x_scaled.reshape(1, -1))
    return (
        float(result.reconstruction_error[0]),
        bool(result.is_normal[0]),
        float(result.confidence[0]),
    )
