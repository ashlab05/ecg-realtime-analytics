"""Tests for the autoencoder, threshold logic, and two-stage prediction."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_core.features import N_FEATURES
from ecg_core.labels import CLASS_NAMES
from ecg_core.models.autoencoder import ECGAutoencoder, load_autoencoder, load_threshold
from inference.anomaly_detector import detect_anomalies, single_anomaly
from inference.predictor import predict_features


def test_autoencoder_shapes():
    model = ECGAutoencoder()
    x = torch.randn(4, N_FEATURES)
    out = model(x)
    assert out.shape == (4, N_FEATURES)
    assert model.encode(x).shape == (4, 16)
    assert model.latent(x).shape == (4, 16)


def test_reconstruction_error_is_mse():
    X = np.array([[1.0, 2.0, 3.0, 4.0]])
    Xr = np.array([[1.0, 2.0, 3.0, 5.0]])
    err = ECGAutoencoder.reconstruction_error(X, Xr)
    assert err.shape == (1,)
    assert err[0] == pytest.approx(0.25)  # mean of [0,0,0,1]


def test_threshold_persists(models_dir):
    threshold = load_threshold(models_dir / "autoencoder")
    meta_threshold = load_autoencoder(models_dir / "autoencoder")  # loads without error
    assert isinstance(threshold, float)
    assert threshold > 0
    assert meta_threshold is not None


def test_anomaly_detector_normal_below_threshold(test_bundle, synthetic_df):
    """Scaled normal samples from the training distribution score below the threshold."""
    from ecg_core.preprocessing import frame_to_matrix

    X = frame_to_matrix(synthetic_df[synthetic_df["type"] == "N"])
    X_scaled = test_bundle.transform(X)
    result = detect_anomalies(test_bundle, X_scaled)
    # The synthetic normal cluster is compact; most samples must pass Stage 1.
    assert result.is_normal.mean() > 0.5
    assert (result.reconstruction_error <= result.threshold).sum() == result.is_normal.sum()


def test_anomaly_detector_shifted_is_abnormal(test_bundle):
    """Far-shifted feature vectors must exceed the reconstruction threshold."""
    X = np.full((3, N_FEATURES), 25.0)
    X_scaled = test_bundle.transform(X)
    result = detect_anomalies(test_bundle, X_scaled)
    assert not result.is_normal.any()
    assert (result.reconstruction_error > result.threshold).all()


def test_single_anomaly_wrapper(test_bundle):
    x = np.zeros(N_FEATURES)
    err, is_normal, conf = single_anomaly(test_bundle, test_bundle.transform(x.reshape(1, -1))[0])
    assert err >= 0
    assert 0.0 <= conf <= 1.0
    assert isinstance(is_normal, bool)


def test_predict_features_end_to_end(test_bundle, synthetic_df):
    """Two-stage pipeline returns a complete prediction for each class."""
    from ecg_core.preprocessing import frame_to_matrix

    for cls in CLASS_NAMES:
        row = synthetic_df[synthetic_df["type"] == cls].iloc[0]
        features = frame_to_matrix(pd_row_to_df(row))
        result = predict_features(
            features, bundle=test_bundle, event_id="test-123",
            record_id="100", true_label=cls, explain=True,
        )
        assert result.event_id == "test-123"
        assert result.predicted_class in list(CLASS_NAMES) + ["UNKNOWN"]
        assert result.is_abnormal == (result.predicted_class != "N")
        assert result.reconstruction_error >= 0
        assert result.anomaly_threshold > 0
        assert result.latency_ms >= 0
        assert result.confidence > 0
        # Explanations only for Stage-2 (abnormal) predictions.
        if result.stage == "stage2_xgboost":
            assert result.top_features is not None and len(result.top_features) > 0


def test_predict_features_wrong_dimension(test_bundle):
    with pytest.raises(ValueError, match="expected"):
        predict_features(np.zeros(31), bundle=test_bundle)


def pd_row_to_df(row):
    import pandas as pd

    return pd.DataFrame([row])
