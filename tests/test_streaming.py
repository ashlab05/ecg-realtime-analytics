"""Tests for the streaming layer: event validation, payload conversion, producer."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ecg_core.features import FEATURE_COLUMNS, N_FEATURES
from streaming.feature_pipeline import (
    EventValidationError,
    error_event,
    event_to_feature_vector,
    prediction_event,
    validate_event,
)
from streaming.kafka_producer import DatasetSampler, row_to_event_payload
from tests.conftest import make_synthetic_df


def _base_event():
    lead = {
        "pre_rr": 0.76, "post_rr": 0.81,
        "p_peak": 0.12, "t_peak": 0.35, "r_peak": 1.10, "s_peak": -0.20, "q_peak": -0.15,
        "qrs_interval": 94, "pq_interval": 18, "qt_interval": 66, "st_interval": 7,
        "qrs_morph_0": 0.1, "qrs_morph_1": 0.2, "qrs_morph_2": 0.15, "qrs_morph_3": 0.05,
        "qrs_morph_4": 0.08,
    }
    return {
        "event_id": "evt-1",
        "timestamp": "2026-09-15T10:00:00+00:00",
        "record_id": "100",
        "true_label": "N",
        "lead_ii": dict(lead),
        "lead_v5": dict(lead, r_peak=0.95),
    }


def test_validate_event_ok():
    validated = validate_event(_base_event())
    assert validated.event_id == "evt-1"
    assert validated.features.shape == (N_FEATURES,)
    assert all(np.isfinite(validated.features))


def test_validate_event_feature_order():
    """Flattened feature vector must follow the canonical training order."""
    event = _base_event()
    event["lead_ii"]["r_peak"] = 1.234
    validated = validate_event(event)
    r_peak_ii_index = FEATURE_COLUMNS.index("0_rPeak")
    assert validated.features[r_peak_ii_index] == pytest.approx(1.234)


def test_validate_event_missing_field():
    event = _base_event()
    del event["lead_v5"]["qrs_morph_3"]
    with pytest.raises(EventValidationError, match="lead_v5.qrs_morph_3"):
        validate_event(event)


def test_validate_event_non_numeric():
    event = _base_event()
    event["lead_ii"]["r_peak"] = "high"
    with pytest.raises(EventValidationError, match="numeric"):
        validate_event(event)


def test_validate_event_non_finite():
    event = _base_event()
    event["lead_ii"]["r_peak"] = float("inf")
    with pytest.raises(EventValidationError, match="not finite"):
        validate_event(event)


def test_validate_event_not_an_object():
    with pytest.raises(EventValidationError, match="JSON object"):
        validate_event([1, 2, 3])


def test_event_to_feature_vector():
    vec = event_to_feature_vector(_base_event())
    assert vec.shape == (N_FEATURES,)


def test_row_to_event_payload_roundtrip(synthetic_df):
    """A dataset row -> JSON payload -> validated feature vector must round-trip."""
    row = synthetic_df.iloc[0]
    event = row_to_event_payload(row)
    validated = validate_event(event)
    original = row[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    np.testing.assert_allclose(validated.features, original, rtol=1e-6)


def test_error_event_serialization():
    msg = error_event({"bad": "payload"}, "some failure")
    parsed = json.loads(msg)
    assert parsed["error"] == "event_validation_failed"
    assert "some failure" in parsed["message"]


def test_prediction_event_serialization():
    msg = json.loads(prediction_event({"event_id": "e1", "predicted_class": "VEB"}, source="test"))
    assert msg["predicted_class"] == "VEB"
    assert msg["source"] == "test"


def test_dataset_sampler(synthetic_df):
    sampler = DatasetSampler(df=synthetic_df)
    counts = sampler.available_classes()
    assert counts["N"] > 0 and counts["VEB"] > 0

    event = sampler.sample_by_mode("normal")
    assert event["true_label"] == "N"

    event = sampler.sample_by_mode("abnormal")
    assert event["true_label"] in ("SVEB", "VEB", "F", "Q")

    # Every sampled event must pass validation.
    validate_event(event)


def test_dataset_sampler_specific_class(synthetic_df):
    sampler = DatasetSampler(df=synthetic_df)
    event = sampler.sample("VEB")
    assert event["true_label"] == "VEB"


def test_dataset_sampler_unknown_class(synthetic_df):
    sampler = DatasetSampler(df=synthetic_df)
    with pytest.raises(ValueError):
        sampler.sample("Z")
