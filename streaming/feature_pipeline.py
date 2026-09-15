"""Feature pipeline: ECG event validation and feature extraction.

Used by the Spark streaming job (and reusable anywhere raw JSON events
need to become validated 32-dim feature vectors). Deliberately free of
heavy dependencies so it can run inside Spark executors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional
from uuid import uuid4

import numpy as np

from ecg_core.features import (
    LEAD_FEATURE_STEMS,
    LEAD_II_PAYLOAD_KEY,
    LEAD_V_PAYLOAD_KEY,
    N_FEATURES,
    STEM_TO_PAYLOAD_KEY,
)


class EventValidationError(ValueError):
    """Raised when an incoming ECG event does not match the event schema."""


@dataclass
class ValidatedEvent:
    """A validated ECG event ready for inference."""

    event_id: str
    timestamp: str
    record_id: Optional[str]
    true_label: Optional[str]
    features: np.ndarray  # raw (unscaled), canonical feature order, shape (32,)

    def feature_dict(self) -> Dict[str, float]:
        from ecg_core.features import FEATURE_COLUMNS

        return dict(zip(FEATURE_COLUMNS, self.features.tolist()))


def _validate_lead(lead_payload: Dict, lead_key: str) -> Dict[str, float]:
    if not isinstance(lead_payload, dict):
        raise EventValidationError(f"'{lead_key}' must be an object with {len(LEAD_FEATURE_STEMS)} numeric fields")
    values: Dict[str, float] = {}
    for stem in LEAD_FEATURE_STEMS:
        key = STEM_TO_PAYLOAD_KEY[stem]
        if key not in lead_payload:
            raise EventValidationError(f"Missing field '{lead_key}.{key}'")
        value = lead_payload[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EventValidationError(f"Field '{lead_key}.{key}' must be numeric, got {type(value).__name__}")
        value = float(value)
        if not np.isfinite(value):
            raise EventValidationError(f"Field '{lead_key}.{key}' is not finite ({value})")
        values[key] = value
    return values


def validate_event(event: Dict) -> ValidatedEvent:
    """Validate an ECG event dict and extract the raw feature vector.

    Raises EventValidationError with a human-readable message on any problem.
    """
    if not isinstance(event, dict):
        raise EventValidationError("Event must be a JSON object")

    event_id = event.get("event_id") or str(uuid4())
    if not isinstance(event_id, str) or not event_id.strip():
        raise EventValidationError("'event_id' must be a non-empty string")

    timestamp = event.get("timestamp") or datetime.now(timezone.utc).isoformat()
    if not isinstance(timestamp, str):
        raise EventValidationError("'timestamp' must be an ISO-8601 string")

    record_id = event.get("record_id")
    if record_id is not None and not isinstance(record_id, (str, int)):
        raise EventValidationError("'record_id' must be a string or integer")

    true_label = event.get("true_label")
    if true_label is not None and not isinstance(true_label, str):
        raise EventValidationError("'true_label' must be a string")

    lead_ii = _validate_lead(event.get(LEAD_II_PAYLOAD_KEY), LEAD_II_PAYLOAD_KEY)
    lead_v = _validate_lead(event.get(LEAD_V_PAYLOAD_KEY), LEAD_V_PAYLOAD_KEY)

    features = np.array(
        [lead_ii[STEM_TO_PAYLOAD_KEY[s]] for s in LEAD_FEATURE_STEMS]
        + [lead_v[STEM_TO_PAYLOAD_KEY[s]] for s in LEAD_FEATURE_STEMS],
        dtype=np.float64,
    )
    assert features.shape == (N_FEATURES,)

    return ValidatedEvent(
        event_id=event_id,
        timestamp=timestamp,
        record_id=str(record_id) if record_id is not None else None,
        true_label=true_label,
        features=features,
    )


def event_to_feature_vector(event: Dict) -> np.ndarray:
    """Convenience: validate + return just the raw feature vector."""
    return validate_event(event).features


def prediction_event(
    prediction: Dict,
    source: str = "kafka",
    processing_time_ms: Optional[float] = None,
) -> str:
    """Serialize a prediction dict to a Kafka-ready JSON message."""
    payload = {**prediction, "source": source}
    if processing_time_ms is not None:
        payload["processing_time_ms"] = round(processing_time_ms, 3)
    return json.dumps(payload)


def error_event(original: Dict, error_message: str) -> str:
    """Serialize a malformed-event report for the ecg-errors topic."""
    try:
        original_str = json.dumps(original)[:2000]
    except (TypeError, ValueError):
        original_str = str(original)[:2000]
    return json.dumps(
        {
            "error": "event_validation_failed",
            "message": error_message[:500],
            "original_event": original_str,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
