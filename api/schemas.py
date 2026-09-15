"""Pydantic schemas for API request/response validation.

The ECG event schema mirrors streaming/feature_pipeline.py exactly —
16 numeric features per lead, 32 total, in canonical order.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ecg_core.features import LEAD_FEATURE_STEMS, STEM_TO_PAYLOAD_KEY

PAYLOAD_FIELDS = [STEM_TO_PAYLOAD_KEY[stem] for stem in LEAD_FEATURE_STEMS]


class ECGLeadPayload(BaseModel):
    """16 ECG features for a single lead."""

    pre_rr: float = Field(..., description="Pre-RR interval (s)", examples=[0.76])
    post_rr: float = Field(..., description="Post-RR interval (s)", examples=[0.81])
    p_peak: float = Field(..., description="P peak amplitude (mV)")
    t_peak: float = Field(..., description="T peak amplitude (mV)")
    r_peak: float = Field(..., description="R peak amplitude (mV)")
    s_peak: float = Field(..., description="S peak amplitude (mV)")
    q_peak: float = Field(..., description="Q peak amplitude (mV)")
    qrs_interval: float = Field(..., description="QRS duration (samples)", examples=[94])
    pq_interval: float = Field(..., description="PQ interval (samples)")
    qt_interval: float = Field(..., description="QT interval (samples)")
    st_interval: float = Field(..., description="ST interval (samples)")
    qrs_morph_0: float = Field(..., description="QRS morphology feature 0")
    qrs_morph_1: float = Field(..., description="QRS morphology feature 1")
    qrs_morph_2: float = Field(..., description="QRS morphology feature 2")
    qrs_morph_3: float = Field(..., description="QRS morphology feature 3")
    qrs_morph_4: float = Field(..., description="QRS morphology feature 4")

    @field_validator("*")
    @classmethod
    def check_finite(cls, v: float) -> float:
        import math

        if not math.isfinite(v):
            raise ValueError("feature values must be finite numbers")
        return v


class ECGEventPayload(BaseModel):
    """Full ECG event as published to Kafka (topic ecg-events)."""

    event_id: Optional[str] = Field(None, description="Client-supplied id; generated when omitted")
    timestamp: Optional[datetime] = Field(
        None, description="Event time (ISO-8601); server time used when omitted"
    )
    record_id: Optional[str] = Field(None, description="MIT-BIH record id (for replay)")
    true_label: Optional[str] = Field(
        None, description="Ground-truth class, only for replay/simulation demos"
    )
    lead_ii: ECGLeadPayload
    lead_v5: ECGLeadPayload


class ECGSendResponse(BaseModel):
    event_id: str
    published_to_kafka: bool
    topic: str
    message: str
    prediction: Optional[Dict] = Field(
        None,
        description="Immediate prediction (only in direct fallback mode when Kafka is unavailable)",
    )


class SimulationStartRequest(BaseModel):
    mode: str = Field("random", description="normal | abnormal | random | N | SVEB | VEB | F | Q")
    events_per_second: float = Field(2.0, ge=0.1, le=50.0)
    duration_seconds: float = Field(30.0, ge=1.0, le=3600.0)
    n_events: Optional[int] = Field(None, ge=1, le=100000,
                                    description="Optional hard cap on the number of events")


class SimulationStatus(BaseModel):
    running: bool
    mode: Optional[str] = None
    events_sent: int = 0
    events_per_second: Optional[float] = None
    duration_seconds: Optional[float] = None
    started_at: Optional[str] = None


class PredictionOut(BaseModel):
    event_id: str
    timestamp: Optional[str] = None
    record_id: Optional[str] = None
    predicted_class: str
    is_abnormal: bool
    confidence: Optional[float] = None
    reconstruction_error: Optional[float] = None
    anomaly_threshold: Optional[float] = None
    inference_latency_ms: Optional[float] = None
    processing_time_ms: Optional[float] = None
    source: Optional[str] = None
    true_label: Optional[str] = None
    stage: Optional[str] = None


class StatisticsOut(BaseModel):
    total_events: int
    normal_events: int
    abnormal_events: int
    abnormal_rate: float
    average_latency_ms: Optional[float] = None
    class_distribution: Dict[str, int]
    events_per_second: Optional[float] = None
    latency: Optional[Dict[str, float]] = None


class ComponentStatus(BaseModel):
    status: str  # "up" | "down" | "unknown" | "degraded"
    detail: Optional[str] = None


class SystemStatusOut(BaseModel):
    api: ComponentStatus
    kafka: ComponentStatus
    spark: ComponentStatus
    database: ComponentStatus
    models: ComponentStatus
    timestamp: str


class ThresholdOut(BaseModel):
    anomaly_threshold: float
    threshold_percentile: Optional[float] = None
    model: str = "autoencoder"
    description: str = (
        "Reconstruction errors at or below this threshold are classified as Normal (Stage 1)."
    )


class MetricsOut(BaseModel):
    total_events: int
    latency_ms: Dict[str, Optional[float]]
    throughput_events_per_second: Optional[float]
    class_distribution: Dict[str, int]
    window_seconds: int


class ExplanationOut(BaseModel):
    event_id: str
    predicted_class: Optional[str] = None
    is_abnormal: Optional[bool] = None
    explanation_type: str = "model_feature_contribution"
    top_features: Optional[List[Dict]] = None
    note: str = (
        "Feature contributions computed with SHAP values for the XGBoost stage. "
        "This is a model feature contribution explanation, not a clinical interpretation."
    )
