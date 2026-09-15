"""SQLAlchemy models and database session management."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Double,
    Index,
    String,
    create_engine,
    desc,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from ecg_core.config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class ECGPrediction(Base):
    """One processed ECG event and its prediction."""

    __tablename__ = "ecg_predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    record_id: Mapped[Optional[str]] = mapped_column(String(16))
    predicted_class: Mapped[str] = mapped_column(String(8), nullable=False)
    is_abnormal: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence: Mapped[Optional[float]] = mapped_column(Double)
    reconstruction_error: Mapped[Optional[float]] = mapped_column(Double)
    anomaly_threshold: Mapped[Optional[float]] = mapped_column(Double)
    inference_latency_ms: Mapped[Optional[float]] = mapped_column(Double)
    processing_time_ms: Mapped[Optional[float]] = mapped_column(Double)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="kafka")
    true_label: Mapped[Optional[str]] = mapped_column(String(8))
    stage: Mapped[Optional[str]] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_ecg_predictions_timestamp", "timestamp"),
        Index("idx_ecg_predictions_class", "predicted_class"),
        Index("idx_ecg_predictions_abnormal", "is_abnormal"),
        Index("idx_ecg_predictions_event_id", "event_id"),
    )


_engine = None
_SessionFactory: Optional[sessionmaker] = None


def get_engine():
    """Lazily create the SQLAlchemy engine (shared across the API process)."""
    global _engine, _SessionFactory
    if _engine is None:
        _engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
        _SessionFactory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def init_db() -> None:
    """Create tables if they do not exist (idempotent)."""
    get_engine()
    Base.metadata.create_all(_engine)
    logger.info("Database schema ensured at %s", settings.database_url.split("@")[-1])


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope; commits on success, rolls back on error."""
    get_engine()
    session: Session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def insert_prediction(session: Session, prediction: dict) -> ECGPrediction:
    """Insert one prediction dict as a row. Caller manages the transaction."""
    ts = prediction.get("timestamp")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts is None:
        ts = datetime.now(timezone.utc)
    row = ECGPrediction(
        event_id=prediction["event_id"],
        timestamp=ts,
        record_id=prediction.get("record_id"),
        predicted_class=prediction["predicted_class"],
        is_abnormal=bool(prediction.get("is_abnormal", False)),
        confidence=prediction.get("confidence"),
        reconstruction_error=prediction.get("reconstruction_error"),
        anomaly_threshold=prediction.get("anomaly_threshold"),
        inference_latency_ms=prediction.get("latency_ms") or prediction.get("inference_latency_ms"),
        processing_time_ms=prediction.get("processing_time_ms"),
        source=prediction.get("source", "kafka"),
        true_label=prediction.get("true_label"),
        stage=prediction.get("stage"),
    )
    session.add(row)
    return row


def latest_predictions(session: Session, limit: int = 50) -> list[ECGPrediction]:
    return list(
        session.execute(
            select(ECGPrediction).order_by(desc(ECGPrediction.timestamp)).limit(limit)
        ).scalars()
    )


def get_prediction_by_event_id(session: Session, event_id: str) -> Optional[ECGPrediction]:
    return session.execute(
        select(ECGPrediction).where(ECGPrediction.event_id == event_id)
    ).scalar_one_or_none()


def compute_statistics(session: Session) -> dict:
    """Aggregate live statistics for the dashboard."""
    total = session.execute(select(func.count(ECGPrediction.id))).scalar() or 0
    abnormal = (
        session.execute(select(func.count(ECGPrediction.id)).where(ECGPrediction.is_abnormal)).scalar() or 0
    )
    avg_latency = session.execute(
        select(func.avg(ECGPrediction.inference_latency_ms))
    ).scalar()
    by_class = session.execute(
        select(ECGPrediction.predicted_class, func.count(ECGPrediction.id))
        .group_by(ECGPrediction.predicted_class)
    ).all()
    return {
        "total_events": int(total),
        "abnormal_events": int(abnormal),
        "normal_events": int(total) - int(abnormal),
        "abnormal_rate": (abnormal / total) if total else 0.0,
        "average_latency_ms": float(avg_latency) if avg_latency is not None else None,
        "class_distribution": {cls: int(cnt) for cls, cnt in by_class},
    }
