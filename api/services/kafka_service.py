"""Kafka wiring for the API service.

* :class:`EventPublisher`       — publishes ECG events to ecg-events.
* :class:`PredictionConsumer`   — background thread consuming
                                  ecg-predictions, persisting them to
                                  PostgreSQL and broadcasting to
                                  WebSocket clients.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from concurrent import futures
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional

from ecg_core.config import settings
from streaming.feature_pipeline import validate_event

logger = logging.getLogger(__name__)

KAFKA_IMPORT_ERROR_HINT = (
    "kafka-python is not installed. Run `pip install kafka-python` "
    "or use the Docker Compose stack."
)

# Cached, non-blocking Kafka reachability probe. The admin client's
# bootstrap can block for tens of seconds when no broker is running,
# so the check runs in a worker thread with a hard timeout and its
# result is cached briefly.
_KAFKA_CHECK_TIMEOUT_S = 2.0
_KAFKA_CACHE_TTL_S = 5.0
_kafka_status_cache: Dict[str, object] = {"ok": False, "checked_at": 0.0}


def _probe_kafka() -> bool:
    from kafka import KafkaAdminClient

    client = KafkaAdminClient(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        request_timeout_ms=1500,
    )
    client.close()
    return True


def kafka_available() -> bool:
    """Cheap connectivity check used by /api/system/status (cached, non-blocking)."""
    now = time.time()
    if now - float(_kafka_status_cache["checked_at"]) < _KAFKA_CACHE_TTL_S:
        return bool(_kafka_status_cache["ok"])

    ok = False
    try:
        executor = futures.ThreadPoolExecutor(max_workers=1)
        try:
            ok = executor.submit(_probe_kafka).result(timeout=_KAFKA_CHECK_TIMEOUT_S)
        finally:
            # Do NOT wait for the probe thread: it may still be stuck in
            # the admin client's bootstrap retry loop.
            executor.shutdown(wait=False)
    except ImportError:
        logger.warning(KAFKA_IMPORT_ERROR_HINT)
    except Exception:  # noqa: BLE001 - any failure means Kafka is effectively down
        ok = False

    _kafka_status_cache.update(ok=ok, checked_at=now)
    return ok


class EventPublisher:
    """Publishes validated ECG events to the ecg-events topic."""

    def __init__(self) -> None:
        self._producer = None
        self._lock = threading.Lock()

    @property
    def producer(self):
        with self._lock:
            if self._producer is None:
                from kafka import KafkaProducer

                self._producer = KafkaProducer(
                    bootstrap_servers=settings.kafka_bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
                    key_serializer=lambda k: (k or "").encode("utf-8"),
                    acks="all",
                    retries=3,
                    request_timeout_ms=5000,
                )
            return self._producer

    def publish(self, event: dict) -> str:
        """Validate + publish; raises on validation error or broker failure."""
        validated = validate_event(event)
        future = self.producer.send(
            settings.kafka_topic_events, key=validated.event_id, value=event
        )
        future.get(timeout=10)  # surface broker errors to the caller
        return validated.event_id

    def flush(self) -> None:
        with self._lock:
            if self._producer is not None:
                self._producer.flush(timeout=10)

    def close(self) -> None:
        with self._lock:
            if self._producer is not None:
                try:
                    self._producer.flush(timeout=5)
                    self._producer.close()
                except Exception:  # noqa: BLE001
                    pass
                self._producer = None


class LatencyTracker:
    """Rolling window of inference latencies + event timestamps for /api/metrics."""

    def __init__(self, window: int = 1000):
        self._latencies: Deque[float] = deque(maxlen=window)
        self._event_times: Deque[float] = deque(maxlen=window)
        self._class_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def record(self, latency_ms: float, predicted_class: str) -> None:
        now = time.time()
        with self._lock:
            self._latencies.append(float(latency_ms))
            self._event_times.append(now)
            self._class_counts[predicted_class] = self._class_counts.get(predicted_class, 0) + 1

    def snapshot(self, window_seconds: int = 60) -> dict:
        import numpy as np

        with self._lock:
            latencies = list(self._latencies)
            times = list(self._event_times)
            counts = dict(self._class_counts)

        if latencies:
            arr = np.asarray(latencies)
            lat = {
                "mean": float(arr.mean()),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
            }
        else:
            lat = {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}

        # Events per second over the trailing window.
        if times:
            cutoff = time.time() - window_seconds
            recent = [t for t in times if t >= cutoff]
            span = max((recent[-1] - recent[0]), 1e-9) if len(recent) > 1 else window_seconds
            eps = len(recent) / span if span > 0 else 0.0
        else:
            eps = 0.0

        return {
            "total_events": len(latencies),
            "latency_ms": lat,
            "events_per_second": round(eps, 3),
            "class_distribution": counts,
        }


class PredictionConsumer:
    """Background thread: ecg-predictions -> PostgreSQL + WebSocket + logs.

    Kafka outages never crash the API — the consumer retries with
    backoff until the broker returns.
    """

    def __init__(
        self,
        on_prediction: Optional[Callable[[dict], None]] = None,
        bootstrap_servers: Optional[str] = None,
    ):
        self._on_prediction = on_prediction
        self._bootstrap = bootstrap_servers or settings.kafka_bootstrap_servers
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_message_at: Optional[float] = None
        self.connected = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="kafka-prediction-consumer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            # Short join: the thread may be blocked inside the Kafka
            # client's bootstrap retries; it is a daemon and dies with us.
            self._thread.join(timeout=2)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._consume()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001 - keep the API alive
                self.connected = False
                logger.warning("Prediction consumer error (%s); retrying in %.0fs", exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def _consume(self) -> None:
        from kafka import KafkaConsumer

        consumer = KafkaConsumer(
            settings.kafka_topic_predictions,
            bootstrap_servers=self._bootstrap,
            group_id=f"{settings.kafka_consumer_group}-api",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            auto_offset_reset="latest",
            consumer_timeout_ms=1000,
        )
        self.connected = True
        logger.info("Prediction consumer connected to %s (topic=%s)", self._bootstrap, settings.kafka_topic_predictions)
        try:
            for message in consumer:
                if self._stop.is_set():
                    break
                self.last_message_at = time.time()
                try:
                    self._handle(message.value)
                except Exception:  # noqa: BLE001 - one bad message must not kill the loop
                    logger.exception("Failed to handle prediction message: %r", message.value)
        finally:
            consumer.close()
            self.connected = False

    def _handle(self, prediction: dict) -> None:
        # Normalize field names (Kafka message latency_ms -> DB inference_latency_ms).
        normalized = dict(prediction)
        normalized.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if "latency_ms" in normalized and "inference_latency_ms" not in normalized:
            normalized["inference_latency_ms"] = normalized.get("latency_ms")
        if "event_timestamp" in normalized and "timestamp" not in prediction:
            normalized["timestamp"] = prediction["event_timestamp"]

        try:
            from database.models import insert_prediction, session_scope

            with session_scope() as session:
                insert_prediction(session, normalized)
        except Exception:  # noqa: BLE001 - DB outage must not break the WS feed
            logger.exception("Failed to persist prediction to PostgreSQL")

        if self._on_prediction:
            try:
                self._on_prediction(normalized)
            except Exception:  # noqa: BLE001
                logger.exception("Prediction callback failed")


event_publisher = EventPublisher()
latency_tracker = LatencyTracker()
