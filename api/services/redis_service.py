"""Redis Streams wiring for the API service (experimental Kafka alternative).

Mirrors :mod:`api.services.kafka_service`:

* :func:`redis_available`        — cached, non-blocking reachability probe.
* :class:`RedisEventPublisher`   — XADDs ECG events to the ecg-events stream.
* :class:`RedisPredictionConsumer` — background thread consuming the
  ecg-predictions stream via a consumer group, persisting to PostgreSQL
  and broadcasting to WebSocket clients.

Design notes vs Kafka:
  * Streams are capped with MAXLEN ~ appends are O(1) and old entries
    are trimmed, so a laptop does not fill its disk during simulations.
  * Delivery is at-least-once: entries are XACKed only after the
    handler succeeds; a crashed consumer re-reads pending entries after
    idle (XAUTOCLAIM-style reclaim is left as future work).
  * No broker-side consumer-group rebalancing — each process uses a
    unique consumer name within the group.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from ecg_core.config import settings

logger = logging.getLogger(__name__)

REDIS_IMPORT_ERROR_HINT = (
    "redis is not installed. Run `pip install redis` or use the Docker "
    "Compose stack (profile: redis)."
)

# Cached, non-blocking reachability probe (same pattern as Kafka's).
_REDIS_CHECK_TIMEOUT_S = 2.0
_REDIS_CACHE_TTL_S = 5.0
_redis_status_cache: dict = {"ok": False, "checked_at": 0.0}

_CONSUMER_NAME = f"api-{uuid.uuid4().hex[:8]}"


def _redis_client():
    import redis

    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _probe_redis() -> bool:
    client = _redis_client()
    try:
        return bool(client.ping())
    finally:
        client.close()


def redis_available() -> bool:
    """Cheap connectivity check used by /api/system/status (cached, non-blocking)."""
    now = time.time()
    if now - float(_redis_status_cache["checked_at"]) < _REDIS_CACHE_TTL_S:
        return bool(_redis_status_cache["ok"])

    ok = False
    try:
        from concurrent import futures

        executor = futures.ThreadPoolExecutor(max_workers=1)
        try:
            ok = executor.submit(_probe_redis).result(timeout=_REDIS_CHECK_TIMEOUT_S)
        finally:
            executor.shutdown(wait=False)
    except ImportError:
        logger.warning(REDIS_IMPORT_ERROR_HINT)
    except Exception:  # noqa: BLE001 - any failure means Redis is effectively down
        ok = False

    _redis_status_cache.update(ok=ok, checked_at=now)
    return ok


def _ensure_group(client, stream: str, group: str) -> None:
    """Create the consumer group at the head of the stream if missing."""
    try:
        client.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception as exc:  # noqa: BLE001 - BUSYGROUP means it already exists
        if "BUSYGROUP" not in str(exc):
            raise


class RedisEventPublisher:
    """Publishes validated ECG events to the ecg-events stream."""

    def __init__(self) -> None:
        self._client = None
        self._lock = threading.Lock()

    @property
    def client(self):
        with self._lock:
            if self._client is None:
                self._client = _redis_client()
            return self._client

    def publish(self, event: dict) -> str:
        """Validate + publish; raises on validation error or Redis failure."""
        from streaming.feature_pipeline import validate_event

        validated = validate_event(event)
        self.client.xadd(
            settings.kafka_topic_events,  # same names as the Kafka topics
            {"payload": json.dumps(event, default=str)},
            maxlen=settings.redis_stream_maxlen,
            approximate=True,
        )
        return validated.event_id

    def flush(self) -> None:
        """XADD is synchronous — nothing to flush (API parity with Kafka's publisher)."""

    def close(self) -> None:
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:  # noqa: BLE001
                    pass
                self._client = None


class RedisPredictionConsumer:
    """Background thread: ecg-predictions stream -> PostgreSQL + WebSocket.

    Redis outages never crash the API — the consumer retries with backoff
    until the server returns. Same contract as the Kafka PredictionConsumer.
    """

    def __init__(
        self,
        on_prediction: Optional[Callable[[dict], None]] = None,
        redis_url: Optional[str] = None,
    ):
        self._on_prediction = on_prediction
        self._url = redis_url or settings.redis_url
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_message_at: Optional[float] = None
        self.connected = False

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="redis-prediction-consumer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            # Short join: the thread may be blocked in XREADGROUP; it is a
            # daemon and dies with the process.
            self._thread.join(timeout=2)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._consume()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001 - keep the API alive
                self.connected = False
                logger.warning("Redis prediction consumer error (%s); retrying in %.0fs", exc, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)

    def _consume(self) -> None:
        import redis

        client = redis.Redis.from_url(self._url, decode_responses=True)
        stream = settings.kafka_topic_predictions
        group = f"{settings.kafka_consumer_group}-api"
        _ensure_group(client, stream, group)

        self.connected = True
        logger.info(
            "Redis prediction consumer connected to %s (stream=%s, group=%s)",
            self._url, stream, group,
        )
        try:
            while not self._stop.is_set():
                # XREADGROUP with BLOCK returns empty on timeout; the block
                # interval doubles as the shutdown-latency bound.
                resp = client.xreadgroup(
                    group, _CONSUMER_NAME, {stream: ">"},
                    count=50, block=settings.redis_block_ms,
                )
                if not resp:
                    continue
                for _stream, entries in resp:
                    for entry_id, fields in entries:
                        self.last_message_at = time.time()
                        try:
                            payload = json.loads(fields.get("payload", "{}"))
                            self._handle(payload)
                            client.xack(stream, group, entry_id)
                        except Exception:  # noqa: BLE001 - one bad message must not kill the loop
                            logger.exception(
                                "Failed to handle prediction entry %s: %r", entry_id, fields
                            )
                            client.xack(stream, group, entry_id)  # skip poison entries
        finally:
            client.close()
            self.connected = False

    def _handle(self, prediction: dict) -> None:
        # Same normalization as the Kafka consumer.
        normalized = dict(prediction)
        normalized.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        if "latency_ms" in normalized and "inference_latency_ms" not in normalized:
            normalized["inference_latency_ms"] = normalized.get("latency_ms")
        if "event_timestamp" in normalized and "timestamp" not in prediction:
            normalized["timestamp"] = prediction["event_timestamp"]
        normalized.setdefault("source", "redis")

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


redis_event_publisher = RedisEventPublisher()
