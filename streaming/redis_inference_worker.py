#!/usr/bin/env python3
"""Redis Streams inference worker — the Spark-leg replacement in Redis mode.

Consumes the ``ecg-events`` stream via a consumer group, runs the same
two-stage inference (autoencoder gate -> XGBoost) the Spark job
performs, and appends predictions to ``ecg-predictions`` (malformed
events go to ``ecg-errors``). Every handled entry is XACKed.

Why not Spark? Spark has no first-class Redis Streams source; pulling in
the third-party spark-redis connector defeats the "lighter local stack"
motivation of this experiment. This worker keeps the exact same
contracts (topics, payloads, error handling) with a plain Redis client.

Run:
    MESSAGE_BROKER=redis python -m streaming.redis_inference_worker

Batch metrics are appended to the same logs/streaming/batch_metrics.csv
as the Spark job, so /api/system/status keeps working unchanged.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ecg_core.config import settings  # noqa: E402
from streaming.feature_pipeline import error_event, validate_event  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("redis-inference-worker")

CONSUMER_NAME = f"worker-{uuid.uuid4().hex[:8]}"


class BatchMetricsWriter:
    """Appends per-poll batch metrics to the Spark-compatible CSV."""

    def __init__(self, out_path: Path):
        self.path = out_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "query", "batch_id", "num_input_rows",
                     "input_rows_per_second", "processed_rows_per_second",
                     "trigger_duration_ms"]
                )

    def write(self, batch_id: int, n_rows: int, duration_ms: float) -> None:
        try:
            with self.path.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        "redis-worker",
                        batch_id,
                        n_rows,
                        round(n_rows / max(duration_ms / 1000.0, 1e-9), 3),
                        round(n_rows / max(duration_ms / 1000.0, 1e-9), 3),
                        round(duration_ms, 1),
                    ]
                )
        except OSError:
            logger.exception("Failed to write batch metrics")


def _redis_client():
    import redis

    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


def _ensure_group(client, stream: str, group: str) -> None:
    try:
        client.xgroup_create(stream, group, id="$", mkstream=True)
    except Exception as exc:  # noqa: BLE001 - BUSYGROUP means it already exists
        if "BUSYGROUP" not in str(exc):
            raise


def _predict_one(event: dict) -> dict:
    """Same two-stage inference the Spark UDF performs."""
    import numpy as np

    from inference.model_loader import get_model_bundle
    from inference.predictor import predict_features

    validated = validate_event(event)
    result = predict_features(
        validated.features,
        event_id=validated.event_id,
        record_id=validated.record_id,
        true_label=validated.true_label,
    )
    payload = dict(vars(result))
    payload["source"] = "redis-worker"
    payload["event_timestamp"] = event.get("timestamp")
    return payload


def run() -> None:
    import json

    client = _redis_client()
    events_stream = settings.kafka_topic_events
    predictions_stream = settings.kafka_topic_predictions
    errors_stream = settings.kafka_topic_errors
    group = f"{settings.kafka_consumer_group}-worker"

    _ensure_group(client, events_stream, group)
    metrics = BatchMetricsWriter(settings.spark_stream_batch_log)

    logger.info(
        "Redis inference worker started (url=%s, stream=%s, group=%s, consumer=%s)",
        settings.redis_url, events_stream, group, CONSUMER_NAME,
    )

    batch_id = 0
    backoff = 1.0
    while True:
        try:
            resp = client.xreadgroup(
                group, CONSUMER_NAME, {events_stream: ">"},
                count=100, block=settings.redis_block_ms,
            )
            if not resp:
                backoff = 1.0
                continue

            started = time.perf_counter()
            n_ok = n_err = 0
            for _stream, entries in resp:
                for entry_id, fields in entries:
                    try:
                        event = json.loads(fields.get("payload", "{}"))
                        prediction = _predict_one(event)
                        client.xadd(
                            predictions_stream,
                            {"payload": json.dumps(prediction, default=str)},
                            maxlen=settings.redis_stream_maxlen,
                            approximate=True,
                        )
                        n_ok += 1
                    except Exception as exc:  # noqa: BLE001 - route bad events to the error stream
                        n_err += 1
                        logger.warning("Failed to process entry %s: %s", entry_id, exc)
                        try:
                            original = json.loads(fields.get("payload", "null"))
                        except ValueError:
                            original = fields.get("payload")
                        client.xadd(
                            errors_stream,
                            {"payload": json.dumps(error_event(original, str(exc)), default=str)},
                            maxlen=settings.redis_stream_maxlen,
                            approximate=True,
                        )
                    client.xack(events_stream, group, entry_id)

            duration_ms = (time.perf_counter() - started) * 1000.0
            batch_id += 1
            metrics.write(batch_id, n_ok + n_err, duration_ms)
            if n_ok or n_err:
                logger.info(
                    "Batch %d: %d ok, %d errors in %.1fms", batch_id, n_ok, n_err, duration_ms
                )
            backoff = 1.0
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as exc:  # noqa: BLE001 - Redis outage: retry with backoff
            logger.warning("Worker error (%s); retrying in %.0fs", exc, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30.0)
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass
            client = _redis_client()
            _ensure_group(client, events_stream, group)

    client.close()


if __name__ == "__main__":
    run()
