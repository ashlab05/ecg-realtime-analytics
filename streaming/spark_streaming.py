#!/usr/bin/env python3
"""Spark Structured Streaming job: ECG event ingestion and inference.

Architecture
------------
    Kafka (ecg-events)
        -> Spark Structured Streaming (micro-batch)
            1. parse JSON payload with an explicit schema
            2. route malformed events to the ecg-errors Kafka topic
            3. extract the 32 ECG features (canonical order)
            4. run two-stage inference (autoencoder -> XGBoost) in a
               vectorized pandas UDF; the model bundle is loaded once
               per Python worker process
            5. publish prediction events to Kafka (ecg-predictions)
        -> the FastAPI service consumes ecg-predictions, persists them
           to PostgreSQL, and pushes them to dashboard WebSockets.

Event time vs processing time
-----------------------------
* ``event_time``   — the timestamp embedded in the ECG event payload
                     (when the heartbeat was produced/simulated).
* ``processing_time`` — the moment Spark processed the record.
The difference (end-to-end lag) is included in every prediction event.

Trigger interval, checkpointing, and topic names are configurable via
environment variables (see .env.example).

Usage:
  python streaming/spark_streaming.py
  SPARK_TRIGGER_INTERVAL='2 seconds' python streaming/spark_streaming.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    StringType,
    StructField,
    StructType,
)

from ecg_core.config import settings
from ecg_core.features import (
    LEAD_FEATURE_STEMS,
    LEAD_II_CSV_PREFIX,
    LEAD_II_PAYLOAD_KEY,
    LEAD_V_CSV_PREFIX,
    LEAD_V_PAYLOAD_KEY,
    STEM_TO_PAYLOAD_KEY,
)
from ecg_core.models.autoencoder import load_threshold
from ecg_core.preprocessing import load_artifacts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("spark_streaming")

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

LEAD_SCHEMA = StructType(
    [StructField(STEM_TO_PAYLOAD_KEY[stem], DoubleType(), True) for stem in LEAD_FEATURE_STEMS]
)

EVENT_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("record_id", StringType(), True),
        StructField("true_label", StringType(), True),
        StructField(LEAD_II_PAYLOAD_KEY, LEAD_SCHEMA, True),
        StructField(LEAD_V_PAYLOAD_KEY, LEAD_SCHEMA, True),
    ]
)

PREDICTION_SCHEMA = StructType(
    [
        StructField("event_id", StringType(), True),
        StructField("predicted_class", StringType(), True),
        StructField("is_abnormal", BooleanType(), True),
        StructField("confidence", DoubleType(), True),
        StructField("reconstruction_error", DoubleType(), True),
        StructField("anomaly_threshold", DoubleType(), True),
        StructField("latency_ms", DoubleType(), True),
        StructField("end_to_end_lag_ms", DoubleType(), True),
        StructField("stage", StringType(), True),
        StructField("stage2_probability", DoubleType(), True),
        StructField("true_label", StringType(), True),
        StructField("record_id", StringType(), True),
        StructField("event_timestamp", StringType(), True),
        StructField("processing_timestamp", StringType(), True),
    ]
)

# Canonical feature order as flattened payload field paths.
FEATURE_FIELDS = [
    f"{lead_key}.{STEM_TO_PAYLOAD_KEY[stem]}"
    for lead_key, _ in ((LEAD_II_PAYLOAD_KEY, LEAD_II_CSV_PREFIX), (LEAD_V_PAYLOAD_KEY, LEAD_V_CSV_PREFIX))
    for stem in LEAD_FEATURE_STEMS
]

# ---------------------------------------------------------------------------
# Inference UDF (model bundle loaded once per Python worker)
# ---------------------------------------------------------------------------

_bundle_cache = None


def _get_bundle():
    """Lazy per-process model bundle: weights are deserialized exactly once."""
    global _bundle_cache
    if _bundle_cache is None:
        from inference.model_loader import get_model_bundle

        _bundle_cache = get_model_bundle()
    return _bundle_cache


def make_predict_udf(spark):
    """Build the vectorized two-stage inference pandas UDF."""

    @F.pandas_udf(returnType=PREDICTION_SCHEMA)
    def predict_udf(event_ids, timestamps, record_ids, true_labels, features):
        import time
        from datetime import datetime, timezone

        import numpy as np
        import pandas as pd

        bundle = _get_bundle()
        X_raw = np.vstack(features.to_numpy())
        X_scaled = bundle.transform(X_raw)

        started = time.perf_counter()
        recon = bundle.autoencoder.reconstruct(X_scaled)
        errors = np.mean((X_scaled - recon.astype(np.float64)) ** 2, axis=1)
        threshold = bundle.anomaly_threshold
        is_normal = errors <= threshold

        predicted = np.full(len(X_raw), "N", dtype=object)
        confidence = np.zeros(len(X_raw))
        stage = np.full(len(X_raw), "stage1_normal", dtype=object)
        stage2_prob = np.full(len(X_raw), np.nan)

        abnormal_idx = np.where(~is_normal)[0]
        if bundle.xgboost is not None and len(abnormal_idx) > 0:
            sub = X_scaled[abnormal_idx]
            preds_idx = bundle.xgboost.predict(sub)
            probs = bundle.xgboost.predict_proba(sub)
            for j, i in enumerate(abnormal_idx):
                predicted[i] = bundle.xgboost_classes[int(preds_idx[j])]
                stage2_prob[i] = float(probs[j, int(preds_idx[j])])
                stage[i] = "stage2_xgboost"

        # Confidence: margin-based for Stage 1, blended with class
        # probability for Stage 2.
        margin = np.abs(errors - threshold) / max(threshold, 1e-12)
        ae_conf = 0.5 + 0.5 * np.tanh(margin)
        confidence = np.where(
            np.isnan(stage2_prob),
            ae_conf,
            0.5 * ae_conf + 0.5 * np.nan_to_num(stage2_prob),
        )

        latency_ms = (time.perf_counter() - started) * 1000.0 / max(len(X_raw), 1)
        now = datetime.now(timezone.utc).isoformat()

        def _lag_ms(ts):
            try:
                event_dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                return max(0.0, (datetime.now(timezone.utc) - event_dt).total_seconds() * 1000.0)
            except (ValueError, TypeError):
                return float("nan")

        return pd.DataFrame(
            {
                "event_id": event_ids,
                "predicted_class": predicted,
                "is_abnormal": predicted != "N",
                "confidence": np.round(confidence, 4),
                "reconstruction_error": np.round(errors, 6),
                "anomaly_threshold": float(threshold),
                "latency_ms": round(float(latency_ms), 3),
                "end_to_end_lag_ms": [round(_lag_ms(t), 3) for t in timestamps],
                "stage": stage,
                "stage2_probability": np.round(stage2_prob, 4),
                "true_label": true_labels,
                "record_id": record_ids,
                "event_timestamp": timestamps,
                "processing_timestamp": now,
            }
        )

    return predict_udf


# ---------------------------------------------------------------------------
# Batch metrics listener (logs/streaming/batch_metrics.csv)
# ---------------------------------------------------------------------------

class BatchMetricsLogger:
    """StreamingQueryListener that records per-micro-batch throughput."""

    def __init__(self, out_path: Path):
        import csv

        self.path = out_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.writer(f).writerow(
                    ["timestamp", "query", "batch_id", "num_input_rows",
                     "input_rows_per_second", "processed_rows_per_second",
                     "trigger_duration_ms"]
                )
        self._csv = csv

    def onQueryStarted(self, event):
        logger.info("Streaming query started: %s", event.id)

    def onQueryProgress(self, event):
        p = event.progress
        try:
            from datetime import datetime, timezone

            with self.path.open("a", newline="") as f:
                self._csv.writer(f).writerow(
                    [
                        datetime.now(timezone.utc).isoformat(),
                        str(event.id),
                        p.batchId,
                        p.numInputRows,
                        round(p.inputRowsPerSecond or 0.0, 3),
                        round(p.processedRowsPerSecond or 0.0, 3),
                        (p.durationMs or {}).get("triggerExecution"),
                    ]
                )
        except OSError:
            logger.exception("Failed to write batch metrics")

    def onQueryTerminated(self, event):
        logger.info("Streaming query terminated: %s (exception=%s)", event.id, event.exception)


# ---------------------------------------------------------------------------
# Main streaming application
# ---------------------------------------------------------------------------

def build_spark_session(app_name: str = "ECG-RealTime-Analytics") -> SparkSession:
    return (
        SparkSession.builder.master(settings.spark_master)
        .appName(app_name)
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
        )
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


def run(spark: SparkSession | None = None) -> None:
    spark = spark or build_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    # Fail fast with a clear message if models are not trained yet.
    load_artifacts()
    load_threshold()
    logger.info(
        "Starting ECG streaming job | kafka=%s | topics: events=%s predictions=%s errors=%s",
        settings.kafka_bootstrap_servers,
        settings.kafka_topic_events,
        settings.kafka_topic_predictions,
        settings.kafka_topic_errors,
    )

    # ---- 1. Ingest raw events --------------------------------------------
    raw = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("subscribe", settings.kafka_topic_events)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
        .selectExpr("CAST(value AS STRING) AS json_str")
    )

    # ---- 2. Parse JSON; keep malformed rows for the errors route ---------
    parsed = raw.select(
        F.from_json(F.col("json_str"), EVENT_SCHEMA).alias("event"),
        F.col("json_str"),
    )
    valid = parsed.filter(F.col("event").isNotNull()).select("event.*")
    malformed = parsed.filter(F.col("event").isNull()).select("json_str")

    # ---- 3/4. Feature extraction + two-stage inference (pandas UDF) ------
    features_col = F.array(*[F.col(field) for field in FEATURE_FIELDS])
    predictions = valid.select(
        F.coalesce(F.col("event_id"), F.expr("uuid()")).alias("event_id"),
        F.coalesce(F.col("timestamp"), F.expr("current_timestamp().cast(string)")).alias("timestamp"),
        F.col("record_id"),
        F.col("true_label"),
        features_col.alias("features"),
    )

    predict_udf = make_predict_udf(spark)
    result = predictions.select(
        predict_udf(
            F.col("event_id"), F.col("timestamp"), F.col("record_id"),
            F.col("true_label"), F.col("features"),
        ).alias("pred")
    ).select("pred.*")

    # ---- 5. Publish prediction events to Kafka ----------------------------
    kafka_predictions = result.select(
        F.col("event_id").cast("string").alias("key"),
        F.to_json(F.struct("*")).alias("value"),
    )

    query_predictions = (
        kafka_predictions.writeStream.format("kafka")
        .option("kafka.bootstrap.servers", settings.kafka_bootstrap_servers)
        .option("topic", settings.kafka_topic_predictions)
        .option("checkpointLocation", f"{settings.spark_checkpoint_dir}/predictions")
        .trigger(processingTime=settings.spark_trigger_interval)
        .outputMode("append")
        .start()
    )

    # ---- Malformed events -> ecg-errors topic ------------------------------
    def send_errors(batch_df, batch_id):
        count = batch_df.count()
        if count == 0:
            return
        import json

        from kafka import KafkaProducer

        from streaming.feature_pipeline import error_event

        producer = KafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: v.encode("utf-8"),
        )
        try:
            for row in batch_df.collect():
                try:
                    original = json.loads(row["json_str"])
                except (ValueError, TypeError):
                    original = {"raw": str(row["json_str"])[:500]}
                producer.send(
                    settings.kafka_topic_errors,
                    value=error_event(original, "event failed JSON schema validation"),
                )
            producer.flush()
            logger.warning("Batch %d: routed %d malformed events to %s",
                           batch_id, count, settings.kafka_topic_errors)
        finally:
            producer.close()

    query_errors = (
        malformed.writeStream.foreachBatch(send_errors)
        .option("checkpointLocation", f"{settings.spark_checkpoint_dir}/errors")
        .trigger(processingTime=settings.spark_trigger_interval)
        .outputMode("append")
        .start()
    )

    metrics_logger = BatchMetricsLogger(settings.spark_stream_batch_log)
    spark.streams.addListener(metrics_logger)

    logger.info("Streaming queries running (trigger=%s). Ctrl+C to stop.", settings.spark_trigger_interval)
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Interrupt received; stopping streaming queries gracefully.")
    finally:
        query_predictions.stop()
        query_errors.stop()
        spark.stop()


if __name__ == "__main__":
    run()
