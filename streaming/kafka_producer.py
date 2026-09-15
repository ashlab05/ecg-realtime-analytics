#!/usr/bin/env python3
"""Kafka producer for ECG events.

Provides:
  * ECGEventProducer  — publishes validated JSON events to the
                        ecg-events topic (used by the FastAPI service
                        and this module's CLI).
  * DatasetSampler    — draws real heartbeats from the MIT-BIH feature
                        CSV and converts them to the JSON event schema.
  * CLI               — replay/simulate ECG events to Kafka at a
                        controlled rate for standalone testing.

Usage examples:
  python streaming/kafka_producer.py --mode random --rate 2 --duration 30
  python streaming/kafka_producer.py --mode class --class VEB --n 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from ecg_core.config import settings
from ecg_core.features import (
    LEAD_FEATURE_STEMS,
    LEAD_II_CSV_PREFIX,
    LEAD_II_PAYLOAD_KEY,
    LEAD_V_CSV_PREFIX,
    LEAD_V_PAYLOAD_KEY,
    STEM_TO_PAYLOAD_KEY,
)
from ecg_core.labels import CLASS_DESCRIPTIONS, CLASS_NAMES, normalize_label
from streaming.feature_pipeline import EventValidationError, validate_event

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("kafka_producer")


def row_to_event_payload(row: pd.Series, true_label: Optional[str] = None) -> Dict:
    """Convert one dataset row (canonical feature columns) to the JSON event schema."""
    lead_payloads: Dict[str, Dict[str, float]] = {
        LEAD_II_PAYLOAD_KEY: {},
        LEAD_V_PAYLOAD_KEY: {},
    }
    for lead_csv_prefix, payload_key in (
        (LEAD_II_CSV_PREFIX, LEAD_II_PAYLOAD_KEY),
        (LEAD_V_CSV_PREFIX, LEAD_V_PAYLOAD_KEY),
    ):
        for stem in LEAD_FEATURE_STEMS:
            payload_key_field = STEM_TO_PAYLOAD_KEY[stem]
            lead_payloads[payload_key][payload_key_field] = float(row[f"{lead_csv_prefix}_{stem}"])

    return {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_id": str(row.get("record", "")) if "record" in row.index else None,
        "true_label": true_label or (str(row["type"]) if "type" in row.index else None),
        LEAD_II_PAYLOAD_KEY: lead_payloads[LEAD_II_PAYLOAD_KEY],
        LEAD_V_PAYLOAD_KEY: lead_payloads[LEAD_V_PAYLOAD_KEY],
    }


class ECGEventProducer:
    """Thin wrapper around kafka-python's KafkaProducer with JSON serialization."""

    def __init__(self, bootstrap_servers: Optional[str] = None, topic: Optional[str] = None):
        self.bootstrap_servers = bootstrap_servers or settings.kafka_bootstrap_servers
        self.topic = topic or settings.kafka_topic_events
        self._producer = None

    @property
    def producer(self):
        if self._producer is None:
            from kafka import KafkaProducer

            self._producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks="all",
                retries=3,
            )
        return self._producer

    def send_event(self, event: Dict) -> str:
        """Validate and publish one event; returns its event_id."""
        validated = validate_event(event)  # raises EventValidationError
        self.producer.send(self.topic, key=validated.event_id, value=event)
        return validated.event_id

    def flush(self, timeout: float = 10.0) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=timeout)

    def close(self) -> None:
        if self._producer is not None:
            self._producer.flush(timeout=10.0)
            self._producer.close()
            self._producer = None


class DatasetSampler:
    """Samples real heartbeats from the dataset for simulation/replay."""

    def __init__(self, df: Optional[pd.DataFrame] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._df = df
        self._class_frames: Dict[str, pd.DataFrame] = {}

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            from ecg_core.preprocessing import load_dataset

            self._df = load_dataset()
        return self._df

    def _frame_for(self, label: str) -> pd.DataFrame:
        if label not in self._class_frames:
            self._class_frames[label] = self.df[self.df["type"] == label]
        return self._class_frames[label]

    def available_classes(self) -> Dict[str, int]:
        """Class -> number of available samples."""
        counts = self.df["type"].value_counts().to_dict()
        return {cls: int(counts.get(cls, 0)) for cls in CLASS_NAMES}

    def sample(self, label: Optional[str] = None) -> Dict:
        """Sample one heartbeat as a JSON event payload.

        label=None -> random class weighted by natural prevalence.
        """
        if label is None:
            label = str(self.rng.choice(list(self.df["type"].unique())))
        label = normalize_label(label)
        frame = self._frame_for(label)
        if frame.empty:
            raise ValueError(f"No samples available for class {label}")
        row = frame.iloc[self.rng.integers(len(frame))]
        return row_to_event_payload(row, true_label=label)

    def sample_abnormal(self) -> Dict:
        """Sample a random abnormal heartbeat (SVEB/VEB/F/Q)."""
        abnormal_labels = [c for c in CLASS_NAMES if c != "N"]
        return self.sample(str(self.rng.choice(abnormal_labels)))

    def sample_by_mode(self, mode: str) -> Dict:
        """mode: normal | abnormal | random | one of the class names."""
        mode = mode.lower()
        if mode == "normal":
            return self.sample("N")
        if mode == "abnormal":
            return self.sample_abnormal()
        if mode == "random":
            return self.sample(None)
        return self.sample(mode)


def simulate(
    mode: str = "random",
    events_per_second: float = 2.0,
    duration_seconds: float = 30.0,
    n_events: Optional[int] = None,
    producer: Optional[ECGEventProducer] = None,
    sampler: Optional[DatasetSampler] = None,
    on_sent=None,
) -> List[str]:
    """Publish simulated ECG events to Kafka at a controlled rate."""
    producer = producer or ECGEventProducer()
    sampler = sampler or DatasetSampler()
    sent: List[str] = []
    interval = 1.0 / max(events_per_second, 0.01)
    start = time.time()

    while True:
        elapsed = time.time() - start
        if duration_seconds and elapsed >= duration_seconds:
            break
        if n_events and len(sent) >= n_events:
            break
        event = sampler.sample_by_mode(mode)
        try:
            event_id = producer.send_event(event)
            sent.append(event_id)
            if on_sent:
                on_sent(event, event_id)
        except EventValidationError as exc:
            logger.error("Skipping invalid sampled event: %s", exc)
        # Simple fixed-rate pacing (good enough for a local prototype).
        time.sleep(max(0.0, interval - (time.time() - start - elapsed)))

    producer.flush()
    logger.info("Sent %d events (mode=%s, rate=%.2f/s)", len(sent), mode, events_per_second)
    return sent


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish simulated ECG events to Kafka")
    parser.add_argument("--mode", default="random",
                        help="normal | abnormal | random | N | SVEB | VEB | F | Q")
    parser.add_argument("--rate", type=float, default=2.0, help="events per second")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds (0 = until n_events)")
    parser.add_argument("--n", type=int, default=None, help="stop after n events")
    args = parser.parse_args()

    sent = simulate(
        mode=args.mode,
        events_per_second=args.rate,
        duration_seconds=args.duration if args.duration > 0 else None,
        n_events=args.n,
    )
    print(json.dumps({"sent": len(sent), "mode": args.mode}))


if __name__ == "__main__":
    main()
