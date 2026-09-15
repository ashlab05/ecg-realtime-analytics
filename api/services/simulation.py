"""Simulation engine: streams dataset-sampled ECG events at a controlled rate.

Runs inside the FastAPI event loop as an asyncio task; each tick
samples a real heartbeat from the dataset, publishes it to Kafka, and
caches its features so /api/explain/{event_id} can serve SHAP values.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Dict, Optional

import numpy as np

from api.services.kafka_service import EventPublisher
from ecg_core.config import settings
from streaming.feature_pipeline import validate_event
from streaming.kafka_producer import DatasetSampler, row_to_event_payload

logger = logging.getLogger(__name__)


class FeatureCache:
    """Small LRU of recent events' feature vectors (for /api/explain)."""

    def __init__(self, capacity: int = 500):
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._capacity = capacity

    def put(self, event_id: str, features: np.ndarray) -> None:
        self._cache[event_id] = features
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)

    def get(self, event_id: str) -> Optional[np.ndarray]:
        if event_id in self._cache:
            self._cache.move_to_end(event_id)
            return self._cache[event_id]
        return None


feature_cache = FeatureCache()


class SimulationEngine:
    """Start/stop streaming simulations of ECG events."""

    def __init__(self, publisher):
        """`publisher` is any broker publisher exposing publish()/flush()."""
        self._publisher = publisher
        self._sampler: Optional[DatasetSampler] = None
        self._task: Optional[asyncio.Task] = None
        self._state = {
            "running": False,
            "mode": None,
            "events_sent": 0,
            "events_per_second": None,
            "duration_seconds": None,
            "started_at": None,
        }

    @property
    def sampler(self) -> DatasetSampler:
        if self._sampler is None:
            self._sampler = DatasetSampler()
        return self._sampler

    @property
    def status(self) -> Dict:
        return dict(self._state)

    def available_classes(self) -> Dict[str, int]:
        return self.sampler.available_classes()

    def sample_event(self, mode: str = "random") -> Dict:
        """Sample (but do not send) one event — used by 'send single event'."""
        return self.sampler.sample_by_mode(mode)

    async def start(
        self,
        mode: str = "random",
        events_per_second: float = 2.0,
        duration_seconds: float = 30.0,
        n_events: Optional[int] = None,
    ) -> Dict:
        if self._state["running"]:
            raise RuntimeError("A simulation is already running; stop it first.")
        self._task = asyncio.create_task(
            self._run(mode, events_per_second, duration_seconds, n_events),
            name="ecg-simulation",
        )
        return self.status

    def stop(self) -> Dict:
        if self._task and not self._task.done():
            self._task.cancel()
        self._state["running"] = False
        return self.status

    async def _run(
        self,
        mode: str,
        events_per_second: float,
        duration_seconds: float,
        n_events: Optional[int],
    ) -> None:
        self._state.update(
            {
                "running": True,
                "mode": mode,
                "events_sent": 0,
                "events_per_second": events_per_second,
                "duration_seconds": duration_seconds,
                "started_at": time.time(),
            }
        )
        interval = 1.0 / max(events_per_second, 0.1)
        start = time.time()
        logger.info("Simulation started: mode=%s rate=%.2f/s duration=%ss", mode, events_per_second, duration_seconds)

        try:
            while True:
                if time.time() - start >= duration_seconds:
                    break
                if n_events and self._state["events_sent"] >= n_events:
                    break
                try:
                    event = self.sampler.sample_by_mode(mode)
                    event_id = self._publish(event)
                    self._state["events_sent"] += 1
                except Exception as exc:  # noqa: BLE001 - keep simulating past one bad sample
                    logger.warning("Simulation tick failed: %s", exc)
                # Pace by wall clock so slow ticks do not drift.
                next_tick = start + (self._state["events_sent"] + 1) * interval
                await asyncio.sleep(max(0.0, next_tick - time.time()))
        except asyncio.CancelledError:
            logger.info("Simulation cancelled by user")
        finally:
            self._state["running"] = False
            self._publisher.flush()
            logger.info("Simulation finished: %d events sent", self._state["events_sent"])

    def _publish(self, event: Dict) -> str:
        validated = validate_event(event)
        self._publisher.publish(event)
        feature_cache.put(validated.event_id, validated.features)
        return validated.event_id


def send_direct_prediction(event: Dict) -> Dict:
    """Fallback path when Kafka is unavailable: infer in-process.

    Runs the same two-stage inference the Spark job performs, marks the
    result with source='direct', and returns the prediction dict for
    the caller to persist and broadcast.
    """
    from inference.predictor import predict_features

    validated = validate_event(event)
    feature_cache.put(validated.event_id, validated.features)
    result = predict_features(
        validated.features,
        event_id=validated.event_id,
        record_id=validated.record_id,
        true_label=validated.true_label,
    )
    payload = dict(vars(result))
    payload["source"] = "direct"
    return payload


simulation_engine: Optional[SimulationEngine] = None


def get_simulation_engine() -> SimulationEngine:
    global simulation_engine
    if simulation_engine is None:
        if settings.use_redis:
            from api.services.redis_service import RedisEventPublisher

            simulation_engine = SimulationEngine(RedisEventPublisher())
        else:
            simulation_engine = SimulationEngine(EventPublisher())
    return simulation_engine
