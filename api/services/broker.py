"""Broker facade: picks Kafka or Redis Streams based on MESSAGE_BROKER.

The rest of the API talks to this module instead of the concrete broker
services, so switching brokers is a single env var — no route or
dashboard changes needed:

    MESSAGE_BROKER=redis   # use Redis Streams
    MESSAGE_BROKER=kafka   # default: Kafka

The dashboard and all REST schemas are unchanged; only the transport
underneath differs.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ecg_core.config import settings

logger = logging.getLogger(__name__)


def broker_available() -> bool:
    """Is the *active* broker reachable?"""
    if settings.use_redis:
        from api.services.redis_service import redis_available

        return redis_available()
    from api.services.kafka_service import kafka_available

    return kafka_available()


def broker_name() -> str:
    return "redis" if settings.use_redis else "kafka"


def publish_event(event: dict) -> str:
    """Publish one validated event; returns the event id; raises on failure."""
    if settings.use_redis:
        from api.services.redis_service import redis_event_publisher

        return redis_event_publisher.publish(event)
    from api.services.kafka_service import event_publisher

    return event_publisher.publish(event)


def make_prediction_consumer(on_prediction: Callable[[dict], None]):
    """Build the background prediction consumer for the active broker."""
    if settings.use_redis:
        from api.services.redis_service import RedisPredictionConsumer

        return RedisPredictionConsumer(on_prediction=on_prediction)
    from api.services.kafka_service import PredictionConsumer

    return PredictionConsumer(on_prediction=on_prediction)


def stop_publishers() -> None:
    """Flush/close the active publisher (called on shutdown)."""
    try:
        if settings.use_redis:
            from api.services.redis_service import redis_event_publisher

            redis_event_publisher.close()
        else:
            from api.services.kafka_service import event_publisher

            event_publisher.close()
    except Exception:  # noqa: BLE001
        logger.warning("Failed to close event publisher", exc_info=True)
