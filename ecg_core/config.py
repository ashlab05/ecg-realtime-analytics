"""Central configuration loaded from environment variables with sane defaults.

No absolute paths are hardcoded — everything resolves relative to the
project root (the directory containing this package) unless overridden
via environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional; keeps ecg_core importable without python-dotenv
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _path_from_env(key: str, default: str) -> Path:
    """Resolve a possibly-relative path against the project root."""
    value = os.getenv(key, default)
    p = Path(value)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # --- General -----------------------------------------------------
    project_name: str = os.getenv("PROJECT_NAME", "ecg-realtime-analytics")
    random_seed: int = int(os.getenv("RANDOM_SEED", "42"))

    # --- Paths -------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _path_from_env("DATA_DIR", "data"))
    models_dir: Path = field(default_factory=lambda: _path_from_env("MODELS_DIR", "models"))
    logs_dir: Path = field(default_factory=lambda: _path_from_env("LOGS_DIR", "logs"))
    raw_data_dir: Path = field(default_factory=lambda: _path_from_env("DATA_DIR", "data") / "raw")
    processed_data_dir: Path = field(
        default_factory=lambda: _path_from_env("DATA_DIR", "data") / "processed"
    )

    # --- Dataset -----------------------------------------------------
    dataset_path: Path = field(
        default_factory=lambda: _path_from_env(
            "ECG_DATASET_PATH", "data/raw/MIT-BIH Arrhythmia Database.csv"
        )
    )

    # --- Kafka -------------------------------------------------------
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic_events: str = os.getenv("KAFKA_TOPIC_EVENTS", "ecg-events")
    kafka_topic_predictions: str = os.getenv("KAFKA_TOPIC_PREDICTIONS", "ecg-predictions")
    kafka_topic_errors: str = os.getenv("KAFKA_TOPIC_ERRORS", "ecg-errors")
    kafka_consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "ecg-analytics")

    # --- Redis Streams (experimental broker alternative) ----------------
    # MESSAGE_BROKER=kafka | redis selects which broker the API publishes
    # to and which prediction consumer it runs. Streams reuse the Kafka
    # topic names (ecg-events / ecg-predictions / ecg-errors).
    message_broker: str = os.getenv("MESSAGE_BROKER", "kafka")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_stream_maxlen: int = int(os.getenv("REDIS_STREAM_MAXLEN", "100000"))
    redis_block_ms: int = int(os.getenv("REDIS_BLOCK_MS", "1000"))

    @property
    def use_redis(self) -> bool:
        return self.message_broker.strip().lower() == "redis"

    # --- PostgreSQL ----------------------------------------------------
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://ecg:ecg_secret@localhost:5432/ecg_analytics",
    )
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "ecg_analytics")
    postgres_user: str = os.getenv("POSTGRES_USER", "ecg")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "ecg_secret")

    # --- API -----------------------------------------------------------
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))

    # --- Spark ---------------------------------------------------------
    spark_master: str = os.getenv("SPARK_MASTER", "local[*]")
    spark_checkpoint_dir: str = os.getenv("SPARK_CHECKPOINT_DIR", "./spark-checkpoints/ecg")
    spark_trigger_interval: str = os.getenv("SPARK_TRIGGER_INTERVAL", "5 seconds")
    spark_stream_batch_log: Path = field(
        default_factory=lambda: _path_from_env("LOGS_DIR", "logs") / "streaming" / "batch_metrics.csv"
    )

    # --- Model ---------------------------------------------------------
    ae_threshold_percentile: float = float(os.getenv("AE_THRESHOLD_PERCENTILE", "95"))
    ae_batch_size: int = int(os.getenv("AE_BATCH_SIZE", "256"))
    ae_epochs: int = int(os.getenv("AE_EPOCHS", "50"))
    ae_learning_rate: float = float(os.getenv("AE_LEARNING_RATE", "0.001"))

    # --- Derived artifact locations --------------------------------------
    @property
    def autoencoder_dir(self) -> Path:
        return self.models_dir / "autoencoder"

    @property
    def xgboost_dir(self) -> Path:
        return self.models_dir / "xgboost"

    @property
    def preprocessing_dir(self) -> Path:
        return self.models_dir / "preprocessing"

    @property
    def mlp_dir(self) -> Path:
        return self.models_dir / "mlp"

    @property
    def training_logs_dir(self) -> Path:
        return self.logs_dir / "training"

    @property
    def inference_logs_dir(self) -> Path:
        return self.logs_dir / "inference"

    @property
    def streaming_logs_dir(self) -> Path:
        return self.logs_dir / "streaming"

    def ensure_dirs(self) -> None:
        """Create all runtime directories if missing."""
        for d in (
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.autoencoder_dir,
            self.xgboost_dir,
            self.preprocessing_dir,
            self.mlp_dir,
            self.logs_dir,
            self.training_logs_dir,
            self.inference_logs_dir,
            self.streaming_logs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
