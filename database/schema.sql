-- ============================================================
-- ECG Real-Time Analytics — PostgreSQL schema
-- Apply with: psql -f database/schema.sql  (or automatically by the
-- FastAPI service at startup via SQLAlchemy metadata.create_all)
-- ============================================================

CREATE TABLE IF NOT EXISTS ecg_predictions (
    id                    BIGSERIAL PRIMARY KEY,
    event_id              VARCHAR(64)  NOT NULL,
    timestamp             TIMESTAMPTZ  NOT NULL,
    record_id             VARCHAR(16),
    predicted_class       VARCHAR(8)   NOT NULL,
    is_abnormal           BOOLEAN      NOT NULL,
    confidence            DOUBLE PRECISION,
    reconstruction_error  DOUBLE PRECISION,
    anomaly_threshold     DOUBLE PRECISION,
    inference_latency_ms  DOUBLE PRECISION,
    processing_time_ms    DOUBLE PRECISION,
    source                VARCHAR(32)  NOT NULL DEFAULT 'kafka',
    true_label            VARCHAR(8),
    stage                 VARCHAR(32),
    created_at            TIMESTAMPTZ  NOT NULL DEFAULT now()
);

-- Indexes for the dashboard's common query patterns.
CREATE INDEX IF NOT EXISTS idx_ecg_predictions_timestamp     ON ecg_predictions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ecg_predictions_class         ON ecg_predictions (predicted_class);
CREATE INDEX IF NOT EXISTS idx_ecg_predictions_abnormal      ON ecg_predictions (is_abnormal);
CREATE INDEX IF NOT EXISTS idx_ecg_predictions_event_id      ON ecg_predictions (event_id);
