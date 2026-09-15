"""API tests: endpoints, payload validation, direct-mode inference, persistence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ecg_core.features import FEATURE_COLUMNS, N_FEATURES


@pytest.fixture()
def client(test_bundle, models_dir, tmp_path, monkeypatch):
    """TestClient with: injected test models, sqlite DB, Kafka disabled."""
    import api.routes.ecg as ecg_route
    import database.models as db_models
    import inference.model_loader as model_loader

    # 1. Model bundle injection (skips disk loading in get_model_bundle).
    monkeypatch.setattr(model_loader, "_bundle", test_bundle, raising=False)

    # 2. Point model-status endpoint at the test artifacts.
    monkeypatch.setattr(model_loader, "settings", _test_settings(models_dir), raising=False)

    # 3. SQLite database (URL is read lazily per engine creation).
    monkeypatch.setattr(
        db_models, "settings", _test_settings(tmp_path, sqlite=True), raising=False
    )
    monkeypatch.setattr(db_models, "_engine", None, raising=False)
    monkeypatch.setattr(db_models, "_SessionFactory", None, raising=False)

    # 4. Kafka always "down" so /api/ecg/send uses the direct fallback path.
    monkeypatch.setattr(ecg_route, "kafka_available", lambda: False, raising=False)

    from api.main import app

    with TestClient(app) as c:
        yield c


def _test_settings(tmp_path, sqlite: bool = False):
    from ecg_core.config import Settings

    s = Settings()
    # Derived dirs (preprocessing/autoencoder/xgboost) are read-only
    # properties computed from models_dir, so patching the base is enough.
    object.__setattr__(s, "models_dir", tmp_path)
    object.__setattr__(s, "logs_dir", tmp_path / "logs")
    if sqlite:
        object.__setattr__(s, "database_url", f"sqlite:///{tmp_path}/test.db")
    return s


def _valid_payload():
    lead = {
        "pre_rr": 0.76, "post_rr": 0.81,
        "p_peak": 0.12, "t_peak": 0.35, "r_peak": 1.10, "s_peak": -0.20, "q_peak": -0.15,
        "qrs_interval": 94, "pq_interval": 18, "qt_interval": 66, "st_interval": 7,
        "qrs_morph_0": 0.1, "qrs_morph_1": 0.2, "qrs_morph_2": 0.15, "qrs_morph_3": 0.05,
        "qrs_morph_4": 0.08,
    }
    lead_v = dict(lead, r_peak=0.95)
    return {
        "event_id": "evt-test-1",
        "record_id": "100",
        "true_label": "N",
        "lead_ii": lead,
        "lead_v5": lead_v,
    }


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_send_ecg_event_direct_mode(client):
    resp = client.post("/api/ecg/send", json=_valid_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["published_to_kafka"] is False  # Kafka disabled in tests
    assert body["event_id"] == "evt-test-1"
    pred = body["prediction"]
    assert pred["event_id"] == "evt-test-1"
    assert pred["predicted_class"] in ("N", "SVEB", "VEB", "F", "Q", "UNKNOWN")
    assert pred["anomaly_threshold"] > 0
    assert pred["reconstruction_error"] >= 0
    assert pred["latency_ms"] >= 0


def test_send_ecg_event_invalid_payload(client):
    payload = _valid_payload()
    del payload["lead_ii"]["r_peak"]  # missing a required feature
    resp = client.post("/api/ecg/send", json=payload)
    assert resp.status_code == 422


def test_send_ecg_event_non_finite(client):
    payload = _valid_payload()
    payload["lead_ii"]["r_peak"] = float("inf")
    # Strict JSON serialization refuses non-finite floats before the
    # request is even sent — the API never sees infinities over the wire.
    with pytest.raises(ValueError, match="not JSON compliant"):
        client.post("/api/ecg/send", json=payload)


def test_models_status(client):
    resp = client.get("/api/models/status")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) >= {"preprocessing", "autoencoder", "xgboost_abnormal", "feature_columns"}
    assert len(body["feature_columns"]) == N_FEATURES


def test_threshold_endpoint(client):
    resp = client.get("/api/threshold")
    # 200 when artifacts exist under the configured models dir (real or test),
    # 503 with a helpful message when they do not.
    assert resp.status_code in (200, 503)
    if resp.status_code == 200:
        assert resp.json()["anomaly_threshold"] > 0


def test_predictions_flow(client):
    """Send an event (direct mode) then read it back from /api/predictions."""
    client.post("/api/ecg/send", json=_valid_payload())
    resp = client.get("/api/predictions", params={"limit": 10})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) >= 1
    assert rows[0]["event_id"] == "evt-test-1"

    latest = client.get("/api/predictions/latest")
    assert latest.status_code == 200
    assert latest.json()["event_id"] == "evt-test-1"


def test_statistics(client):
    client.post("/api/ecg/send", json=_valid_payload())
    resp = client.get("/api/statistics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_events"] >= 1
    assert body["normal_events"] + body["abnormal_events"] == body["total_events"]


def test_system_status(client):
    resp = client.get("/api/system/status")
    assert resp.status_code == 200
    body = resp.json()
    for component in ("api", "kafka", "spark", "database", "models"):
        assert component in body
    assert body["api"]["status"] == "up"


def test_ecg_classes(client):
    resp = client.get("/api/ecg/classes")
    assert resp.status_code == 200
    classes = {c["code"] for c in resp.json()["classes"]}
    assert {"N", "SVEB", "VEB", "F", "Q"} <= classes


def test_feature_columns_consistency(client):
    """API feature metadata must match the canonical training order."""
    resp = client.get("/api/models/status")
    assert resp.json()["feature_columns"] == FEATURE_COLUMNS
