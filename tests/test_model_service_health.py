"""Model service /health body + 503 model_not_loaded behavior.

services.model_service is imported inside the fixture with load_all patched
out, so TestClient startup never loads a real backend — tests drive
model_manager.model_states directly.
"""

import os

import pytest
from fastapi.testclient import TestClient

from managers.model_manager import ModelManager

# services.model_service imports api.settings_routes, which resolves the auth
# URL at import time (same pattern as tests/test_settings_routes.py).
os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")


@pytest.fixture
def model_service(monkeypatch):
    """Fresh singleton bound into services.model_service with no-op load_all."""
    ModelManager._instance = None
    ModelManager._initialized = False

    import services.model_service as ms

    fresh = ModelManager(auto_load=False)
    monkeypatch.setattr(fresh, "load_all", lambda: None)
    monkeypatch.setattr(ms, "model_manager", fresh)
    # Let the startup hook run for THIS test (records uptime, marks loading).
    monkeypatch.setattr(ms, "_startup_done", False)

    yield ms

    ModelManager._instance = None
    ModelManager._initialized = False


def test_health_ok_when_live_ready(model_service):
    ms = model_service
    with TestClient(ms.app) as client:
        ms.model_manager._set_slot_state("live", "ready")
        ms.model_manager._set_slot_state("background", "ready")

        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    # Backward-compatible keys keep their exact meaning.
    assert body["status"] == "ok"
    assert "models" in body
    assert "aliases" in body
    # New keys.
    assert body["slots"]["live"]["status"] == "ready"
    assert body["slots"]["background"]["status"] == "ready"
    assert "last_attempt_seconds_ago" in body["slots"]["live"]
    assert "last_attempt_monotonic" not in body["slots"]["live"]
    assert body["uptime_s"] is not None and body["uptime_s"] >= 0
    assert body["started_at"] is not None


def test_health_degraded_when_live_failed(model_service):
    ms = model_service
    with TestClient(ms.app) as client:
        ms.model_manager._set_slot_state(
            "live", "failed", error="RuntimeError: boom", count_attempt=True
        )

        resp = client.get("/health")

    assert resp.status_code == 200  # process alive → 200; API layer aggregates
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["slots"]["live"]["status"] == "failed"
    assert "boom" in body["slots"]["live"]["error"]
    assert body["slots"]["live"]["last_attempt_seconds_ago"] is not None


def test_health_loading_while_models_load(model_service):
    ms = model_service
    with TestClient(ms.app) as client:
        # startup hook marks slots "loading" before the loader thread runs
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "loading"
    assert body["slots"]["live"]["status"] == "loading"
    assert "models" in body and "aliases" in body


def test_chat_returns_503_when_live_not_loaded(model_service, monkeypatch):
    ms = model_service
    monkeypatch.setenv("MODEL_SERVICE_TOKEN", "test-token")

    with TestClient(ms.app) as client:
        ms.model_manager._set_slot_state(
            "live", "failed", error="RuntimeError: boom", count_attempt=True
        )
        assert ms.model_manager.live_model is None

        resp = client.post(
            "/internal/model/chat",
            headers={"X-Internal-Token": "test-token"},
            json={
                "model": "live",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 503
    err = resp.json()["detail"]["error"]
    assert err["code"] == "model_not_loaded"
    assert err["type"] == "model_not_loaded"
    assert err["slot"] == "live"
    assert "failed" in err["message"]
    assert err["state"]["status"] == "failed"


def test_chat_stream_returns_503_when_live_not_loaded(model_service, monkeypatch):
    ms = model_service
    monkeypatch.setenv("MODEL_SERVICE_TOKEN", "test-token")

    with TestClient(ms.app) as client:
        ms.model_manager._set_slot_state(
            "live", "failed", error="RuntimeError: boom", count_attempt=True
        )

        resp = client.post(
            "/internal/model/chat/stream",
            headers={"X-Internal-Token": "test-token"},
            json={
                "model": "live",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 503
    assert resp.json()["detail"]["error"]["code"] == "model_not_loaded"
