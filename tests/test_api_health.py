"""API-side /health HTTP status semantics (api/health_routes.py).

The docker healthcheck does urlopen('http://localhost:7704/health') and fails
only on non-2xx — these tests pin the status codes that finally make a dead
model service surface as an unhealthy container.

respx is not in the venv, so httpx.AsyncClient is stubbed directly.
"""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.health_routes as health_routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(health_routes.router)
    return TestClient(app)


@pytest.fixture
def model_service_url(monkeypatch):
    monkeypatch.setattr(
        health_routes, "get_setting", lambda key, env, default: "http://ms:7705"
    )


@pytest.fixture
def default_timeout(monkeypatch):
    """Simulate the (insane-for-healthchecks) 60s configured inference timeout."""
    monkeypatch.setattr(
        health_routes, "get_float_setting", lambda key, env, default: 60.0
    )


def _stub_httpx(monkeypatch, response=None, exc=None) -> dict:
    """Replace httpx.AsyncClient with a stub; returns captured ctor kwargs."""
    captured: dict = {}
    inner = AsyncMock()
    if exc is not None:
        inner.get.side_effect = exc
    else:
        inner.get.return_value = response

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return inner

        async def __aexit__(self, *exc_info):
            return False

    monkeypatch.setattr(health_routes.httpx, "AsyncClient", FakeAsyncClient)
    return captured


def _response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    resp.text = str(body)
    return resp


# ---------------------------------------------------------------------------
# The five status-code scenarios
# ---------------------------------------------------------------------------


def test_no_model_service_url_is_deliberate_200_degraded(client, monkeypatch):
    """macOS dev passthrough posture: keep the container green."""
    monkeypatch.setattr(health_routes, "get_setting", lambda key, env, default: "")

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "degraded", "reason": "MODEL_SERVICE_URL not set"}


def test_live_ready_is_200_healthy(client, monkeypatch, model_service_url, default_timeout):
    body = {
        "status": "ok",
        "models": ["m"],
        "aliases": {"live": "m"},
        "slots": {"live": {"status": "ready"}},
        "uptime_s": 42.0,
    }
    _stub_httpx(monkeypatch, response=_response(body))

    resp = client.get("/health")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "healthy"
    assert payload["model_service"]["slots"]["live"]["status"] == "ready"


def test_loading_within_grace_is_200_initializing(
    client, monkeypatch, model_service_url, default_timeout
):
    body = {
        "status": "loading",
        "models": [],
        "aliases": {},
        "slots": {"live": {"status": "loading"}},
        "uptime_s": 42.0,
    }
    _stub_httpx(monkeypatch, response=_response(body))

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "initializing"


def test_loading_without_uptime_is_treated_as_within_grace(
    client, monkeypatch, model_service_url, default_timeout
):
    """Older model service body lacks uptime_s → assume within grace."""
    body = {
        "status": "loading",
        "models": [],
        "aliases": {},
        "slots": {"live": {"status": "loading"}},
    }
    _stub_httpx(monkeypatch, response=_response(body))

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "initializing"


def test_live_failed_is_503(client, monkeypatch, model_service_url, default_timeout):
    body = {
        "status": "degraded",
        "models": [],
        "aliases": {},
        "slots": {"live": {"status": "failed", "error": "RuntimeError: boom"}},
        "uptime_s": 42.0,
    }
    _stub_httpx(monkeypatch, response=_response(body))

    resp = client.get("/health")

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["status"] == "degraded"
    assert "reason" in payload


def test_loading_past_grace_is_503(client, monkeypatch, model_service_url, default_timeout):
    body = {
        "status": "loading",
        "models": [],
        "aliases": {},
        "slots": {"live": {"status": "loading"}},
        "uptime_s": 1200.0,  # > 900s grace window
    }
    _stub_httpx(monkeypatch, response=_response(body))

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"


def test_model_service_unreachable_is_503(
    client, monkeypatch, model_service_url, default_timeout
):
    _stub_httpx(monkeypatch, exc=httpx.ConnectError("connection refused"))

    resp = client.get("/health")

    assert resp.status_code == 503
    payload = resp.json()
    assert payload["status"] == "degraded"
    assert "reason" in payload


def test_model_service_timeout_is_503(
    client, monkeypatch, model_service_url, default_timeout
):
    _stub_httpx(monkeypatch, exc=httpx.TimeoutException("timed out"))

    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["reason"] == "Model service health check timed out"


def test_model_service_error_status_is_503(
    client, monkeypatch, model_service_url, default_timeout
):
    _stub_httpx(monkeypatch, response=_response({}, status_code=500))

    resp = client.get("/health")

    assert resp.status_code == 503
    assert "Model service error 500" in resp.json()["reason"]


def test_old_model_service_body_without_slots_is_healthy(
    client, monkeypatch, model_service_url, default_timeout
):
    """Older model service always reported status 'ok' with no slots key."""
    body = {"status": "ok", "models": ["m"], "aliases": {}}
    _stub_httpx(monkeypatch, response=_response(body))

    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


# ---------------------------------------------------------------------------
# Probe timeout cap
# ---------------------------------------------------------------------------


def test_probe_timeout_capped_at_5s(client, monkeypatch, model_service_url, default_timeout):
    body = {"status": "ok", "models": [], "aliases": {}}
    captured = _stub_httpx(monkeypatch, response=_response(body))

    client.get("/health")

    assert captured["timeout"] <= 5.0


def test_probe_timeout_uses_configured_value_when_lower(
    client, monkeypatch, model_service_url
):
    monkeypatch.setattr(
        health_routes, "get_float_setting", lambda key, env, default: 2.0
    )
    body = {"status": "ok", "models": [], "aliases": {}}
    captured = _stub_httpx(monkeypatch, response=_response(body))

    client.get("/health")

    assert captured["timeout"] == 2.0


def test_legacy_v1_health_route_same_semantics(
    client, monkeypatch, model_service_url, default_timeout
):
    _stub_httpx(monkeypatch, exc=httpx.ConnectError("connection refused"))

    resp = client.get("/v1/health")

    assert resp.status_code == 503
