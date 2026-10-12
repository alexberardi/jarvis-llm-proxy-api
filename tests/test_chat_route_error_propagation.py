"""Non-200 responses from the model service must keep their own status class.

`api/chat_routes.py` flattened EVERY non-200 from `/internal/model/chat` into
`internal_server_error` / 500:

    if resp.status_code != 200:
        openai_error("internal_server_error", f"Model service error {...}", 500)

So a caller mistake surfaced as a server fault. The clearest live example:
posting an image to the non-vision `background` slot. The model service
correctly answers 400 `invalid_request_error` — "Model 'background' does not
support images" — and the gateway reported it as 500 with the real body
stringified inside the message. Clients cannot tell "you sent something
unsupported" from "the server is broken", and retry logic built on 5xx will
hammer a request that can never succeed.

Fix: pass the upstream status through, and unwrap the model service's
OpenAI-shaped error body instead of nesting it as text.

Run with:
    pytest tests/test_chat_route_error_propagation.py -v
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.chat_routes as chat_routes

MODEL_SERVICE_400 = {
    "detail": {
        "error": {
            "type": "invalid_request_error",
            "message": "Model 'background' does not support images. Use a vision-capable model instead.",
            "code": None,
        }
    }
}


def _app() -> FastAPI:
    app = FastAPI()
    # The router depends on require_app_auth, which calls out to jarvis-auth.
    # Replace it with a no-op so these tests exercise only error propagation.
    app.include_router(chat_routes.router)
    app.dependency_overrides[chat_routes.require_app_auth] = lambda: None
    return app


class _FakeResponse:
    def __init__(self, status: int, payload: dict):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    """Minimal stand-in for httpx.AsyncClient used as an async context manager.

    A MagicMock does not survive `async with` here: `client.post` ends up an
    AsyncMock attribute, so `resp.status_code` comes back as an AsyncMock
    instead of an int and the assertions pass/fail for the wrong reason.
    """

    def __init__(self, resp: _FakeResponse, *a, **k):
        self._resp = resp
        self.aclose = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, *a, **k):
        return self._resp


def _mock_post(status: int, payload: dict) -> _FakeAsyncClient:
    return _FakeAsyncClient(_FakeResponse(status, payload))


def _call(client_factory, body: dict | None = None):
    app = _app()
    headers = {"X-Jarvis-App-Id": "a", "X-Jarvis-App-Key": "b"}
    payload = body or {
        "model": "background",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
    }
    with patch.object(chat_routes.httpx, "AsyncClient", return_value=client_factory):
        return TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions", json=payload, headers=headers
        )


def test_auth_dependency_is_overridden():
    """Guard: if auth stops being overridden, every test here passes for the
    wrong reason (a 500 from the auth call, not from error flattening)."""
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    r = _call(_mock_post(200, {"content": "hi", "usage": usage, "tool_calls": None,
                               "date_keys": None, "finish_reason": "stop"}))
    assert r.status_code == 200, r.text


def test_model_service_400_surfaces_as_400():
    r = _call(_mock_post(400, MODEL_SERVICE_400))
    assert r.status_code == 400, f"client error reported as {r.status_code}"


def test_model_service_400_preserves_error_type():
    r = _call(_mock_post(400, MODEL_SERVICE_400))
    err = r.json()["detail"]["error"]
    assert err["type"] == "invalid_request_error"
    assert "does not support images" in err["message"]


def test_model_service_400_message_is_not_double_encoded():
    """The nested body must be unwrapped, not stringified into the message."""
    r = _call(_mock_post(400, MODEL_SERVICE_400))
    msg = r.json()["detail"]["error"]["message"]
    assert "{\"detail\"" not in msg, f"payload still nested as text: {msg!r}"
    assert "Model service error 400" not in msg


def test_model_service_422_surfaces_as_422():
    r = _call(_mock_post(422, MODEL_SERVICE_400))
    assert r.status_code == 422


def test_model_service_429_surfaces_as_429():
    """Retry logic keys off 429; flattening it to 500 breaks backoff."""
    r = _call(_mock_post(429, {"detail": {"error": {"type": "rate_limit_error",
                                                    "message": "slow down", "code": None}}}))
    assert r.status_code == 429


def test_model_service_500_stays_500():
    r = _call(_mock_post(500, {"detail": {"error": {"type": "internal_server_error",
                                                    "message": "boom", "code": None}}}))
    assert r.status_code == 500


def test_model_service_503_stays_503():
    r = _call(_mock_post(503, {"detail": {"error": {"type": "unavailable",
                                                    "message": "no slot", "code": None}}}))
    assert r.status_code == 503


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 429, 500, 502, 503, 504])
def test_all_status_codes_pass_through_unchanged(status):
    r = _call(_mock_post(status, {"detail": {"error": {"type": "x", "message": "y", "code": None}}}))
    assert r.status_code == status
