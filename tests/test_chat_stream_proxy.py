"""API-hop streaming proxy: error frames must be valid JSON, upstream must
close deterministically, and X-Request-Id must flow through both ways.

Pins the P0 failure-ladder item: the upstream-status error frame was a
single-quoted f-string (`data: {'error': ...}`) that no JSON parser accepts —
CC's client silently dropped it and hung with no done event.
"""

import importlib
import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    from tests.conftest import apply_auth_mock
    import main

    monkeypatch.setenv("MODEL_SERVICE_URL", "http://model-service.test")
    monkeypatch.setenv("MODEL_SERVICE_TOKEN", "test-token")
    importlib.reload(main)
    apply_auth_mock(main.app)
    return TestClient(main.app)


class _StubStreamResponse:
    def __init__(self, status_code: int, lines: list[str], headers: dict | None = None):
        self.status_code = status_code
        self._lines = lines
        self.headers = headers or {}
        self.closed = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def aclose(self):
        self.closed = True


class _StubClient:
    """Stands in for httpx.AsyncClient inside api.chat_routes."""

    last_response: _StubStreamResponse | None = None
    last_request_headers: dict | None = None
    next_status: int = 200
    next_lines: list[str] = []
    next_headers: dict = {}
    next_raise: Exception | None = None

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aclose(self):
        pass

    def stream(self, method, url, json=None, headers=None):
        _StubClient.last_request_headers = dict(headers or {})
        resp = _StubStreamResponse(
            _StubClient.next_status, _StubClient.next_lines, _StubClient.next_headers
        )
        _StubClient.last_response = resp

        class _Ctx:
            async def __aenter__(_self):
                if _StubClient.next_raise is not None:
                    raise _StubClient.next_raise
                return resp

            async def __aexit__(_self, *exc):
                await resp.aclose()
                return False

        return _Ctx()


@pytest.fixture
def stub_httpx(monkeypatch):
    import api.chat_routes as cr

    monkeypatch.setattr(cr.httpx, "AsyncClient", _StubClient)
    _StubClient.last_response = None
    _StubClient.last_request_headers = None
    _StubClient.next_status = 200
    _StubClient.next_lines = []
    _StubClient.next_headers = {}
    _StubClient.next_raise = None
    yield _StubClient


def _stream_body() -> dict:
    return {
        "model": "live",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }


def test_upstream_error_frame_is_valid_json(client, stub_httpx):
    stub_httpx.next_status = 503

    with client.stream("POST", "/v1/chat/completions", json=_stream_body()) as resp:
        frames = [
            json.loads(line[len("data: "):])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]

    assert len(frames) == 1
    assert "error" in frames[0]
    assert "503" in frames[0]["error"]


def test_transport_error_yields_error_frame(client, stub_httpx):
    """Connect/read failures must end the stream with a valid {"error"} frame,
    not a bare truncation the consumer can't distinguish from success."""
    import httpx as real_httpx

    stub_httpx.next_raise = real_httpx.ConnectError("connection refused")

    with client.stream("POST", "/v1/chat/completions", json=_stream_body()) as resp:
        frames = [
            json.loads(line[len("data: "):])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]

    assert len(frames) == 1
    assert "connection" in frames[0]["error"].lower()


def test_stream_passthrough_preserves_frames(client, stub_httpx):
    stub_httpx.next_lines = [
        'data: {"delta": "hel"}',
        'data: {"delta": "lo"}',
        'data: {"done": true, "content": "hello", "usage": {}}',
    ]

    with client.stream("POST", "/v1/chat/completions", json=_stream_body()) as resp:
        frames = [
            json.loads(line[len("data: "):])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]

    assert frames[0] == {"delta": "hel"}
    assert frames[-1]["done"] is True
    # Upstream stream context exited → explicitly closed.
    assert stub_httpx.last_response.closed


def test_request_id_forwarded_and_echoed(client, stub_httpx):
    stub_httpx.next_lines = ['data: {"done": true, "content": "", "usage": {}}']

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json=_stream_body(),
        headers={"X-Request-Id": "phone-turn-7"},
    ) as resp:
        assert resp.headers.get("X-Request-Id") == "phone-turn-7"
        for _ in resp.iter_lines():
            pass

    assert stub_httpx.last_request_headers.get("X-Request-Id") == "phone-turn-7"


def test_request_id_generated_when_absent(client, stub_httpx):
    stub_httpx.next_lines = ['data: {"done": true, "content": "", "usage": {}}']

    with client.stream("POST", "/v1/chat/completions", json=_stream_body()) as resp:
        generated = resp.headers.get("X-Request-Id")
        assert generated
        for _ in resp.iter_lines():
            pass

    # The same generated id was forwarded upstream.
    assert stub_httpx.last_request_headers.get("X-Request-Id") == generated


def test_cancel_proxy_forwards_to_model_service(client, monkeypatch):
    import api.chat_routes as cr

    calls: dict = {}

    class _CancelClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None):
            calls["url"] = url
            calls["headers"] = dict(headers or {})

            class R:
                status_code = 200

                @staticmethod
                def json():
                    return {"status": "cancelling", "request_id": "abc"}

            return R()

    monkeypatch.setattr(cr.httpx, "AsyncClient", _CancelClient)

    resp = client.post("/v1/chat/completions/cancel/abc")

    assert resp.status_code == 200
    assert resp.json()["status"] == "cancelling"
    assert calls["url"].endswith("/internal/model/cancel/abc")
    assert calls["headers"].get("X-Internal-Token") == "test-token"
