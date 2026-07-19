"""Streaming disconnect-abort, cancel endpoint, and event-loop liveness.

Regression pins for the 2026-07-18 phone-call P0 incident: an abandoned
/internal/model/chat/stream generation kept grinding inside llama.cpp while
holding the backend lock, wedging every later request — and a sync
non-streaming request blocking the event loop on that same lock could
deadlock the service permanently (/health included).

The fake backend below mimics gguf_backend's lock discipline exactly: the
WHOLE generation (every yield) runs under one threading.Lock, and the
non-streaming path acquires the same lock.
"""

import asyncio
import json
import os
import threading
import time

import pytest
from fastapi.testclient import TestClient

from managers.chat_types import ChatResult
from managers.model_manager import ModelConfig, ModelManager

# services.model_service imports api.settings_routes, which resolves the auth
# URL at import time (same pattern as tests/test_model_service_health.py).
os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

TOKEN = "test-token"
AUTH = {"X-Internal-Token": TOKEN}


class FakeLockedBackend:
    """Sync streaming backend with gguf_backend's lock discipline."""

    def __init__(self, tokens: list[str] | None = None, delay: float = 0.01):
        self._lock = threading.Lock()
        self.generator_closed = threading.Event()
        self.tokens = tokens  # None → stream forever
        self.delay = delay

    def generate_text_chat_stream(self, model_cfg, messages, params):
        with self._lock:
            try:
                if self.tokens is None:
                    i = 0
                    while True:
                        time.sleep(self.delay)
                        yield {"delta": f"tok{i} "}
                        i += 1
                else:
                    content = ""
                    for tok in self.tokens:
                        time.sleep(self.delay)
                        content += tok
                        yield {"delta": tok}
                    yield {
                        "done": True,
                        "content": content,
                        "usage": {"completion_tokens": len(self.tokens)},
                        "tool_calls": None,
                        "finish_reason": "stop",
                    }
            finally:
                self.generator_closed.set()

    def generate_text_chat(self, model_cfg, messages, params):
        with self._lock:
            return ChatResult(content="sync-response", usage={})


@pytest.fixture
def model_service(monkeypatch):
    """Fresh singleton bound into services.model_service with no-op load_all."""
    ModelManager._instance = None
    ModelManager._initialized = False

    import services.model_service as ms

    fresh = ModelManager(auto_load=False)
    monkeypatch.setattr(fresh, "load_all", lambda: None)
    monkeypatch.setattr(ms, "model_manager", fresh)
    monkeypatch.setattr(ms, "_startup_done", False)
    monkeypatch.setenv("MODEL_SERVICE_TOKEN", TOKEN)

    yield ms

    ModelManager._instance = None
    ModelManager._initialized = False


def _install_fake_backend(ms, backend) -> None:
    cfg = ModelConfig("fake-live", "MOCK", backend)
    ms.model_manager.registry["fake-live"] = cfg
    ms.model_manager.aliases["live"] = "fake-live"
    ms.model_manager.live_model = backend
    ms.model_manager._set_slot_state("live", "ready")


def _chat_body() -> dict:
    return {"model": "live", "messages": [{"role": "user", "content": "hi"}]}


# ---------------------------------------------------------------------------
# Raw-ASGI stream driver
#
# TestClient cannot simulate a mid-stream client disconnect: it joins the
# response task instead of cancelling it. Real servers (uvicorn) cancel the
# task when the socket closes — that cancellation is exactly the signal the
# abort machinery keys off, so these tests drive the ASGI app directly.
# ---------------------------------------------------------------------------


class AsgiStream:
    """One in-flight streaming request against the raw ASGI app."""

    def __init__(self, app, path: str, body: dict, headers: dict[str, str]):
        self._app = app
        self._path = path
        self._body = json.dumps(body).encode()
        self._headers = headers
        self._chunks: asyncio.Queue = asyncio.Queue()
        self.response_headers: dict[str, str] = {}
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        body = self._body
        sent = False

        async def receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            await asyncio.Event().wait()  # no http.disconnect; parity = cancel

        async def send(message):
            await self._chunks.put(message)

        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ] + [(k.lower().encode(), v.encode()) for k, v in self._headers.items()]

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": self._path,
            "raw_path": self._path.encode(),
            "query_string": b"",
            "root_path": "",
            "headers": headers,
            "client": ("testclient", 123),
            "server": ("testserver", 80),
        }
        self.task = asyncio.create_task(self._app(scope, receive, send))

        start_msg = await asyncio.wait_for(self._chunks.get(), timeout=5.0)
        assert start_msg["type"] == "http.response.start", start_msg
        assert start_msg["status"] == 200, start_msg
        self.response_headers = {
            k.decode(): v.decode() for k, v in start_msg.get("headers", [])
        }

    async def read_data_frames(self, n: int) -> list[dict]:
        """Read until n SSE data frames arrive (buffers partial chunks)."""
        frames: list[dict] = []
        buffer = ""
        while len(frames) < n:
            msg = await asyncio.wait_for(self._chunks.get(), timeout=5.0)
            if msg["type"] != "http.response.body":
                continue
            buffer += msg.get("body", b"").decode()
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if block.startswith("data: "):
                    frames.append(json.loads(block[len("data: "):]))
            if not msg.get("more_body", True) and len(frames) < n:
                raise AssertionError(
                    f"stream ended after {len(frames)} frames, wanted {n}"
                )
        return frames

    async def read_to_end(self) -> list[dict]:
        """Read every remaining frame until the response body completes."""
        frames: list[dict] = []
        buffer = ""
        while True:
            msg = await asyncio.wait_for(self._chunks.get(), timeout=5.0)
            if msg["type"] != "http.response.body":
                continue
            buffer += msg.get("body", b"").decode()
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                if block.startswith("data: "):
                    frames.append(json.loads(block[len("data: "):]))
            if not msg.get("more_body", True):
                return frames

    async def disconnect(self) -> None:
        """What uvicorn does when the client socket closes mid-response."""
        assert self.task is not None
        self.task.cancel()
        try:
            await self.task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Baseline: the stream still works end-to-end after the pump rewrite
# ---------------------------------------------------------------------------


def test_stream_completes_normally(model_service):
    ms = model_service
    backend = FakeLockedBackend(tokens=["a", "b", "c"])

    with TestClient(ms.app) as client:
        _install_fake_backend(ms, backend)

        events = []
        with client.stream(
            "POST", "/internal/model/chat/stream", headers=AUTH, json=_chat_body()
        ) as resp:
            assert resp.status_code == 200
            assert resp.headers.get("X-Request-Id")
            for line in resp.iter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[len("data: "):]))

    assert [e["delta"] for e in events[:3]] == ["a", "b", "c"]
    assert events[-1]["done"] is True
    assert events[-1]["content"] == "abc"
    # Normal completion also finalizes the generator (lock released).
    assert backend.generator_closed.wait(2.0)
    assert backend._lock.acquire(timeout=2.0)
    backend._lock.release()


# ---------------------------------------------------------------------------
# The wedge: client disconnect must abort generation and release the lock
# ---------------------------------------------------------------------------


def test_disconnect_aborts_generation_and_releases_lock(model_service):
    ms = model_service
    backend = FakeLockedBackend(tokens=None)  # streams forever
    _install_fake_backend(ms, backend)

    async def drive():
        stream = AsgiStream(
            ms.app, "/internal/model/chat/stream", _chat_body(), AUTH
        )
        await stream.start()
        await stream.read_data_frames(3)
        await stream.disconnect()  # abandon mid-generation

    asyncio.run(drive())

    # Disconnect must finalize the generator (GeneratorExit at the next
    # yield) and release the backend lock — bounded, not GC-dependent.
    assert backend.generator_closed.wait(5.0), (
        "generator was never closed after client disconnect"
    )
    assert backend._lock.acquire(timeout=5.0), (
        "backend lock still held after client disconnect"
    )
    backend._lock.release()


def test_orphaned_stream_does_not_poison_next_request(model_service):
    """The P0 incident shape: abandon a stream, then require the very next
    non-streaming request to succeed within a bounded time."""
    ms = model_service
    backend = FakeLockedBackend(tokens=None)
    _install_fake_backend(ms, backend)

    async def orphan():
        stream = AsgiStream(
            ms.app, "/internal/model/chat/stream", _chat_body(), AUTH
        )
        await stream.start()
        await stream.read_data_frames(1)  # first token is enough — abandon
        await stream.disconnect()

    asyncio.run(orphan())

    with TestClient(ms.app) as client:
        result: dict = {}

        def _next_request():
            r = client.post("/internal/model/chat", headers=AUTH, json=_chat_body())
            result["status"] = r.status_code
            result["body"] = r.json()

        t = threading.Thread(target=_next_request, daemon=True)
        t.start()
        t.join(timeout=10.0)

        assert not t.is_alive(), (
            "non-streaming request wedged behind an orphaned stream generation"
        )
        assert result["status"] == 200
        assert result["body"]["content"] == "sync-response"


# ---------------------------------------------------------------------------
# Cancel endpoint
# ---------------------------------------------------------------------------


def test_cancel_endpoint_aborts_stream(model_service):
    ms = model_service
    backend = FakeLockedBackend(tokens=None)
    _install_fake_backend(ms, backend)

    events: list[dict] = []
    request_id_box: dict = {}

    async def drive():
        stream = AsgiStream(
            ms.app, "/internal/model/chat/stream", _chat_body(), AUTH
        )
        await stream.start()
        request_id = stream.response_headers.get("x-request-id")
        assert request_id
        request_id_box["id"] = request_id

        events.extend(await stream.read_data_frames(3))

        # Cancel while the stream is live (thread, so this loop stays free
        # to keep consuming — mirrors a second caller hitting the endpoint).
        def _cancel():
            with TestClient(ms.app) as client:
                r = client.post(
                    f"/internal/model/cancel/{request_id}", headers=AUTH
                )
                assert r.status_code == 200
                assert r.json()["status"] == "cancelling"

        cancel_thread = threading.Thread(target=_cancel, daemon=True)
        cancel_thread.start()

        events.extend(await stream.read_to_end())
        cancel_thread.join(timeout=5.0)
        assert not cancel_thread.is_alive()
        assert stream.task is not None
        await stream.task

    asyncio.run(drive())

    assert backend.generator_closed.wait(5.0)
    assert events[-1].get("cancelled") is True

    # Registry entry is gone → a second cancel 404s.
    with TestClient(ms.app) as client:
        resp = client.post(
            f"/internal/model/cancel/{request_id_box['id']}", headers=AUTH
        )
        assert resp.status_code == 404


def test_cancel_unknown_request_id_404s(model_service):
    ms = model_service
    with TestClient(ms.app) as client:
        resp = client.post("/internal/model/cancel/nope", headers=AUTH)
        assert resp.status_code == 404


def test_caller_supplied_request_id_is_honored(model_service):
    ms = model_service
    backend = FakeLockedBackend(tokens=["x"])

    with TestClient(ms.app) as client:
        _install_fake_backend(ms, backend)
        with client.stream(
            "POST",
            "/internal/model/chat/stream",
            headers={**AUTH, "X-Request-Id": "phone-turn-42"},
            json=_chat_body(),
        ) as resp:
            assert resp.headers.get("X-Request-Id") == "phone-turn-42"
            for _ in resp.iter_lines():
                pass


# ---------------------------------------------------------------------------
# Event-loop liveness: sync generation must not block /health
# ---------------------------------------------------------------------------


def test_health_responsive_during_sync_generation(model_service):
    ms = model_service

    class SlowSyncBackend(FakeLockedBackend):
        def generate_text_chat(self, model_cfg, messages, params):
            with self._lock:
                time.sleep(1.5)
                return ChatResult(content="slow-response", usage={})

    backend = SlowSyncBackend()

    with TestClient(ms.app) as client:
        _install_fake_backend(ms, backend)

        chat_done = threading.Event()

        def _slow_chat():
            client.post("/internal/model/chat", headers=AUTH, json=_chat_body())
            chat_done.set()

        t = threading.Thread(target=_slow_chat, daemon=True)
        t.start()
        time.sleep(0.2)  # let the generation start and take the lock

        start = time.monotonic()
        resp = client.get("/health")
        elapsed = time.monotonic() - start

        assert resp.status_code == 200
        assert elapsed < 1.0, (
            f"/health took {elapsed:.2f}s — event loop blocked by sync generation"
        )
        assert chat_done.wait(10.0)
