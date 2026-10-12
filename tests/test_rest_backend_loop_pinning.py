"""RestClient must pin ALL httpx work to one stable event loop.

`RestClient` keeps a long-lived `httpx.AsyncClient`. Its connection pool holds
asyncio primitives that bind to whichever loop first uses them. `generate_text_chat`
already forces its work onto a dedicated background loop for exactly this reason.

`generate_vision_chat` was async and awaited `self.chat_completion()` directly, so
it ran on the *caller's* loop — the same client on two loops. The pooled connection
then raised "Event loop is closed" / "<Event> is bound to a different event loop",
surfacing as intermittent 500s on TEXT requests (whichever call grabbed a poisoned
connection first).

This was latent while `supports_images` was hardcoded False; enabling vision made
`generate_vision_chat` reachable and the collision real.

Run with:
    pytest tests/test_rest_backend_loop_pinning.py -v
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams
from models.api_models import Message
from services.message_service import normalize_message


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "backends.rest_backend.get_setting",
        lambda key, env_fallback, default, **k: (
            "openai" if key == "rest.provider" else default
        ),
    )
    c = RestClient(base_url="http://llama:8080/", model_name="qwen3-vl")
    c.client = MagicMock()
    c.client.post = AsyncMock(
        return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={"choices": [{"message": {"content": "ok"}}], "usage": {}}
            ),
        )
    )
    return c


def _vision_messages():
    return [
        normalize_message(
            Message(
                role="user",
                content=[{"type": "text", "text": "what cards?"}],
            )
        )
    ]


def _text_messages():
    return [
        normalize_message(
            Message(
                role="user",
                content=[{"type": "text", "text": "hello"}],
            )
        )
    ]


def test_vision_request_runs_on_background_loop(client):
    """The httpx call for a vision request must NOT use the caller's loop."""
    seen = {}
    real_post = client.client.post

    async def spy(*a, **k):
        seen["loop"] = asyncio.get_running_loop()
        return await real_post(*a, **k)

    client.client.post = spy

    asyncio.run(client.generate_vision_chat(MagicMock(), _vision_messages(), GenerationParams()))

    bg = client._bg_loop
    assert bg is not None, "background loop should have been created"
    assert seen["loop"] is bg


def test_text_request_runs_on_background_loop(client):
    seen = {}
    real_post = client.client.post

    async def spy(*a, **k):
        seen["loop"] = asyncio.get_running_loop()
        return await real_post(*a, **k)

    client.client.post = spy

    # generate_text_chat is sync (it blocks on the bg-loop future)
    client.generate_text_chat(MagicMock(), _text_messages(), GenerationParams())

    assert seen["loop"] is client._bg_loop


def test_vision_and_text_share_one_loop(client):
    """The whole point: one client, one loop, no cross-loop primitives."""
    loops = []
    real_post = client.client.post

    async def spy(*a, **k):
        loops.append(asyncio.get_running_loop())
        return await real_post(*a, **k)

    client.client.post = spy

    async def both():
        await client.generate_vision_chat(MagicMock(), _vision_messages(), GenerationParams())
        await asyncio.to_thread(
            client.generate_text_chat, MagicMock(), _text_messages(), GenerationParams()
        )
        await client.generate_vision_chat(MagicMock(), _vision_messages(), GenerationParams())

    asyncio.run(both())

    assert len(loops) == 3
    assert all(loop is client._bg_loop for loop in loops)
    assert len({id(loop) for loop in loops}) == 1


def test_vision_still_returns_content(client):
    result = asyncio.run(
        client.generate_vision_chat(MagicMock(), _vision_messages(), GenerationParams())
    )
    assert result.content == "ok"


def test_generate_vision_chat_remains_a_coroutine_function(client):
    """chat_runner branches on this; changing it would reroute to the sync path."""
    assert asyncio.iscoroutinefunction(client.generate_vision_chat)


def test_background_loop_is_shared_across_calls(client):
    asyncio.run(client.generate_vision_chat(MagicMock(), _vision_messages(), GenerationParams()))
    first = client._bg_loop
    client.generate_text_chat(MagicMock(), _text_messages(), GenerationParams())
    assert client._bg_loop is first


def test_repeated_vision_calls_do_not_leak_loops(client):
    """Each call must reuse the same daemon loop, not spawn a fresh one."""
    seen = []
    real_post = client.client.post

    async def spy(*a, **k):
        seen.append(asyncio.get_running_loop())
        return await real_post(*a, **k)

    client.client.post = spy

    for _ in range(5):
        asyncio.run(
            client.generate_vision_chat(MagicMock(), _vision_messages(), GenerationParams())
        )

    assert len(seen) == 5
    # One loop for every call — not one per call.
    assert len({id(loop) for loop in seen}) == 1
    assert client._bg_loop is not None and client._bg_loop.is_running()
