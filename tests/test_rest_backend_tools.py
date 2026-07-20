"""Tests for native tool-calling passthrough in the REST backend.

The REST backend previously dropped tool calls (always returned
``tool_calls=None``), so a cloud model's tool decision never reached the
caller. These tests pin that, when ``GenerationParams.tools`` is set, the
backend (a) forwards ``tools``/``tool_choice`` in the OpenAI request and
(b) parses the structured ``tool_calls`` + ``finish_reason`` back out.

No network/API key: httpx is mocked with a canned OpenAI response.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams, NormalizedMessage, TextPart


SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a countdown timer.",
            "parameters": {
                "type": "object",
                "properties": {"duration_minutes": {"type": "integer"}},
                "required": ["duration_minutes"],
            },
        },
    }
]

TOOL_CALL_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "set_timer",
                            "arguments": json.dumps({"duration_minutes": 5}),
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 50, "completion_tokens": 8, "total_tokens": 58},
}

PLAIN_RESPONSE = {
    "choices": [{"message": {"content": "It is sunny."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
}


def _client() -> RestClient:
    client = RestClient(
        base_url="https://api.openai.com",
        model_name="gpt-4.1-nano",
        model_type="live",
    )
    client.provider = "openai"  # deterministic OpenAI response parsing
    return client


def _mock_post(client: RestClient, response_json: dict) -> AsyncMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value=response_json)
    post = AsyncMock(return_value=resp)
    client.client.post = post
    return post


def _messages() -> list[NormalizedMessage]:
    return [
        NormalizedMessage(role="system", content=[TextPart(text="You route tools.")]),
        NormalizedMessage(role="user", content=[TextPart(text="set a 5 minute timer")]),
    ]


class TestRestToolPassthrough:
    def test_tool_calls_are_parsed_back(self) -> None:
        client = _client()
        _mock_post(client, TOOL_CALL_RESPONSE)

        params = GenerationParams(
            temperature=0.4, max_tokens=256, tools=SAMPLE_TOOLS, tool_choice="auto"
        )
        result = client.generate_text_chat(None, _messages(), params)

        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert result.tool_calls[0]["function"]["name"] == "set_timer"
        assert result.usage["total_tokens"] == 58

    def test_request_forwards_tools_and_tool_choice(self) -> None:
        client = _client()
        post = _mock_post(client, TOOL_CALL_RESPONSE)

        params = GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto", max_tokens=256)
        client.generate_text_chat(None, _messages(), params)

        sent = post.call_args.kwargs["json"]
        assert sent["tools"] == SAMPLE_TOOLS
        assert sent["tool_choice"] == "auto"
        assert sent["max_tokens"] == 256

    def test_no_tools_keeps_content_path(self) -> None:
        # Backward-compatible: no tools -> content-only, tool_calls stays None,
        # and no `tools` key is sent in the request.
        client = _client()
        post = _mock_post(client, PLAIN_RESPONSE)

        result = client.generate_text_chat(None, _messages(), GenerationParams())

        assert result.tool_calls is None
        assert result.content == "It is sunny."
        assert "tools" not in post.call_args.kwargs["json"]


class TestSyncBridgeEventLoop:
    """Regression: the model service calls the SYNC ``generate_text_chat`` from
    its ASYNC ``/internal/model/chat`` endpoint, on one long-lived shared
    RestClient. The old bridge ran each call on a throwaway ``asyncio.run`` loop
    whenever the caller was already in an async context, so reusing the
    persistent ``self.client`` (its connection pool binds to the loop it was
    first used on) across those per-call loops raised intermittent
    ``RuntimeError: Event loop is closed`` on connection cleanup — surfacing as
    flaky 500s under any multi-request load. The fix routes all async-context
    calls onto a single dedicated background loop so the client stays valid.
    """

    def test_repeated_calls_from_async_context_share_client(self) -> None:
        # We are NOT in an async context here; drive the sync bridge from inside
        # a running loop to exercise the background-loop path the model service
        # hits. Several calls on the SAME client must all succeed.
        client = _client()
        _mock_post(client, TOOL_CALL_RESPONSE)
        params = GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto", max_tokens=256)

        async def _drive() -> list:
            # Inside this coroutine asyncio.get_running_loop() succeeds, so
            # generate_text_chat takes the background-loop branch. The blocking
            # .result() waits on the bg loop (a separate thread), so it returns
            # rather than deadlocking the running loop.
            return [
                client.generate_text_chat(None, _messages(), params)
                for _ in range(5)
            ]

        results = asyncio.run(_drive())

        assert len(results) == 5
        for result in results:
            assert result.finish_reason == "tool_calls"
            assert result.tool_calls is not None
            assert result.tool_calls[0]["function"]["name"] == "set_timer"
        # The async-context path runs on the dedicated background loop, not the
        # caller's throwaway loop — this is what keeps the shared client valid.
        assert client._bg_loop is not None
        assert client._bg_loop.is_running()

    def test_sync_context_still_works(self) -> None:
        # A sync caller (scripts, tests) still gets a correct result.
        #
        # This used to additionally assert `client._bg_loop is None`, pinning
        # the old "no running loop => inline asyncio.run" branch. That branch
        # WAS the bug: once chat_runner began offloading sync generations with
        # asyncio.to_thread, the model service reached this same no-running-loop
        # state with a PERSISTENT client, and each throwaway loop poisoned the
        # next call's pooled connection ("Event loop is closed", alternating
        # 200/500 in the CI behavior corpus 2026-07-20). The bridge now always
        # uses the dedicated loop, so the assertion is inverted here on purpose.
        client = _client()
        _mock_post(client, TOOL_CALL_RESPONSE)
        params = GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto", max_tokens=256)

        result = client.generate_text_chat(None, _messages(), params)

        assert result.finish_reason == "tool_calls"
        assert client._bg_loop is not None
        assert client._bg_loop.is_running()
