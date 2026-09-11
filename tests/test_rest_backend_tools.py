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


ANTHROPIC_TOOL_RESPONSE = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [
        {"type": "text", "text": "Setting that now."},
        {
            "type": "tool_use",
            "id": "toolu_01",
            "name": "set_timer",
            "input": {"duration_minutes": 5},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 50, "output_tokens": 8},
}


def _anthropic_client() -> RestClient:
    client = RestClient(
        base_url="https://api.anthropic.com",
        model_name="claude-sonnet-5",
        model_type="live",
    )
    client.provider = "anthropic"
    # Existing deployments were documented with auth_type=bearer; Anthropic
    # still needs x-api-key, so the provider must win over the auth type.
    client.auth_type = "bearer"
    client.auth_token = "sk-ant-test"
    client.headers = client._setup_headers()
    return client


class TestAnthropicToolPath:
    """Native tool calling against the Anthropic Messages API.

    `provider=anthropic` routed to `/v1/messages` but still built an OpenAI
    `/v1/chat/completions` body (tools, tool_choice, seed, choices[] parsing)
    and still sent `Authorization: Bearer`, so every tool-enabled turn died on
    a 400 from `raise_for_status()`.
    """

    def test_posts_to_the_messages_endpoint(self) -> None:
        client = _anthropic_client()
        post = _mock_post(client, ANTHROPIC_TOOL_RESPONSE)

        params = GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto", max_tokens=256)
        client.generate_text_chat(None, _messages(), params)

        url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
        assert url.endswith("/v1/messages")

    def test_payload_is_anthropic_shaped(self) -> None:
        client = _anthropic_client()
        post = _mock_post(client, ANTHROPIC_TOOL_RESPONSE)

        params = GenerationParams(
            temperature=0.4,
            max_tokens=256,
            top_p=0.9,
            seed=7,
            tools=SAMPLE_TOOLS,
            tool_choice="auto",
        )
        client.generate_text_chat(None, _messages(), params)

        sent = post.call_args.kwargs["json"]
        assert sent["system"] == "You route tools."
        assert sent["max_tokens"] == 256
        assert sent["tools"] == [
            {
                "name": "set_timer",
                "description": "Set a countdown timer.",
                "input_schema": SAMPLE_TOOLS[0]["function"]["parameters"],
            }
        ]
        assert sent["tool_choice"] == {"type": "auto"}
        assert sent["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "set a 5 minute timer"}]}
        ]
        # OpenAI-only keys must not leak into the Messages API body.
        assert "seed" not in sent
        assert "top_p" not in sent
        assert "chat_template_kwargs" not in sent

    def test_max_tokens_defaults_when_absent(self) -> None:
        client = _anthropic_client()
        post = _mock_post(client, ANTHROPIC_TOOL_RESPONSE)

        client.generate_text_chat(
            None, _messages(), GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto")
        )

        # Anthropic REQUIRES max_tokens; omitting it is a 400.
        assert post.call_args.kwargs["json"]["max_tokens"] == 4096

    def test_headers_use_x_api_key_not_bearer(self) -> None:
        client = _anthropic_client()

        assert client.headers["x-api-key"] == "sk-ant-test"
        assert client.headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in client.headers
        assert client.headers["Content-Type"] == "application/json"

    def test_result_is_normalized_to_the_openai_shape(self) -> None:
        client = _anthropic_client()
        _mock_post(client, ANTHROPIC_TOOL_RESPONSE)

        params = GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto")
        result = client.generate_text_chat(None, _messages(), params)

        assert result.content == "Setting that now."
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None and len(result.tool_calls) == 1
        call = result.tool_calls[0]
        assert call["id"] == "toolu_01"
        assert call["type"] == "function"
        assert call["function"]["name"] == "set_timer"
        assert json.loads(call["function"]["arguments"]) == {"duration_minutes": 5}
        assert result.usage["prompt_tokens"] == 50
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 58

    def test_plain_text_reply_normalizes_to_stop(self) -> None:
        client = _anthropic_client()
        _mock_post(
            client,
            {
                "content": [{"type": "text", "text": "Timer set."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 9, "output_tokens": 3},
            },
        )

        result = client.generate_text_chat(
            None, _messages(), GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto")
        )

        assert result.content == "Timer set."
        assert result.tool_calls is None
        assert result.finish_reason == "stop"


class TestOpenAIPathUnchangedByAnthropicSupport:
    """The Anthropic branch must not touch the OpenAI request byte-for-byte."""

    def test_openai_payload_and_headers_are_untouched(self) -> None:
        client = _client()
        client.auth_type = "bearer"
        client.auth_token = "sk-openai"
        client.headers = client._setup_headers()
        post = _mock_post(client, TOOL_CALL_RESPONSE)

        params = GenerationParams(tools=SAMPLE_TOOLS, tool_choice="auto", max_tokens=256)
        client.generate_text_chat(None, _messages(), params)

        sent = post.call_args.kwargs["json"]
        assert sent["tools"] == SAMPLE_TOOLS, "OpenAI keeps the OpenAI tool shape"
        assert sent["tool_choice"] == "auto"
        assert sent["messages"][0] == {"role": "system", "content": "You route tools."}
        assert "system" not in sent
        assert client.headers["Authorization"] == "Bearer sk-openai"
        assert "x-api-key" not in client.headers
