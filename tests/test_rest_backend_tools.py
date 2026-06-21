"""Tests for native tool-calling passthrough in the REST backend.

The REST backend previously dropped tool calls (always returned
``tool_calls=None``), so a cloud model's tool decision never reached the
caller. These tests pin that, when ``GenerationParams.tools`` is set, the
backend (a) forwards ``tools``/``tool_choice`` in the OpenAI request and
(b) parses the structured ``tool_calls`` + ``finish_reason`` back out.

No network/API key: httpx is mocked with a canned OpenAI response.
"""

from __future__ import annotations

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
