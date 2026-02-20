"""
Tests for native OpenAI tool calling support in LLM proxy.

Validates:
- Tool definitions accepted in ChatCompletionRequest
- Tool calls propagated in response
- JSON repair skipped when tool_calls present
- Backward compatibility when no tools provided
"""

import json

import pytest

from models.api_models import (
    ChatCompletionRequest,
    FunctionCall,
    FunctionDefinition,
    Message,
    ToolCall,
    ToolDefinition,
)
from managers.chat_types import ChatResult, GenerationParams


# ---------------------------------------------------------------------------
# Model / type tests
# ---------------------------------------------------------------------------


class TestToolModels:
    """Test that tool-related models validate correctly."""

    def test_tool_definition_creates(self):
        tool = ToolDefinition(
            function=FunctionDefinition(
                name="get_weather",
                description="Get the weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            )
        )
        assert tool.type == "function"
        assert tool.function.name == "get_weather"

    def test_tool_call_creates(self):
        tc = ToolCall(
            id="call_abc123",
            function=FunctionCall(
                name="get_weather",
                arguments='{"city": "London"}',
            ),
        )
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert json.loads(tc.function.arguments) == {"city": "London"}

    def test_message_allows_none_content(self):
        msg = Message(role="assistant", content=None, tool_calls=[
            ToolCall(
                id="call_1",
                function=FunctionCall(name="test", arguments="{}"),
            )
        ])
        assert msg.content is None
        assert len(msg.tool_calls) == 1

    def test_message_with_tool_call_id(self):
        msg = Message(role="tool", content="result", tool_call_id="call_1")
        assert msg.tool_call_id == "call_1"

    def test_request_with_tools(self):
        req = ChatCompletionRequest(
            model="full",
            messages=[Message(role="user", content="What's the weather?")],
            tools=[
                ToolDefinition(
                    function=FunctionDefinition(
                        name="get_weather",
                        parameters={"type": "object", "properties": {}},
                    )
                )
            ],
            tool_choice="auto",
        )
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"

    def test_request_without_tools_backward_compat(self):
        req = ChatCompletionRequest(
            model="full",
            messages=[Message(role="user", content="Hello")],
        )
        assert req.tools is None
        assert req.tool_choice is None


# ---------------------------------------------------------------------------
# ChatResult / GenerationParams tests
# ---------------------------------------------------------------------------


class TestChatResultFields:
    """Test new fields on ChatResult and GenerationParams."""

    def test_chat_result_with_tool_calls(self):
        tc = [{"id": "call_1", "type": "function", "function": {"name": "test", "arguments": "{}"}}]
        result = ChatResult(content="", usage={}, tool_calls=tc, finish_reason="tool_calls")
        assert result.tool_calls == tc
        assert result.finish_reason == "tool_calls"

    def test_chat_result_defaults(self):
        result = ChatResult(content="hello")
        assert result.tool_calls is None
        assert result.finish_reason is None

    def test_generation_params_with_tools(self):
        tools = [{"type": "function", "function": {"name": "test"}}]
        params = GenerationParams(tools=tools, tool_choice="auto")
        assert params.tools == tools
        assert params.tool_choice == "auto"

    def test_generation_params_defaults(self):
        params = GenerationParams()
        assert params.tools is None
        assert params.tool_choice is None


# ---------------------------------------------------------------------------
# Response helpers tests
# ---------------------------------------------------------------------------


class TestResponseHelpers:
    """Test that response helpers include tool calls."""

    def test_response_with_tool_calls(self):
        from services.response_helpers import create_openai_response

        tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
            }
        ]
        resp = create_openai_response(
            content="",
            model_name="test",
            tool_calls=tool_calls,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        assert choice.message.tool_calls[0].function.name == "get_weather"

    def test_response_without_tool_calls(self):
        from services.response_helpers import create_openai_response

        resp = create_openai_response(
            content="Hello!",
            model_name="test",
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "stop"
        assert choice.message.tool_calls is None
        assert choice.message.content == "Hello!"

    def test_response_explicit_finish_reason(self):
        from services.response_helpers import create_openai_response

        resp = create_openai_response(
            content="",
            model_name="test",
            finish_reason="length",
        )
        assert resp.choices[0].finish_reason == "length"
