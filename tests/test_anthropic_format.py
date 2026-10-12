"""Unit tests for the Anthropic Messages API translation layer.

``backends/anthropic_format.py`` is the seam between the OpenAI shape every
caller in this repo speaks and the ``/v1/messages`` shape Anthropic requires.
Both directions are pure functions so they can be pinned without a socket:

- OUT: OpenAI tools / tool_choice / messages -> Anthropic request payload
  (system hoisted out of ``messages``, ``tool_calls`` -> ``tool_use`` blocks,
  ``role: "tool"`` -> a user ``tool_result`` block, consecutive same-role
  messages merged, ``max_tokens`` always present, every sampling knob and
  ``seed`` dropped, thinking disabled).
- IN: Anthropic content blocks / stop_reason / usage -> the OpenAI-normalized
  ``ChatResult`` downstream (``services/response_helpers.py`` and the
  command-center service) already understands.
"""

from __future__ import annotations

import json
import logging

import pytest

from backends.anthropic_format import (
    DEFAULT_MAX_TOKENS,
    AnthropicStreamAccumulator,
    build_anthropic_payload,
    map_stop_reason,
    normalize_usage,
    openai_messages_to_anthropic,
    openai_tool_choice_to_anthropic,
    openai_tools_to_anthropic,
    parse_anthropic_message,
)
from managers.chat_types import GenerationParams

SET_TIMER_TOOL = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": "Set a countdown timer.",
        "parameters": {
            "type": "object",
            "properties": {"minutes": {"type": "integer"}},
            "required": ["minutes"],
        },
    },
}


class TestToolTranslation:
    def test_function_tool_becomes_name_description_input_schema(self) -> None:
        out = openai_tools_to_anthropic([SET_TIMER_TOOL])

        assert out == [
            {
                "name": "set_timer",
                "description": "Set a countdown timer.",
                "input_schema": {
                    "type": "object",
                    "properties": {"minutes": {"type": "integer"}},
                    "required": ["minutes"],
                },
            }
        ]

    def test_missing_parameters_fall_back_to_an_empty_object_schema(self) -> None:
        out = openai_tools_to_anthropic(
            [{"type": "function", "function": {"name": "now"}}]
        )

        assert out == [{"name": "now", "input_schema": {"type": "object", "properties": {}}}]

    def test_empty_tool_list_stays_empty(self) -> None:
        assert openai_tools_to_anthropic([]) == []


class TestToolChoiceTranslation:
    def test_auto(self) -> None:
        assert openai_tool_choice_to_anthropic("auto") == {"type": "auto"}

    def test_required_becomes_any(self) -> None:
        assert openai_tool_choice_to_anthropic("required") == {"type": "any"}

    def test_none(self) -> None:
        assert openai_tool_choice_to_anthropic("none") == {"type": "none"}

    def test_named_function_becomes_tool(self) -> None:
        choice = {"type": "function", "function": {"name": "set_timer"}}

        assert openai_tool_choice_to_anthropic(choice) == {
            "type": "tool",
            "name": "set_timer",
        }

    def test_nameless_tool_object_becomes_any(self) -> None:
        # ``{"type": "tool"}`` with no name is not a legal Anthropic tool_choice.
        assert openai_tool_choice_to_anthropic({"type": "tool"}) == {"type": "any"}
        assert openai_tool_choice_to_anthropic({"type": "tool", "name": ""}) == {"type": "any"}

    def test_none_value_is_omitted(self) -> None:
        assert openai_tool_choice_to_anthropic(None) is None


class TestMessageTranslation:
    def test_multiple_system_messages_are_hoisted_and_joined(self) -> None:
        system, messages = openai_messages_to_anthropic(
            [
                {"role": "system", "content": "You are Jarvis."},
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ]
        )

        assert system == "You are Jarvis.\n\nBe terse."
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def test_no_system_message_yields_none(self) -> None:
        system, messages = openai_messages_to_anthropic([{"role": "user", "content": "hi"}])

        assert system is None
        assert len(messages) == 1

    def test_assistant_tool_calls_become_tool_use_blocks(self) -> None:
        _, messages = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "set a timer"},
                {
                    "role": "assistant",
                    "content": "Sure.",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {
                                "name": "set_timer",
                                "arguments": json.dumps({"minutes": 5}),
                            },
                        }
                    ],
                },
            ]
        )

        assert messages[1] == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Sure."},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "set_timer",
                    "input": {"minutes": 5},
                },
            ],
        }

    def test_tool_call_with_empty_or_invalid_arguments_falls_back_to_empty_input(
        self,
    ) -> None:
        _, messages = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "one", "arguments": ""}},
                        {"id": "b", "function": {"name": "two", "arguments": "{not json"}},
                    ],
                },
            ]
        )

        blocks = messages[1]["content"]
        assert [b["type"] for b in blocks] == ["tool_use", "tool_use"]
        assert blocks[0]["input"] == {}
        assert blocks[1]["input"] == {}

    def test_empty_assistant_message_without_tool_calls_is_dropped(self) -> None:
        _, messages = openai_messages_to_anthropic(
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]
        )

        assert [m["role"] for m in messages] == ["user"]

    def test_tool_role_messages_merge_into_one_user_message(self) -> None:
        # Parallel tool calls must come back as ONE user turn holding every
        # tool_result block; Anthropic rejects the split form.
        _, messages = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "a", "function": {"name": "one", "arguments": "{}"}},
                        {"id": "b", "function": {"name": "two", "arguments": "{}"}},
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "first"},
                {"role": "tool", "tool_call_id": "b", "content": {"ok": True}},
            ]
        )

        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[2]["content"] == [
            {"type": "tool_result", "tool_use_id": "a", "content": "first"},
            {"type": "tool_result", "tool_use_id": "b", "content": json.dumps({"ok": True})},
        ]

    def test_consecutive_same_role_messages_are_merged(self) -> None:
        _, messages = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
                {"role": "assistant", "content": "ack"},
            ]
        )

        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == [
            {"type": "text", "text": "one"},
            {"type": "text", "text": "two"},
        ]

    def test_missing_tool_call_fields_do_not_explode(self) -> None:
        # NormalizedMessage -> dict conversion upstream can drop the tool-call
        # id; an assistant tool_use still translates, with a generated id.
        _, messages = openai_messages_to_anthropic(
            [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "content": "plain",
                    "tool_calls": [{"function": {"name": "one", "arguments": "{}"}}],
                },
            ]
        )

        assert messages[1]["content"][0] == {"type": "text", "text": "plain"}
        block = messages[1]["content"][1]
        assert block["type"] == "tool_use"
        assert block["name"] == "one"
        assert block["id"], "a tool_use block with an empty id is a 400"

    def test_orphan_tool_result_without_an_id_raises(self) -> None:
        # ``tool_use_id: ""`` is a 400 from the API; fail loudly instead of
        # shipping a request that cannot succeed.
        with pytest.raises(ValueError, match="tool_call_id"):
            openai_messages_to_anthropic(
                [{"role": "user", "content": "go"}, {"role": "tool", "content": "orphan"}]
            )

    def test_leading_assistant_messages_are_dropped(self) -> None:
        # The Messages API requires the first message to be ``user``.
        _, messages = openai_messages_to_anthropic(
            [
                {"role": "assistant", "content": "hello there"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ack"},
            ]
        )

        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[0]["content"] == [{"type": "text", "text": "hi"}]


class TestPayload:
    def test_max_tokens_defaults_when_the_request_omits_it(self) -> None:
        payload = build_anthropic_payload(
            "claude-sonnet-5", [{"role": "user", "content": "hi"}], GenerationParams(), False
        )

        assert payload["max_tokens"] == DEFAULT_MAX_TOKENS

    def test_explicit_max_tokens_wins(self) -> None:
        payload = build_anthropic_payload(
            "claude-sonnet-5",
            [{"role": "user", "content": "hi"}],
            GenerationParams(max_tokens=128),
            False,
        )

        assert payload["max_tokens"] == 128

    def test_sampling_params_are_dropped_and_no_openai_only_keys_leak(self) -> None:
        params = GenerationParams(temperature=0.3, top_p=0.9, seed=11, reasoning_budget=0)
        payload = build_anthropic_payload(
            "claude-sonnet-5", [{"role": "user", "content": "hi"}], params, False
        )

        # Current Claude models 400 on temperature/top_p/top_k outright.
        assert "temperature" not in payload
        assert "top_p" not in payload
        assert "top_k" not in payload
        assert "seed" not in payload
        assert "chat_template_kwargs" not in payload
        assert "reasoning_budget" not in payload

    def test_thinking_is_disabled_by_default(self) -> None:
        payload = build_anthropic_payload(
            "claude-sonnet-5", [{"role": "user", "content": "hi"}], GenerationParams(), False
        )

        # Adaptive thinking is the default otherwise, and thinking blocks would
        # then have to be replayed verbatim on the tool-result round trip.
        assert payload["thinking"] == {"type": "disabled"}

    @pytest.mark.parametrize(
        "model",
        ["claude-fable-5", "CLAUDE-FABLE-5.1", "claude-mythos-5.1", "us.anthropic.claude-fable-5"],
    )
    def test_thinking_is_omitted_on_fable_and_mythos(self, model) -> None:
        # Thinking is always on there; ``disabled`` is a 400.
        payload = build_anthropic_payload(
            model, [{"role": "user", "content": "hi"}], GenerationParams(), False
        )

        assert "thinking" not in payload

    def test_default_max_tokens_override_is_honoured(self) -> None:
        payload = build_anthropic_payload(
            "claude-sonnet-5",
            [{"role": "user", "content": "hi"}],
            GenerationParams(),
            False,
            default_max_tokens=777,
        )

        assert payload["max_tokens"] == 777

    def test_explicit_max_tokens_beats_the_default_override(self) -> None:
        payload = build_anthropic_payload(
            "claude-sonnet-5",
            [{"role": "user", "content": "hi"}],
            GenerationParams(max_tokens=128),
            False,
            default_max_tokens=777,
        )

        assert payload["max_tokens"] == 128

    def test_empty_message_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one user/assistant message"):
            build_anthropic_payload(
                "claude-sonnet-5", [{"role": "system", "content": "be terse"}],
                GenerationParams(), False,
            )

    def test_system_is_hoisted_and_stream_flag_is_set(self) -> None:
        payload = build_anthropic_payload(
            "claude-sonnet-5",
            [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
            GenerationParams(),
            True,
        )

        assert payload["system"] == "be terse"
        assert payload["stream"] is True
        assert payload["model"] == "claude-sonnet-5"
        assert payload["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]

    def test_tools_are_translated_only_when_present(self) -> None:
        without = build_anthropic_payload(
            "claude-sonnet-5", [{"role": "user", "content": "hi"}], GenerationParams(), False
        )
        assert "tools" not in without
        assert "tool_choice" not in without

        with_tools = build_anthropic_payload(
            "claude-sonnet-5",
            [{"role": "user", "content": "hi"}],
            GenerationParams(tools=[SET_TIMER_TOOL], tool_choice="auto"),
            False,
        )
        assert with_tools["tools"][0]["name"] == "set_timer"
        assert "input_schema" in with_tools["tools"][0]
        assert with_tools["tool_choice"] == {"type": "auto"}


class TestStopReasonMapping:
    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("refusal", "content_filter"),
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("pause_turn", "stop"),
            ("something_new", "stop"),
            (None, "stop"),
        ],
    )
    def test_mapping(self, stop_reason, expected) -> None:
        assert map_stop_reason(stop_reason) == expected

    def test_refusal_is_logged_with_details(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            result = parse_anthropic_message(
                {
                    "content": [],
                    "stop_reason": "refusal",
                    "stop_details": {"type": "safety"},
                }
            )

        assert result.finish_reason == "content_filter"
        assert any("refusal" in r.message and "safety" in r.message for r in caplog.records)

    def test_pause_turn_is_logged(self, caplog) -> None:
        with caplog.at_level(logging.WARNING):
            parse_anthropic_message({"content": [], "stop_reason": "pause_turn"})

        assert any("pause_turn" in r.message for r in caplog.records)

    def test_stream_refusal_is_logged(self, caplog) -> None:
        acc = AnthropicStreamAccumulator()
        acc.feed(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "refusal", "stop_details": {"type": "safety"}},
                "usage": {"output_tokens": 0},
            }
        )
        with caplog.at_level(logging.WARNING):
            result = acc.result()

        assert result["finish_reason"] == "content_filter"
        assert any("refusal" in r.message for r in caplog.records)


class TestUsageNormalization:
    def test_input_output_tokens_become_prompt_completion_total(self) -> None:
        usage = normalize_usage({"input_tokens": 25, "output_tokens": 17})

        assert usage["prompt_tokens"] == 25
        assert usage["completion_tokens"] == 17
        assert usage["total_tokens"] == 42

    def test_missing_usage_is_zeroed(self) -> None:
        usage = normalize_usage(None)

        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0

    def test_cache_tokens_count_towards_prompt_tokens(self) -> None:
        # Cached reads/writes are billed input; leaving them out understates
        # prompt_tokens by the whole cached prefix.
        usage = normalize_usage(
            {
                "input_tokens": 25,
                "output_tokens": 17,
                "cache_read_input_tokens": 1000,
                "cache_creation_input_tokens": 40,
            }
        )

        assert usage["prompt_tokens"] == 1065
        assert usage["completion_tokens"] == 17
        assert usage["total_tokens"] == 1082

    def test_missing_cache_keys_default_to_zero(self) -> None:
        usage = normalize_usage({"input_tokens": 3, "output_tokens": 4})

        assert usage["prompt_tokens"] == 3
        assert usage["total_tokens"] == 7

    def test_raw_anthropic_keys_are_not_the_only_keys(self) -> None:
        usage = normalize_usage({"input_tokens": 1, "output_tokens": 2})

        assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(usage)


class TestParseAnthropicMessage:
    def test_mixed_text_and_two_tool_use_blocks(self) -> None:
        data = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [
                {"type": "text", "text": "Sure, "},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "set_timer",
                    "input": {"minutes": 5},
                },
                {"type": "text", "text": "and also"},
                {"type": "tool_use", "id": "toolu_2", "name": "get_time", "input": {}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 8},
        }

        result = parse_anthropic_message(data)

        assert result.content == "Sure, and also"
        assert result.finish_reason == "tool_calls"
        assert result.usage["total_tokens"] == 58
        assert result.tool_calls == [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {
                    "name": "set_timer",
                    "arguments": json.dumps({"minutes": 5}),
                },
            },
            {
                "id": "toolu_2",
                "type": "function",
                "function": {"name": "get_time", "arguments": json.dumps({})},
            },
        ]

    def test_plain_text_reply_has_no_tool_calls(self) -> None:
        result = parse_anthropic_message(
            {
                "content": [{"type": "text", "text": "It is sunny."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 3, "output_tokens": 4},
            }
        )

        assert result.content == "It is sunny."
        assert result.tool_calls is None
        assert result.finish_reason == "stop"

    def test_truncated_reply_maps_to_length(self) -> None:
        result = parse_anthropic_message(
            {"content": [{"type": "text", "text": "abc"}], "stop_reason": "max_tokens"}
        )

        assert result.finish_reason == "length"


def _tool_call_stream_events() -> list:
    return [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 25}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "On it"},
        },
        {"type": "ping"},
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "."},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "set_timer", "input": {}},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"minu'},
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": 'tes": 5}'},
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {"type": "tool_use", "id": "toolu_2", "name": "get_time", "input": {}},
        },
        {"type": "content_block_stop", "index": 2},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 17},
        },
        {"type": "message_stop"},
    ]


class TestStreamAccumulator:
    def test_full_sequence_with_text_and_two_tool_uses(self) -> None:
        acc = AnthropicStreamAccumulator()
        deltas = [d for d in (acc.feed(e) for e in _tool_call_stream_events()) if d]

        assert deltas == ["On it", "."]

        result = acc.result()
        assert result["content"] == "On it."
        assert result["finish_reason"] == "tool_calls"
        assert result["usage"] == {
            "prompt_tokens": 25,
            "completion_tokens": 17,
            "total_tokens": 42,
        }
        assert [c["id"] for c in result["tool_calls"]] == ["toolu_1", "toolu_2"]
        assert [c["function"]["name"] for c in result["tool_calls"]] == [
            "set_timer",
            "get_time",
        ]
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"minutes": 5}
        # A tool_use block with no input_json_delta means "no arguments".
        assert json.loads(result["tool_calls"][1]["function"]["arguments"]) == {}

    def test_message_start_cache_tokens_count_towards_prompt_tokens(self) -> None:
        acc = AnthropicStreamAccumulator()
        acc.feed(
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 25,
                        "cache_read_input_tokens": 1000,
                        "cache_creation_input_tokens": 40,
                    }
                },
            }
        )
        acc.feed(
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 17}}
        )

        usage = acc.result()["usage"]
        assert usage["prompt_tokens"] == 1065
        assert usage["total_tokens"] == 1082

    def test_ping_is_ignored(self) -> None:
        acc = AnthropicStreamAccumulator()

        assert acc.feed({"type": "ping"}) is None

    def test_plain_text_stream_has_no_tool_calls(self) -> None:
        acc = AnthropicStreamAccumulator()
        for event in [
            {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]:
            acc.feed(event)

        result = acc.result()
        assert result["content"] == "hi"
        assert result["tool_calls"] is None
        assert result["finish_reason"] == "stop"

    def test_error_event_raises(self) -> None:
        acc = AnthropicStreamAccumulator()

        with pytest.raises(RuntimeError, match="Overloaded"):
            acc.feed({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})

    def test_empty_stream_still_produces_a_normalized_result(self) -> None:
        result = AnthropicStreamAccumulator().result()

        assert result == {
            "content": "",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "tool_calls": None,
            "finish_reason": "stop",
        }
