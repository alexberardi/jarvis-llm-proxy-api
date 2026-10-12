"""REST backend streaming — `generate_text_chat_stream`.

Prod 2026-08-30: every phone-call turn died with

    POST /internal/model/chat/stream -> 501 "Backend does not support streaming"
    LLM turn 1/2/3 failed: Model service error 501

and the caller's fallback line ("Sorry, I'm having a little trouble hearing
that") made it look like an audio fault. It was not: the model never ran.

Cause: prod moved `live` onto the llama-server sidecars on 2026-08-27
(commit 42543fc, "Qwen3.5 sidecar serving (llama-server via REST backend)"),
which routes generation through this backend — and this backend had never
implemented streaming. Voice was unaffected because it uses the
non-streaming `/internal/model/chat`, so nothing surfaced until a call.

The capability check in model_service is STRUCTURAL:

    stream_fn = getattr(backend, "generate_text_chat_stream", None)
    if stream_fn is None or not inspect.isgeneratorfunction(...): 501

so the method must be a real generator function, not a coroutine and not a
function returning an iterator.
"""

import inspect
import json
import logging
from unittest.mock import patch

import pytest

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams, NormalizedMessage, TextPart


def _msgs(text="hello"):
    return [NormalizedMessage(role="user", content=[TextPart(text=text)])]


def _sse(chunks):
    """Render OpenAI-style SSE lines from a list of chunk dicts."""
    out = []
    for c in chunks:
        out.append(f"data: {json.dumps(c)}")
        out.append("")
    out.append("data: [DONE]")
    out.append("")
    return out


class _FakeStreamResponse:
    def __init__(self, lines, status=200, body=""):
        self._lines = lines
        self.status_code = status
        # httpx exposes these on a streamed response; the backend reads the
        # body before raise_for_status() drops it.
        self.is_error = status >= 400
        self.text = body

    async def aread(self):
        return self.text.encode()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_lines(self):
        for line in self._lines:
            yield line

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _client_with_stream(lines, status=200, body=""):
    client = RestClient(base_url="http://llama-server:8080", model_name="live")

    def _stream(method, url, **kw):
        return _FakeStreamResponse(lines, status, body)

    client.client.stream = _stream  # type: ignore[assignment]
    return client


class TestStreamingCapability:
    """model_service gates on isgeneratorfunction — the shape is the contract."""

    def test_is_a_real_generator_function(self):
        fn = getattr(RestClient, "generate_text_chat_stream", None)
        assert fn is not None, "REST backend must expose generate_text_chat_stream"
        assert inspect.isgeneratorfunction(fn), (
            "must be a generator function — model_service rejects anything else "
            "with 501 even though hasattr() is true"
        )

    def test_not_a_coroutine(self):
        fn = RestClient.generate_text_chat_stream
        assert not inspect.iscoroutinefunction(fn)


class TestStreamingYields:
    def test_yields_deltas_then_a_done_event(self):
        lines = _sse([
            {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 7, "completion_tokens": 2}},
        ])
        client = _client_with_stream(lines)
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))

        deltas = [e["delta"] for e in events if "delta" in e]
        assert deltas == ["Hel", "lo"]

        done = events[-1]
        assert done["done"] is True
        assert done["content"] == "Hello"
        assert done["finish_reason"] == "stop"
        assert done["usage"]["completion_tokens"] == 2

    def test_done_event_is_last_and_only_one(self):
        lines = _sse([{"choices": [{"delta": {"content": "x"}, "finish_reason": "stop"}]}])
        client = _client_with_stream(lines)
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))
        assert sum(1 for e in events if e.get("done")) == 1
        assert events[-1].get("done") is True

    def test_empty_stream_still_terminates_with_done(self):
        client = _client_with_stream(["data: [DONE]", ""])
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))
        assert events == [{
            "done": True, "content": "", "usage": {},
            "tool_calls": None, "finish_reason": "stop",
        }]

    def test_ignores_blank_lines_and_non_data_lines(self):
        lines = [
            "", ": ping", "event: message",
            'data: {"choices":[{"delta":{"content":"a"}}]}', "",
            "data: [DONE]",
        ]
        client = _client_with_stream(lines)
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))
        assert [e["delta"] for e in events if "delta" in e] == ["a"]

    def test_malformed_json_chunk_is_skipped_not_fatal(self):
        lines = [
            "data: {not json}",
            'data: {"choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        client = _client_with_stream(lines)
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))
        assert [e["delta"] for e in events if "delta" in e] == ["ok"]
        assert events[-1]["content"] == "ok"


class TestStreamingErrors:
    def test_http_error_propagates(self):
        client = _client_with_stream([], status=500)
        with pytest.raises(Exception):
            list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))

    def test_error_body_is_logged_before_raise_for_status(self, caplog):
        # raise_for_status() discards the body, and on a streamed response the
        # body has not even been read yet — so a provider 400 ("max_tokens:
        # field required") was invisible. Read and log it first.
        body = '{"type":"error","error":{"message":"max_tokens: field required"}}'
        client = _client_with_stream([], status=400, body=body)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))

        logged = " ".join(r.message for r in caplog.records)
        assert "400" in logged
        assert "max_tokens: field required" in logged

    def test_error_body_log_is_truncated(self, caplog):
        client = _client_with_stream([], status=400, body="x" * 5000)

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))

        logged = " ".join(r.message for r in caplog.records)
        assert "x" * 2000 in logged
        assert "x" * 2001 not in logged


class TestStreamingRequest:
    def test_request_sets_stream_true_and_forwards_sampling(self):
        seen = {}

        client = RestClient(base_url="http://llama-server:8080", model_name="live")

        def _stream(method, url, **kw):
            seen["method"] = method
            seen["url"] = url
            seen["json"] = kw.get("json")
            return _FakeStreamResponse(["data: [DONE]"])

        client.client.stream = _stream  # type: ignore[assignment]
        params = GenerationParams(temperature=0.3, max_tokens=64, top_p=0.9, seed=11)
        list(client.generate_text_chat_stream(None, _msgs("hi"), params))

        body = seen["json"]
        assert body["stream"] is True, "the whole point"
        assert body["model"] == "live"
        assert body["temperature"] == 0.3
        assert body["max_tokens"] == 64
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "hi"
        assert seen["method"] == "POST"


class TestStreamedToolCalls:
    """Streamed tool calls arrive split across frames and keyed by `index`.

    The name lands once, the JSON arguments dribble in a few characters at a
    time, and parallel calls interleave. Concatenating in arrival order without
    honouring `index` welds two calls into one.
    """

    def test_arguments_are_reassembled(self):
        lines = _sse([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "book", "arguments": ""}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '{"when":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '"tue"}'}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])
        client = _client_with_stream(lines)
        done = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))[-1]

        assert done["finish_reason"] == "tool_calls"
        assert len(done["tool_calls"]) == 1
        call = done["tool_calls"][0]
        assert call["id"] == "c1"
        assert call["function"]["name"] == "book"
        assert json.loads(call["function"]["arguments"]) == {"when": "tue"}

    def test_parallel_calls_stay_separate(self):
        lines = _sse([
            {"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "a", "function": {"name": "one", "arguments": '{"x":'}},
                {"index": 1, "id": "b", "function": {"name": "two", "arguments": '{"y":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [
                {"index": 1, "function": {"arguments": "2}"}},
                {"index": 0, "function": {"arguments": "1}"}}]}}]},
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ])
        client = _client_with_stream(lines)
        done = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))[-1]

        names = [c["function"]["name"] for c in done["tool_calls"]]
        assert names == ["one", "two"], "index must key the merge, not arrival order"
        assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
        assert json.loads(done["tool_calls"][1]["function"]["arguments"]) == {"y": 2}

    def test_no_tool_calls_yields_none(self):
        lines = _sse([{"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]}])
        client = _client_with_stream(lines)
        done = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))[-1]
        assert done["tool_calls"] is None

    def test_tools_are_forwarded_in_the_request(self):
        seen = {}
        client = RestClient(base_url="http://llama-server:8080", model_name="live")

        def _stream(method, url, **kw):
            seen["json"] = kw.get("json")
            return _FakeStreamResponse(["data: [DONE]"])

        client.client.stream = _stream  # type: ignore[assignment]
        tools = [{"type": "function", "function": {"name": "book"}}]
        params = GenerationParams(tools=tools, tool_choice="auto")
        list(client.generate_text_chat_stream(None, _msgs(), params))

        assert seen["json"]["tools"] == tools
        assert seen["json"]["tool_choice"] == "auto"


def _anthropic_sse(events):
    """Render Anthropic Messages SSE: `event:`/`data:` pairs, no [DONE]."""
    out = []
    for e in events:
        out.append(f"event: {e['type']}")
        out.append(f"data: {json.dumps(e)}")
        out.append("")
    return out


def _anthropic_client_with_stream(lines, seen=None, status=200):
    client = RestClient(base_url="https://api.anthropic.com", model_name="claude-sonnet-5")
    client.provider = "anthropic"
    client.auth_type = "bearer"
    client.auth_token = "sk-ant-test"
    client.headers = client._setup_headers()

    def _stream(method, url, **kw):
        if seen is not None:
            seen["method"] = method
            seen["url"] = url
            seen["json"] = kw.get("json")
            seen["headers"] = kw.get("headers")
        return _FakeStreamResponse(lines, status)

    client.client.stream = _stream  # type: ignore[assignment]
    return client


ANTHROPIC_TOOL_STREAM = [
    {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 25}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "On "}},
    {"type": "ping"},
    {"type": "content_block_delta", "index": 0,
     "delta": {"type": "text_delta", "text": "it."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1,
     "content_block": {"type": "tool_use", "id": "toolu_a", "name": "one", "input": {}}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": '{"x":'}},
    {"type": "content_block_delta", "index": 1,
     "delta": {"type": "input_json_delta", "partial_json": "1}"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "content_block_start", "index": 2,
     "content_block": {"type": "tool_use", "id": "toolu_b", "name": "two", "input": {}}},
    {"type": "content_block_delta", "index": 2,
     "delta": {"type": "input_json_delta", "partial_json": '{"y": 2}'}},
    {"type": "content_block_stop", "index": 2},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"},
     "usage": {"output_tokens": 17}},
    {"type": "message_stop"},
]


class TestAnthropicStreamedToolCalls:
    """Anthropic streams typed SSE events, not OpenAI `choices[].delta` chunks.

    No `[DONE]` sentinel either: the stream ends at `message_stop`. Feeding
    those frames to the OpenAI parser yields an empty answer at best, and the
    request itself 400s because the body was OpenAI-shaped.
    """

    def test_request_is_anthropic_shaped_and_streaming(self):
        seen = {}
        client = _anthropic_client_with_stream(_anthropic_sse(ANTHROPIC_TOOL_STREAM), seen)
        tools = [{"type": "function", "function": {"name": "one",
                                                   "parameters": {"type": "object"}}}]
        params = GenerationParams(temperature=0.3, top_p=0.9, seed=11,
                                  tools=tools, tool_choice="auto")

        list(client.generate_text_chat_stream(None, _msgs("hi"), params))

        body = seen["json"]
        assert body["stream"] is True
        # Falls back to the `inference.general.max_tokens` setting, not a
        # second hard-coded ceiling.
        assert body["max_tokens"] == client._default_max_tokens
        assert body["model"] == "claude-sonnet-5"
        assert body["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        assert body["tools"][0]["name"] == "one"
        assert "input_schema" in body["tools"][0]
        assert body["tool_choice"] == {"type": "auto"}
        assert body["thinking"] == {"type": "disabled"}
        assert "temperature" not in body
        assert "top_p" not in body
        assert "seed" not in body
        assert seen["url"].endswith("/v1/messages")
        assert seen["headers"]["x-api-key"] == "sk-ant-test"
        assert "Authorization" not in seen["headers"]

    def test_text_deltas_are_yielded(self):
        client = _anthropic_client_with_stream(_anthropic_sse(ANTHROPIC_TOOL_STREAM))
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))

        assert [e["delta"] for e in events if "delta" in e] == ["On ", "it."]
        assert events[-1]["content"] == "On it."

    def test_parallel_tool_uses_reassemble_separately(self):
        client = _anthropic_client_with_stream(_anthropic_sse(ANTHROPIC_TOOL_STREAM))
        done = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))[-1]

        assert done["done"] is True
        assert done["finish_reason"] == "tool_calls"
        assert [c["id"] for c in done["tool_calls"]] == ["toolu_a", "toolu_b"]
        assert [c["function"]["name"] for c in done["tool_calls"]] == ["one", "two"]
        assert json.loads(done["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
        assert json.loads(done["tool_calls"][1]["function"]["arguments"]) == {"y": 2}

    def test_usage_is_normalized(self):
        client = _anthropic_client_with_stream(_anthropic_sse(ANTHROPIC_TOOL_STREAM))
        done = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))[-1]

        assert done["usage"]["prompt_tokens"] == 25
        assert done["usage"]["completion_tokens"] == 17
        assert done["usage"]["total_tokens"] == 42

    def test_plain_text_stream_ends_with_stop_and_no_tool_calls(self):
        lines = _anthropic_sse([
            {"type": "message_start", "message": {"usage": {"input_tokens": 4}}},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"},
             "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ])
        client = _anthropic_client_with_stream(lines)
        events = list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))

        assert [e["delta"] for e in events if "delta" in e] == ["hello"]
        assert events[-1]["tool_calls"] is None
        assert events[-1]["finish_reason"] == "stop"

    def test_error_event_surfaces_as_an_exception(self):
        lines = _anthropic_sse([
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
        ])
        client = _anthropic_client_with_stream(lines)

        with pytest.raises(RuntimeError, match="Overloaded"):
            list(client.generate_text_chat_stream(None, _msgs(), GenerationParams()))
