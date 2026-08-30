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
    def __init__(self, lines, status=200):
        self._lines = lines
        self.status_code = status

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


def _client_with_stream(lines, status=200):
    client = RestClient(base_url="http://llama-server:8080", model_name="live")

    def _stream(method, url, **kw):
        return _FakeStreamResponse(lines, status)

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
