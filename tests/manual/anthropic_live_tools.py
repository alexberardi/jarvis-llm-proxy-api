"""Live smoke test for native tool calling against the Anthropic Messages API.

Not a pytest module — it talks to the real API and costs tokens. Run it by
hand when changing ``backends/anthropic_format.py`` or the Anthropic branches
of ``backends/rest_backend.py``.

    export ANTHROPIC_API_KEY=sk-ant-...
    # DATABASE_URL only has to be parseable; nothing here touches the DB.
    DATABASE_URL=postgresql://x:y@localhost/z \
        ANTHROPIC_MODEL=claude-sonnet-5 \
        python tests/manual/anthropic_live_tools.py

Checks:
  a) non-stream tool call    -> finish_reason "tool_calls" + one set_timer call
  b) tool-result round trip  -> finish_reason "stop" + a spoken confirmation
  c) streaming tool call     -> text/tool deltas + a terminal "tool_calls" frame

This doubles as the pre-fix reproduction: before the Anthropic translation
landed, (a) died with an HTTP 400 from ``/v1/messages`` because the backend
posted an OpenAI ``/v1/chat/completions`` body.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backends.rest_backend import RestClient  # noqa: E402
from managers.chat_types import GenerationParams, NormalizedMessage, TextPart  # noqa: E402

DEFAULT_MODEL = "claude-sonnet-5"

SET_TIMER_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "set_timer",
        "description": "Set a countdown timer for the user.",
        "parameters": {
            "type": "object",
            "properties": {"minutes": {"type": "integer", "description": "Duration in minutes."}},
            "required": ["minutes"],
        },
    },
}


def _client(model: str, api_key: str) -> RestClient:
    client = RestClient(
        base_url="https://api.anthropic.com",
        model_name=model,
        model_type="live",
    )
    client.provider = "anthropic"
    client.auth_type = "api_key"
    client.auth_token = api_key
    # __init__ silently overrides model_name from JARVIS_REST_MODEL_NAME /
    # model.main.rest_model_name, so pin it back to what was asked for.
    client.model_name = model
    # Headers are computed in __init__, before these overrides.
    client.headers = client._setup_headers()
    return client


def _msg(role: str, text: str) -> NormalizedMessage:
    return NormalizedMessage(role=role, content=[TextPart(text=text)])


def _report(name: str, ok: bool, detail: str) -> Dict[str, Any]:
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return {"check": name, "passed": ok, "detail": detail}


def check_non_stream_tool_call(client: RestClient) -> Dict[str, Any]:
    """(a) The model should choose set_timer on its own (no forced choice)."""
    messages = [
        _msg("system", "You control timers. Use the tools you are given."),
        _msg("user", "Set a timer for 5 minutes"),
    ]
    params = GenerationParams(
        temperature=0.0, max_tokens=512, tools=[SET_TIMER_TOOL], tool_choice="auto"
    )
    try:
        result = client.generate_text_chat(None, messages, params)
    except Exception as exc:  # noqa: BLE001 — this IS the reproduction
        return _report("non_stream_tool_call", False, f"{type(exc).__name__}: {exc}")

    calls = result.tool_calls or []
    if result.finish_reason != "tool_calls" or not calls:
        return _report(
            "non_stream_tool_call",
            False,
            f"finish_reason={result.finish_reason!r} tool_calls={calls!r}",
        )
    call = calls[0]
    try:
        args = json.loads(call["function"]["arguments"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        return _report("non_stream_tool_call", False, f"unparseable arguments: {exc}")
    if call["function"]["name"] != "set_timer":
        return _report("non_stream_tool_call", False, f"wrong tool: {call['function']['name']}")

    return _report(
        "non_stream_tool_call",
        True,
        f"set_timer({args}) id={call.get('id')} usage={result.usage}",
    )


def check_tool_result_round_trip(client: RestClient) -> Dict[str, Any]:
    """(b) Feed the tool result back and expect a normal spoken answer.

    Exercises the assistant tool_use + user tool_result block translation,
    which is the half of the conversion a one-shot call never touches.
    """
    messages = [
        _msg("system", "You control timers. Use the tools you are given."),
        _msg("user", "Set a timer for 5 minutes"),
    ]
    params = GenerationParams(
        temperature=0.0, max_tokens=512, tools=[SET_TIMER_TOOL], tool_choice="auto"
    )
    try:
        first = client.generate_text_chat(None, messages, params)
    except Exception as exc:  # noqa: BLE001
        return _report("tool_result_round_trip", False, f"first turn failed: {exc}")

    calls = first.tool_calls or []
    if not calls:
        return _report("tool_result_round_trip", False, "first turn produced no tool call")

    # NOTE: this calls the private ``_chat_completion_with_tools`` directly
    # because NormalizedMessage (managers/chat_types.py) carries only role +
    # content parts — it drops ``tool_calls`` and ``tool_call_id`` — so the
    # round trip is NOT reachable through the public /v1/chat/completions yet.
    # The assistant turn and the tool results go in as raw dicts instead.
    dict_messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You control timers. Use the tools you are given."},
        {"role": "user", "content": "Set a timer for 5 minutes"},
        {"role": "assistant", "content": first.content, "tool_calls": calls},
    ]
    # EVERY tool_use block needs a matching tool_result, not just the first:
    # a parallel-call turn 400s if any of them is left unanswered.
    for call in calls:
        dict_messages.append(
            {"role": "tool", "tool_call_id": call.get("id"), "content": "Timer set"}
        )

    try:
        second = asyncio.run_coroutine_threadsafe(
            client._chat_completion_with_tools(dict_messages, params),
            client._get_background_loop(),
        ).result()
    except Exception as exc:  # noqa: BLE001
        return _report("tool_result_round_trip", False, f"second turn failed: {exc}")

    ok = second.finish_reason == "stop" and bool(second.content.strip())
    return _report(
        "tool_result_round_trip",
        ok,
        f"finish_reason={second.finish_reason!r} content={second.content[:80]!r}",
    )


def check_streaming_tool_call(client: RestClient) -> Dict[str, Any]:
    """(c) The same ask over SSE: deltas plus a terminal tool_calls frame."""
    messages = [
        _msg("system", "You control timers. Use the tools you are given."),
        _msg("user", "Set a timer for 5 minutes"),
    ]
    params = GenerationParams(
        temperature=0.0, max_tokens=512, tools=[SET_TIMER_TOOL], tool_choice="auto"
    )
    try:
        events = list(client.generate_text_chat_stream(None, messages, params))
    except Exception as exc:  # noqa: BLE001
        return _report("streaming_tool_call", False, f"{type(exc).__name__}: {exc}")

    if not events or not events[-1].get("done"):
        return _report("streaming_tool_call", False, "no terminal done frame")

    done = events[-1]
    deltas = [e["delta"] for e in events if "delta" in e]
    calls = done.get("tool_calls") or []
    ok = done.get("finish_reason") == "tool_calls" and bool(calls)
    return _report(
        "streaming_tool_call",
        ok,
        f"{len(deltas)} deltas, finish_reason={done.get('finish_reason')!r}, "
        f"tools={[c['function']['name'] for c in calls]}, usage={done.get('usage')}",
    )


def main() -> int:
    logging.basicConfig(level=logging.INFO)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set — nothing to do.")
        return 2

    model = os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    client = _client(model, api_key)
    print(f"Model: {model}  Base URL: {client.base_url}")

    results = [
        check_non_stream_tool_call(client),
        check_tool_result_round_trip(client),
        check_streaming_tool_call(client),
    ]

    passed = sum(1 for r in results if r["passed"])
    print(json.dumps({"model": model, "passed": passed, "total": len(results),
                      "results": results}, indent=2))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
