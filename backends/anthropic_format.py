"""Translation between the OpenAI chat shape and the Anthropic Messages API.

Everything in this service — ``GenerationParams.tools``, the ``tool_calls``
array on ``ChatResult``, the terminal streaming frame, ``response_helpers`` and
the command-center service downstream — speaks OpenAI. Anthropic's
``POST /v1/messages`` does not: tools carry an ``input_schema`` instead of
``parameters``, the system prompt is a top-level field rather than a message,
tool calls and tool results are content *blocks*, and streaming is a sequence
of typed SSE events with no ``[DONE]`` sentinel.

This module is the seam. Outbound it builds the Messages payload; inbound it
normalises the response back to the OpenAI shape so nothing downstream has to
learn a second vocabulary:

    content        concatenation of every text block, in order
    tool_calls     [{"id", "type": "function",
                     "function": {"name", "arguments": <JSON string>}}] or None
    finish_reason  tool_use -> "tool_calls", max_tokens /
                   model_context_window_exceeded -> "length",
                   refusal -> "content_filter", else "stop"
    usage          {"prompt_tokens", "completion_tokens", "total_tokens"}

Kept free of HTTP so every rule above is unit-testable without a socket; the
REST backend only chooses between this and the OpenAI path.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from managers.chat_types import ChatResult, GenerationParams

# Anthropic REQUIRES max_tokens on every request; OpenAI treats it as optional
# and most callers here omit it. Last-resort default only — RestClient resolves
# the real fallback from the `inference.general.max_tokens` setting and passes
# it in as `default_max_tokens`.
DEFAULT_MAX_TOKENS: int = 4096

# Pinned per the Messages API docs; sent alongside x-api-key on every request.
ANTHROPIC_VERSION: str = "2023-06-01"

_EMPTY_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}}

# Models whose extended thinking is always on: sending thinking.type="disabled"
# is a 400. Everything else gets thinking disabled explicitly (see
# build_anthropic_payload).
_ALWAYS_THINKING_MODELS: Tuple[str, ...] = ("fable", "mythos")

logger = logging.getLogger("uvicorn")


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    """Decode an OpenAI ``function.arguments`` JSON string into a dict.

    Models emit ``""`` for a no-argument call and occasionally emit truncated
    JSON; neither should take down the turn, so both degrade to ``{}``.
    """
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str) or not arguments.strip():
        return {}
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _stringify(value: Any) -> str:
    """Coerce tool-result content to the string Anthropic expects."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _text_blocks(content: Any) -> List[Dict[str, Any]]:
    """Turn OpenAI message content into Anthropic ``text`` blocks.

    Empty text yields no blocks — an empty block is a 400 from the API.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        blocks: List[Dict[str, Any]] = []
        for part in content:
            if isinstance(part, str):
                if part:
                    blocks.append({"type": "text", "text": part})
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text:
                    blocks.append({"type": "text", "text": text})
        return blocks
    return [{"type": "text", "text": str(content)}]


def openai_tools_to_anthropic(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """``{"type":"function","function":{...}}`` -> ``{"name","description","input_schema"}``."""
    out: List[Dict[str, Any]] = []
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(fn, dict):
            # Already Anthropic-shaped (or a bare function dict) — take it as is.
            fn = tool if isinstance(tool, dict) else {}
        name = fn.get("name")
        if not name:
            continue
        schema = fn.get("parameters")
        if not isinstance(schema, dict):
            schema = fn.get("input_schema")
        if not isinstance(schema, dict):
            schema = dict(_EMPTY_SCHEMA)
        translated: Dict[str, Any] = {"name": name}
        description = fn.get("description")
        if description:
            translated["description"] = description
        translated["input_schema"] = schema
        out.append(translated)
    return out


def openai_tool_choice_to_anthropic(tool_choice: Any) -> Optional[Dict[str, Any]]:
    """Map an OpenAI ``tool_choice`` onto Anthropic's object form.

    Returns None when the caller did not specify one, so the key is omitted
    entirely and the model keeps its default (auto).

    Note: forced tool use was removed on Claude Fable 5.1 / Mythos 5.1, which
    reject ``{"type": "any"}`` and ``{"type": "tool"}`` outright. Every caller
    in this repo only ever sends ``"auto"``, so the forced forms below exist
    for completeness rather than for a path we exercise.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        lowered = tool_choice.lower()
        if lowered == "required":
            return {"type": "any"}
        if lowered in ("none", "any", "auto"):
            return {"type": lowered}
        return {"type": "auto"}
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            name = (tool_choice.get("function") or {}).get("name")
            if name:
                return {"type": "tool", "name": name}
            return {"type": "any"}
        if tool_choice.get("type") == "tool" and not tool_choice.get("name"):
            # ``tool`` without a name is not a legal tool_choice; "any" is the
            # closest legal intent ("call some tool").
            return {"type": "any"}
        if tool_choice.get("type") in ("auto", "any", "none", "tool"):
            # Already Anthropic-shaped.
            return dict(tool_choice)
    return {"type": "auto"}


def openai_messages_to_anthropic(
    messages: List[Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Split OpenAI messages into ``(system, messages)`` for the Messages API.

    - ``role: "system"`` messages are hoisted out and joined with blank lines;
      Anthropic has no system role inside ``messages``.
    - Assistant ``tool_calls`` become ``tool_use`` blocks (arguments decoded
      from their JSON string), ``role: "tool"`` results become ``tool_result``
      blocks on a *user* message.
    - Consecutive same-role messages are merged into one. This is what makes
      parallel tool calls legal: every ``tool_result`` for one assistant turn
      must arrive in a SINGLE user message.
    - Messages that would carry no blocks at all are dropped rather than sent
      empty (a 400), and leading ``assistant`` messages are dropped because the
      Messages API requires the first message to be ``user``.

    Raises:
        ValueError: for a ``role: "tool"`` message with no ``tool_call_id`` —
            ``tool_use_id: ""`` cannot be matched to a tool_use block and is a
            guaranteed 400, so fail here rather than on the wire.
    """
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []

    def _append(role: str, blocks: List[Dict[str, Any]]) -> None:
        if not blocks:
            return
        if not out and role == "assistant":
            # The first message must be "user"; a leading assistant turn (a
            # greeting replayed from history) is a 400.
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role") or "user"
        content = msg.get("content")

        if role == "system":
            text = "".join(b["text"] for b in _text_blocks(content))
            if text.strip():
                system_parts.append(text)
            continue

        if role == "tool":
            tool_use_id = msg.get("tool_call_id") or msg.get("id")
            if not tool_use_id:
                raise ValueError(
                    "Anthropic tool_result requires tool_call_id; "
                    "a tool-role message arrived without one"
                )
            _append(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": _stringify(content),
                    }
                ],
            )
            continue

        if role == "assistant":
            blocks = _text_blocks(content)
            for call in msg.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        # An empty id is a 400; synthesise a stable one when the
                        # upstream conversion dropped it.
                        "id": call.get("id") or f"toolu_generated_{len(out)}_{len(blocks)}",
                        "name": fn.get("name") or call.get("name") or "",
                        "input": _parse_tool_arguments(fn.get("arguments")),
                    }
                )
            _append("assistant", blocks)
            continue

        _append("user", _text_blocks(content))

    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def build_anthropic_payload(
    model: str,
    messages: List[Dict[str, Any]],
    params: GenerationParams,
    stream: bool,
    default_max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    """Build a ``POST /v1/messages`` body from OpenAI-shaped inputs.

    Deliberately omits every sampling knob the caller may have set:

    - ``temperature`` / ``top_p`` / ``top_k``: current Claude models (Sonnet 5,
      Opus 5 / 4.8 / 4.7, Fable 5 / 5.1) return 400 when any of them is present
      — Sonnet 5 rejects any non-default value outright.
    - ``seed`` and the llama-server-only reasoning fields: no such parameters.

    Thinking is sent as ``{"type": "disabled"}``, because omitting it runs
    ADAPTIVE thinking by default on Sonnet 5 / Opus 5 / Fable, and thinking
    blocks must then be replayed verbatim on the tool-result round trip —
    something the OpenAI-shaped ``ChatResult`` cannot carry. The exception is
    Fable / Mythos, where thinking is always on and ``disabled`` is itself a
    400: the key is omitted there, so tool round trips on those models are NOT
    supported yet (they would need thinking-block replay).

    Raises:
        ValueError: when no user/assistant message survives translation.
    """
    system, anthropic_messages = openai_messages_to_anthropic(messages)
    if not anthropic_messages:
        raise ValueError("Anthropic request needs at least one user/assistant message")

    payload: Dict[str, Any] = {
        "model": model,
        "messages": anthropic_messages,
        "max_tokens": params.max_tokens if params.max_tokens is not None else default_max_tokens,
        "stream": stream,
    }
    # No temperature/top_p/top_k: current Claude models 400 on them (see above).
    if not any(token in model.lower() for token in _ALWAYS_THINKING_MODELS):
        payload["thinking"] = {"type": "disabled"}
    if system:
        payload["system"] = system
    if params.tools:
        payload["tools"] = openai_tools_to_anthropic(params.tools)
        tool_choice = openai_tool_choice_to_anthropic(params.tool_choice)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    return payload


def map_stop_reason(stop_reason: Optional[str]) -> str:
    """Anthropic ``stop_reason`` -> OpenAI ``finish_reason``.

    ``end_turn``, ``stop_sequence``, ``pause_turn`` and an absent stop_reason
    all read as a normal end of turn downstream.
    """
    if stop_reason == "tool_use":
        return "tool_calls"
    if stop_reason in ("max_tokens", "model_context_window_exceeded"):
        return "length"
    if stop_reason == "refusal":
        # OpenAI's finish_reason for a model-side refusal.
        return "content_filter"
    return "stop"


def _log_notable_stop_reason(
    stop_reason: Optional[str], stop_details: Optional[Dict[str, Any]] = None
) -> None:
    """Warn on the two stop reasons that mean the turn did not really finish."""
    if stop_reason == "refusal":
        logger.warning(
            f"⚠️ Anthropic stop_reason=refusal (the model declined to continue); "
            f"stop_details={stop_details}"
        )
    elif stop_reason == "pause_turn":
        logger.warning(
            "⚠️ Anthropic stop_reason=pause_turn — the turn was paused mid-flight "
            "and would need to be continued; treating it as a normal stop"
        )


def normalize_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Anthropic ``input_tokens``/``output_tokens`` -> OpenAI token counts.

    Downstream metrics read ``prompt_tokens``/``completion_tokens``/
    ``total_tokens``; leaking the raw Anthropic keys reads as zero usage.
    """
    usage = usage or {}
    # Cache reads/writes are billed input tokens and are reported SEPARATELY
    # from input_tokens; omitting them understates the prompt by the whole
    # cached prefix.
    prompt = (
        int(usage.get("input_tokens") or 0)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )
    completion = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _tool_call_from_block(block: Dict[str, Any]) -> Dict[str, Any]:
    """One ``tool_use`` block -> one OpenAI tool call."""
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        tool_input = {}
    return {
        "id": block.get("id") or "",
        "type": "function",
        "function": {
            "name": block.get("name") or "",
            "arguments": json.dumps(tool_input),
        },
    }


def parse_anthropic_message(data: Dict[str, Any]) -> ChatResult:
    """Parse a non-streaming Messages response into a normalized ChatResult.

    ``content`` may interleave any number of ``text`` and ``tool_use`` blocks,
    so text is concatenated in order and every tool_use becomes a tool call.
    """
    content_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            content_parts.append(block.get("text") or "")
        elif block.get("type") == "tool_use":
            tool_calls.append(_tool_call_from_block(block))

    stop_reason = data.get("stop_reason")
    _log_notable_stop_reason(stop_reason, data.get("stop_details"))

    return ChatResult(
        content="".join(content_parts),
        usage=normalize_usage(data.get("usage")),
        tool_calls=tool_calls or None,
        finish_reason=map_stop_reason(stop_reason),
    )


class AnthropicStreamAccumulator:
    """Fold an Anthropic SSE event stream into one normalized result.

    The stream is a sequence of typed events rather than OpenAI's
    ``choices[].delta`` chunks, and it ends at ``message_stop`` with no
    ``[DONE]`` sentinel:

        message_start
        content_block_start / content_block_delta* / content_block_stop  (per block)
        message_delta (stop_reason + output_tokens)
        message_stop

    Blocks are keyed by ``index``, not arrival order, and a tool call's
    arguments arrive as ``input_json_delta`` fragments that are only valid JSON
    once concatenated — so accumulate per index and parse at the end. ``ping``
    may appear anywhere; an ``error`` frame can arrive mid-stream and must not
    be mistaken for a normal end.
    """

    def __init__(self) -> None:
        self._blocks: Dict[int, Dict[str, Any]] = {}
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._stop_reason: Optional[str] = None
        self._stop_details: Optional[Dict[str, Any]] = None

    def feed(self, event: Dict[str, Any]) -> Optional[str]:
        """Consume one SSE frame; return a text delta to yield, or None.

        Raises:
            RuntimeError: on an ``error`` frame (overloaded, invalid request …).
        """
        event_type = event.get("type")

        if event_type == "error":
            err = event.get("error") or {}
            message = err.get("message") or "unknown error"
            raise RuntimeError(f"Anthropic stream error ({err.get('type', 'error')}): {message}")

        if event_type == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            # Cached input is reported separately but billed as prompt tokens.
            self._prompt_tokens = (
                int(usage.get("input_tokens") or 0)
                + int(usage.get("cache_read_input_tokens") or 0)
                + int(usage.get("cache_creation_input_tokens") or 0)
            )
            output = usage.get("output_tokens")
            if output:
                self._completion_tokens = int(output)
            return None

        if event_type == "content_block_start":
            block = event.get("content_block") or {}
            self._blocks[int(event.get("index") or 0)] = {
                "type": block.get("type"),
                "id": block.get("id"),
                "name": block.get("name"),
                "text": "",
                "partial_json": "",
            }
            return None

        if event_type == "content_block_delta":
            index = int(event.get("index") or 0)
            slot = self._blocks.setdefault(
                index,
                {"type": "text", "id": None, "name": None, "text": "", "partial_json": ""},
            )
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                text = delta.get("text") or ""
                if text:
                    slot["text"] += text
                    return text
            elif delta.get("type") == "input_json_delta":
                slot["partial_json"] += delta.get("partial_json") or ""
            return None

        if event_type == "message_delta":
            delta_obj = event.get("delta") or {}
            stop_reason = delta_obj.get("stop_reason")
            if stop_reason:
                self._stop_reason = stop_reason
            stop_details = delta_obj.get("stop_details")
            if stop_details:
                self._stop_details = stop_details
            usage = event.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self._completion_tokens = int(usage.get("output_tokens") or 0)
            if usage.get("input_tokens"):
                self._prompt_tokens = int(usage["input_tokens"])
            return None

        # content_block_stop / message_stop / ping / anything unknown: nothing
        # to emit. The block buffers already hold everything needed.
        return None

    def result(self) -> Dict[str, Any]:
        """The terminal frame's payload, in the OpenAI-normalized shape."""
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for index in sorted(self._blocks):
            slot = self._blocks[index]
            if slot.get("type") == "tool_use":
                tool_calls.append(
                    _tool_call_from_block(
                        {
                            "id": slot.get("id"),
                            "name": slot.get("name"),
                            "input": _parse_tool_arguments(slot.get("partial_json")),
                        }
                    )
                )
            else:
                content_parts.append(slot.get("text") or "")

        _log_notable_stop_reason(self._stop_reason, self._stop_details)

        return {
            # _prompt_tokens already folds in the cache_* counters.
            "content": "".join(content_parts),
            "usage": normalize_usage(
                {
                    "input_tokens": self._prompt_tokens,
                    "output_tokens": self._completion_tokens,
                }
            ),
            "tool_calls": tool_calls or None,
            "finish_reason": map_stop_reason(self._stop_reason),
        }
