"""
Custom chat format handlers for llama-cpp-python.

Registers additional chat_format options that can be used via the
model.main.chat_format setting. Import this module before creating
Llama instances so handlers are available in the registry.

Registered formats:
  - gemma4: Gemma 4 native template with thinking disabled.
"""

from typing import Any, List

from llama_cpp import llama_types
from llama_cpp.llama_chat_format import (
    ChatFormatterResponse,
    LlamaChatCompletionHandlerRegistry,
    chat_formatter_to_chat_completion_handler,
)


# ---------------------------------------------------------------------------
# gemma4 — native Gemma 4 template, thinking disabled
# ---------------------------------------------------------------------------
#
# Uses <start_of_turn>/<end_of_turn> markers (native) instead of ChatML.
# Unlike the built-in "gemma" handler, this one:
#   1. Supports system messages (Gemma 4 added system role support).
#   2. Prefills an empty thinking channel (<|channel>thought\n<channel|>)
#      at the start of the model turn so the model skips reasoning and
#      responds immediately — ~2-5x faster on 31B.
#
# Google recommends the empty thinking channel for 31B/26B-A4B to suppress
# "ghost" thought blocks that appear even when thinking is off.
#
# Ref: https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
# Ref: https://ai.google.dev/gemma/docs/capabilities/thinking


def _format_gemma4(
    messages: List[llama_types.ChatCompletionRequestMessage],
    **kwargs: Any,
) -> ChatFormatterResponse:
    """Format messages using Gemma 4's native template without thinking."""
    parts: list[str] = []

    for msg in messages:
        role: str = msg["role"]  # type: ignore[typeddict-item]
        content: str = msg.get("content", "") or ""  # type: ignore[arg-type]

        if role == "system":
            parts.append(f"<start_of_turn>system\n{content}<end_of_turn>\n")
        elif role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>\n")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>\n")

    # Generation prompt: start model turn with empty thinking channel.
    # The model sees "thinking already happened (nothing to think about)"
    # and jumps straight to producing the response.
    parts.append("<start_of_turn>model\n<|channel>thought\n<channel|>")

    return ChatFormatterResponse(
        prompt="".join(parts),
        stop="<end_of_turn>",
    )


# Register on import
_registry = LlamaChatCompletionHandlerRegistry()
_handler = chat_formatter_to_chat_completion_handler(_format_gemma4)
_registry.register_chat_completion_handler("gemma4", _handler)
