from __future__ import annotations

import os
import time
from typing import Any, Dict, Generator, List

from managers.chat_types import ChatResult, ImagePart, NormalizedMessage, TextPart
from backends.base import LLMBackendBase


class MockBackend(LLMBackendBase):
    """Lightweight backend used for tests and local development."""

    def __init__(self, name: str = "mock-backend") -> None:
        self.name = name
        self.model_name = name
        self.inference_engine = "mock"
        self.last_usage = None

    async def generate_text_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: Any,
    ) -> ChatResult:
        text_parts: list[str] = []
        for message in messages:
            for part in message.content:
                if isinstance(part, TextPart):
                    text_parts.append(part.text)
        combined = " ".join(text_parts).strip()
        content = f"[mock-text:{getattr(model_cfg, 'model_id', 'model')}] {combined}"
        usage = {
            "prompt_tokens": len(combined.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(combined.split()) + len(content.split()),
        }
        return ChatResult(content=content, usage=usage, tool_calls=None, finish_reason="stop")

    async def generate_vision_chat(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: Any,
    ) -> ChatResult:
        image_count = 0
        text_parts: list[str] = []
        for message in messages:
            for part in message.content:
                if isinstance(part, ImagePart):
                    image_count += 1
                elif isinstance(part, TextPart):
                    text_parts.append(part.text)
        combined = " ".join(text_parts).strip()
        content = (
            f"[mock-vision:{getattr(model_cfg, 'model_id', 'model')}] "
            f"images={image_count} text={combined}"
        )
        usage = {
            "prompt_tokens": len(combined.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(combined.split()) + len(content.split()),
        }
        return ChatResult(content=content, usage=usage, tool_calls=None, finish_reason="stop")

    def generate_text_chat_stream(
        self,
        model_cfg: Any,
        messages: List[NormalizedMessage],
        params: Any,
    ) -> Generator[Dict[str, Any], None, None]:
        """Sync streaming generator matching the gguf/mlx frame contract:
        {"delta": tok} per token, then the {"done": true, ...} final frame.

        MOCK_STREAM_TOKENS / MOCK_STREAM_DELAY_S control length and pacing so
        smoke tests can catch a stream mid-flight (defaults: 20 × 50 ms).
        """
        n_tokens = int(os.getenv("MOCK_STREAM_TOKENS", "20"))
        delay_s = float(os.getenv("MOCK_STREAM_DELAY_S", "0.05"))

        content = ""
        for i in range(n_tokens):
            time.sleep(delay_s)
            tok = f"mock{i} "
            content += tok
            yield {"delta": tok}

        usage = {
            "prompt_tokens": 0,
            "completion_tokens": n_tokens,
            "total_tokens": n_tokens,
        }
        self.last_usage = usage
        yield {
            "done": True,
            "content": content.strip(),
            "usage": usage,
            "tool_calls": None,
            "finish_reason": "stop",
        }

    def unload(self) -> None:
        """No-op for mock backend."""
        pass

