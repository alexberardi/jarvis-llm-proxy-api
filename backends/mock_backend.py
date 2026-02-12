from __future__ import annotations

from typing import Any, List

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
        return ChatResult(content=content, usage=usage)

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
        return ChatResult(content=content, usage=usage)

    def unload(self) -> None:
        """No-op for mock backend."""
        pass

