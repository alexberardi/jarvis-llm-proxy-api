from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, List, Optional, Union


@dataclass
class ImagePart:
    data: bytes
    mime_type: str
    detail: Optional[str] = None

    def to_data_url(self) -> str:
        """Reconstruct a data URL from the stored bytes and mime type."""
        b64 = base64.b64encode(self.data).decode("utf-8")
        return f"data:{self.mime_type};base64,{b64}"


@dataclass
class TextPart:
    text: str


ContentPart = Union[TextPart, ImagePart]


@dataclass
class NormalizedMessage:
    role: str
    content: List[ContentPart]


@dataclass
class GenerationParams:
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    response_format: Optional[dict] = None  # For JSON output: {"type": "json_object"}
    # Per-request adapter settings: {"hash": str, "scale": float, "enabled": bool}
    adapter_settings: Optional[dict] = None
    tools: Optional[list] = None  # OpenAI-format tool definitions for native tool calling
    tool_choice: Optional[Any] = None  # "auto", "none", or {"type": "function", "function": {"name": "..."}}


@dataclass
class ChatResult:
    content: str
    usage: Optional[dict] = None
    tool_calls: Optional[list] = None  # Structured tool calls from native tool calling
    finish_reason: Optional[str] = None  # "stop" or "tool_calls"

