from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import List, Optional, Union


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
    grammar: Optional[str] = None  # GBNF grammar for constrained generation
    # Per-request adapter settings: {"hash": str, "scale": float, "enabled": bool}
    adapter_settings: Optional[dict] = None


@dataclass
class ChatResult:
    content: str
    usage: Optional[dict] = None

