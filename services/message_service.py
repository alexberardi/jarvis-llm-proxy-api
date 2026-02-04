"""Message normalization and processing service.

Provides utilities for converting OpenAI-style messages to internal normalized format,
handling both text-only and multimodal (text + images) content.
"""

import base64
import re
from typing import List, Tuple, Union

from fastapi import HTTPException

from managers.chat_types import (
    NormalizedMessage,
    TextPart,
    ImagePart,
)
from models.api_models import Message
from services.json_repair_service import JSON_SYSTEM_MESSAGE


def parse_data_url(data_url: str) -> Tuple[bytes, str]:
    """Parse a data URL and return (image_bytes, mime_type).

    Expected format: data:image/png;base64,iVBORw0KG...
    """
    match = re.match(r"^data:([^;]+);base64,(.+)$", data_url)
    if not match:
        raise ValueError(
            "Invalid data URL format. Expected: data:<mime_type>;base64,<data>"
        )

    mime_type = match.group(1)
    b64_data = match.group(2)

    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {e}")

    return image_bytes, mime_type


def normalize_message(message: Message) -> NormalizedMessage:
    """Convert an OpenAI-style message to a NormalizedMessage.

    Handles both:
    - content as string: {"role": "user", "content": "Hello"}
    - content as array: {"role": "user", "content": [{"type": "text", "text": "..."}, {"type": "image_url", ...}]}
    """
    content_parts: List[Union[TextPart, ImagePart]] = []

    if isinstance(message.content, str):
        # Simple string content
        content_parts.append(TextPart(text=message.content))
    elif isinstance(message.content, list):
        # Structured content array
        for part in message.content:
            if part.type == "text" and part.text is not None:
                content_parts.append(TextPart(text=part.text))
            elif part.type == "image_url" and part.image_url is not None:
                url = part.image_url.url
                # For now, only support data URLs
                if not url.startswith("data:"):
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": {
                                "type": "invalid_request_error",
                                "message": "Only data URLs are supported for images. HTTP(S) URLs are not yet supported.",
                                "code": None,
                            }
                        },
                    )
                image_bytes, mime_type = parse_data_url(url)
                content_parts.append(
                    ImagePart(
                        data=image_bytes,
                        mime_type=mime_type,
                        detail=part.image_url.detail,
                    )
                )
            else:
                # Unknown part type or missing required field
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": {
                            "type": "invalid_request_error",
                            "message": f"Unsupported content part type or missing field: {part.type}",
                            "code": None,
                        }
                    },
                )
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request_error",
                    "message": "Message content must be either a string or an array of content parts",
                    "code": None,
                }
            },
        )

    return NormalizedMessage(role=message.role, content=content_parts)


def normalize_messages(messages: List[Message]) -> List[NormalizedMessage]:
    """Convert a list of OpenAI-style messages to NormalizedMessage format."""
    return [normalize_message(msg) for msg in messages]


def inject_json_system_message(
    messages: List[NormalizedMessage],
) -> List[NormalizedMessage]:
    """Inject or augment a system message to request JSON output.

    If a system message exists, append JSON instructions to it.
    If no system message exists, prepend one.
    """
    # Check if there's already a system message
    has_system = any(msg.role == "system" for msg in messages)

    if has_system:
        # Augment existing system message
        augmented = []
        for msg in messages:
            if msg.role == "system":
                # Get existing text content
                existing_text = ""
                for part in msg.content:
                    if isinstance(part, TextPart):
                        existing_text += part.text + "\n"

                # Check if JSON instructions are already present
                if (
                    "JSON" not in existing_text.upper()
                    or "valid json" not in existing_text.lower()
                ):
                    augmented_text = (
                        existing_text.strip() + "\n\n" + JSON_SYSTEM_MESSAGE
                    )
                else:
                    augmented_text = existing_text  # Already has JSON instructions

                augmented.append(
                    NormalizedMessage(
                        role="system", content=[TextPart(text=augmented_text)]
                    )
                )
            else:
                augmented.append(msg)
        return augmented
    else:
        # Prepend new system message
        system_msg = NormalizedMessage(
            role="system", content=[TextPart(text=JSON_SYSTEM_MESSAGE)]
        )
        return [system_msg] + messages


def has_images(messages: List[NormalizedMessage]) -> bool:
    """Check if any message contains an ImagePart."""
    for msg in messages:
        for part in msg.content:
            if isinstance(part, ImagePart):
                return True
    return False


def request_has_images(messages: List[Message]) -> bool:
    """Check if any raw request message contains image content."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.type == "image_url":
                    return True
    return False
