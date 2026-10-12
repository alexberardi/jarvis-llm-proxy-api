"""
Tests for message normalization in services/vision_inference.py.

Regression coverage for base64 image handling:
  - ImagePart takes `mime_type`, not `media_type`
  - the mime type is read from the data URL header instead of being hardcoded

Run with:
    pytest tests/test_vision_message_parsing.py -v
"""

import base64

from managers.chat_types import ImagePart, TextPart
from services.vision_inference import _parse_messages


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode()


def test_data_url_produces_image_part_with_real_mime_type():
    """A JPEG data URL must survive as image/jpeg, not be hardcoded to png."""
    raw = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what cards?"},
                {"type": "image_url", "image_url": {"url": _data_url("image/jpeg", raw)}},
            ],
        }
    ]

    parsed = _parse_messages(messages)

    image_parts = [p for p in parsed[0].content if isinstance(p, ImagePart)]
    assert len(image_parts) == 1
    assert image_parts[0].data == raw
    assert image_parts[0].mime_type == "image/jpeg"


def test_image_part_reconstructs_original_data_url():
    """to_data_url() must round-trip the mime type the caller sent."""
    raw = b"png-bytes"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_url("image/png", raw)}},
            ],
        }
    ]

    image_part = next(p for p in _parse_messages(messages)[0].content if isinstance(p, ImagePart))
    assert image_part.to_data_url() == _data_url("image/png", raw)


def test_png_data_url_uses_png():
    raw = b"png-bytes"
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _data_url("image/png", raw)}}],
        }
    ]

    image_part = next(p for p in _parse_messages(messages)[0].content if isinstance(p, ImagePart))
    assert image_part.mime_type == "image/png"


def test_direct_image_type_uses_mime_type_field():
    """The `image` content type accepts mime_type as well as media_type."""
    raw = b"webp-bytes"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "data": base64.b64encode(raw).decode(),
                    "mime_type": "image/webp",
                }
            ],
        }
    ]

    image_part = next(p for p in _parse_messages(messages)[0].content if isinstance(p, ImagePart))
    assert image_part.data == raw
    assert image_part.mime_type == "image/webp"


def test_direct_image_type_accepts_media_type_alias():
    """Legacy callers sending `media_type` still get a working ImagePart."""
    raw = b"jpeg-bytes"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "data": base64.b64encode(raw).decode(),
                    "media_type": "image/jpeg",
                }
            ],
        }
    ]

    image_part = next(p for p in _parse_messages(messages)[0].content if isinstance(p, ImagePart))
    assert image_part.data == raw
    assert image_part.mime_type == "image/jpeg"


def test_string_content_still_yields_text_part():
    parsed = _parse_messages([{"role": "user", "content": "hello"}])
    assert parsed[0].content == [TextPart(text="hello")]


def test_malformed_base64_does_not_raise():
    """Bad base64 is skipped with a warning rather than blowing up the request."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "still here?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,!!!not-base64!!!"}},
            ],
        }
    ]

    parsed = _parse_messages(messages)

    assert not [p for p in parsed[0].content if isinstance(p, ImagePart)]
    assert TextPart(text="still here?") in parsed[0].content
