"""End-to-end multimodal request path for the REST backend.

Covers the chain that a card-reader call actually takes:

    OpenAI-style message (base64 JPEG)
      -> message_service.normalize_messages  -> ImagePart(mime_type="image/jpeg")
      -> RestClient.generate_vision_chat     -> structured content parts
      -> chat_completion                     -> POST /v1/chat/completions

The point of the test is that the *mime type survives the round trip*. The
llama-server endpoint rejects images when no --mmproj is loaded, and a JPEG
mislabelled as PNG is the kind of thing that only shows up as a 500 in prod.

Run with:
    pytest tests/test_rest_vision_e2e.py -v
"""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from backends.rest_backend import RestClient
from managers.chat_types import GenerationParams
from models.api_models import Message
from services.message_service import normalize_message

FAKE_JPEG = b"\xff\xd8\xff\xe0not-really-a-jpeg"
FAKE_PNG = b"\x89PNG\r\n\x1a\nnot-really-a-png"


def _data_url(mime: str, payload: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(payload).decode()


@pytest.fixture
def rest_client(monkeypatch):
    """A RestClient whose HTTP layer is mocked; we assert on the posted payload."""
    monkeypatch.setattr(
        "backends.rest_backend.get_setting",
        lambda key, env_fallback, default, **k: (
            "openai" if key == "rest.provider" else default
        ),
    )
    client = RestClient(base_url="http://llama-vision:8080/", model_name="qwen3-vl")
    client.client = MagicMock()
    client.client.post = AsyncMock(
        return_value=MagicMock(
            raise_for_status=MagicMock(),
            json=MagicMock(
                return_value={
                    "choices": [{"message": {"content": "A clubs, K hearts"}}],
                    "usage": {"completion_tokens": 5},
                }
            ),
        )
    )
    return client


def _msg(content) -> list:
    """normalize one user Message from raw content parts."""
    return [normalize_message(Message(role="user", content=content))]


def _posted_messages(rest_client: RestClient):
    payload = rest_client.client.post.call_args.kwargs["json"]
    return payload["messages"]


def test_jpeg_reaches_provider_with_correct_mime_type(rest_client):
    """A JPEG must arrive as image/jpeg, not be rewritten to png."""
    messages = _msg(
        [
            {"type": "text", "text": "which cards?"},
            {"type": "image_url", "image_url": {"url": _data_url("image/jpeg", FAKE_JPEG)}},
        ]
    )

    result = _run(rest_client, messages)

    assert result.content == "A clubs, K hearts"
    parts = _posted_messages(rest_client)[0]["content"]
    image_parts = [p for p in parts if p["type"] == "image_url"]
    assert len(image_parts) == 1
    assert image_parts[0]["image_url"]["url"] == _data_url("image/jpeg", FAKE_JPEG)


def test_png_reaches_provider_with_correct_mime_type(rest_client):
    messages = _msg([{"type": "image_url", "image_url": {"url": _data_url("image/png", FAKE_PNG)}}])

    _run(rest_client, messages)

    parts = _posted_messages(rest_client)[0]["content"]
    assert parts[0]["image_url"]["url"] == _data_url("image/png", FAKE_PNG)


def test_image_bytes_are_not_corrupted_in_transit(rest_client):
    """base64 -> bytes -> base64 must be lossless (a decode on the wrong
    alphabet would silently truncate)."""
    messages = _msg([{"type": "image_url", "image_url": {"url": _data_url("image/jpeg", FAKE_JPEG)}}])

    _run(rest_client, messages)

    url = _posted_messages(rest_client)[0]["content"][0]["image_url"]["url"]
    _, b64 = url.split(",", 1)
    assert base64.b64decode(b64) == FAKE_JPEG


def test_text_and_image_both_preserved_in_order(rest_client):
    messages = _msg(
        [
            {"type": "text", "text": "before"},
            {"type": "image_url", "image_url": {"url": _data_url("image/jpeg", FAKE_JPEG)}},
            {"type": "text", "text": "after"},
        ]
    )

    _run(rest_client, messages)

    kinds = [p["type"] for p in _posted_messages(rest_client)[0]["content"]]
    assert kinds == ["text", "image_url", "text"]


def test_model_name_is_sent_to_provider(rest_client):
    """llama-server keys off the model field; dropping it 500s."""
    messages = [normalize_message(Message(role="user", content="hi"))]
    _run(rest_client, messages)
    assert rest_client.client.post.call_args.kwargs["json"]["model"] == "qwen3-vl"


def _run(rest_client: RestClient, messages):
    """Drive the async vision call from a sync test."""
    params = GenerationParams(temperature=0.0, max_tokens=32)
    return asyncio.run(
        rest_client.generate_vision_chat(MagicMock(), messages, params)
    )
