import base64
import importlib
import os

import pytest
from fastapi.testclient import TestClient

from managers.model_manager import ModelManager


@pytest.fixture
def set_mock_env(monkeypatch):
    envs = {
        "JARVIS_MODEL_NAME": "jarvis-text-8b",
        "JARVIS_LIGHTWEIGHT_MODEL_NAME": "jarvis-text-1b",
        "JARVIS_VISION_MODEL_NAME": "jarvis-vision-11b",
        "JARVIS_MODEL_BACKEND": "MOCK",
        "JARVIS_LIGHTWEIGHT_MODEL_BACKEND": "MOCK",
        "JARVIS_VISION_MODEL_BACKEND": "MOCK",
    }
    for key, value in envs.items():
        monkeypatch.setenv(key, value)
    # Ensure downstream env lookups default to mock as well
    monkeypatch.delenv("JARVIS_REST_MODEL_URL", raising=False)
    yield


@pytest.fixture
def client(set_mock_env, mock_auth, mock_model_service):
    from tests.conftest import apply_auth_mock
    import main

    importlib.reload(main)
    apply_auth_mock(main.app)
    return TestClient(main.app)


def test_model_alias_resolution(set_mock_env):
    manager = ModelManager()
    assert manager.get_model_config("full").model_id == "jarvis-text-8b"
    assert manager.get_model_config("lightweight").model_id == "jarvis-text-1b"
    vision_cfg = manager.get_model_config("vision")
    assert vision_cfg.model_id == "jarvis-vision-11b"
    assert vision_cfg.supports_images is True


def test_chat_with_string_content(client):
    """Verify simple string content is accepted and processed."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "full",
            "messages": [{"role": "user", "content": "hello there"}],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "choices" in payload
    assert payload["choices"][0]["message"]["role"] == "assistant"
    assert payload["choices"][0]["message"]["content"]  # Has some content


def test_chat_with_structured_text_content(client):
    """Verify structured content array format is accepted and processed."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "lightweight",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "structured hello"}],
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "choices" in payload
    assert payload["choices"][0]["message"]["content"]  # Has some content


def test_chat_with_image_requires_vision_model(client):
    """Verify that images sent to non-vision models are rejected."""
    image_b64 = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "full",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 400
    payload = response.json()
    # FastAPI wraps HTTPException detail in "detail" key
    assert payload["detail"]["error"]["type"] == "invalid_request_error"


def test_chat_with_image_uses_vision_model(client):
    """Verify that vision model accepts image content."""
    image_b64 = base64.b64encode(b"fake-image-bytes").decode("utf-8")
    data_url = f"data:image/png;base64,{image_b64}"
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "vision",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert "choices" in payload
    assert payload["choices"][0]["message"]["content"]  # Has some content

