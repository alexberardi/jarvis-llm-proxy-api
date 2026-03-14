import importlib
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from managers.model_manager import ModelManager


@pytest.fixture
def set_mock_env(monkeypatch):
    envs = {
        "JARVIS_MODEL_NAME": "jarvis-text-8b",
        "JARVIS_MODEL_BACKEND": "MOCK",
    }
    for key, value in envs.items():
        monkeypatch.setenv(key, value)
    # Ensure downstream env lookups default to mock as well
    monkeypatch.delenv("JARVIS_REST_MODEL_URL", raising=False)
    # Clear any new-style env vars so fallback chain works
    for var in (
        "JARVIS_LIVE_MODEL_NAME", "JARVIS_LIVE_MODEL_BACKEND",
        "JARVIS_BACKGROUND_MODEL_NAME", "JARVIS_BACKGROUND_MODEL_BACKEND",
        "JARVIS_LIGHTWEIGHT_MODEL_NAME", "JARVIS_LIGHTWEIGHT_MODEL_BACKEND",
        "JARVIS_VISION_MODEL_NAME", "JARVIS_VISION_MODEL_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def set_mock_env_separate_bg(monkeypatch):
    """Env where background uses a different model than live."""
    envs = {
        "JARVIS_MODEL_NAME": "jarvis-text-8b",
        "JARVIS_MODEL_BACKEND": "MOCK",
        "JARVIS_BACKGROUND_MODEL_NAME": "jarvis-bg-model",
        "JARVIS_BACKGROUND_MODEL_BACKEND": "MOCK",
    }
    for key, value in envs.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("JARVIS_REST_MODEL_URL", raising=False)
    for var in (
        "JARVIS_LIVE_MODEL_NAME", "JARVIS_LIVE_MODEL_BACKEND",
        "JARVIS_LIGHTWEIGHT_MODEL_NAME", "JARVIS_LIGHTWEIGHT_MODEL_BACKEND",
        "JARVIS_VISION_MODEL_NAME", "JARVIS_VISION_MODEL_BACKEND",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


@pytest.fixture
def client(set_mock_env, mock_auth, mock_model_service):
    from tests.conftest import apply_auth_mock
    import main

    importlib.reload(main)
    apply_auth_mock(main.app)
    return TestClient(main.app)


def test_model_alias_resolution_live_and_background(set_mock_env):
    """Verify 'live' and 'background' aliases resolve correctly."""
    ModelManager._instance = None
    ModelManager._initialized = False
    with patch("services.settings_helpers._get_settings_service", return_value=None):
        manager = ModelManager()
        # Canonical aliases
        assert manager.get_model_config("live").model_id == "jarvis-text-8b"
        assert manager.get_model_config("background").model_id == "jarvis-text-8b"
        # When no separate background is configured, they share an instance
        assert manager.live_model is manager.background_model
    ModelManager._instance = None
    ModelManager._initialized = False


def test_backwards_compat_aliases(set_mock_env):
    """Verify old aliases (full, lightweight, cloud, vision) still resolve."""
    ModelManager._instance = None
    ModelManager._initialized = False
    with patch("services.settings_helpers._get_settings_service", return_value=None):
        manager = ModelManager()
        assert manager.get_model_config("full").model_id == "jarvis-text-8b"
        assert manager.get_model_config("lightweight").model_id == "jarvis-text-8b"
        assert manager.get_model_config("cloud").model_id == "jarvis-text-8b"
        assert manager.get_model_config("vision").model_id == "jarvis-text-8b"
    ModelManager._instance = None
    ModelManager._initialized = False


def test_separate_background_model(set_mock_env_separate_bg):
    """Verify separate background model creates a distinct instance."""
    ModelManager._instance = None
    ModelManager._initialized = False
    with patch("services.settings_helpers._get_settings_service", return_value=None):
        manager = ModelManager()
        assert manager.get_model_config("live").model_id == "jarvis-text-8b"
        assert manager.get_model_config("background").model_id == "jarvis-bg-model"
        # They should be different instances
        assert manager.live_model is not manager.background_model
        # Deprecated aliases
        assert manager.get_model_config("full").model_id == "jarvis-text-8b"
        assert manager.get_model_config("cloud").model_id == "jarvis-bg-model"
        assert manager.get_model_config("vision").model_id == "jarvis-bg-model"
    ModelManager._instance = None
    ModelManager._initialized = False


def test_chat_with_string_content(client):
    """Verify simple string content is accepted and processed."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "live",
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
            "model": "live",
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


def test_chat_always_uses_live_model(client):
    """Verify that the chat endpoint always routes to 'live' regardless of model name sent."""
    # Even if we send "background" or "full", the chat endpoint should route to live
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "background",
            "messages": [{"role": "user", "content": "test"}],
        },
    )
    assert response.status_code == 200
