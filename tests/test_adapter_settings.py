"""
Tests for AdapterSettings model and request flow.

Phase 1b: Validates that adapter_settings:
1. Can be parsed from ChatCompletionRequest
2. Flows through to GenerationParams
3. Is accessible in backends (logged but not used yet)

Run with:
    PYTHONPATH=. pytest tests/test_adapter_settings.py -v
"""

import pytest
from pydantic import ValidationError

from managers.chat_types import GenerationParams
from models.api_models import AdapterSettings, ChatCompletionRequest, Message


class TestAdapterSettingsModel:
    """Test AdapterSettings Pydantic model validation."""

    def test_valid_adapter_settings_minimal(self) -> None:
        """Test minimal valid AdapterSettings with only required field."""
        settings = AdapterSettings(hash="abc123")

        assert settings.hash == "abc123"
        assert settings.scale == 1.0  # Default
        assert settings.enabled is True  # Default

    def test_valid_adapter_settings_full(self) -> None:
        """Test AdapterSettings with all fields specified."""
        settings = AdapterSettings(
            hash="b2b8ccb4",
            scale=0.5,
            enabled=False,
        )

        assert settings.hash == "b2b8ccb4"
        assert settings.scale == 0.5
        assert settings.enabled is False

    def test_adapter_settings_missing_hash_fails(self) -> None:
        """Test that hash is required."""
        with pytest.raises(ValidationError) as exc_info:
            AdapterSettings(scale=1.0)  # type: ignore

        errors = exc_info.value.errors()
        assert any(e["loc"] == ("hash",) for e in errors)

    def test_adapter_settings_to_dict(self) -> None:
        """Test model_dump() for passing to GenerationParams."""
        settings = AdapterSettings(hash="test123", scale=0.8)
        settings_dict = settings.model_dump()

        assert settings_dict == {
            "hash": "test123",
            "scale": 0.8,
            "enabled": True,
        }


class TestChatCompletionRequestWithAdapter:
    """Test ChatCompletionRequest with adapter_settings."""

    def test_request_without_adapter_settings(self) -> None:
        """Test that adapter_settings is optional."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
        )

        assert request.adapter_settings is None

    def test_request_with_adapter_settings(self) -> None:
        """Test request with adapter_settings specified."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            adapter_settings=AdapterSettings(hash="abc123", scale=0.7),
        )

        assert request.adapter_settings is not None
        assert request.adapter_settings.hash == "abc123"
        assert request.adapter_settings.scale == 0.7
        assert request.adapter_settings.enabled is True

    def test_request_from_dict(self) -> None:
        """Test parsing request from dict (simulates JSON payload)."""
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "adapter_settings": {
                "hash": "b2b8ccb4",
                "scale": 1.0,
                "enabled": True,
            },
        }

        request = ChatCompletionRequest(**payload)

        assert request.adapter_settings is not None
        assert request.adapter_settings.hash == "b2b8ccb4"

    def test_request_with_disabled_adapter(self) -> None:
        """Test request with adapter explicitly disabled."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            adapter_settings=AdapterSettings(hash="abc123", enabled=False),
        )

        assert request.adapter_settings is not None
        assert request.adapter_settings.enabled is False


class TestGenerationParamsWithAdapter:
    """Test GenerationParams with adapter_settings."""

    def test_generation_params_without_adapter(self) -> None:
        """Test default GenerationParams has no adapter_settings."""
        params = GenerationParams()

        assert params.adapter_settings is None

    def test_generation_params_with_adapter_dict(self) -> None:
        """Test GenerationParams with adapter_settings dict."""
        adapter_dict = {"hash": "abc123", "scale": 0.8, "enabled": True}
        params = GenerationParams(
            temperature=0.7,
            adapter_settings=adapter_dict,
        )

        assert params.adapter_settings is not None
        assert params.adapter_settings["hash"] == "abc123"
        assert params.adapter_settings["scale"] == 0.8

    def test_adapter_settings_flow_from_request(self) -> None:
        """Test the full flow: Request -> model_dump -> GenerationParams."""
        # Simulate incoming request
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            adapter_settings=AdapterSettings(hash="test456", scale=0.5),
        )

        # Simulate chat_runner.py logic
        adapter_settings_dict = None
        if request.adapter_settings and request.adapter_settings.enabled:
            adapter_settings_dict = request.adapter_settings.model_dump()

        params = GenerationParams(
            temperature=request.temperature or 0.7,
            adapter_settings=adapter_settings_dict,
        )

        # Verify flow
        assert params.adapter_settings is not None
        assert params.adapter_settings["hash"] == "test456"
        assert params.adapter_settings["scale"] == 0.5

    def test_disabled_adapter_not_passed(self) -> None:
        """Test that disabled adapter is not passed to GenerationParams."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[Message(role="user", content="Hello")],
            adapter_settings=AdapterSettings(hash="test789", enabled=False),
        )

        # Simulate chat_runner.py logic
        adapter_settings_dict = None
        if request.adapter_settings and request.adapter_settings.enabled:
            adapter_settings_dict = request.adapter_settings.model_dump()

        params = GenerationParams(
            temperature=0.7,
            adapter_settings=adapter_settings_dict,
        )

        # Disabled adapter should not be passed
        assert params.adapter_settings is None


class TestAdapterSettingsEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_hash_is_valid(self) -> None:
        """Test that empty string hash is technically valid (may fail later)."""
        # Pydantic allows empty string - validation of hash format
        # should happen at adapter resolution time
        settings = AdapterSettings(hash="")
        assert settings.hash == ""

    def test_scale_boundaries(self) -> None:
        """Test scale at boundary values."""
        # Zero scale (effectively disables adapter)
        settings_zero = AdapterSettings(hash="test", scale=0.0)
        assert settings_zero.scale == 0.0

        # Negative scale (unusual but may be valid for some use cases)
        settings_neg = AdapterSettings(hash="test", scale=-0.5)
        assert settings_neg.scale == -0.5

        # Large scale
        settings_large = AdapterSettings(hash="test", scale=10.0)
        assert settings_large.scale == 10.0

    def test_extra_fields_allowed(self) -> None:
        """Test that extra fields are allowed (OpenAI compatibility)."""
        payload = {
            "hash": "test",
            "scale": 1.0,
            "custom_field": "should be allowed",
        }

        settings = AdapterSettings(**payload)
        assert settings.hash == "test"
        # Extra fields are allowed due to APIModel config


if __name__ == "__main__":
    """Run tests manually."""
    print("Running AdapterSettings tests...")

    # Model tests
    test_model = TestAdapterSettingsModel()
    test_model.test_valid_adapter_settings_minimal()
    print("  test_valid_adapter_settings_minimal")
    test_model.test_valid_adapter_settings_full()
    print("  test_valid_adapter_settings_full")
    test_model.test_adapter_settings_to_dict()
    print("  test_adapter_settings_to_dict")

    # Request tests
    test_request = TestChatCompletionRequestWithAdapter()
    test_request.test_request_without_adapter_settings()
    print("  test_request_without_adapter_settings")
    test_request.test_request_with_adapter_settings()
    print("  test_request_with_adapter_settings")
    test_request.test_request_from_dict()
    print("  test_request_from_dict")
    test_request.test_request_with_disabled_adapter()
    print("  test_request_with_disabled_adapter")

    # GenerationParams tests
    test_params = TestGenerationParamsWithAdapter()
    test_params.test_generation_params_without_adapter()
    print("  test_generation_params_without_adapter")
    test_params.test_generation_params_with_adapter_dict()
    print("  test_generation_params_with_adapter_dict")
    test_params.test_adapter_settings_flow_from_request()
    print("  test_adapter_settings_flow_from_request")
    test_params.test_disabled_adapter_not_passed()
    print("  test_disabled_adapter_not_passed")

    print("\nAll tests passed!")
