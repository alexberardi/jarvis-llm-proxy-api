"""Tests for the SettingsService.

These tests cover:
- Type coercion
- Caching behavior
- Environment variable fallback
- Bulk operations
"""

import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Import the functions and classes we're testing
from services.settings_service import (
    SettingsService,
    SettingDefinition,
    SETTINGS_DEFINITIONS,
    _coerce_value,
    _serialize_value,
    get_settings_service,
)


class TestCoerceValue:
    """Tests for the _coerce_value function."""

    def test_string_type(self):
        assert _coerce_value("hello", "string", "") == "hello"
        assert _coerce_value("", "string", "default") == "default"
        assert _coerce_value(None, "string", "default") == "default"

    def test_int_type(self):
        assert _coerce_value("42", "int", 0) == 42
        assert _coerce_value("-10", "int", 0) == -10
        assert _coerce_value("", "int", 99) == 99
        assert _coerce_value(None, "int", 99) == 99
        assert _coerce_value("not_a_number", "int", 99) == 99

    def test_float_type(self):
        assert _coerce_value("3.14", "float", 0.0) == 3.14
        assert _coerce_value("-2.5", "float", 0.0) == -2.5
        assert _coerce_value("", "float", 1.0) == 1.0
        assert _coerce_value(None, "float", 1.0) == 1.0
        assert _coerce_value("not_a_number", "float", 1.0) == 1.0

    def test_bool_type(self):
        # True values
        assert _coerce_value("true", "bool", False) is True
        assert _coerce_value("True", "bool", False) is True
        assert _coerce_value("TRUE", "bool", False) is True
        assert _coerce_value("1", "bool", False) is True
        assert _coerce_value("yes", "bool", False) is True
        assert _coerce_value("on", "bool", False) is True

        # False values
        assert _coerce_value("false", "bool", True) is False
        assert _coerce_value("0", "bool", True) is False
        assert _coerce_value("no", "bool", True) is False
        assert _coerce_value("off", "bool", True) is False
        assert _coerce_value("random", "bool", True) is False  # Not a truthy value

        # Empty/None
        assert _coerce_value("", "bool", True) is True  # Returns default
        assert _coerce_value(None, "bool", True) is True

    def test_json_type(self):
        assert _coerce_value('{"key": "value"}', "json", {}) == {"key": "value"}
        assert _coerce_value("[1, 2, 3]", "json", []) == [1, 2, 3]
        assert _coerce_value("null", "json", {}) is None
        assert _coerce_value("invalid json", "json", {"default": True}) == {"default": True}
        assert _coerce_value("", "json", {"default": True}) == {"default": True}


class TestSerializeValue:
    """Tests for the _serialize_value function."""

    def test_string(self):
        assert _serialize_value("hello", "string") == "hello"
        assert _serialize_value("", "string") == ""

    def test_int(self):
        assert _serialize_value(42, "int") == "42"
        assert _serialize_value(-10, "int") == "-10"

    def test_float(self):
        assert _serialize_value(3.14, "float") == "3.14"

    def test_bool(self):
        assert _serialize_value(True, "bool") == "true"
        assert _serialize_value(False, "bool") == "false"

    def test_json(self):
        assert _serialize_value({"key": "value"}, "json") == '{"key": "value"}'
        assert _serialize_value([1, 2, 3], "json") == "[1, 2, 3]"

    def test_none(self):
        assert _serialize_value(None, "string") is None
        assert _serialize_value(None, "int") is None


class TestSettingsServiceCache:
    """Tests for SettingsService caching behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        SettingsService._instance = None

    def test_cache_hit(self):
        """Test that cached values are returned without DB query."""
        service = SettingsService()

        # Manually populate cache
        from services.settings_service import SettingValue

        service._cache["model.main.name"] = SettingValue(
            value="cached_value",
            value_type="string",
            requires_reload=True,
            is_secret=False,
            env_fallback="JARVIS_MODEL_NAME",
            from_db=True,
            cached_at=time.time(),
        )

        # Should return cached value without DB query
        result = service.get("model.main.name")
        assert result == "cached_value"

    def test_cache_expiry(self):
        """Test that expired cache entries are not used."""
        service = SettingsService()

        # Populate cache with expired entry
        from services.settings_service import SettingValue

        service._cache["model.main.name"] = SettingValue(
            value="expired_value",
            value_type="string",
            requires_reload=True,
            is_secret=False,
            env_fallback="JARVIS_MODEL_NAME",
            from_db=True,
            cached_at=time.time() - 120,  # 2 minutes ago (expired)
        )

        # Should fall through to env/default since cache is expired
        with patch.dict(os.environ, {"JARVIS_MODEL_NAME": "env_value"}):
            result = service.get("model.main.name")
            # Will get from env since DB is not available in test
            assert result == "env_value"

    def test_invalidate_single_key(self):
        """Test invalidating a single cache key."""
        service = SettingsService()

        from services.settings_service import SettingValue

        service._cache["key1"] = SettingValue(
            value="value1",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )
        service._cache["key2"] = SettingValue(
            value="value2",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )

        service.invalidate_cache("key1")

        assert "key1" not in service._cache
        assert "key2" in service._cache

    def test_invalidate_all(self):
        """Test invalidating entire cache."""
        service = SettingsService()

        from services.settings_service import SettingValue

        service._cache["key1"] = SettingValue(
            value="value1",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )
        service._cache["key2"] = SettingValue(
            value="value2",
            value_type="string",
            requires_reload=False,
            is_secret=False,
            env_fallback=None,
            from_db=True,
            cached_at=time.time(),
        )

        service.invalidate_cache()

        assert len(service._cache) == 0


class TestSettingsServiceEnvFallback:
    """Tests for environment variable fallback."""

    def setup_method(self):
        """Reset singleton before each test."""
        SettingsService._instance = None

    def test_env_fallback_when_db_unavailable(self):
        """Test that env vars are used when DB is unavailable."""
        service = SettingsService()

        with patch.dict(os.environ, {"JARVIS_MODEL_NAME": "env_model_path"}):
            result = service.get("model.main.name")
            assert result == "env_model_path"

    def test_default_when_no_env(self):
        """Test that defaults are used when no env var is set."""
        service = SettingsService()

        # Clear any env var
        with patch.dict(os.environ, {}, clear=True):
            result = service.get("model.main.context_window")
            # Should return definition default (8192)
            assert result == 8192

    def test_unknown_key_returns_none(self):
        """Test that unknown keys return None."""
        service = SettingsService()

        result = service.get("unknown.key")
        assert result is None

    def test_unknown_key_returns_provided_default(self):
        """Test that unknown keys return provided default."""
        service = SettingsService()

        result = service.get("unknown.key", "my_default")
        assert result == "my_default"


class TestSettingsServiceTypedGetters:
    """Tests for typed getter methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        SettingsService._instance = None

    def test_get_int(self):
        """Test get_int method."""
        service = SettingsService()

        with patch.dict(os.environ, {"JARVIS_MODEL_CONTEXT_WINDOW": "4096"}):
            result = service.get_int("model.main.context_window", 0)
            assert result == 4096
            assert isinstance(result, int)

    def test_get_float(self):
        """Test get_float method."""
        service = SettingsService()

        with patch.dict(os.environ, {"JARVIS_VLLM_GPU_MEMORY_UTILIZATION": "0.85"}):
            result = service.get_float("inference.vllm.gpu_memory_utilization", 0.0)
            assert result == 0.85
            assert isinstance(result, float)

    def test_get_bool(self):
        """Test get_bool method."""
        service = SettingsService()

        with patch.dict(os.environ, {"JARVIS_FLASH_ATTN": "true"}):
            result = service.get_bool("inference.gguf.flash_attn", False)
            assert result is True

    def test_get_str(self):
        """Test get_str method."""
        service = SettingsService()

        with patch.dict(os.environ, {"JARVIS_MODEL_NAME": "test_model"}):
            result = service.get_str("model.main.name", "")
            assert result == "test_model"
            assert isinstance(result, str)


class TestSettingsServiceListMethods:
    """Tests for listing methods."""

    def setup_method(self):
        """Reset singleton before each test."""
        SettingsService._instance = None

    def test_list_categories(self):
        """Test list_categories returns unique categories."""
        service = SettingsService()

        categories = service.list_categories()

        assert isinstance(categories, list)
        assert len(categories) > 0
        assert "model.main" in categories
        assert "inference.vllm" in categories
        assert "training" in categories
        # Should be sorted
        assert categories == sorted(categories)

    def test_list_all(self):
        """Test list_all returns all settings."""
        service = SettingsService()

        settings = service.list_all()

        assert isinstance(settings, list)
        assert len(settings) == len(SETTINGS_DEFINITIONS)

        # Check structure of first setting
        first = settings[0]
        assert "key" in first
        assert "value" in first
        assert "value_type" in first
        assert "category" in first
        assert "from_db" in first

    def test_list_all_with_category_filter(self):
        """Test list_all with category filter."""
        service = SettingsService()

        settings = service.list_all(category="model.main")

        assert all(s["category"] == "model.main" for s in settings)
        assert len(settings) > 0


class TestSettingsServiceModelConfig:
    """Tests for get_model_config and get_inference_config."""

    def setup_method(self):
        """Reset singleton before each test."""
        SettingsService._instance = None

    def test_get_model_config(self):
        """Test get_model_config returns expected keys."""
        service = SettingsService()

        with patch.dict(os.environ, {
            "JARVIS_MODEL_NAME": "test_model",
            "JARVIS_MODEL_BACKEND": "VLLM",
            "JARVIS_MODEL_CONTEXT_WINDOW": "4096",
        }):
            config = service.get_model_config("main")

            assert "name" in config
            assert "backend" in config
            assert "context_window" in config
            assert config["name"] == "test_model"
            assert config["backend"] == "VLLM"
            assert config["context_window"] == 4096

    def test_get_inference_config(self):
        """Test get_inference_config returns expected keys."""
        service = SettingsService()

        with patch.dict(os.environ, {
            "JARVIS_VLLM_GPU_MEMORY_UTILIZATION": "0.85",
            "JARVIS_VLLM_TENSOR_PARALLEL_SIZE": "2",
        }):
            config = service.get_inference_config("vllm")

            assert "gpu_memory_utilization" in config
            assert "tensor_parallel_size" in config
            assert config["gpu_memory_utilization"] == 0.85
            assert config["tensor_parallel_size"] == 2


class TestSettingsDefinitions:
    """Tests for settings definitions."""

    def test_all_definitions_have_required_fields(self):
        """Test that all definitions have required fields."""
        for definition in SETTINGS_DEFINITIONS:
            assert definition.key, f"Missing key for definition"
            assert definition.category, f"Missing category for {definition.key}"
            assert definition.value_type in ("string", "int", "float", "bool", "json"), \
                f"Invalid value_type for {definition.key}: {definition.value_type}"
            # env_fallback can be None

    def test_no_duplicate_keys(self):
        """Test that there are no duplicate keys."""
        keys = [d.key for d in SETTINGS_DEFINITIONS]
        assert len(keys) == len(set(keys)), "Duplicate keys found in SETTINGS_DEFINITIONS"

    def test_key_format(self):
        """Test that keys follow the expected format."""
        for definition in SETTINGS_DEFINITIONS:
            # Keys should be lowercase with dots
            assert "." in definition.key, f"Key should contain dots: {definition.key}"
            assert definition.key == definition.key.lower(), \
                f"Key should be lowercase: {definition.key}"


class TestSingleton:
    """Tests for singleton behavior."""

    def setup_method(self):
        """Reset singleton before each test."""
        SettingsService._instance = None

    def test_singleton_instance(self):
        """Test that get_settings_service returns same instance."""
        service1 = get_settings_service()
        service2 = get_settings_service()

        assert service1 is service2

    def test_direct_construction_returns_same_instance(self):
        """Test that direct construction also returns same instance."""
        service1 = SettingsService()
        service2 = SettingsService()

        assert service1 is service2
