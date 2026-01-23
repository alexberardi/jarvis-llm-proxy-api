"""
Tests for vLLM dynamic per-request adapter loading (Phase 1e).

These tests mock vLLM to verify the adapter resolution and LoRARequest creation
logic without requiring actual GPU resources.

Run with:
    pytest tests/test_vllm_dynamic_adapter.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestVLLMDynamicAdapterLoading:
    """Tests for per-request adapter loading in vLLM backend."""

    def test_generate_text_chat_with_adapter_settings(self) -> None:
        """Test that adapter_settings triggers LoRARequest creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test123"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("backends.vllm_backend.adapter_cache") as mock_cache:
                mock_cache.get_adapter_path.return_value = adapter_path

                with patch("backends.vllm_backend.LLM"):
                    from backends.vllm_backend import VLLMClient
                    from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

                    with patch.object(VLLMClient, "__init__", lambda self, *args, **kwargs: None):
                        client = VLLMClient.__new__(VLLMClient)
                        client._lora_enabled = True
                        client.model = MagicMock()
                        client.chat_format = "llama3"
                        client.stop_tokens = []

                        captured_lora = {}

                        def mock_generate(*args, **kwargs):
                            captured_lora["request"] = kwargs.get("lora_request")
                            return "Generated text", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

                        client.generate = mock_generate

                        params = GenerationParams(
                            temperature=0.7,
                            adapter_settings={"hash": "test123", "scale": 1.0, "enabled": True},
                        )

                        messages = [
                            NormalizedMessage(role="user", content=[TextPart(text="Hello")])
                        ]

                        result = client.generate_text_chat(None, messages, params)

                        mock_cache.get_adapter_path.assert_called_once_with("test123")
                        assert captured_lora["request"] is not None
                        assert "test123" in captured_lora["request"].lora_name
                        assert str(adapter_path) == captured_lora["request"].lora_path

    def test_generate_text_chat_without_adapter_settings(self) -> None:
        """Test that no LoRARequest is created when adapter_settings is None."""
        with patch("backends.vllm_backend.adapter_cache") as mock_cache:
            with patch("backends.vllm_backend.LLM"):
                from backends.vllm_backend import VLLMClient
                from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

                with patch.object(VLLMClient, "__init__", lambda self, *args, **kwargs: None):
                    client = VLLMClient.__new__(VLLMClient)
                    client._lora_enabled = True
                    client.model = MagicMock()
                    client.chat_format = "llama3"
                    client.stop_tokens = []

                    captured_lora = {}

                    def mock_generate(*args, **kwargs):
                        captured_lora["request"] = kwargs.get("lora_request")
                        return "Generated text", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

                    client.generate = mock_generate

                    params = GenerationParams(temperature=0.7)
                    messages = [NormalizedMessage(role="user", content=[TextPart(text="Hello")])]

                    result = client.generate_text_chat(None, messages, params)

                    mock_cache.get_adapter_path.assert_not_called()
                    assert captured_lora["request"] is None

    def test_generate_text_chat_with_disabled_adapter(self) -> None:
        """Test that disabled adapter_settings doesn't create LoRARequest."""
        with patch("backends.vllm_backend.adapter_cache") as mock_cache:
            with patch("backends.vllm_backend.LLM"):
                from backends.vllm_backend import VLLMClient
                from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

                with patch.object(VLLMClient, "__init__", lambda self, *args, **kwargs: None):
                    client = VLLMClient.__new__(VLLMClient)
                    client._lora_enabled = True
                    client.model = MagicMock()
                    client.chat_format = "llama3"
                    client.stop_tokens = []

                    captured_lora = {}

                    def mock_generate(*args, **kwargs):
                        captured_lora["request"] = kwargs.get("lora_request")
                        return "Generated text", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

                    client.generate = mock_generate

                    params = GenerationParams(
                        temperature=0.7,
                        adapter_settings={"hash": "test123", "enabled": False},
                    )
                    messages = [NormalizedMessage(role="user", content=[TextPart(text="Hello")])]

                    result = client.generate_text_chat(None, messages, params)

                    mock_cache.get_adapter_path.assert_not_called()
                    assert captured_lora["request"] is None

    def test_generate_text_chat_with_missing_adapter(self) -> None:
        """Test graceful handling when adapter is not found."""
        with patch("backends.vllm_backend.adapter_cache") as mock_cache:
            mock_cache.get_adapter_path.return_value = None

            with patch("backends.vllm_backend.LLM"):
                from backends.vllm_backend import VLLMClient
                from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

                with patch.object(VLLMClient, "__init__", lambda self, *args, **kwargs: None):
                    client = VLLMClient.__new__(VLLMClient)
                    client._lora_enabled = True
                    client.model = MagicMock()
                    client.chat_format = "llama3"
                    client.stop_tokens = []

                    captured_lora = {}

                    def mock_generate(*args, **kwargs):
                        captured_lora["request"] = kwargs.get("lora_request")
                        return "Generated text", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

                    client.generate = mock_generate

                    params = GenerationParams(
                        temperature=0.7,
                        adapter_settings={"hash": "nonexistent", "enabled": True},
                    )
                    messages = [NormalizedMessage(role="user", content=[TextPart(text="Hello")])]

                    result = client.generate_text_chat(None, messages, params)

                    mock_cache.get_adapter_path.assert_called_once_with("nonexistent")
                    assert captured_lora["request"] is None

    def test_generate_text_chat_lora_disabled(self) -> None:
        """Test that adapter_settings is ignored when LoRA is disabled."""
        with patch("backends.vllm_backend.adapter_cache") as mock_cache:
            with patch("backends.vllm_backend.LLM"):
                from backends.vllm_backend import VLLMClient
                from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

                with patch.object(VLLMClient, "__init__", lambda self, *args, **kwargs: None):
                    client = VLLMClient.__new__(VLLMClient)
                    client._lora_enabled = False  # LoRA disabled
                    client.model = MagicMock()
                    client.chat_format = "llama3"
                    client.stop_tokens = []

                    captured_lora = {}

                    def mock_generate(*args, **kwargs):
                        captured_lora["request"] = kwargs.get("lora_request")
                        return "Generated text", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

                    client.generate = mock_generate

                    params = GenerationParams(
                        temperature=0.7,
                        adapter_settings={"hash": "test123", "enabled": True},
                    )
                    messages = [NormalizedMessage(role="user", content=[TextPart(text="Hello")])]

                    result = client.generate_text_chat(None, messages, params)

                    mock_cache.get_adapter_path.assert_not_called()
                    assert captured_lora["request"] is None

    def test_generate_text_chat_with_adapter_scale(self) -> None:
        """Test that adapter scale is recognized (though not yet used by vLLM)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "scaled"
            adapter_path.mkdir()
            (adapter_path / "adapter_config.json").write_text("{}")

            with patch("backends.vllm_backend.adapter_cache") as mock_cache:
                mock_cache.get_adapter_path.return_value = adapter_path

                with patch("backends.vllm_backend.LLM"):
                    from backends.vllm_backend import VLLMClient
                    from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

                    with patch.object(VLLMClient, "__init__", lambda self, *args, **kwargs: None):
                        client = VLLMClient.__new__(VLLMClient)
                        client._lora_enabled = True
                        client.model = MagicMock()
                        client.chat_format = "llama3"
                        client.stop_tokens = []

                        captured_lora = {}

                        def mock_generate(*args, **kwargs):
                            captured_lora["request"] = kwargs.get("lora_request")
                            return "Generated text", {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

                        client.generate = mock_generate

                        # Include scale parameter
                        params = GenerationParams(
                            temperature=0.7,
                            adapter_settings={"hash": "scaled123", "scale": 0.5, "enabled": True},
                        )
                        messages = [NormalizedMessage(role="user", content=[TextPart(text="Hello")])]

                        result = client.generate_text_chat(None, messages, params)

                        mock_cache.get_adapter_path.assert_called_once_with("scaled123")
                        assert captured_lora["request"] is not None


if __name__ == "__main__":
    print("Running vLLM dynamic adapter tests...")

    test = TestVLLMDynamicAdapterLoading()

    test.test_generate_text_chat_with_adapter_settings()
    print("  ✓ test_generate_text_chat_with_adapter_settings")

    test.test_generate_text_chat_without_adapter_settings()
    print("  ✓ test_generate_text_chat_without_adapter_settings")

    test.test_generate_text_chat_with_disabled_adapter()
    print("  ✓ test_generate_text_chat_with_disabled_adapter")

    test.test_generate_text_chat_with_missing_adapter()
    print("  ✓ test_generate_text_chat_with_missing_adapter")

    test.test_generate_text_chat_lora_disabled()
    print("  ✓ test_generate_text_chat_lora_disabled")

    test.test_generate_text_chat_with_adapter_scale()
    print("  ✓ test_generate_text_chat_with_adapter_scale")

    print("\nAll tests passed!")
