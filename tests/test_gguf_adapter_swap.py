"""
Test llama.cpp (GGUF) per-request adapter swapping.

=============================================================================
THIS TEST SUITE IS FOR FUTURE USE - LLAMA.CPP ADAPTERS ARE CURRENTLY BROKEN
=============================================================================

Bug: GGML_ASSERT(hash_set.size == hash_set.keys.size) in ggml-backend.c
Status: Open as of January 2025
Tracking: https://github.com/ggerganov/llama.cpp/issues/7742
Related: https://github.com/ggerganov/llama.cpp/issues/4485

The llama.cpp scheduler has a bug that causes assertions to fail when
using LoRA adapters. When this is fixed upstream, these tests can be
enabled to validate per-request adapter swapping.

Expected llama-cpp-python API (when bug is fixed):
    from llama_cpp import Llama

    llm = Llama(model_path="model.gguf")
    llm.load_lora_adapter("adapter.gguf", scale=1.0)
    response = llm.create_chat_completion(...)
    llm.unload_lora_adapter()

Run with:
    pytest tests/test_gguf_adapter_swap.py -v
"""

import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from backends.gguf_backend import (
    GGUFAdapterNotSupportedError,
    _GGUFAdapterManager,
)


# ---------------------------------------------------------------------------
# Unit tests (verify the error handling works correctly)
# ---------------------------------------------------------------------------


class TestGGUFAdapterSwapDisabled:
    """Tests verifying that GGUF adapters are properly disabled."""

    def test_adapter_manager_load_raises_error(self) -> None:
        """Verify load_adapter raises GGUFAdapterNotSupportedError."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        with pytest.raises(GGUFAdapterNotSupportedError) as exc_info:
            manager.load_adapter("/path/to/adapter.gguf")

        assert "scheduler bug" in str(exc_info.value)
        assert "7742" in str(exc_info.value)  # GitHub issue number

    def test_adapter_manager_unload_when_not_loaded_succeeds(self) -> None:
        """Verify unload_adapter succeeds when no adapter is loaded."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        # Should return True when no adapter is loaded
        result = manager.unload_adapter()
        assert result is True

    def test_adapter_manager_get_current_adapter_returns_none(self) -> None:
        """Verify get_current_adapter returns None initially."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        assert manager.get_current_adapter() is None
        assert manager.is_adapter_loaded is False

    def test_gguf_client_rejects_adapter_env_var(self) -> None:
        """Verify GGUFClient raises error if JARVIS_ADAPTER_PATH is set."""
        with patch.dict(os.environ, {"JARVIS_ADAPTER_PATH": "/some/adapter"}):
            with pytest.raises(GGUFAdapterNotSupportedError) as exc_info:
                from backends.gguf_backend import GGUFClient

                # This should raise before model loading
                GGUFClient("dummy-model.gguf", "chatml")

            assert "scheduler" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Future tests (uncomment when llama.cpp bug is fixed)
# ---------------------------------------------------------------------------


class TestGGUFAdapterSwapFuture:
    """
    FUTURE TESTS: Enable when llama.cpp scheduler bug is fixed.

    Track: https://github.com/ggerganov/llama.cpp/issues/7742

    These tests document the expected behavior for per-request adapter
    swapping with llama.cpp once the upstream bug is resolved.
    """

    @pytest.mark.skip(reason="llama.cpp adapter bug not yet fixed")
    def test_load_adapter_success(self) -> None:
        """Test successful adapter loading."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        result = manager.load_adapter("/path/to/adapter.gguf")

        assert result is True
        assert manager.get_current_adapter() == "/path/to/adapter.gguf"
        assert manager.is_adapter_loaded is True
        mock_model.load_lora_adapter.assert_called_once()

    @pytest.mark.skip(reason="llama.cpp adapter bug not yet fixed")
    def test_load_adapter_with_scale(self) -> None:
        """Test adapter loading with custom scale."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        result = manager.load_adapter("/path/to/adapter.gguf", scale=0.5)

        assert result is True
        mock_model.load_lora_adapter.assert_called_with(
            "/path/to/adapter.gguf", scale=0.5
        )

    @pytest.mark.skip(reason="llama.cpp adapter bug not yet fixed")
    def test_unload_adapter_success(self) -> None:
        """Test successful adapter unloading."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        # Manually set state as if adapter was loaded
        manager._adapter_loaded = True
        manager._current_adapter_path = "/path/to/adapter.gguf"

        result = manager.unload_adapter()

        assert result is True
        assert manager.get_current_adapter() is None
        assert manager.is_adapter_loaded is False
        mock_model.unload_lora_adapter.assert_called_once()

    @pytest.mark.skip(reason="llama.cpp adapter bug not yet fixed")
    def test_swap_adapters(self) -> None:
        """Test swapping between different adapters."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        # Load adapter A
        manager.load_adapter("/path/to/adapter-a.gguf")
        assert manager.get_current_adapter() == "/path/to/adapter-a.gguf"

        # Swap to adapter B
        manager.load_adapter("/path/to/adapter-b.gguf")
        assert manager.get_current_adapter() == "/path/to/adapter-b.gguf"

        # Should have unloaded A before loading B
        mock_model.unload_lora_adapter.assert_called_once()
        assert mock_model.load_lora_adapter.call_count == 2

    @pytest.mark.skip(reason="llama.cpp adapter bug not yet fixed")
    def test_same_adapter_is_noop(self) -> None:
        """Test that loading same adapter twice is a no-op."""
        mock_model = MagicMock()
        manager = _GGUFAdapterManager(mock_model)

        # Load adapter
        manager.load_adapter("/path/to/adapter.gguf")
        assert mock_model.load_lora_adapter.call_count == 1

        # Load same adapter again - should be no-op
        manager.load_adapter("/path/to/adapter.gguf")
        assert mock_model.load_lora_adapter.call_count == 1  # Not called again


# ---------------------------------------------------------------------------
# Documentation of expected integration flow
# ---------------------------------------------------------------------------


def document_expected_gguf_adapter_flow() -> None:
    """
    Document the expected per-request adapter flow for GGUF/llama.cpp.

    THIS IS DOCUMENTATION ONLY - the actual implementation is blocked by
    the llama.cpp scheduler bug.

    Expected flow when bug is fixed:

    1. Initialize GGUFClient without adapter (base model):
        ```python
        client = GGUFClient("model.gguf", "chatml")
        ```

    2. For each request with adapter_settings:
        ```python
        if request.adapter_settings and request.adapter_settings.hash:
            adapter_path = resolve_adapter_from_hash(request.adapter_settings.hash)
            client.adapter_manager.load_adapter(adapter_path)
        else:
            client.adapter_manager.unload_adapter()
        ```

    3. Generate response:
        ```python
        response = client.chat(messages, temperature)
        ```

    4. Adapter stays loaded for next request (if same adapter needed)
       or gets swapped if different adapter requested.

    Key differences from vLLM:
    - vLLM: LoRARequest passed per-request to generate()
    - llama.cpp: Adapter loaded/unloaded between requests via methods

    Key differences from MLX:
    - MLX: load_adapters() modifies model weights in-place
    - llama.cpp: load_lora_adapter() applies adapter separately

    Performance considerations:
    - Adapter load/unload has latency overhead (~100-500ms typical)
    - Consider caching strategy for frequently-used adapters
    - May need adapter preloading for latency-sensitive paths
    """
    pass


# ---------------------------------------------------------------------------
# Manual testing entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Manual test runner for GGUF adapter swapping.

    Usage:
        python tests/test_gguf_adapter_swap.py

    Since llama.cpp adapters are broken, this will only run the
    disabled/error-checking tests.
    """
    print("=" * 70)
    print("GGUF/llama.cpp Adapter Tests")
    print("=" * 70)
    print()
    print("NOTE: llama.cpp LoRA adapters are currently DISABLED due to")
    print("upstream scheduler bug in llama.cpp.")
    print()
    print("Bug tracking:")
    print("  https://github.com/ggerganov/llama.cpp/issues/7742")
    print("  https://github.com/ggerganov/llama.cpp/issues/4485")
    print()
    print("Running disabled/error-handling tests...")
    print()

    test_instance = TestGGUFAdapterSwapDisabled()

    test_instance.test_adapter_manager_load_raises_error()
    print("  test_adapter_manager_load_raises_error")

    test_instance.test_adapter_manager_unload_when_not_loaded_succeeds()
    print("  test_adapter_manager_unload_when_not_loaded_succeeds")

    test_instance.test_adapter_manager_get_current_adapter_returns_none()
    print("  test_adapter_manager_get_current_adapter_returns_none")

    print()
    print("All disabled/error-handling tests passed!")
    print()
    print("When llama.cpp bug is fixed, enable the TestGGUFAdapterSwapFuture")
    print("tests to validate full adapter swapping functionality.")
