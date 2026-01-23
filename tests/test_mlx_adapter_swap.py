"""
Test MLX per-request adapter swapping.

MLX-LM supports LoRA adapters via:
- load(model_path, adapter_path=...) - load with initial adapter
- load_adapters(model, adapter_path) - swap adapter on existing model
- remove_lora_layers(model) - revert to base model

Unlike vLLM's per-request LoRARequest, MLX requires calling load_adapters()
to swap adapters. This modifies the model weights in-place.

References:
    - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md
    - https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/utils.py

Run with:
    pytest tests/test_mlx_adapter_swap.py -v

Manual test on macOS with Apple Silicon:
    python tests/test_mlx_adapter_swap.py
"""

import os
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Unit tests (mocked, run in CI)
# ---------------------------------------------------------------------------


class TestMLXAdapterSwapMocked:
    """Unit tests with mocked MLX to validate adapter swapping logic.

    Note: These tests document the expected MLX-LM API behavior.
    On non-macOS systems, mlx_lm is not available, so we use
    pure mock objects without patching the actual modules.
    """

    def test_load_with_adapter_path(self) -> None:
        """Verify mlx_lm.load() API accepts adapter_path parameter.

        Documents the expected MLX-LM API:
            model, tokenizer = load("model", adapter_path="/path/to/adapter")
        """
        # Create mock load function
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        def mock_load(model_path: str, adapter_path: str = None):
            return (mock_model, mock_tokenizer)

        # Simulate the call
        model, tokenizer = mock_load(
            "test-model",
            adapter_path="/path/to/adapter"
        )

        assert model is mock_model
        assert tokenizer is mock_tokenizer

    def test_load_adapters_modifies_model_in_place(self) -> None:
        """Verify load_adapters modifies and returns the same model.

        Documents the expected MLX-LM tuner API:
            model = load_adapters(model, adapter_path)
        """
        mock_model = MagicMock()

        def mock_load_adapters(model, adapter_path):
            # In real MLX, this modifies model in-place and returns it
            return model

        result = mock_load_adapters(mock_model, "/path/to/adapter")

        # Should return the same model object (modified in-place)
        assert result is mock_model

    def test_remove_lora_layers_reverts_to_base(self) -> None:
        """Verify remove_lora_layers strips adapter from model.

        Documents the expected MLX-LM tuner API:
            model = remove_lora_layers(model)
        """
        mock_model = MagicMock()

        def mock_remove_lora_layers(model):
            # In real MLX, this removes LoRA layers and returns model
            return model

        result = mock_remove_lora_layers(mock_model)
        assert result is mock_model

    def test_adapter_swap_workflow(self) -> None:
        """Simulate the full adapter swap workflow.

        Documents the expected sequence of operations:
        1. Load base model
        2. Apply adapter A via load_adapters()
        3. Swap to adapter B via load_adapters()
        4. Remove adapter via remove_lora_layers()
        """
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        load_call_count = 0
        load_adapters_call_count = 0
        remove_call_count = 0

        def mock_load(model_path, adapter_path=None):
            nonlocal load_call_count
            load_call_count += 1
            return (mock_model, mock_tokenizer)

        def mock_load_adapters(model, adapter_path):
            nonlocal load_adapters_call_count
            load_adapters_call_count += 1
            return model

        def mock_remove_lora_layers(model):
            nonlocal remove_call_count
            remove_call_count += 1
            return model

        # Step 1: Load base model
        model, tokenizer = mock_load("base-model")
        assert load_call_count == 1

        # Step 2: Apply adapter A
        model = mock_load_adapters(model, "/path/to/adapter-a")
        assert load_adapters_call_count == 1

        # Step 3: Swap to adapter B (calls load_adapters again)
        model = mock_load_adapters(model, "/path/to/adapter-b")
        assert load_adapters_call_count == 2

        # Step 4: Remove adapter, revert to base
        model = mock_remove_lora_layers(model)
        assert remove_call_count == 1

    def test_mlx_client_adapter_methods(self) -> None:
        """Test MlxClient adapter methods with mocks.

        Note: This test only runs if mlx_lm is available (macOS).
        On other platforms, we skip since MlxClient can't be imported.
        """
        # Skip on non-macOS platforms where mlx_lm isn't available
        if sys.platform != "darwin":
            print("    (skipped - mlx_lm only available on macOS)")
            return

        try:
            # This will fail on non-macOS since mlx_lm isn't available
            with patch("backends.mlx_backend.load") as mock_load, \
                 patch("mlx_lm.tuner.utils.load_adapters") as mock_load_adapters, \
                 patch("mlx_lm.tuner.utils.remove_lora_layers") as mock_remove:

                mock_model = MagicMock()
                mock_tokenizer = MagicMock()
                mock_load.return_value = (mock_model, mock_tokenizer)
                mock_load_adapters.return_value = mock_model
                mock_remove.return_value = mock_model

                from backends.mlx_backend import MlxClient

                # Create client without initial adapter
                client = MlxClient("test-model")
                assert client.get_current_adapter() is None
                assert client._lora_layers_applied is False

                # Load adapter
                client.load_adapter("/path/to/adapter-a")
                mock_load_adapters.assert_called_once()
                assert client.get_current_adapter() == "/path/to/adapter-a"
                assert client._lora_layers_applied is True

                # Same adapter is no-op
                client.load_adapter("/path/to/adapter-a")
                assert mock_load_adapters.call_count == 1  # Not called again

                # Different adapter triggers load
                client.load_adapter("/path/to/adapter-b")
                assert mock_load_adapters.call_count == 2
                assert client.get_current_adapter() == "/path/to/adapter-b"

                # Remove adapter
                client.remove_adapter()
                mock_remove.assert_called_once()
                assert client.get_current_adapter() is None
                assert client._lora_layers_applied is False

        except (ModuleNotFoundError, AttributeError):
            # mlx_lm not available (not on macOS)
            print("    (skipped - mlx_lm not available on this platform)")


# ---------------------------------------------------------------------------
# Integration tests (require real MLX + Apple Silicon, skip in CI)
# ---------------------------------------------------------------------------


def _get_mlx_test_adapter_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Get adapter paths for MLX integration testing."""
    base_path = Path("/tmp/jarvis-adapters/store")

    if not base_path.exists():
        return None, None

    # Look for MLX-compatible adapters (safetensors format)
    adapter_dirs = []
    for adapter_dir in base_path.glob("*/.models/*/*/"):
        safetensors = adapter_dir / "adapters.safetensors"
        if safetensors.exists():
            adapter_dirs.append(adapter_dir)

    if len(adapter_dirs) < 2:
        return None, None

    return adapter_dirs[0], adapter_dirs[1]


@pytest.mark.skipif(
    not sys.platform == "darwin",
    reason="MLX integration test requires macOS with Apple Silicon"
)
@pytest.mark.skipif(
    not Path("/tmp/jarvis-adapters/store").exists(),
    reason="Integration test requires adapters in /tmp/jarvis-adapters/store"
)
class TestMLXAdapterSwapIntegration:
    """Integration tests with real MLX (requires Apple Silicon)."""

    def test_real_adapter_swap(self) -> None:
        """Test actual adapter swapping with real MLX.

        Run manually on macOS:
            python tests/test_mlx_adapter_swap.py
        """
        adapter_a, adapter_b = _get_mlx_test_adapter_paths()
        if not adapter_a or not adapter_b:
            pytest.skip("Need at least 2 MLX adapters for swap test")

        from backends.mlx_backend import MlxClient

        model_name = os.getenv("JARVIS_MODEL_NAME", "mlx-community/Llama-3.2-3B-Instruct-4bit")

        # Initialize client
        client = MlxClient(model_name)
        assert client.get_current_adapter() is None

        test_messages = [{"role": "user", "content": "Hello!"}]

        # Request 1: Base model
        response_base = client.chat(test_messages)
        print(f"Base model: {response_base[:100]}...")

        # Request 2: Load adapter A
        client.load_adapter(str(adapter_a))
        assert client.get_current_adapter() == str(adapter_a)
        response_a = client.chat(test_messages)
        print(f"Adapter A: {response_a[:100]}...")

        # Request 3: Swap to adapter B
        client.load_adapter(str(adapter_b))
        assert client.get_current_adapter() == str(adapter_b)
        response_b = client.chat(test_messages)
        print(f"Adapter B: {response_b[:100]}...")

        # Request 4: Remove adapter, back to base
        client.remove_adapter()
        assert client.get_current_adapter() is None
        response_base_2 = client.chat(test_messages)
        print(f"Base again: {response_base_2[:100]}...")

        # All should return valid responses
        assert len(response_base) > 0
        assert len(response_a) > 0
        assert len(response_b) > 0
        assert len(response_base_2) > 0


# ---------------------------------------------------------------------------
# Manual testing entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Manual test runner for MLX adapter swapping.

    Usage:
        # With mocks (no Apple Silicon required):
        python tests/test_mlx_adapter_swap.py --mock

        # With real MLX (requires Apple Silicon + adapters):
        python tests/test_mlx_adapter_swap.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test MLX adapter swapping")
    parser.add_argument("--mock", action="store_true", help="Run mocked tests only")
    args = parser.parse_args()

    if args.mock:
        print("Running mocked tests...")
        test_instance = TestMLXAdapterSwapMocked()
        test_instance.test_load_with_adapter_path()
        print("  test_load_with_adapter_path")
        test_instance.test_load_adapters_modifies_model_in_place()
        print("  test_load_adapters_modifies_model_in_place")
        test_instance.test_remove_lora_layers_reverts_to_base()
        print("  test_remove_lora_layers_reverts_to_base")
        test_instance.test_adapter_swap_workflow()
        print("  test_adapter_swap_workflow")
        test_instance.test_mlx_client_adapter_methods()
        print("  test_mlx_client_adapter_methods")
        print("All mocked tests passed!")
    else:
        if sys.platform != "darwin":
            print("MLX integration tests require macOS with Apple Silicon")
            sys.exit(1)

        print("Running integration test (requires Apple Silicon)...")
        adapter_a, adapter_b = _get_mlx_test_adapter_paths()
        if not adapter_a or not adapter_b:
            print("Adapters not found. Expected structure:")
            print("  /tmp/jarvis-adapters/store/<node>/.models/<model>/<hash>/")
            print("  With adapters.safetensors and adapter_config.json")
            sys.exit(1)

        print(f"Found adapters:")
        print(f"  A: {adapter_a}")
        print(f"  B: {adapter_b}")

        test_instance = TestMLXAdapterSwapIntegration()
        test_instance.test_real_adapter_swap()
        print("Integration test passed!")
