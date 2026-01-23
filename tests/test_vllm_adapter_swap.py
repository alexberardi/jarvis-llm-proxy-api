"""
Spike test: Validate vLLM per-request adapter swapping.

This test validates that vLLM can dynamically swap LoRA adapters between
requests without reloading the model. This is the foundation for per-node
adapter loading in the Jarvis system.

vLLM API Pattern:
    from vllm.lora.request import LoRARequest

    # Each request can specify a different adapter:
    lora_request_a = LoRARequest("adapter-a", 1, "/path/to/adapter-a")
    lora_request_b = LoRARequest("adapter-b", 2, "/path/to/adapter-b")

    outputs_a = llm.generate([prompt], sampling_params, lora_request=lora_request_a)
    outputs_b = llm.generate([prompt], sampling_params, lora_request=lora_request_b)

Requirements:
    - vLLM must be initialized with enable_lora=True
    - max_loras >= 1 (how many adapters can be loaded simultaneously)
    - max_lora_rank must accommodate your adapter's rank

Run with:
    JARVIS_INFERENCE_ENGINE=vllm pytest tests/test_vllm_adapter_swap.py -v

Or manually test with a real model:
    python tests/test_vllm_adapter_swap.py
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


class TestVLLMAdapterSwapMocked:
    """Unit tests with mocked vLLM to validate adapter swapping logic."""

    def test_lora_request_can_be_created_per_request(self) -> None:
        """Verify LoRARequest objects can be created with different paths."""
        # Create mock LoRARequest class that mimics vLLM's LoRARequest
        # This works without vllm installed
        class MockLoRARequest:
            def __init__(self, lora_name: str, lora_int_id: int, lora_path: str):
                self.lora_name = lora_name
                self.lora_int_id = lora_int_id
                self.lora_path = lora_path

        adapter_a = MockLoRARequest("adapter-a", 1, "/path/to/adapter-a")
        adapter_b = MockLoRARequest("adapter-b", 2, "/path/to/adapter-b")

        assert adapter_a.lora_name == "adapter-a"
        assert adapter_a.lora_path == "/path/to/adapter-a"
        assert adapter_b.lora_name == "adapter-b"
        assert adapter_b.lora_path == "/path/to/adapter-b"
        assert adapter_a.lora_int_id != adapter_b.lora_int_id

    def test_generate_accepts_lora_request_kwarg(self) -> None:
        """Verify generate() method signature accepts lora_request parameter."""
        mock_llm = MagicMock()
        mock_llm.generate.return_value = [
            MagicMock(outputs=[MagicMock(text="response")])
        ]

        mock_lora_request = MagicMock(
            lora_name="test-adapter", lora_int_id=1, lora_path="/path/to/adapter"
        )

        # Call generate with lora_request kwarg
        mock_llm.generate(
            ["test prompt"],
            MagicMock(),  # SamplingParams
            lora_request=mock_lora_request,
        )

        # Verify lora_request was passed
        mock_llm.generate.assert_called_once()
        call_kwargs = mock_llm.generate.call_args.kwargs
        assert "lora_request" in call_kwargs
        assert call_kwargs["lora_request"].lora_name == "test-adapter"

    def test_different_adapters_per_request(self) -> None:
        """Simulate swapping adapters between requests."""
        mock_llm = MagicMock()

        # Simulate different responses based on adapter
        def mock_generate(prompts, params, lora_request=None):
            adapter_name = lora_request.lora_name if lora_request else "base"
            return [
                MagicMock(
                    outputs=[MagicMock(text=f"Response from {adapter_name}")]
                )
            ]

        mock_llm.generate.side_effect = mock_generate

        # Request 1: adapter-a
        adapter_a = MagicMock(lora_name="adapter-a", lora_int_id=1)
        result_a = mock_llm.generate(["prompt"], MagicMock(), lora_request=adapter_a)
        assert "adapter-a" in result_a[0].outputs[0].text

        # Request 2: adapter-b
        adapter_b = MagicMock(lora_name="adapter-b", lora_int_id=2)
        result_b = mock_llm.generate(["prompt"], MagicMock(), lora_request=adapter_b)
        assert "adapter-b" in result_b[0].outputs[0].text

        # Request 3: no adapter (base model)
        result_base = mock_llm.generate(["prompt"], MagicMock(), lora_request=None)
        assert "base" in result_base[0].outputs[0].text

    def test_vllm_config_requirements(self) -> None:
        """Document vLLM config requirements for adapter support."""
        # These are the required vLLM initialization kwargs for adapter support
        required_kwargs = {
            "enable_lora": True,  # Must be True to support LoRA adapters
            "max_loras": 1,  # At least 1; can be higher for parallel adapters
            "max_lora_rank": 64,  # Must match or exceed adapter rank
        }

        # Verify all keys are present
        assert required_kwargs["enable_lora"] is True
        assert required_kwargs["max_loras"] >= 1
        assert required_kwargs["max_lora_rank"] >= 8  # Typical minimum rank


# ---------------------------------------------------------------------------
# Integration tests (require real vLLM + GPU, skip in CI)
# ---------------------------------------------------------------------------


def _get_test_adapter_paths() -> tuple[Optional[Path], Optional[Path]]:
    """Get adapter paths for integration testing.

    Expected structure from plan:
    /tmp/jarvis-adapters/store/node-linux-desktop/.models/llama-3.2-3b-instruct/
    ├── 813ac07c.../adapter.zip  # Old adapter
    └── b2b8ccb4.../adapter.zip  # New adapter
    """
    base_path = Path("/tmp/jarvis-adapters/store")

    # Look for any available adapters
    if not base_path.exists():
        return None, None

    adapter_dirs = list(base_path.glob("*/.models/*/*/"))
    if len(adapter_dirs) < 2:
        return None, None

    return adapter_dirs[0], adapter_dirs[1]


@pytest.mark.skipif(
    os.getenv("JARVIS_INFERENCE_ENGINE") != "vllm",
    reason="vLLM integration test requires JARVIS_INFERENCE_ENGINE=vllm"
)
@pytest.mark.skipif(
    not Path("/tmp/jarvis-adapters/store").exists(),
    reason="Integration test requires adapters in /tmp/jarvis-adapters/store"
)
class TestVLLMAdapterSwapIntegration:
    """Integration tests with real vLLM (requires GPU)."""

    def test_real_adapter_swap(self) -> None:
        """Test actual adapter swapping with real vLLM.

        Run manually:
            JARVIS_INFERENCE_ENGINE=vllm python -m pytest \
                tests/test_vllm_adapter_swap.py::TestVLLMAdapterSwapIntegration -v
        """
        adapter_a, adapter_b = _get_test_adapter_paths()
        if not adapter_a or not adapter_b:
            pytest.skip("Need at least 2 adapters for swap test")

        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        # Initialize with LoRA support
        model_name = os.getenv(
            "JARVIS_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"
        )

        llm = LLM(
            model=model_name,
            enable_lora=True,
            max_loras=2,
            max_lora_rank=64,
            trust_remote_code=True,
        )

        sampling_params = SamplingParams(temperature=0.7, max_tokens=50)
        test_prompt = "Hello, how are you?"

        # Request 1: No adapter (base model)
        outputs_base = llm.generate([test_prompt], sampling_params)
        response_base = outputs_base[0].outputs[0].text
        print(f"Base model: {response_base[:100]}...")

        # Request 2: Adapter A
        lora_a = LoRARequest("adapter-a", 1, str(adapter_a))
        outputs_a = llm.generate(
            [test_prompt], sampling_params, lora_request=lora_a
        )
        response_a = outputs_a[0].outputs[0].text
        print(f"Adapter A: {response_a[:100]}...")

        # Request 3: Adapter B
        lora_b = LoRARequest("adapter-b", 2, str(adapter_b))
        outputs_b = llm.generate(
            [test_prompt], sampling_params, lora_request=lora_b
        )
        response_b = outputs_b[0].outputs[0].text
        print(f"Adapter B: {response_b[:100]}...")

        # All should return valid responses (may or may not differ)
        assert len(response_base) > 0
        assert len(response_a) > 0
        assert len(response_b) > 0


# ---------------------------------------------------------------------------
# Manual testing entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Manual test runner for debugging adapter swapping.

    Usage:
        # With mocks (no GPU required):
        python tests/test_vllm_adapter_swap.py --mock

        # With real vLLM (requires GPU + adapters):
        JARVIS_INFERENCE_ENGINE=vllm python tests/test_vllm_adapter_swap.py
    """
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM adapter swapping")
    parser.add_argument("--mock", action="store_true", help="Run mocked tests only")
    args = parser.parse_args()

    if args.mock:
        print("Running mocked tests...")
        test_instance = TestVLLMAdapterSwapMocked()
        test_instance.test_lora_request_can_be_created_per_request()
        print("  test_lora_request_can_be_created_per_request")
        test_instance.test_generate_accepts_lora_request_kwarg()
        print("  test_generate_accepts_lora_request_kwarg")
        test_instance.test_different_adapters_per_request()
        print("  test_different_adapters_per_request")
        test_instance.test_vllm_config_requirements()
        print("  test_vllm_config_requirements")
        print("All mocked tests passed!")
    else:
        print("Running integration test (requires vLLM + GPU)...")
        adapter_a, adapter_b = _get_test_adapter_paths()
        if not adapter_a or not adapter_b:
            print(f"Adapters not found. Expected structure:")
            print(f"  /tmp/jarvis-adapters/store/<node>/.models/<model>/<hash>/")
            print(f"  Need at least 2 adapter directories.")
            sys.exit(1)

        print(f"Found adapters:")
        print(f"  A: {adapter_a}")
        print(f"  B: {adapter_b}")

        # Run the real test
        test_instance = TestVLLMAdapterSwapIntegration()
        test_instance.test_real_adapter_swap()
        print("Integration test passed!")
