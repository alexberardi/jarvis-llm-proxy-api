"""
Test GGUF LoRA adapter support via constructor-based loading.

llama-cpp-python supports lora_path/lora_scale at Llama() construction time.
Adapter switching = model reload. These tests validate:
- _resolve_gguf_adapter() finds GGUF adapter files in cached directories
- _reload_with_adapter() destroys and recreates the model
- generate_text_chat() triggers reload on adapter hash change
- generate_text_chat() skips reload for same adapter
- _detect_adapter_format() correctly identifies GGUF in zip files

Run with:
    pytest tests/test_gguf_adapter_swap.py -v
"""

import sys
import threading
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.adapter_training import _detect_adapter_format


# ---------------------------------------------------------------------------
# Mock llama_cpp module so we can import backends.gguf_backend without it
# ---------------------------------------------------------------------------

_mock_llama_module = MagicMock()
_mock_llama_class = MagicMock()
_mock_llama_module.Llama = _mock_llama_class


@pytest.fixture(autouse=True)
def _mock_llama_cpp():
    """Inject a mock llama_cpp module into sys.modules for all tests."""
    with patch.dict(sys.modules, {"llama_cpp": _mock_llama_module}):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gguf_client_mock():
    """Create a GGUFClient-like mock with the attributes the adapter methods need."""
    from backends.gguf_backend import GGUFClient

    # We can't instantiate GGUFClient without a real model, so we
    # manually construct an object with the right attributes set.
    client = object.__new__(GGUFClient)
    client.model = MagicMock()
    client.model_path = "test-model.gguf"
    client.model_name = "test-model.gguf"
    client.inference_engine = "llama_cpp"
    client._lock = threading.Lock()
    client._current_adapter_hash = None
    client._current_adapter_path = None
    client._current_adapter_scale = 1.0
    client._llama_init_kwargs = {
        "model_path": "test-model.gguf",
        "n_threads": 4,
        "n_gpu_layers": -1,
        "verbose": False,
        "seed": 42,
        "n_ctx": 4096,
        "n_batch": 512,
        "n_ubatch": 512,
        "rope_scaling_type": 0,
        "mul_mat_q": True,
        "f16_kv": True,
        "flash_attn": True,
    }
    client.flash_attn = True
    client.max_tokens = 7000
    client.top_p = 0.95
    client.top_k = 40
    client.repeat_penalty = 1.1
    client.mirostat_mode = 0
    client.mirostat_tau = 5.0
    client.mirostat_eta = 0.1
    client.last_usage = None
    client.stop_tokens = []
    return client


def _make_mock_adapter_cache(return_path):
    """Create a mock adapter_cache module."""
    mock_mod = MagicMock()
    mock_mod.get_adapter_path.return_value = return_path
    return mock_mod


# ---------------------------------------------------------------------------
# _resolve_gguf_adapter tests
# ---------------------------------------------------------------------------


class TestResolveGGUFAdapter:
    """Test _resolve_gguf_adapter() resolution logic."""

    def test_finds_gguf_subdir_adapter(self, tmp_path: Path) -> None:
        """Should find gguf/adapter.gguf (preferred path from dual-format training)."""
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        adapter_file = gguf_dir / "adapter.gguf"
        adapter_file.write_bytes(b"fake gguf data")

        client = _make_gguf_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_gguf_adapter("abc123")

        assert result == str(adapter_file)

    def test_falls_back_to_root_gguf(self, tmp_path: Path) -> None:
        """Should fall back to any *.gguf file in the adapter root directory."""
        adapter_file = tmp_path / "my-adapter.gguf"
        adapter_file.write_bytes(b"fake gguf data")

        client = _make_gguf_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_gguf_adapter("abc123")

        assert result == str(adapter_file)

    def test_returns_none_when_no_gguf_file(self, tmp_path: Path) -> None:
        """Should return None when adapter dir has no .gguf files."""
        (tmp_path / "adapter_model.safetensors").write_bytes(b"fake")
        (tmp_path / "adapter_config.json").write_text("{}")

        client = _make_gguf_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_gguf_adapter("abc123")

        assert result is None

    def test_returns_none_when_adapter_not_in_cache(self) -> None:
        """Should return None when adapter hash is not found in cache."""
        client = _make_gguf_client_mock()
        mock_cache = _make_mock_adapter_cache(None)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_gguf_adapter("unknown_hash")

        assert result is None


# ---------------------------------------------------------------------------
# _reload_with_adapter tests
# ---------------------------------------------------------------------------


class TestReloadWithAdapter:
    """Test _reload_with_adapter() model recreation."""

    def test_reload_destroys_and_recreates_model(self) -> None:
        """Should delete the old model and create a new Llama instance."""
        client = _make_gguf_client_mock()

        mock_new_model = MagicMock()
        mock_new_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        MockLlama = MagicMock(return_value=mock_new_model)

        with patch.dict(sys.modules, {"llama_cpp": MagicMock(Llama=MockLlama)}):
            client._reload_with_adapter("/path/to/adapter.gguf", 0.8)

        # New model was created with lora_path and lora_scale
        call_kwargs = MockLlama.call_args[1]
        assert call_kwargs["lora_path"] == "/path/to/adapter.gguf"
        assert call_kwargs["lora_scale"] == 0.8
        assert call_kwargs["model_path"] == "test-model.gguf"

        # New model is set
        assert client.model is mock_new_model

    def test_reload_without_adapter(self) -> None:
        """Should recreate model without lora_path when adapter is None."""
        client = _make_gguf_client_mock()

        mock_new_model = MagicMock()
        mock_new_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        MockLlama = MagicMock(return_value=mock_new_model)

        with patch.dict(sys.modules, {"llama_cpp": MagicMock(Llama=MockLlama)}):
            client._reload_with_adapter(None)

        call_kwargs = MockLlama.call_args[1]
        assert "lora_path" not in call_kwargs
        assert "lora_scale" not in call_kwargs


# ---------------------------------------------------------------------------
# generate_text_chat adapter switching tests
# ---------------------------------------------------------------------------


class TestGenerateTextChatAdapterSwitch:
    """Test that generate_text_chat triggers adapter reload when hash changes."""

    def _make_client_with_mock_model(self):
        """Create a client with a mock model that returns valid responses."""
        client = _make_gguf_client_mock()
        client.model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "test response"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return client

    def test_triggers_reload_on_hash_change(self, tmp_path: Path) -> None:
        """Should reload model when adapter hash changes."""
        from managers.chat_types import NormalizedMessage, TextPart, GenerationParams

        # Set up adapter directory with GGUF file
        gguf_dir = tmp_path / "gguf"
        gguf_dir.mkdir()
        (gguf_dir / "adapter.gguf").write_bytes(b"fake")

        client = self._make_client_with_mock_model()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        params = GenerationParams(
            temperature=0.7,
            adapter_settings={"hash": "new_hash_123", "scale": 0.9, "enabled": True},
        )
        messages = [NormalizedMessage(role="user", content=[TextPart(text="hello")])]

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}), \
             patch.object(client, "_reload_with_adapter") as mock_reload:
            client.generate_text_chat(None, messages, params)

        mock_reload.assert_called_once_with(str(gguf_dir / "adapter.gguf"), 0.9)

    def test_skips_reload_for_same_hash(self) -> None:
        """Should not reload when adapter hash matches current."""
        from managers.chat_types import NormalizedMessage, TextPart, GenerationParams

        client = self._make_client_with_mock_model()
        client._current_adapter_hash = "same_hash"

        params = GenerationParams(
            temperature=0.7,
            adapter_settings={"hash": "same_hash", "scale": 1.0, "enabled": True},
        )
        messages = [NormalizedMessage(role="user", content=[TextPart(text="hello")])]

        with patch.object(client, "_reload_with_adapter") as mock_reload:
            client.generate_text_chat(None, messages, params)

        mock_reload.assert_not_called()

    def test_no_adapter_settings_keeps_current(self) -> None:
        """Should keep current adapter when no adapter_settings in params."""
        from managers.chat_types import NormalizedMessage, TextPart, GenerationParams

        client = self._make_client_with_mock_model()
        client._current_adapter_hash = "existing_hash"
        client._current_adapter_path = "/some/adapter.gguf"

        params = GenerationParams(temperature=0.7)
        messages = [NormalizedMessage(role="user", content=[TextPart(text="hello")])]

        with patch.object(client, "_reload_with_adapter") as mock_reload:
            client.generate_text_chat(None, messages, params)

        mock_reload.assert_not_called()
        assert client._current_adapter_hash == "existing_hash"

    def test_disabled_adapter_keeps_current(self) -> None:
        """Should keep current adapter when adapter_settings.enabled is False."""
        from managers.chat_types import NormalizedMessage, TextPart, GenerationParams

        client = self._make_client_with_mock_model()
        client._current_adapter_hash = "existing_hash"

        params = GenerationParams(
            temperature=0.7,
            adapter_settings={"hash": "new_hash", "scale": 1.0, "enabled": False},
        )
        messages = [NormalizedMessage(role="user", content=[TextPart(text="hello")])]

        with patch.object(client, "_reload_with_adapter") as mock_reload:
            client.generate_text_chat(None, messages, params)

        mock_reload.assert_not_called()


# ---------------------------------------------------------------------------
# _detect_adapter_format tests
# ---------------------------------------------------------------------------


class TestDetectAdapterFormat:
    """Test _detect_adapter_format() zip inspection."""

    def test_detects_gguf_in_zip(self, tmp_path: Path) -> None:
        """Should return 'peft_lora+gguf' when zip contains gguf/*.gguf."""
        zip_path = tmp_path / "adapter.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_config.json", "{}")
            zf.writestr("adapter_model.safetensors", "fake")
            zf.writestr("gguf/adapter.gguf", "fake gguf")

        result = _detect_adapter_format(zip_path)
        assert result == "peft_lora+gguf"

    def test_detects_peft_only(self, tmp_path: Path) -> None:
        """Should return 'peft_lora' when zip has no GGUF files."""
        zip_path = tmp_path / "adapter.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("adapter_config.json", "{}")
            zf.writestr("adapter_model.safetensors", "fake")

        result = _detect_adapter_format(zip_path)
        assert result == "peft_lora"

    def test_handles_bad_zip(self, tmp_path: Path) -> None:
        """Should return 'peft_lora' for corrupted zip files."""
        bad_zip = tmp_path / "bad.zip"
        bad_zip.write_bytes(b"not a zip file")

        result = _detect_adapter_format(bad_zip)
        assert result == "peft_lora"

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        """Should return 'peft_lora' when file doesn't exist."""
        result = _detect_adapter_format(tmp_path / "nonexistent.zip")
        assert result == "peft_lora"


# ---------------------------------------------------------------------------
# load_adapter / remove_adapter / get_current_adapter tests
# ---------------------------------------------------------------------------


class TestAdapterLifecycle:
    """Test the base class adapter interface implementation."""

    def test_load_adapter_sets_state(self) -> None:
        """load_adapter should set adapter path and reload model."""
        client = _make_gguf_client_mock()
        mock_new_model = MagicMock()
        mock_new_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        MockLlama = MagicMock(return_value=mock_new_model)

        with patch.dict(sys.modules, {"llama_cpp": MagicMock(Llama=MockLlama)}):
            client.load_adapter("/path/to/adapter.gguf", scale=0.5)

        assert client.get_current_adapter() == "/path/to/adapter.gguf"
        assert client._current_adapter_scale == 0.5

    def test_remove_adapter_clears_state(self) -> None:
        """remove_adapter should clear adapter state and reload base model."""
        client = _make_gguf_client_mock()
        client._current_adapter_hash = "some_hash"
        client._current_adapter_path = "/path/to/adapter.gguf"
        client._current_adapter_scale = 0.5

        mock_new_model = MagicMock()
        mock_new_model.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "hi"}}]
        }
        MockLlama = MagicMock(return_value=mock_new_model)

        with patch.dict(sys.modules, {"llama_cpp": MagicMock(Llama=MockLlama)}):
            client.remove_adapter()

        assert client.get_current_adapter() is None
        assert client._current_adapter_hash is None
        assert client._current_adapter_scale == 1.0

    def test_remove_adapter_noop_when_no_adapter(self) -> None:
        """remove_adapter should be a no-op when no adapter is loaded."""
        client = _make_gguf_client_mock()
        # Should not raise or call _reload_with_adapter
        client.remove_adapter()
        assert client.get_current_adapter() is None
