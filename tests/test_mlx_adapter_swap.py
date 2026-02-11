"""
Test MLX LoRA adapter support via in-place adapter loading.

mlx-lm supports dynamic adapter swapping via load_adapters() and
removal via remove_lora_layers(). These tests validate:
- _resolve_mlx_adapter() finds PEFT adapter files in cached directories
- _handle_adapter_switch() triggers load on hash change
- _handle_adapter_switch() skips load for same adapter
- generate_text_chat() integrates adapter switching
- remove_adapter() clears hash tracking

Run with:
    pytest tests/test_mlx_adapter_swap.py -v
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock mlx_lm modules so we can import backends.mlx_backend without them
# ---------------------------------------------------------------------------

_mock_mlx_lm = MagicMock()
_mock_generate_mod = MagicMock()
_mock_sample_utils = MagicMock()
_mock_utils = MagicMock()
_mock_tuner_utils = MagicMock()
_mock_pil = MagicMock()

_mock_mlx_lm.generate = _mock_generate_mod
_mock_mlx_lm.sample_utils = _mock_sample_utils
_mock_mlx_lm.utils = _mock_utils


@pytest.fixture(autouse=True)
def _mock_mlx_modules():
    """Inject mock mlx_lm and PIL modules for all tests."""
    with patch.dict(sys.modules, {
        "mlx_lm": _mock_mlx_lm,
        "mlx_lm.generate": _mock_generate_mod,
        "mlx_lm.sample_utils": _mock_sample_utils,
        "mlx_lm.utils": _mock_utils,
        "mlx_lm.tuner": MagicMock(),
        "mlx_lm.tuner.utils": _mock_tuner_utils,
        "PIL": _mock_pil,
        "PIL.Image": MagicMock(),
    }):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mlx_client_mock():
    """Create an MlxClient-like mock with the attributes the adapter methods need."""
    from backends.mlx_backend import MlxClient

    client = object.__new__(MlxClient)
    client.model = MagicMock()
    client.tokenizer = MagicMock()
    client.model_name = "test-model"
    client.model_path = "test-model"
    client.inference_engine = "mlx"
    client._current_adapter_path = None
    client._current_adapter_hash = None
    client._lora_layers_applied = False
    client.last_usage = None
    return client


def _make_mock_adapter_cache(return_path):
    """Create a mock adapter_cache module."""
    mock_mod = MagicMock()
    mock_mod.get_adapter_path.return_value = return_path
    return mock_mod


# ---------------------------------------------------------------------------
# _resolve_mlx_adapter tests
# ---------------------------------------------------------------------------


class TestResolveMlxAdapter:
    """Test _resolve_mlx_adapter() resolution logic."""

    def test_finds_mlx_style_adapter(self, tmp_path: Path) -> None:
        """Should find adapters.safetensors + adapter_config.json (MLX naming)."""
        (tmp_path / "adapters.safetensors").write_bytes(b"fake weights")
        (tmp_path / "adapter_config.json").write_text("{}")

        client = _make_mlx_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_mlx_adapter("abc123")

        assert result == str(tmp_path)

    def test_finds_hf_style_adapter(self, tmp_path: Path) -> None:
        """Should find adapter_model.safetensors + adapter_config.json (HF naming)."""
        (tmp_path / "adapter_model.safetensors").write_bytes(b"fake weights")
        (tmp_path / "adapter_config.json").write_text("{}")

        client = _make_mlx_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_mlx_adapter("abc123")

        assert result == str(tmp_path)

    def test_returns_none_when_missing_config(self, tmp_path: Path) -> None:
        """Should return None when adapter_config.json is missing."""
        (tmp_path / "adapters.safetensors").write_bytes(b"fake weights")

        client = _make_mlx_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_mlx_adapter("abc123")

        assert result is None

    def test_returns_none_when_missing_weights(self, tmp_path: Path) -> None:
        """Should return None when safetensors files are missing."""
        (tmp_path / "adapter_config.json").write_text("{}")

        client = _make_mlx_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_mlx_adapter("abc123")

        assert result is None

    def test_returns_none_when_not_in_cache(self) -> None:
        """Should return None when adapter hash is not found in cache."""
        client = _make_mlx_client_mock()
        mock_cache = _make_mock_adapter_cache(None)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}):
            result = client._resolve_mlx_adapter("unknown_hash")

        assert result is None


# ---------------------------------------------------------------------------
# _handle_adapter_switch tests
# ---------------------------------------------------------------------------


class TestHandleAdapterSwitch:
    """Test _handle_adapter_switch() logic."""

    def test_loads_adapter_on_new_hash(self, tmp_path: Path) -> None:
        """Should call load_adapter when hash changes."""
        (tmp_path / "adapters.safetensors").write_bytes(b"fake")
        (tmp_path / "adapter_config.json").write_text("{}")

        client = _make_mlx_client_mock()
        mock_cache = _make_mock_adapter_cache(tmp_path)

        with patch.dict(sys.modules, {"services.adapter_cache": mock_cache}), \
             patch.object(client, "load_adapter") as mock_load:
            client._handle_adapter_switch({"hash": "new_hash", "scale": 1.0, "enabled": True})

        mock_load.assert_called_once_with(str(tmp_path))
        assert client._current_adapter_hash == "new_hash"

    def test_skips_for_same_hash(self) -> None:
        """Should not call load_adapter when hash matches current."""
        client = _make_mlx_client_mock()
        client._current_adapter_hash = "same_hash"

        with patch.object(client, "load_adapter") as mock_load:
            client._handle_adapter_switch({"hash": "same_hash", "scale": 1.0, "enabled": True})

        mock_load.assert_not_called()

    def test_skips_when_disabled(self) -> None:
        """Should not switch adapter when enabled is False."""
        client = _make_mlx_client_mock()

        with patch.object(client, "load_adapter") as mock_load:
            client._handle_adapter_switch({"hash": "new_hash", "scale": 1.0, "enabled": False})

        mock_load.assert_not_called()
        assert client._current_adapter_hash is None

    def test_skips_when_no_hash(self) -> None:
        """Should keep current adapter when hash is missing."""
        client = _make_mlx_client_mock()
        client._current_adapter_hash = "existing"

        with patch.object(client, "load_adapter") as mock_load:
            client._handle_adapter_switch({"scale": 1.0, "enabled": True})

        mock_load.assert_not_called()
        assert client._current_adapter_hash == "existing"


# ---------------------------------------------------------------------------
# generate_text_chat integration tests
# ---------------------------------------------------------------------------


class TestGenerateTextChatAdapterSwitch:
    """Test that generate_text_chat triggers adapter switch."""

    def _make_client_with_mock_chat(self):
        """Create a client with a mock chat method."""
        client = _make_mlx_client_mock()
        # Mock tokenizer with chat template
        client.tokenizer.chat_template = "fake template"
        client.tokenizer.apply_chat_template.return_value = "formatted prompt"
        # Mock generate to return a response
        _mock_generate_mod.generate.return_value = "test response"
        _mock_sample_utils.make_sampler.return_value = MagicMock()
        return client

    def test_triggers_switch_on_adapter_settings(self) -> None:
        """Should call _handle_adapter_switch when adapter_settings present."""
        import asyncio

        from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

        client = self._make_client_with_mock_chat()

        params = GenerationParams(
            temperature=0.7,
            adapter_settings={"hash": "new_hash", "scale": 1.0, "enabled": True},
        )
        messages = [NormalizedMessage(role="user", content=[TextPart(text="hello")])]

        with patch.object(client, "_handle_adapter_switch") as mock_switch:
            asyncio.get_event_loop().run_until_complete(
                client.generate_text_chat(None, messages, params)
            )

        mock_switch.assert_called_once_with({"hash": "new_hash", "scale": 1.0, "enabled": True})

    def test_no_switch_without_adapter_settings(self) -> None:
        """Should not call _handle_adapter_switch when no adapter_settings."""
        import asyncio

        from managers.chat_types import GenerationParams, NormalizedMessage, TextPart

        client = self._make_client_with_mock_chat()

        params = GenerationParams(temperature=0.7)
        messages = [NormalizedMessage(role="user", content=[TextPart(text="hello")])]

        with patch.object(client, "_handle_adapter_switch") as mock_switch:
            asyncio.get_event_loop().run_until_complete(
                client.generate_text_chat(None, messages, params)
            )

        mock_switch.assert_not_called()


# ---------------------------------------------------------------------------
# remove_adapter clears hash tests
# ---------------------------------------------------------------------------


class TestRemoveAdapterClearsHash:
    """Test that remove_adapter clears _current_adapter_hash."""

    def test_remove_clears_hash(self) -> None:
        """remove_adapter should clear adapter hash state."""
        client = _make_mlx_client_mock()
        client._current_adapter_hash = "some_hash"
        client._current_adapter_path = "/some/path"
        client._lora_layers_applied = True

        _mock_tuner_utils.remove_lora_layers.return_value = client.model

        client.remove_adapter()

        assert client._current_adapter_hash is None
        assert client._current_adapter_path is None
        assert client._lora_layers_applied is False

    def test_remove_noop_when_no_adapter(self) -> None:
        """remove_adapter should be a no-op when no adapter loaded."""
        client = _make_mlx_client_mock()
        client.remove_adapter()
        assert client._current_adapter_hash is None
