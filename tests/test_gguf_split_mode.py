"""GGUFClient split_mode resolution: the -1 AUTO sentinel vs explicit values.

Regression for the dual-RTX-3090 prod outage: defaulting split_mode to 0
(SPLIT_MODE_NONE) silently confined BOTH models to GPU0 -> cudaMalloc OOM at
boot while GPU1 sat idle. The new default is -1 = AUTO: layer-split when >=2
CUDA GPUs are visible, single-GPU otherwise. Explicit DB/env values (0/1/2)
must pass through untouched and must NOT consult the topology probe.

No GPU / no DB touched: llama_cpp.Llama is mocked to capture ctor kwargs
(gguf_backend does `from llama_cpp import Llama` inside _init_llama_cpp, so
patching the llama_cpp module attribute works), the settings getters are
monkeypatched in the backends.gguf_backend namespace, and PowerMetrics is
stubbed out (its real __init__ shells out to `sudo -n powermetrics`).
"""

from unittest.mock import patch

import pytest

import backends.gguf_backend as gguf_backend


class _DummyPowerMetrics:
    """No subprocess probe, no background thread."""

    def start_monitoring(self) -> None:
        pass

    def stop_monitoring(self) -> None:
        pass


def _capture_llama_kwargs(monkeypatch, settings: dict[str, object]) -> dict[str, object]:
    """Instantiate GGUFClient with everything stubbed; return the Llama ctor kwargs."""
    monkeypatch.setattr(gguf_backend, "PowerMetrics", _DummyPowerMetrics)

    def fake_get_setting(key, env_fallback, default, value_type="string"):
        return settings.get(key, default)

    monkeypatch.setattr(gguf_backend, "get_setting", fake_get_setting)
    monkeypatch.setattr(
        gguf_backend, "get_int_setting", lambda k, e, d: int(settings.get(k, d))
    )
    monkeypatch.setattr(
        gguf_backend, "get_float_setting", lambda k, e, d: float(settings.get(k, d))
    )
    monkeypatch.setattr(
        gguf_backend, "get_bool_setting", lambda k, e, d: bool(settings.get(k, d))
    )

    with patch("llama_cpp.Llama") as mock_llama:
        gguf_backend.GGUFClient("dummy-model.gguf", "chatml")
    assert mock_llama.call_count == 1
    return mock_llama.call_args.kwargs


def test_auto_sentinel_two_gpus_resolves_to_layer(monkeypatch):
    # DB/env resolve to -1 (the new default) + 2 CUDA GPUs -> LAYER split.
    monkeypatch.setattr(gguf_backend, "auto_gguf_split_mode", lambda: 1)
    kwargs = _capture_llama_kwargs(
        monkeypatch, {"inference.gguf.split_mode": -1}
    )
    assert kwargs["split_mode"] == 1


def test_auto_sentinel_single_gpu_resolves_to_none(monkeypatch):
    monkeypatch.setattr(gguf_backend, "auto_gguf_split_mode", lambda: 0)
    kwargs = _capture_llama_kwargs(
        monkeypatch, {"inference.gguf.split_mode": -1}
    )
    assert kwargs["split_mode"] == 0


@pytest.mark.parametrize("explicit", [0, 1, 2])
def test_explicit_split_mode_passes_through_untouched(monkeypatch, explicit):
    def _must_not_probe() -> int:
        raise AssertionError("auto probe must not run for an explicit split_mode")

    monkeypatch.setattr(gguf_backend, "auto_gguf_split_mode", _must_not_probe)
    kwargs = _capture_llama_kwargs(
        monkeypatch, {"inference.gguf.split_mode": explicit}
    )
    assert kwargs["split_mode"] == explicit


def test_tensor_split_parsing_unaffected(monkeypatch):
    monkeypatch.setattr(gguf_backend, "auto_gguf_split_mode", lambda: 1)
    kwargs = _capture_llama_kwargs(
        monkeypatch,
        {
            "inference.gguf.split_mode": -1,
            "inference.gguf.tensor_split": "0.5,0.5",
        },
    )
    assert kwargs["split_mode"] == 1
    assert kwargs["tensor_split"] == [0.5, 0.5]
