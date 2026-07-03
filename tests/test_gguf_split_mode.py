"""GGUFClient split_mode resolution: the -1 AUTO sentinel vs explicit values.

Regression for the dual-RTX-3090 prod outage: defaulting split_mode to 0
(SPLIT_MODE_NONE) silently confined BOTH models to GPU0 -> cudaMalloc OOM at
boot while GPU1 sat idle. The new default is -1 = AUTO: layer-split when the
CUDA topology is split-worthy, single-GPU otherwise. Explicit DB/env values
(0/1/2) must pass through untouched and must NOT consult the topology probe,
and the bare default path (no DB row, no env) must resolve to the sentinel.

CI-safe: llama-cpp-python is NOT installed in the Tests workflow, and
backends.gguf_backend imports it transitively (backends.chat_formats does
`from llama_cpp import llama_types` at module load). So — same pattern as
test_gguf_adapter_swap.py — an autouse fixture stubs sys.modules['llama_cpp']
(+ submodules) and the gguf_backend import is deferred until the stub is
active. The stubbed Llama captures ctor kwargs. The settings getters are
monkeypatched in the backends.gguf_backend namespace, and PowerMetrics is
stubbed out (its real __init__ shells out to `sudo -n powermetrics`).
"""

import sys
from typing import Callable
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock llama_cpp module so we can import backends.gguf_backend without it
# ---------------------------------------------------------------------------
# backends.chat_formats does `from llama_cpp.llama_chat_format import ...`
# at module load, so we also need to stub the submodules — patching only
# sys.modules["llama_cpp"] with a MagicMock leaves it not-a-package and the
# submodule import fails.

_mock_llama_module = MagicMock()
_mock_llama_module.Llama = MagicMock()

_mock_llama_chat_format = MagicMock()
_mock_llama_types = MagicMock()


@pytest.fixture(autouse=True)
def _mock_llama_cpp():
    """Inject mock llama_cpp + submodules into sys.modules for all tests."""
    with patch.dict(
        sys.modules,
        {
            "llama_cpp": _mock_llama_module,
            "llama_cpp.llama_chat_format": _mock_llama_chat_format,
            "llama_cpp.llama_types": _mock_llama_types,
        },
    ):
        yield


class _DummyPowerMetrics:
    """No subprocess probe, no background thread."""

    def start_monitoring(self) -> None:
        pass

    def stop_monitoring(self) -> None:
        pass


def _capture_llama_kwargs(
    monkeypatch,
    settings: dict[str, object],
    auto: Callable[[], int] | None = None,
) -> dict[str, object]:
    """Instantiate GGUFClient with everything stubbed; return the Llama ctor kwargs.

    Imported here — not at module scope — so collection succeeds without
    llama-cpp-python installed (the autouse stub is active by the time we run).
    """
    import backends.gguf_backend as gguf_backend

    monkeypatch.setattr(gguf_backend, "PowerMetrics", _DummyPowerMetrics)
    if auto is not None:
        monkeypatch.setattr(gguf_backend, "auto_gguf_split_mode", auto)

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

    # _init_llama_cpp does `from llama_cpp import Llama` at call time, which
    # resolves through the stubbed sys.modules entry — a fresh mock per call
    # keeps call_count isolated between tests.
    mock_llama = MagicMock()
    monkeypatch.setattr(_mock_llama_module, "Llama", mock_llama)
    gguf_backend.GGUFClient("dummy-model.gguf", "chatml")
    assert mock_llama.call_count == 1
    return mock_llama.call_args.kwargs


def test_auto_sentinel_two_gpus_resolves_to_layer(monkeypatch):
    # DB/env resolve to -1 (the new default) + split-worthy topology -> LAYER.
    kwargs = _capture_llama_kwargs(
        monkeypatch, {"inference.gguf.split_mode": -1}, auto=lambda: 1
    )
    assert kwargs["split_mode"] == 1


def test_auto_sentinel_single_gpu_resolves_to_none(monkeypatch):
    kwargs = _capture_llama_kwargs(
        monkeypatch, {"inference.gguf.split_mode": -1}, auto=lambda: 0
    )
    assert kwargs["split_mode"] == 0


def test_default_split_mode_is_auto_sentinel(monkeypatch):
    # NO split_mode in DB/env at all (empty settings dict): the call-site
    # default must be -1 = AUTO, i.e. the probe result — a distinctive marker
    # here — reaches the Llama ctor. Pins the default against a regression back
    # to 0, which would silently kill the dual-3090 fix while every
    # explicit-value test stayed green.
    kwargs = _capture_llama_kwargs(monkeypatch, {}, auto=lambda: 2)
    assert kwargs["split_mode"] == 2


def test_settings_definition_default_is_auto_sentinel():
    # The DB seed default must match the call-site default: -1 = AUTO.
    from services.settings_service import SETTINGS_DEFINITIONS

    definition = next(
        d for d in SETTINGS_DEFINITIONS if d.key == "inference.gguf.split_mode"
    )
    assert definition.default == -1


@pytest.mark.parametrize("explicit", [0, 1, 2])
def test_explicit_split_mode_passes_through_untouched(monkeypatch, explicit):
    def _must_not_probe() -> int:
        raise AssertionError("auto probe must not run for an explicit split_mode")

    kwargs = _capture_llama_kwargs(
        monkeypatch, {"inference.gguf.split_mode": explicit}, auto=_must_not_probe
    )
    assert kwargs["split_mode"] == explicit


def test_tensor_split_parsing_unaffected(monkeypatch):
    kwargs = _capture_llama_kwargs(
        monkeypatch,
        {
            "inference.gguf.split_mode": -1,
            "inference.gguf.tensor_split": "0.5,0.5",
        },
        auto=lambda: 1,
    )
    assert kwargs["split_mode"] == 1
    assert kwargs["tensor_split"] == [0.5, 0.5]
