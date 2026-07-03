"""Per-slot fault isolation + retry behavior of ModelManager.

Covers the 2026-07 incident class: a failed model load (e.g. the background
32B) must never crash construction — it lands in `model_states` and is
re-attempted by `retry_failed_loads()` with exponential cooldown.
"""

import threading
import time as real_time

import pytest

import managers.model_manager as mm_module
from managers.model_manager import ModelManager

# Settings keys resolved by _initialize_models — MOCK backends load instantly
# with no deps. Distinct paths so live/background do NOT share an instance.
LIVE_ONLY_SETTINGS = {
    "model.live.backend": "MOCK",
    "model.live.name": "mock-live",
}
SPLIT_SETTINGS = {
    **LIVE_ONLY_SETTINGS,
    "model.background.backend": "MOCK",
    "model.background.name": "mock-bg",
}


@pytest.fixture
def reset_singleton():
    """Reset the ModelManager singleton around each test.

    REQUIRED — the class-level `_instance`/`_initialized` flags otherwise
    leak one test's manager (and its patched factories) into the next.
    """
    ModelManager._instance = None
    ModelManager._initialized = False
    yield
    ModelManager._instance = None
    ModelManager._initialized = False


def _patch_settings(monkeypatch, values: dict) -> None:
    """Force deterministic settings in the managers.model_manager namespace."""

    def fake_get_setting(key, env_fallback, default, value_type="string"):
        return values.get(key, default)

    def fake_get_int_setting(key, env_fallback, default):
        return int(values.get(key, default))

    monkeypatch.setattr(mm_module, "get_setting", fake_get_setting)
    monkeypatch.setattr(mm_module, "get_int_setting", fake_get_int_setting)


def _install_factory(
    monkeypatch,
    fail_slots: set[str],
    slow_slots: set[str] | None = None,
    delay_s: float = 0.3,
) -> dict:
    """Patch _create_backend: count calls, fail for model_type in fail_slots.

    `fail_slots` is consulted at call time, so tests can mutate it to make the
    factory "stop raising" for retry scenarios.
    """
    calls = {"count": 0}
    real_create = ModelManager._create_backend

    def factory(
        self,
        backend_type,
        model_path,
        chat_format,
        stop_tokens,
        context_window,
        rest_url,
        model_type,
    ):
        calls["count"] += 1
        if model_type in fail_slots:
            raise RuntimeError(f"boom-{model_type}")
        if slow_slots and model_type in slow_slots:
            real_time.sleep(delay_s)
        return real_create(
            self,
            backend_type=backend_type,
            model_path=model_path,
            chat_format=chat_format,
            stop_tokens=stop_tokens,
            context_window=context_window,
            rest_url=rest_url,
            model_type=model_type,
        )

    monkeypatch.setattr(ModelManager, "_create_backend", factory)
    return calls


class _FakeTime:
    """Stand-in for the `time` module inside managers.model_manager only."""

    def __init__(self, offset_s: float) -> None:
        self._offset = offset_s

    def monotonic(self) -> float:
        return real_time.monotonic() + self._offset

    def time(self) -> float:
        return real_time.time() + self._offset


# ---------------------------------------------------------------------------
# Fault isolation
# ---------------------------------------------------------------------------


def test_background_failure_is_isolated(reset_singleton, monkeypatch):
    """Background load failure must not kill construction or the live slot."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    _install_factory(monkeypatch, fail_slots={"background"})

    manager = ModelManager()  # must NOT raise

    assert manager.live_model is not None
    assert manager.background_model is None
    assert manager.model_states["live"]["status"] == "ready"
    assert manager.model_states["background"]["status"] == "failed"
    assert "boom-background" in manager.model_states["background"]["error"]
    assert manager.model_states["background"]["attempts"] == 1
    assert "mock-live" in manager.registry
    assert "mock-bg" not in manager.registry
    assert manager.aliases.get("live") == "mock-live"


def test_live_failure_is_isolated(reset_singleton, monkeypatch):
    """Live load failure leaves the background slot fully usable."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    _install_factory(monkeypatch, fail_slots={"live"})

    manager = ModelManager()

    assert manager.live_model is None
    assert manager.background_model is not None
    assert manager.model_states["live"]["status"] == "failed"
    assert "boom-live" in manager.model_states["live"]["error"]
    assert manager.model_states["background"]["status"] == "ready"
    assert "mock-bg" in manager.registry
    assert "mock-live" not in manager.registry
    assert manager.aliases.get("background") == "mock-bg"


def test_shared_config_failure_single_attempt(reset_singleton, monkeypatch):
    """Shared live/background config: one load attempt, both slots mirror it."""
    _patch_settings(monkeypatch, LIVE_ONLY_SETTINGS)
    calls = _install_factory(monkeypatch, fail_slots={"live", "background"})

    manager = ModelManager()

    assert calls["count"] == 1  # background mirrors live — no second load
    assert manager.live_model is None
    assert manager.background_model is None
    assert manager.model_states["live"]["status"] == "failed"
    assert manager.model_states["background"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Retry + cooldown
# ---------------------------------------------------------------------------


def test_retry_cooldown_then_success(reset_singleton, monkeypatch):
    """Immediate retry no-ops (cooldown); after backoff elapses it reloads."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    fail_slots = {"background"}
    _install_factory(monkeypatch, fail_slots)

    manager = ModelManager()
    assert manager.model_states["background"]["attempts"] == 1

    # Within the 60s cooldown: no-op, attempts unchanged.
    assert manager.retry_failed_loads() is False
    assert manager.model_states["background"]["attempts"] == 1

    # Move the module's clock past the first backoff window and stop failing.
    fail_slots.clear()
    monkeypatch.setattr(mm_module, "time", _FakeTime(offset_s=120.0))

    assert manager.retry_failed_loads() is True
    assert manager.model_states["background"]["status"] == "ready"
    assert manager.background_model is not None
    assert "mock-bg" in manager.registry
    assert manager.aliases.get("background") == "mock-bg"


def test_retry_force_bypasses_cooldown(reset_singleton, monkeypatch):
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    fail_slots = {"background"}
    _install_factory(monkeypatch, fail_slots)

    manager = ModelManager()
    assert manager.retry_failed_loads() is False  # cooldown active

    fail_slots.clear()
    assert manager.retry_failed_loads(force=True) is True
    assert manager.model_states["background"]["status"] == "ready"


def test_concurrent_retries_do_not_double_load(reset_singleton, monkeypatch):
    """A retry already in flight makes concurrent callers no-op fast."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    fail_slots = {"background"}
    calls = _install_factory(
        monkeypatch, fail_slots, slow_slots={"background"}, delay_s=0.3
    )

    manager = ModelManager()
    initial_calls = calls["count"]
    assert initial_calls == 2  # live ok + background failed

    fail_slots.clear()
    results: list[bool] = []

    def worker() -> None:
        results.append(manager.retry_failed_loads(force=True))

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    real_time.sleep(0.05)  # ensure t2 lands while t1's slow load is in flight
    t2.start()
    t1.join()
    t2.join()

    assert calls["count"] == initial_calls + 1  # exactly ONE retry load
    assert sorted(results) == [False, True]
    assert manager.model_states["background"]["status"] == "ready"


def test_retry_noops_while_load_all_in_flight(reset_singleton, monkeypatch):
    """load_all() and retry_failed_loads() serialize on the same lock — the
    60s retry loop must never run a second full-weight load concurrently with
    an in-flight initial load (double VRAM → native OOM)."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    calls = _install_factory(monkeypatch, set(), slow_slots={"live"}, delay_s=0.3)

    manager = ModelManager(auto_load=False)
    loader = threading.Thread(target=manager.load_all)
    loader.start()
    real_time.sleep(0.05)  # land while the slow live load is in flight

    assert manager.retry_failed_loads(force=True) is False

    loader.join()
    assert calls["count"] == 2  # only load_all's two attempts, no retry load


# ---------------------------------------------------------------------------
# Wedged-loading protection (exceptions outside per-slot isolation)
# ---------------------------------------------------------------------------


def test_settings_preamble_failure_marks_slots_failed_not_loading(
    reset_singleton, monkeypatch
):
    """An exception in the settings-resolution preamble (e.g. a malformed DB
    value) escapes the per-slot fault isolation — the slots must land in
    'failed' (retryable) rather than wedge in 'loading' forever, and
    construction must not raise."""

    def broken_get_setting(key, env_fallback, default, value_type="string"):
        return 12345  # non-string DB value → .upper() AttributeError

    monkeypatch.setattr(mm_module, "get_setting", broken_get_setting)

    manager = ModelManager()  # must NOT raise

    for slot in ("live", "background"):
        state = manager.model_states[slot]
        assert state["status"] == "failed"
        assert "AttributeError" in state["error"]
        assert state["attempts"] == 1  # cooldown applies — no hot retry loop


def test_retry_heals_settings_preamble_failure(reset_singleton, monkeypatch):
    """After a preamble failure there are NO frozen configs — a retry must
    re-run settings resolution (nothing frozen to replay) and recover once
    settings are readable again."""
    values = dict(SPLIT_SETTINGS)
    _patch_settings(monkeypatch, values)
    _install_factory(monkeypatch, set())

    resolve_calls = {"broken": True}
    real_resolve = ModelManager._resolve_slot_configs

    def flaky_resolve(self):
        if resolve_calls["broken"]:
            raise RuntimeError("settings DB unreachable")
        real_resolve(self)

    monkeypatch.setattr(ModelManager, "_resolve_slot_configs", flaky_resolve)

    manager = ModelManager()
    assert manager.model_states["live"]["status"] == "failed"
    assert manager._slot_configs == {}

    resolve_calls["broken"] = False
    assert manager.retry_failed_loads(force=True) is True
    assert manager.model_states["live"]["status"] == "ready"
    assert manager.model_states["background"]["status"] == "ready"
    assert "mock-live" in manager.registry


# ---------------------------------------------------------------------------
# Frozen configs: replay valid ones, re-resolve unloadable ones
# ---------------------------------------------------------------------------


def test_retry_re_resolves_unloadable_frozen_config(reset_singleton, monkeypatch):
    """A frozen config with no model path (settings DB unreachable at resolve
    time, empty env fallback) can NEVER load — the retry must re-resolve
    settings instead of replaying the frozen garbage forever."""
    values = {"model.live.backend": "GGUF"}  # no model name anywhere
    _patch_settings(monkeypatch, values)
    seen_paths: list = []

    def factory(
        self,
        backend_type,
        model_path,
        chat_format,
        stop_tokens,
        context_window,
        rest_url,
        model_type,
    ):
        seen_paths.append(model_path)
        if not model_path:
            raise RuntimeError("no model path")
        return object()  # stand-in backend

    monkeypatch.setattr(ModelManager, "_create_backend", factory)

    manager = ModelManager()
    assert manager.model_states["live"]["status"] == "failed"
    assert seen_paths == [None]

    # Operator fixes the settings DB; the next retry re-reads it.
    values["model.live.name"] = "fixed-model"
    assert manager.retry_failed_loads(force=True) is True
    assert manager.model_states["live"]["status"] == "ready"
    assert seen_paths[-1] == "fixed-model"
    assert "fixed-model" in manager.registry


def test_registry_uses_frozen_config_after_settings_change(
    reset_singleton, monkeypatch
):
    """A successful retry loads the FROZEN config — the registry and aliases
    must advertise the weights actually loaded, not a settings value that
    changed between the failure and the retry."""
    values = dict(SPLIT_SETTINGS)
    _patch_settings(monkeypatch, values)
    fail_slots = {"background"}
    _install_factory(monkeypatch, fail_slots)

    manager = ModelManager()

    # Operator points background at a NEW model while the retry is pending.
    values["model.background.name"] = "mock-bg-v2"
    fail_slots.clear()

    assert manager.retry_failed_loads(force=True) is True
    assert manager.aliases["background"] == "mock-bg"
    assert "mock-bg" in manager.registry
    assert "mock-bg-v2" not in manager.registry


def test_ready_slot_is_immediately_addressable(reset_singleton, monkeypatch):
    """No ready-before-registry window: as soon as a slot reports 'ready',
    get_model_config must resolve it (otherwise /internal/model/chat 404s —
    a non-retryable signal — while the sibling slot is still loading)."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    _install_factory(monkeypatch, set(), slow_slots={"background"}, delay_s=0.5)

    manager = ModelManager(auto_load=False)
    loader = threading.Thread(target=manager.load_all)
    loader.start()
    try:
        deadline = real_time.monotonic() + 5.0
        while (
            real_time.monotonic() < deadline
            and manager.model_states["live"]["status"] != "ready"
        ):
            real_time.sleep(0.01)
        assert manager.model_states["live"]["status"] == "ready"
        assert manager.get_model_config("live") is not None
        assert manager.aliases.get("live") == "mock-live"
    finally:
        loader.join()


# ---------------------------------------------------------------------------
# Pause (vision/training GPU handoff) + shutdown (reload retirement)
# ---------------------------------------------------------------------------


def test_paused_manager_never_retries(reset_singleton, monkeypatch):
    """While /internal/model/unload has freed the GPU for a vision/training
    job, the retry machinery must not reload weights into it."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    fail_slots = {"background"}
    calls = _install_factory(monkeypatch, fail_slots)

    manager = ModelManager()
    initial_calls = calls["count"]
    fail_slots.clear()

    manager.pause_loads()
    assert manager.retry_failed_loads(force=True) is False
    assert calls["count"] == initial_calls
    assert manager.model_states["background"]["status"] == "failed"

    manager.resume_loads()
    assert manager.retry_failed_loads(force=True) is True
    assert manager.model_states["background"]["status"] == "ready"


def test_begin_shutdown_waits_for_in_flight_load_and_blocks_new_ones(
    reset_singleton, monkeypatch
):
    """Reload retires the old manager: begin_shutdown() must block until the
    in-flight retry finishes (no concurrent double-load with the fresh
    manager) and prevent any load from starting afterwards."""
    _patch_settings(monkeypatch, SPLIT_SETTINGS)
    fail_slots = {"background"}
    calls = _install_factory(
        monkeypatch, fail_slots, slow_slots={"background"}, delay_s=0.3
    )

    manager = ModelManager()
    fail_slots.clear()

    retry_thread = threading.Thread(
        target=manager.retry_failed_loads, kwargs={"force": True}
    )
    retry_thread.start()
    real_time.sleep(0.05)  # let the slow retry get in flight

    manager.begin_shutdown()  # must block until the retry completes
    assert manager.model_states["background"]["status"] == "ready"
    retry_thread.join()

    # No new loads on a retired manager.
    calls_after_shutdown = calls["count"]
    manager._set_slot_state("background", "failed", error="boom")
    assert manager.retry_failed_loads(force=True) is False
    manager.load_all()
    assert calls["count"] == calls_after_shutdown
