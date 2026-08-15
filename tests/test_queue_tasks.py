"""Queue worker in-process fallback (queues/tasks.py).

When model_service.url / MODEL_SERVICE_URL is unset, _process_chat_job runs
inference in-process via the ModelManager. services.model_service constructs
that manager with auto_load=False (models load in the FastAPI startup hook,
which never runs in a worker process) — the job path must trigger the initial
load itself or every in-process job fails on an empty registry.
"""

import os

import pytest

import managers.model_manager as mm_module
from managers.model_manager import ModelManager

# services.model_service imports api.settings_routes, which resolves the auth
# URL at import time (same pattern as tests/test_model_service_health.py).
os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")


@pytest.fixture
def fresh_unloaded_manager(monkeypatch):
    """Fresh auto_load=False singleton bound into services.model_service,
    with MOCK settings so load_all() is instant when the worker triggers it."""
    ModelManager._instance = None
    ModelManager._initialized = False

    values = {
        "model.live.backend": "MOCK",
        "model.live.name": "mock-live",
    }
    monkeypatch.setattr(
        mm_module,
        "get_setting",
        lambda key, env_fallback, default, value_type="string": values.get(key, default),
    )
    monkeypatch.setattr(
        mm_module,
        "get_int_setting",
        lambda key, env_fallback, default: int(values.get(key, default)),
    )

    import services.model_service as ms

    manager = ModelManager(auto_load=False)
    monkeypatch.setattr(ms, "model_manager", manager)

    yield manager

    ModelManager._instance = None
    ModelManager._initialized = False


def test_in_process_fallback_loads_models_before_first_job(
    fresh_unloaded_manager, monkeypatch
):
    import queues.tasks as tasks

    # No model service configured → the in-process branch.
    monkeypatch.setattr(tasks, "get_setting", lambda key, env_fallback, default: "")

    payload = {
        "job_id": "job-1",
        "job_type": "chat",
        "request": {"messages": [{"role": "user", "content": "hello"}]},
        "callback": {},  # no URL → callback skipped, envelope returned
    }

    envelope = tasks._process_chat_job(payload)

    assert envelope["status"] == "succeeded", envelope.get("error")
    assert envelope["result"]["content"]
    assert fresh_unloaded_manager.model_states["background"]["status"] == "ready"
    assert fresh_unloaded_manager.background_model is not None


def test_job_forwards_reasoning_budget_and_sampling(
    fresh_unloaded_manager, monkeypatch
):
    """reasoning_budget / sampling.top_p / sampling.seed from the job payload must
    reach the ChatCompletionRequest the worker builds (they were previously
    dropped by the temperature/max_tokens-only whitelist)."""
    import services.chat_runner as chat_runner
    import queues.tasks as tasks

    # No model service configured → the in-process branch.
    monkeypatch.setattr(tasks, "get_setting", lambda key, env_fallback, default: "")

    captured = {}

    async def _fake_run(manager, req, allow_images=False):
        captured["req"] = req

        class _R:
            content = "ok"

        return _R()

    monkeypatch.setattr(chat_runner, "run_chat_completion", _fake_run)

    payload = {
        "job_id": "job-2",
        "job_type": "chat",
        "request": {
            "messages": [{"role": "user", "content": "hello"}],
            "sampling": {"temperature": 0.0, "top_p": 0.9, "seed": 7},
            "reasoning_budget": 0,
        },
        "callback": {},
    }

    envelope = tasks._process_chat_job(payload)

    assert envelope["status"] == "succeeded", envelope.get("error")
    req = captured["req"]
    assert req.model == "background"
    assert req.reasoning_budget == 0
    assert req.top_p == 0.9
    assert req.seed == 7
    assert req.temperature == 0.0
