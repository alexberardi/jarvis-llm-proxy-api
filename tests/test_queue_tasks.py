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
