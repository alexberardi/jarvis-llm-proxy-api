"""Image-capability plumbing on ModelManager's slot registry.

Before this, `supports_images` was hardcoded False for both slots, so
chat_runner rejected every image request even though RestBackend already
implemented generate_vision_chat. The capability is now a setting, frozen
alongside the rest of the slot config (the registry must describe the
weights that are actually loaded, per _populate_registry's docstring).

Run with:
    pytest tests/test_model_supports_images.py -v
"""

import pytest

import managers.model_manager as mm_module
from managers.model_manager import ModelManager


@pytest.fixture
def reset_singleton():
    ModelManager._instance = None
    ModelManager._initialized = False
    yield
    ModelManager._instance = None
    ModelManager._initialized = False


def _patch_settings(monkeypatch, values: dict) -> None:
    def fake_get_setting(key, env_fallback, default, value_type="string"):
        return values.get(key, default)

    def fake_get_int_setting(key, env_fallback, default):
        return int(values.get(key, default))

    monkeypatch.setattr(mm_module, "get_setting", fake_get_setting)
    monkeypatch.setattr(mm_module, "get_int_setting", fake_get_int_setting)


def _patch_backends(monkeypatch, slots: dict[str, object]) -> None:
    """Patch _create_backend to hand back a sentinel per slot."""
    def factory(self, backend_type, model_path, *args, **kwargs):
        return slots[model_path]

    monkeypatch.setattr(ModelManager, "_create_backend", factory)


VISION_SETTINGS = {
    "model.live.backend": "MOCK",
    "model.live.name": "mock-vision",
    "model.live.supports_images": True,
    "model.background.backend": "MOCK",
    "model.background.name": "mock-text",
    "model.background.supports_images": False,
}

TEXT_ONLY_SETTINGS = {
    "model.live.backend": "MOCK",
    "model.live.name": "mock-text",
    "model.background.backend": "MOCK",
    "model.background.name": "mock-text2",
}


def _build_manager(monkeypatch, settings: dict) -> ModelManager:
    _patch_settings(monkeypatch, settings)
    _patch_backends(
        monkeypatch,
        {
            "mock-vision": object(),
            "mock-text": object(),
            "mock-text2": object(),
        },
    )
    return ModelManager()


def test_supports_images_frozen_into_slot_config(monkeypatch, reset_singleton):
    mgr = _build_manager(monkeypatch, VISION_SETTINGS)
    assert mgr._slot_configs["live"]["supports_images"] is True
    assert mgr._slot_configs["background"]["supports_images"] is False


def test_registry_reports_live_as_image_capable(monkeypatch, reset_singleton):
    mgr = _build_manager(monkeypatch, VISION_SETTINGS)
    cfg = mgr.get_model_config("live")
    assert cfg is not None
    assert cfg.supports_images is True


def test_registry_reports_background_as_text_only(monkeypatch, reset_singleton):
    mgr = _build_manager(monkeypatch, VISION_SETTINGS)
    cfg = mgr.get_model_config("background")
    assert cfg is not None
    assert cfg.supports_images is False


def test_supports_images_defaults_false_when_unset(monkeypatch, reset_singleton):
    """Existing deployments that never set the key keep rejecting images."""
    mgr = _build_manager(monkeypatch, TEXT_ONLY_SETTINGS)
    assert mgr._slot_configs["live"]["supports_images"] is False
    assert mgr._slot_configs["background"]["supports_images"] is False
    assert mgr.get_model_config("live").supports_images is False


def test_list_models_advertises_image_support(monkeypatch, reset_singleton):
    """/v1/models consumers need to see which models take images."""
    mgr = _build_manager(monkeypatch, VISION_SETTINGS)
    by_id = {m["id"]: m for m in mgr.list_models()}
    live_id = mgr.aliases["live"]
    bg_id = mgr.aliases["background"]
    assert by_id[live_id]["supports_images"] is True
    assert by_id[bg_id]["supports_images"] is False


def test_string_true_from_settings_is_coerced(monkeypatch, reset_singleton):
    """Settings rows arrive as strings; "true"/"1" must not stay truthy-ambiguous."""
    mgr = _build_manager(
        monkeypatch,
        {**VISION_SETTINGS, "model.live.supports_images": "true"},
    )
    assert mgr.get_model_config("live").supports_images is True
