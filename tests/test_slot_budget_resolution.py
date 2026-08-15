"""resolve_slot_reasoning_budget — request-boundary slot default resolution.

The backend instance's own reasoning-budget default is keyed on the slot it was
constructed for, which is "live" whenever background mirrors live (shared
instance). Resolving at the request boundary keyed on the REQUESTED slot makes
model.background.reasoning_budget govern background requests in every topology.
"""

import os
from unittest.mock import patch

from services.settings_helpers import resolve_slot_reasoning_budget


class _StubSettings:
    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


def _patch(values):
    return patch(
        "services.settings_helpers._get_settings_service",
        return_value=_StubSettings(values),
    )


def test_background_slot_reads_background_setting():
    with _patch({"model.background.reasoning_budget": "-1"}), patch.dict(
        os.environ, {}, clear=True
    ):
        assert resolve_slot_reasoning_budget("background") == -1


def test_live_slot_not_polluted_by_background_setting():
    values = {
        "model.live.reasoning_budget": "0",
        "model.background.reasoning_budget": "-1",
    }
    with _patch(values), patch.dict(os.environ, {}, clear=True):
        assert resolve_slot_reasoning_budget("live") == 0
        assert resolve_slot_reasoning_budget("background") == -1


def test_blank_slot_setting_falls_back_to_env():
    with _patch({}), patch.dict(
        os.environ, {"JARVIS_REST_REASONING_BUDGET": "-1"}, clear=True
    ):
        assert resolve_slot_reasoning_budget("background") == -1


def test_non_slot_model_id_resolves_none():
    with _patch({"model.background.reasoning_budget": "-1"}), patch.dict(
        os.environ, {}, clear=True
    ):
        assert resolve_slot_reasoning_budget("gpt-4o") is None
        assert resolve_slot_reasoning_budget(None) is None
        assert resolve_slot_reasoning_budget("") is None


def test_blank_everything_resolves_none():
    with _patch({}), patch.dict(os.environ, {}, clear=True):
        assert resolve_slot_reasoning_budget("background") is None


def test_unparseable_value_resolves_none():
    with _patch({"model.background.reasoning_budget": "banana"}), patch.dict(
        os.environ, {}, clear=True
    ):
        assert resolve_slot_reasoning_budget("background") is None
