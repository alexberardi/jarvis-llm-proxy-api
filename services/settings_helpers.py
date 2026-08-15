"""Helpers for settings access with env fallback."""

from __future__ import annotations

import os
from typing import Any


def _get_settings_service():
    try:
        from services.settings_service import get_settings_service

        return get_settings_service()
    except (ImportError, RuntimeError):
        return None


def _coerce_value(raw: str, value_type: str, default: Any) -> Any:
    if raw is None or raw == "":
        return default
    try:
        if value_type == "int":
            return int(raw)
        if value_type == "float":
            return float(raw)
        if value_type == "bool":
            return raw.lower() in ("true", "1", "yes", "on")
        return raw
    except ValueError:
        return default


def get_setting(
    key: str,
    env_fallback: str | None,
    default: Any,
    value_type: str = "string",
) -> Any:
    """Get a setting value with fallback to environment variable."""
    settings = _get_settings_service()
    if settings:
        try:
            value = settings.get(key)
            if value is not None:
                return value
        except Exception:
            pass
    if env_fallback:
        raw = os.getenv(env_fallback)
        if raw is not None and raw != "":
            return _coerce_value(raw, value_type, default)
    return default


def get_int_setting(key: str, env_fallback: str | None, default: int) -> int:
    return int(get_setting(key, env_fallback, default, "int"))


def get_float_setting(key: str, env_fallback: str | None, default: float) -> float:
    return float(get_setting(key, env_fallback, default, "float"))


def get_bool_setting(key: str, env_fallback: str | None, default: bool) -> bool:
    value = get_setting(key, env_fallback, default, "bool")
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def resolve_slot_reasoning_budget(model: str | None) -> int | None:
    """Default thinking budget for a slot-aliased request ("live"/"background").

    Mirrors RestClient's init-time resolution but keyed on the REQUESTED slot.
    The instance-level default is keyed on the slot the backend was constructed
    for — when the two slots share one instance (mirrored config) that is
    always "live", so model.background.* would never govern background requests
    without this request-boundary resolution. Returns None for non-slot model
    ids and for blank/unparseable settings (backend default applies).
    """
    slot = (model or "").strip().lower()
    if slot not in ("live", "background"):
        return None
    rb = get_setting(f"model.{slot}.reasoning_budget", "", "")
    if str(rb).strip() == "":
        rb = get_setting(
            "model.main.reasoning_budget", "JARVIS_REST_REASONING_BUDGET", ""
        )
    try:
        return int(rb) if str(rb).strip() not in ("", "None") else None
    except (TypeError, ValueError):
        return None
