"""Settings API routes with dual authentication.

Supports both:
- X-Internal-Token: Service-to-service authentication
- X-Jarvis-Admin-Token: Human operator authentication

All endpoints require one of these authentication methods.
"""

import logging
import os
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.settings_service import get_settings_service, SETTINGS_DEFINITIONS

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/internal/settings", tags=["settings"])


# ===========================================================================
# Authentication
# ===========================================================================


def require_admin_or_internal_token(
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> str:
    """Require either internal token or admin token.

    Returns the token type that was used for logging purposes.
    """
    # Check internal token
    internal_token = os.getenv("MODEL_SERVICE_TOKEN") or os.getenv("LLM_PROXY_INTERNAL_TOKEN")
    if internal_token and x_internal_token == internal_token:
        return "internal"

    # Check admin token
    admin_token = os.getenv("JARVIS_ADMIN_TOKEN")
    if admin_token and x_jarvis_admin_token == admin_token:
        return "admin"

    # Neither token is valid
    raise HTTPException(
        status_code=401,
        detail={
            "error": {
                "type": "unauthorized",
                "message": "Valid X-Internal-Token or X-Jarvis-Admin-Token required",
                "code": "unauthorized",
            }
        },
    )


# ===========================================================================
# Request/Response Models
# ===========================================================================


class SettingUpdate(BaseModel):
    """Request to update a single setting."""

    value: Any


class BulkSettingUpdate(BaseModel):
    """Request to update multiple settings."""

    settings: dict[str, Any]


class SettingResponse(BaseModel):
    """Response for a single setting."""

    key: str
    value: Any
    value_type: str
    category: str
    description: str | None
    requires_reload: bool
    is_secret: bool
    env_fallback: str | None
    from_db: bool


class SettingsListResponse(BaseModel):
    """Response for listing settings."""

    settings: list[SettingResponse]
    total: int


class CategoriesResponse(BaseModel):
    """Response for listing categories."""

    categories: list[str]


class SyncResponse(BaseModel):
    """Response for sync operation."""

    synced: dict[str, bool]
    total_synced: int
    total_skipped: int


class UpdateResponse(BaseModel):
    """Response for update operation."""

    success: bool
    key: str
    requires_reload: bool
    message: str | None = None


class BulkUpdateResponse(BaseModel):
    """Response for bulk update operation."""

    results: dict[str, bool]
    total_updated: int
    total_failed: int
    requires_reload: bool


# ===========================================================================
# Endpoints
# ===========================================================================


@router.get("/", response_model=SettingsListResponse)
def list_settings(
    category: str | None = None,
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> SettingsListResponse:
    """List all settings.

    Optionally filter by category using the `category` query parameter.
    """
    auth_type = require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)
    logger.debug(f"Settings list requested via {auth_type} auth")

    service = get_settings_service()
    settings = service.list_all(category=category)

    return SettingsListResponse(
        settings=[SettingResponse(**s) for s in settings],
        total=len(settings),
    )


@router.get("/categories", response_model=CategoriesResponse)
def list_categories(
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> CategoriesResponse:
    """List all unique setting categories."""
    require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)

    service = get_settings_service()
    categories = service.list_categories()

    return CategoriesResponse(categories=categories)


@router.get("/{key:path}", response_model=SettingResponse)
def get_setting(
    key: str,
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> SettingResponse:
    """Get a single setting by key.

    The key uses dot notation, e.g., `model.main.name`.
    """
    require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)

    service = get_settings_service()

    # Check if key exists in definitions
    if key not in service._definitions:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "not_found",
                    "message": f"Setting not found: {key}",
                    "code": "not_found",
                }
            },
        )

    # Get the setting details
    definition = service._definitions[key]
    value = service.get(key)

    # Check if it's from DB
    with service._cache_lock:
        from_db = key in service._cache and service._cache[key].from_db

    # Mask secrets
    display_value = value
    if definition.is_secret and value:
        display_value = "********"

    return SettingResponse(
        key=key,
        value=display_value,
        value_type=definition.value_type,
        category=definition.category,
        description=definition.description,
        requires_reload=definition.requires_reload,
        is_secret=definition.is_secret,
        env_fallback=definition.env_fallback,
        from_db=from_db,
    )


@router.put("/{key:path}", response_model=UpdateResponse)
def update_setting(
    key: str,
    body: SettingUpdate,
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> UpdateResponse:
    """Update a single setting.

    If the setting has `requires_reload=True`, the model will need to be
    reloaded for the change to take effect. The response indicates this.
    """
    auth_type = require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)

    service = get_settings_service()

    # Check if key exists
    if key not in service._definitions:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "type": "not_found",
                    "message": f"Setting not found: {key}",
                    "code": "not_found",
                }
            },
        )

    definition = service._definitions[key]
    success = service.set(key, body.value)

    if not success:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "type": "internal_error",
                    "message": f"Failed to update setting: {key}",
                    "code": "update_failed",
                }
            },
        )

    logger.info(f"Setting {key} updated via {auth_type} auth")

    message = None
    if definition.requires_reload:
        message = "Model reload required for this change to take effect"

    return UpdateResponse(
        success=True,
        key=key,
        requires_reload=definition.requires_reload,
        message=message,
    )


@router.put("/", response_model=BulkUpdateResponse)
def bulk_update_settings(
    body: BulkSettingUpdate,
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> BulkUpdateResponse:
    """Update multiple settings at once.

    The response indicates whether any settings require a model reload.
    """
    auth_type = require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)

    service = get_settings_service()

    # Validate all keys first
    invalid_keys = [k for k in body.settings.keys() if k not in service._definitions]
    if invalid_keys:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "type": "invalid_request",
                    "message": f"Unknown setting keys: {', '.join(invalid_keys)}",
                    "code": "invalid_keys",
                }
            },
        )

    results = service.set_bulk(body.settings)

    # Check if any updated settings require reload
    requires_reload = False
    for key, success in results.items():
        if success and service._definitions[key].requires_reload:
            requires_reload = True
            break

    total_updated = sum(1 for v in results.values() if v)
    total_failed = sum(1 for v in results.values() if not v)

    logger.info(f"Bulk update: {total_updated} updated, {total_failed} failed via {auth_type} auth")

    return BulkUpdateResponse(
        results=results,
        total_updated=total_updated,
        total_failed=total_failed,
        requires_reload=requires_reload,
    )


@router.post("/sync-from-env", response_model=SyncResponse)
def sync_from_env(
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> SyncResponse:
    """One-time migration: sync settings from environment variables to database.

    This reads all defined settings from environment variables and writes
    them to the database. Useful for initial setup or migration.
    """
    auth_type = require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)
    logger.info(f"Sync from env requested via {auth_type} auth")

    service = get_settings_service()
    results = service.sync_from_env()

    total_synced = sum(1 for v in results.values() if v)
    total_skipped = sum(1 for v in results.values() if not v)

    return SyncResponse(
        synced=results,
        total_synced=total_synced,
        total_skipped=total_skipped,
    )


@router.post("/invalidate-cache")
def invalidate_cache(
    key: str | None = None,
    x_internal_token: str | None = Header(None),
    x_jarvis_admin_token: str | None = Header(None),
) -> dict[str, str]:
    """Invalidate the settings cache.

    If `key` is provided, only that key is invalidated.
    Otherwise, the entire cache is cleared.
    """
    require_admin_or_internal_token(x_internal_token, x_jarvis_admin_token)

    service = get_settings_service()
    service.invalidate_cache(key)

    return {"status": "ok", "invalidated": key or "all"}
