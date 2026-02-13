"""Settings API routes with combined authentication.

Supports authentication via:
- Authorization: Bearer <token>: Superuser JWT (validated via jarvis-auth)
- X-Jarvis-App-Id + X-Jarvis-App-Key: App-to-app authentication

All endpoints require authentication. Settings support multi-tenant scoping:
- System default: no scope parameters
- Household-level: household_id set
- Node-level: household_id + node_id set
- User-level: household_id + node_id + user_id set
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from config.service_config import get_auth_url
from jarvis_settings_client import create_combined_auth
from services.settings_service import get_settings_service

require_app_auth = create_combined_auth(get_auth_url)

logger = logging.getLogger("uvicorn")

router = APIRouter(prefix="/settings", tags=["settings"])


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


class CacheInvalidateResponse(BaseModel):
    """Response for cache invalidation."""

    status: str
    invalidated: str


# ===========================================================================
# Endpoints
# ===========================================================================


@router.get("/", response_model=SettingsListResponse)
async def list_settings(
    request: Request,
    category: str | None = Query(None, description="Filter by category"),
    household_id: str | None = Query(None, description="Household scope"),
    node_id: str | None = Query(None, description="Node scope"),
    user_id: int | None = Query(None, description="User scope"),
    _auth: None = Depends(require_app_auth),
) -> SettingsListResponse:
    """List all settings.

    Optionally filter by category using the `category` query parameter.
    Scope parameters control which values are returned (cascade lookup).
    """
    calling_app = getattr(request.state, "calling_app_id", None)
    logger.debug(f"Settings list requested by app: {calling_app}")

    service = get_settings_service()
    settings = service.list_all(category=category)

    return SettingsListResponse(
        settings=[SettingResponse(**s) for s in settings],
        total=len(settings),
    )


@router.get("/categories", response_model=CategoriesResponse)
async def list_categories(
    request: Request,
    _auth: None = Depends(require_app_auth),
) -> CategoriesResponse:
    """List all unique setting categories."""
    service = get_settings_service()
    categories = service.list_categories()

    return CategoriesResponse(categories=categories)


@router.get("/{key:path}", response_model=SettingResponse)
async def get_setting(
    key: str,
    request: Request,
    household_id: str | None = Query(None, description="Household scope"),
    node_id: str | None = Query(None, description="Node scope"),
    user_id: int | None = Query(None, description="User scope"),
    _auth: None = Depends(require_app_auth),
) -> SettingResponse:
    """Get a single setting by key.

    The key uses dot notation, e.g., `model.main.name`.
    Scope parameters enable cascade lookup (user > node > household > system).
    """
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
async def update_setting(
    key: str,
    body: SettingUpdate,
    request: Request,
    household_id: str | None = Query(None, description="Household scope"),
    node_id: str | None = Query(None, description="Node scope"),
    user_id: int | None = Query(None, description="User scope"),
    _auth: None = Depends(require_app_auth),
) -> UpdateResponse:
    """Update a single setting.

    If the setting has `requires_reload=True`, the model will need to be
    reloaded for the change to take effect. The response indicates this.

    Scope parameters allow setting values for specific scopes:
    - No scope: system default
    - household_id: household-level override
    - household_id + node_id: node-level override
    - household_id + node_id + user_id: user-level override
    """
    calling_app = getattr(request.state, "calling_app_id", None)
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

    logger.info(f"Setting {key} updated by app: {calling_app}")

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
async def bulk_update_settings(
    body: BulkSettingUpdate,
    request: Request,
    _auth: None = Depends(require_app_auth),
) -> BulkUpdateResponse:
    """Update multiple settings at once.

    The response indicates whether any settings require a model reload.
    """
    calling_app = getattr(request.state, "calling_app_id", None)
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

    logger.info(f"Bulk update: {total_updated} updated, {total_failed} failed by app: {calling_app}")

    return BulkUpdateResponse(
        results=results,
        total_updated=total_updated,
        total_failed=total_failed,
        requires_reload=requires_reload,
    )


@router.post("/sync-from-env", response_model=SyncResponse)
async def sync_from_env(
    request: Request,
    _auth: None = Depends(require_app_auth),
) -> SyncResponse:
    """One-time migration: sync settings from environment variables to database.

    This reads all defined settings from environment variables and writes
    them to the database. Useful for initial setup or migration.
    """
    calling_app = getattr(request.state, "calling_app_id", None)
    logger.info(f"Sync from env requested by app: {calling_app}")

    service = get_settings_service()
    results = service.sync_from_env()

    total_synced = sum(1 for v in results.values() if v)
    total_skipped = sum(1 for v in results.values() if not v)

    return SyncResponse(
        synced=results,
        total_synced=total_synced,
        total_skipped=total_skipped,
    )


@router.post("/invalidate-cache", response_model=CacheInvalidateResponse)
async def invalidate_cache(
    request: Request,
    key: str | None = Query(None, description="Key to invalidate (all if omitted)"),
    _auth: None = Depends(require_app_auth),
) -> CacheInvalidateResponse:
    """Invalidate the settings cache.

    If `key` is provided, only that key is invalidated.
    Otherwise, the entire cache is cleared.
    """
    service = get_settings_service()
    service.invalidate_cache(key)

    return CacheInvalidateResponse(status="ok", invalidated=key or "all")
