"""Tests for the settings API routes.

These tests cover:
- Authentication (app-to-app auth)
- CRUD operations
- Error handling
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from jarvis_settings_client import SettingsService
from services.settings_service import SETTINGS_DEFINITIONS

# Ensure JARVIS_AUTH_BASE_URL is set before importing settings_routes,
# since the module creates require_app_auth at import time via get_auth_url().
os.environ.setdefault("JARVIS_AUTH_BASE_URL", "http://localhost:7701")

from api.settings_routes import require_app_auth, router  # noqa: E402


@pytest.fixture
def mock_service():
    """Create a mock settings service."""
    service = SettingsService(
        definitions=SETTINGS_DEFINITIONS,
        get_db_session=lambda: None,
        setting_model=None,
    )
    return service


@pytest.fixture
def client(mock_service):
    """Create test client with mocked dependencies."""
    app = FastAPI()
    app.include_router(router)

    # Override the auth dependency to always succeed — must use the same
    # require_app_auth object that the router uses (from api.settings_routes).
    async def mock_auth():
        return None

    app.dependency_overrides[require_app_auth] = mock_auth

    # Override the settings service
    with patch("api.settings_routes.get_settings_service", return_value=mock_service):
        yield TestClient(app)


@pytest.fixture
def unauthenticated_client():
    """Create test client without auth override (auth will fail)."""
    app = FastAPI()
    app.include_router(router)

    # Don't override auth - let it fail naturally
    with patch.dict(os.environ, {}, clear=False):
        yield TestClient(app)


class TestAuthentication:
    """Tests for authentication requirements."""

    def test_no_auth_returns_401(self, unauthenticated_client):
        """Test that requests without auth return 401."""
        response = unauthenticated_client.get("/settings/")
        assert response.status_code == 401


class TestListSettings:
    """Tests for listing settings."""

    def test_list_all_settings(self, client):
        """Test listing all settings."""
        response = client.get("/settings/")
        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
        assert "total" in data
        assert data["total"] > 0

    def test_list_settings_by_category(self, client):
        """Test filtering settings by category."""
        response = client.get("/settings/?category=model.main")
        assert response.status_code == 200
        data = response.json()
        assert all(s["category"] == "model.main" for s in data["settings"])


class TestListCategories:
    """Tests for listing categories."""

    def test_list_categories(self, client):
        """Test listing categories."""
        response = client.get("/settings/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert "model.main" in data["categories"]
        assert "inference.vllm" in data["categories"]


class TestGetSetting:
    """Tests for getting individual settings."""

    def test_get_existing_setting(self, client):
        """Test getting an existing setting."""
        response = client.get("/settings/model.main.name")
        assert response.status_code == 200
        data = response.json()
        assert data["key"] == "model.main.name"
        assert "value" in data
        assert "value_type" in data

    def test_get_nonexistent_setting(self, client):
        """Test getting a nonexistent setting returns 404."""
        response = client.get("/settings/nonexistent.setting.key")
        assert response.status_code == 404
        assert "not_found" in response.json()["detail"]["error"]["type"]


class TestUpdateSetting:
    """Tests for updating settings."""

    def test_update_nonexistent_setting(self, client):
        """Test updating a nonexistent setting returns 404."""
        response = client.put(
            "/settings/nonexistent.setting.key",
            json={"value": "some_value"},
        )
        assert response.status_code == 404

    def test_update_setting_requires_reload(self, client, mock_service):
        """Test that model.main.name indicates requires_reload."""
        # Mock the set method to return True
        with patch.object(mock_service, "set", return_value=True):
            response = client.put(
                "/settings/model.main.name",
                json={"value": "new_model"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["requires_reload"] is True


class TestSyncFromEnv:
    """Tests for sync-from-env operation."""

    def test_sync_from_env(self, client):
        """Test syncing settings from environment."""
        with patch.dict(os.environ, {
            "JARVIS_MODEL_NAME": "env_model",
            "JARVIS_MODEL_CONTEXT_WINDOW": "8192",
        }):
            response = client.post("/settings/sync-from-env")
            assert response.status_code == 200
            data = response.json()
            assert "synced" in data
            assert "total_synced" in data


class TestInvalidateCache:
    """Tests for cache invalidation."""

    def test_invalidate_all_cache(self, client):
        """Test invalidating all cache."""
        response = client.post("/settings/invalidate-cache")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["invalidated"] == "all"

    def test_invalidate_single_key(self, client):
        """Test invalidating a single key."""
        response = client.post("/settings/invalidate-cache?key=model.main.name")
        assert response.status_code == 200
        data = response.json()
        assert data["invalidated"] == "model.main.name"
