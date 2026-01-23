"""Shared test fixtures for llm-proxy-api tests."""

import sys
from pathlib import Path

# Add project root to path so imports work without PYTHONPATH
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def apply_auth_mock(app: Any) -> None:
    """Apply the auth mock to the given FastAPI app."""
    from auth.app_auth import require_app_auth

    async def no_op_auth() -> None:
        pass

    app.dependency_overrides[require_app_auth] = no_op_auth


@pytest.fixture
def mock_auth() -> Generator[None, None, None]:
    """
    Marker fixture indicating auth should be mocked.

    The actual mocking is done by client fixture after app reload.
    """
    yield


@pytest.fixture
def mock_model_service() -> Generator[MagicMock, None, None]:
    """
    Mock httpx.AsyncClient to avoid hitting the real model service.

    Returns a simple canned response - no logic here.
    The parsing, validation, and routing are tested by the code
    that runs BEFORE this mock is reached.
    """
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "content": "mocked response",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value.post.return_value = mock_response
    mock_client.__aenter__.return_value.get.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client) as mock:
        yield mock
