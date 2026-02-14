"""Tests for MCP client service.

Tests the MCP client that connects to jarvis-mcp for date resolution.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestMcpClientInit:
    """Tests for MCP client initialization."""

    def test_default_url(self):
        from services.mcp_client import McpClient
        client = McpClient()
        assert client.mcp_url == "http://localhost:8011"

    def test_custom_url(self):
        from services.mcp_client import McpClient
        client = McpClient(mcp_url="http://custom:9999")
        assert client.mcp_url == "http://custom:9999"

    def test_initial_state_not_connected(self):
        from services.mcp_client import McpClient
        client = McpClient()
        assert client.is_connected() is False


class TestMcpClientResolveDateKeys:
    """Tests for the resolve_date_keys convenience method."""

    @pytest.mark.asyncio
    async def test_resolve_returns_dict(self):
        from services.mcp_client import McpClient
        client = McpClient()

        # Mock call_tool to return a resolved response
        mock_result = MagicMock()
        mock_result.isError = False
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "resolved": ["2026-02-14T00:00:00Z"],
            "unresolved": [],
        })
        mock_result.content = [mock_content]

        client.call_tool = AsyncMock(return_value=mock_result)
        client._connected = True

        result = await client.resolve_date_keys(["tomorrow"])
        assert result["resolved"] == ["2026-02-14T00:00:00Z"]
        assert result["unresolved"] == []

    @pytest.mark.asyncio
    async def test_resolve_with_timezone(self):
        from services.mcp_client import McpClient
        client = McpClient()

        mock_result = MagicMock()
        mock_result.isError = False
        mock_content = MagicMock()
        mock_content.text = json.dumps({
            "resolved": ["2026-02-14T05:00:00Z"],
            "unresolved": [],
        })
        mock_result.content = [mock_content]

        client.call_tool = AsyncMock(return_value=mock_result)
        client._connected = True

        result = await client.resolve_date_keys(["tomorrow"], timezone="America/New_York")

        # Verify call_tool was called with correct args
        client.call_tool.assert_called_once_with(
            "datetime_resolve",
            {"date_keys": ["tomorrow"], "timezone": "America/New_York"},
        )

    @pytest.mark.asyncio
    async def test_resolve_returns_none_when_not_connected(self):
        from services.mcp_client import McpClient
        client = McpClient()
        # Not connected
        result = await client.resolve_date_keys(["tomorrow"])
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_returns_none_on_error(self):
        from services.mcp_client import McpClient
        client = McpClient()
        client._connected = True

        mock_result = MagicMock()
        mock_result.isError = True
        mock_result.content = [MagicMock(text="Some error")]

        client.call_tool = AsyncMock(return_value=mock_result)

        result = await client.resolve_date_keys(["tomorrow"])
        assert result is None

    @pytest.mark.asyncio
    async def test_resolve_returns_none_on_exception(self):
        from services.mcp_client import McpClient
        client = McpClient()
        client._connected = True

        client.call_tool = AsyncMock(side_effect=Exception("Connection lost"))

        result = await client.resolve_date_keys(["tomorrow"])
        assert result is None


class TestMcpClientGracefulDegradation:
    """Tests for graceful degradation when MCP is unavailable."""

    @pytest.mark.asyncio
    async def test_call_tool_returns_none_when_not_connected(self):
        from services.mcp_client import McpClient
        client = McpClient()
        # Not connected, no session
        result = await client.call_tool("datetime_resolve", {"date_keys": ["tomorrow"]})
        assert result is None


class TestMcpClientModule:
    """Tests for module-level singleton."""

    def test_get_mcp_client_returns_singleton(self):
        from services.mcp_client import get_mcp_client
        client1 = get_mcp_client()
        client2 = get_mcp_client()
        assert client1 is client2

    def test_get_mcp_client_reads_env(self):
        with patch.dict("os.environ", {"JARVIS_MCP_URL": "http://test:1234"}):
            from services import mcp_client
            # Reset singleton to pick up new env
            mcp_client._mcp_client = None
            client = mcp_client.get_mcp_client()
            assert client.mcp_url == "http://test:1234"
            # Clean up
            mcp_client._mcp_client = None


class TestApiModelExtensions:
    """Tests for the new API model fields."""

    def test_request_has_resolve_dates_field(self):
        from models.api_models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
            resolve_dates=True,
        )
        assert req.resolve_dates is True

    def test_request_resolve_dates_defaults_none(self):
        from models.api_models import ChatCompletionRequest
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hello"}],
        )
        assert req.resolve_dates is None

    def test_response_has_resolved_datetimes_field(self):
        from models.api_models import ChatCompletionResponse, ChatCompletionChoice, Message, Usage
        resp = ChatCompletionResponse(
            id="test",
            created=0,
            model="test",
            choices=[ChatCompletionChoice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            resolved_datetimes=["2026-02-14T00:00:00Z"],
        )
        assert resp.resolved_datetimes == ["2026-02-14T00:00:00Z"]

    def test_response_resolved_datetimes_defaults_none(self):
        from models.api_models import ChatCompletionResponse, ChatCompletionChoice, Message, Usage
        resp = ChatCompletionResponse(
            id="test",
            created=0,
            model="test",
            choices=[ChatCompletionChoice(index=0, message=Message(role="assistant", content="hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        assert resp.resolved_datetimes is None


class TestResponseHelpersExtension:
    """Tests for response helpers passing through resolved_datetimes."""

    def test_create_openai_response_with_resolved_datetimes(self):
        from services.response_helpers import create_openai_response
        resp = create_openai_response(
            content="hello",
            model_name="test",
            resolved_datetimes=["2026-02-14T00:00:00Z"],
        )
        assert resp.resolved_datetimes == ["2026-02-14T00:00:00Z"]

    def test_create_openai_response_without_resolved_datetimes(self):
        from services.response_helpers import create_openai_response
        resp = create_openai_response(
            content="hello",
            model_name="test",
        )
        assert resp.resolved_datetimes is None
