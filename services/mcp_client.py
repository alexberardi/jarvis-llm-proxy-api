"""MCP client for connecting to jarvis-mcp.

Provides a persistent connection to jarvis-mcp's SSE endpoint for
calling MCP tools (e.g., datetime_resolve). Designed for graceful
degradation: if jarvis-mcp is unavailable, callers get None and
can fall back to raw date_keys.
"""

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger("uvicorn")

_mcp_client: Optional["McpClient"] = None


class McpClient:
    """MCP client connection to jarvis-mcp.

    Connects to jarvis-mcp via SSE transport and exposes tool calling.
    Connection is established lazily on first use.
    """

    def __init__(self, mcp_url: str | None = None) -> None:
        self.mcp_url = mcp_url or os.getenv("JARVIS_MCP_URL", "http://localhost:8011")
        self._connected: bool = False
        self._session: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None
        self._cm: Any = None  # context manager for sse_client

    def is_connected(self) -> bool:
        """Check if the client is connected to jarvis-mcp."""
        return self._connected

    async def connect(self) -> bool:
        """Connect to jarvis-mcp SSE endpoint.

        Returns True if connected successfully, False otherwise.
        """
        try:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client

            sse_url = f"{self.mcp_url.rstrip('/')}/sse"
            logger.info("Connecting to jarvis-mcp at %s", sse_url)

            self._cm = sse_client(sse_url, timeout=5)
            self._read_stream, self._write_stream = await self._cm.__aenter__()

            self._session = ClientSession(self._read_stream, self._write_stream)
            await self._session.__aenter__()
            await self._session.initialize()

            self._connected = True
            logger.info("Connected to jarvis-mcp successfully")
            return True

        except ImportError:
            logger.warning("mcp package not installed, MCP client disabled")
            return False
        except Exception as e:
            logger.warning("Failed to connect to jarvis-mcp: %s", e)
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from jarvis-mcp."""
        try:
            if self._session:
                await self._session.__aexit__(None, None, None)
                self._session = None
            if self._cm:
                await self._cm.__aexit__(None, None, None)
                self._cm = None
        except Exception as e:
            logger.debug("Error during MCP disconnect: %s", e)
        finally:
            self._connected = False
            self._read_stream = None
            self._write_stream = None

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool on jarvis-mcp.

        Returns the CallToolResult, or None if not connected.
        """
        if not self._connected or not self._session:
            return None

        return await self._session.call_tool(name, arguments)

    async def resolve_date_keys(
        self,
        date_keys: list[str],
        timezone: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve date keys via jarvis-mcp datetime_resolve tool.

        Args:
            date_keys: List of semantic date keys (e.g., ["tomorrow", "morning"])
            timezone: Optional IANA timezone string

        Returns:
            Dict with "resolved" and "unresolved" lists, or None on failure.
        """
        if not self._connected:
            return None

        try:
            args: dict[str, Any] = {"date_keys": date_keys}
            if timezone:
                args["timezone"] = timezone

            result = await self.call_tool("datetime_resolve", args)

            if result is None or result.isError:
                return None

            # Parse the JSON response from the tool
            if result.content and hasattr(result.content[0], "text"):
                return json.loads(result.content[0].text)

            return None

        except Exception as e:
            logger.warning("MCP datetime_resolve failed: %s", e)
            return None


def get_mcp_client() -> McpClient:
    """Get the global MCP client singleton."""
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = McpClient()
    return _mcp_client
