"""MCP client towards the ESP32 device (JSON-RPC 2.0 over xiaozhi WebSocket).

The device runs an MCP server (see main/mcp_server.cc of the firmware). After
the WebSocket hello handshake (device advertises ``features.mcp``), the bridge
sends ``initialize`` + ``tools/list`` to discover device tools, and can later
invoke them with ``tools/call`` — either from the HTTP management API or from
future QwenPaw-side integrations.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger("bridge.mcp")

SendPayload = Callable[[dict], Awaitable[None]]


class McpToolError(RuntimeError):
    def __init__(self, error: dict) -> None:
        self.error = error
        super().__init__(str(error.get("message", "MCP tool error")))


class McpDeviceClient:
    """JSON-RPC 2.0 request/response bookkeeping for one device connection."""

    def __init__(self, send_payload: SendPayload, timeout: float = 30.0) -> None:
        self._send_payload = send_payload
        self._timeout = timeout
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self.tools: list[dict] = []
        self.server_info: dict = {}
        self.initialized = False

    async def _request(
        self,
        method: str,
        params: dict | None = None,
        timeout: float | None = None,
    ) -> Any:
        request_id = next(self._ids)
        payload: dict = {"jsonrpc": "2.0", "method": method, "id": request_id}
        if params is not None:
            payload["params"] = params
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send_payload(payload)
            return await asyncio.wait_for(future, timeout=timeout or self._timeout)
        finally:
            self._pending.pop(request_id, None)

    async def on_payload(self, payload: dict) -> None:
        """Handle a device -> bridge MCP message."""
        request_id = payload.get("id")
        if "method" in payload and request_id is not None:
            # Device-initiated requests (elicitation, sampling, ...) are not
            # supported by the bridge yet.
            logger.debug("Ignoring MCP request from device: %s", payload.get("method"))
            return
        if not isinstance(request_id, int):
            return
        future = self._pending.get(request_id)
        if future is None or future.done():
            return
        if "error" in payload and payload["error"] is not None:
            future.set_exception(McpToolError(payload["error"]))
        else:
            future.set_result(payload.get("result"))

    async def setup(self) -> list[dict]:
        """Initialize the MCP session and discover tools."""
        result = await self._request("initialize", {"capabilities": {}})
        self.server_info = (result or {}).get("serverInfo", {})
        self.initialized = True
        logger.info("MCP initialized: %s", self.server_info)
        tools = await self.discover_tools()
        logger.info("Discovered %d device tools", len(tools))
        return tools

    async def discover_tools(self) -> list[dict]:
        tools: list[dict] = []
        cursor = ""
        while True:
            result = await self._request(
                "tools/list", {"cursor": cursor, "withUserTools": False}
            )
            page = (result or {}).get("tools", [])
            if isinstance(page, list):
                tools.extend(page)
            cursor = (result or {}).get("nextCursor", "")
            if not cursor:
                break
        self.tools = tools
        return tools

    async def call_tool(self, name: str, arguments: dict | None = None) -> Any:
        if not self.initialized:
            raise McpToolError({"message": "device MCP not initialized"})
        return await self._request("tools/call", {"name": name, "arguments": arguments or {}})
