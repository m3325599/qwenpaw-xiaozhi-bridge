"""aiohttp application: xiaozhi WebSocket endpoint + management HTTP API."""

from __future__ import annotations

import asyncio
import logging

from aiohttp import web

from .config import Config
from .mcp import McpToolError
from .qwenpaw import QwenPawClient
from .session import DeviceRegistry, DeviceSession

logger = logging.getLogger("bridge.server")


def create_app(cfg: Config) -> web.Application:
    qwenpaw = QwenPawClient(
        cfg.qwenpaw_base_url, cfg.qwenpaw_agent_id, cfg.qwenpaw_channel, cfg.qwenpaw_api_token
    )
    registry = DeviceRegistry()

    app = web.Application()
    app["cfg"] = cfg
    app["qwenpaw"] = qwenpaw
    app["registry"] = registry

    async def _on_cleanup(app: web.Application) -> None:
        await qwenpaw.close()

    app.on_cleanup.append(_on_cleanup)

    # ------------------------------------------------------------- HTTP API

    async def index(_request: web.Request) -> web.Response:
        return web.json_response(
            {
                "service": "qwenpaw-xiaozhi-bridge",
                "websocket": f"ws://<host>:{cfg.port}/xiaozhi/v1/",
                "devices": len(registry.list()),
                "qwenpaw": cfg.qwenpaw_base_url,
                "agent": cfg.qwenpaw_agent_id,
            }
        )

    async def healthz(_request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def list_devices(_request: web.Request) -> web.Response:
        return web.json_response([s.describe() for s in registry.list()])

    async def device_tools(request: web.Request) -> web.Response:
        session = registry.get(request.match_info["device_id"])
        if session is None:
            raise web.HTTPNotFound(text="device not connected")
        return web.json_response({"tools": session.mcp.tools})

    async def device_tool_call(request: web.Request) -> web.Response:
        session = registry.get(request.match_info["device_id"])
        if session is None:
            raise web.HTTPNotFound(text="device not connected")
        try:
            body = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise web.HTTPBadRequest(text=f"invalid JSON: {exc}") from exc
        name = body.get("name")
        if not name:
            raise web.HTTPBadRequest(text="missing 'name'")
        arguments = body.get("arguments") or {}
        try:
            result = await session.mcp.call_tool(name, arguments)
        except McpToolError as exc:
            return web.json_response({"error": exc.error}, status=502)
        except asyncio.TimeoutError:
            return web.json_response({"error": "device tool call timeout"}, status=504)
        return web.json_response({"result": result})

    # ------------------------------------------------------ xiaozhi WS endpoint

    async def xiaozhi_ws(request: web.Request) -> web.WebSocketResponse:
        # Token check (optional)
        if cfg.token:
            auth = request.headers.get("Authorization", "")
            token = auth[7:].strip() if auth.lower().startswith("bearer") else auth.strip()
            if token != cfg.token:
                raise web.HTTPUnauthorized(text="invalid token")

        device_id = request.headers.get("Device-Id", "").strip() or "unknown-device"
        client_id = request.headers.get("Client-Id", "").strip()
        try:
            protocol_version = int(request.headers.get("Protocol-Version", "1"))
        except ValueError:
            protocol_version = 1
        protocol_version = protocol_version if 1 <= protocol_version <= 3 else 1

        ws = web.WebSocketResponse(heartbeat=30, autoping=True)
        await ws.prepare(request)

        session = DeviceSession(
            cfg, ws, device_id, client_id, protocol_version, qwenpaw, registry
        )
        await session.run()
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/devices", list_devices)
    app.router.add_get("/devices/{device_id}/tools", device_tools)
    app.router.add_post("/devices/{device_id}/tools/call", device_tool_call)
    app.router.add_get("/xiaozhi/v1", xiaozhi_ws)
    app.router.add_get("/xiaozhi/v1/", xiaozhi_ws)
    return app
