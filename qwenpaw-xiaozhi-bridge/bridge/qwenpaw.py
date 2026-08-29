"""QwenPaw REST client: chat via ``POST /api/console/chat`` (SSE stream).

The endpoint follows the AgentScope Runtime protocol: it returns a stream of
``data: {...}`` Server-Sent Events whose ``output`` field carries the
assistant message (cumulative). This client converts the stream into text
deltas and yields them to the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator, Optional

import aiohttp

logger = logging.getLogger("bridge.qwenpaw")


class QwenPawError(RuntimeError):
    pass


class QwenPawClient:
    def __init__(self, base_url: str, agent_id: str, channel: str, token: str = "") -> None:
        self._chat_url = f"{base_url.rstrip('/')}/api/console/chat"
        self._agent_id = agent_id
        self._channel = channel
        self._token = token
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=600)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def chat_stream(
        self,
        text: str,
        session_id: str,
        user_id: str,
    ) -> AsyncIterator[str]:
        """Send one user message and yield assistant text deltas."""
        headers = {"Content-Type": "application/json", "X-Agent-Id": self._agent_id}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        payload = {
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": text}],
                }
            ],
            "session_id": session_id,
            "user_id": user_id,
            "channel": self._channel,
        }

        session = await self._get_session()
        try:
            async with session.post(self._chat_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    raise QwenPawError(f"QwenPaw HTTP {resp.status}: {body}")
                prev_text = ""
                async for event in self._iter_sse(resp):
                    status = event.get("status", "")
                    if status == "failed":
                        error = event.get("error") or {}
                        message = (
                            error.get("message") if isinstance(error, dict) else str(error)
                        ) or "agent failed"
                        raise QwenPawError(f"QwenPaw: {message}")
                    current = _extract_text(event)
                    if current.startswith(prev_text):
                        # Cumulative output: yield only the new suffix.
                        delta = current[len(prev_text):]
                        prev_text = current
                    else:
                        # Delta-style or replaced output: append it all.
                        delta = current
                        prev_text = prev_text + current
                    if delta:
                        yield delta
                    if status == "completed":
                        return
        except aiohttp.ClientError as exc:
            raise QwenPawError(f"无法连接 QwenPaw ({exc})") from exc
        except asyncio.TimeoutError as exc:
            raise QwenPawError("QwenPaw 响应超时") from exc

    @staticmethod
    async def _iter_sse(resp: aiohttp.ClientResponse) -> AsyncIterator[dict]:
        """Parse an SSE body into JSON events."""
        buffer = b""
        async for chunk in resp.content.iter_any():
            buffer += chunk
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                line = line.strip(b"\r")
                if not line or line.startswith(b":"):
                    continue
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if not data or data == b"[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    logger.warning("SSE event is not JSON: %s", data[:200])

    async def ping(self) -> bool:
        """Best-effort connectivity check (used at startup)."""
        try:
            session = await self._get_session()
            url = self._chat_url.rsplit("/api/", 1)[0]
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status < 500
        except Exception:  # noqa: BLE001
            return False


def _extract_text(event: dict) -> str:
    """Concatenate assistant text parts from a response event."""
    output = event.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("role") != "assistant":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)
