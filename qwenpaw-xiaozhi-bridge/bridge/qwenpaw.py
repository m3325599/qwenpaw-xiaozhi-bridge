"""QwenPaw REST client: chat via ``POST /api/console/chat`` (SSE stream).

The endpoint follows the AgentScope Runtime protocol: it returns a stream of
``data: {...}`` Server-Sent Events. Assistant text is emitted as incremental
``{"object": "content", "type": "text", "delta": true, "text": ...}`` events,
each tagged with a ``msg_id`` that links back to its parent message. A message
whose ``type`` is ``reasoning`` is the model's thinking and is ignored; only
``message`` content is streamed to the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncIterator, Optional

import aiohttp

logger = logging.getLogger("bridge.qwenpaw")

# Message ``type`` values that are intermediate steps and must never be spoken:
# the model's thinking plus any tool/plugin/MCP activity in a ReAct loop.
_SKIP_MSG_TYPES = {
    "reasoning",
    "thinking",
    "function_call",
    "function_call_output",
    "plugin_call",
    "plugin_call_output",
    "component_call",
    "component_call_output",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_call",
    "mcp_approval_response",
    "mcp_call_output",
    "tool",
    "tool_call",
    "tool_call_output",
}

# Message ``type`` values that carry the final spoken reply. The runtime's
# protocol uses "message", while some console builds put the role ("assistant")
# into the type field; both are treated as speakable.
_SPEAK_MSG_TYPES = {"message", "assistant", "text"}


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
        started = time.monotonic()
        first_delta_logged = False
        try:
            async with session.post(self._chat_url, json=payload, headers=headers) as resp:
                if resp.status != 200:
                    body = (await resp.text())[:300]
                    raise QwenPawError(f"QwenPaw HTTP {resp.status}: {body}")
                # msg_id -> "speak" (final answer) or "skip" (reasoning/tool
                # activity). Content is only yielded for "speak" messages.
                msg_kind: dict[str, str] = {}
                # msg_ids whose incremental deltas have already been yielded.
                streamed: set[str] = set()
                # Whether any speakable text has been emitted this turn.
                spoke = False
                # Count filtered intermediate text so operators can see it works.
                skipped_chars = 0
                async for event in self._iter_sse(resp):
                    obj = event.get("object")
                    etype = event.get("type")
                    status = event.get("status")
                    eid = event.get("id")
                    msg_id = event.get("msg_id")

                    logger.debug(
                        "QwenPaw SSE object=%s type=%s status=%s id=%s msg_id=%s",
                        obj, etype, status, eid, msg_id,
                    )

                    # Classify each message by its type. Reasoning and any
                    # tool/plugin/MCP step are silent; only the final message
                    # text is spoken.
                    if obj == "message" and eid and etype:
                        if etype in _SKIP_MSG_TYPES:
                            msg_kind[eid] = "skip"
                        elif etype in _SPEAK_MSG_TYPES:
                            msg_kind[eid] = "speak"
                        else:
                            # Unknown type: be conservative and keep it silent
                            # rather than risk reading the model's thinking.
                            msg_kind[eid] = "skip"
                        logger.debug(
                            "QwenPaw message id=%s type=%s -> %s",
                            eid, etype, msg_kind[eid],
                        )

                    if status == "failed":
                        error = event.get("error") or {}
                        message = (
                            error.get("message") if isinstance(error, dict) else str(error)
                        ) or "agent failed"
                        raise QwenPawError(f"QwenPaw: {message}")

                    if obj == "content" and etype == "text" and "text" in event:
                        chunk = event.get("text") or ""
                        if msg_kind.get(msg_id) != "speak":
                            # Thinking / tool narration: drop it (and any text
                            # whose parent message type we couldn't resolve).
                            skipped_chars += len(chunk)
                            continue
                        if event.get("delta") is True:
                            # Incremental chunk: stream it immediately.
                            if chunk:
                                if not first_delta_logged:
                                    first_delta_logged = True
                                    logger.info(
                                        "QwenPaw 首个增量 %.1fs（含思考时间）",
                                        time.monotonic() - started,
                                    )
                                streamed.add(msg_id)
                                spoke = True
                                yield chunk
                        elif msg_id not in streamed and chunk:
                            # No incremental deltas arrived (non-streaming
                            # fallback): emit the full text once.
                            if not first_delta_logged:
                                first_delta_logged = True
                                logger.info(
                                    "QwenPaw 首个增量 %.1fs（非流式整体返回）",
                                    time.monotonic() - started,
                                )
                            streamed.add(msg_id)
                            spoke = True
                            yield chunk

                    if obj == "response" and status == "completed":
                        if skipped_chars:
                            logger.info("QwenPaw 已过滤思考/工具内容 %d 字符", skipped_chars)
                        # Safety net: if nothing was streamed (e.g. a build that
                        # only reports the final output in the completed event),
                        # pull the speakable text straight from ``output``.
                        if not spoke:
                            for output_msg in event.get("output") or []:
                                if output_msg.get("type") not in _SPEAK_MSG_TYPES:
                                    continue
                                for part in output_msg.get("content") or []:
                                    text = part.get("text") if isinstance(part, dict) else ""
                                    if text:
                                        spoke = True
                                        yield text
                        logger.info(
                            "QwenPaw 回复完成，总耗时 %.1fs", time.monotonic() - started
                        )
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
