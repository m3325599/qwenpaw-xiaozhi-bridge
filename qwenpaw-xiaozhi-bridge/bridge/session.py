"""Per-device session: implements the xiaozhi WebSocket protocol.

Pipeline for one voice interaction:

1. Device connects and sends ``hello`` -> bridge replies with its own hello
   (24 kHz Opus downlink) and discovers device MCP tools.
2. ``listen start`` -> start a DashScope streaming ASR session.
3. Binary Opus frames -> decode to PCM -> feed the ASR stream.
   - manual mode: ``listen stop`` finalizes the utterance;
   - auto mode: a recognized sentence followed by ``UTTERANCE_SILENCE``
     seconds of silence finalizes it.
4. Final text -> ``stt`` message to the device -> QwenPaw chat_stream()
   -> text deltas split into sentences -> CosyVoice TTS -> ``tts start``,
   binary Opus frames, ``tts stop``.
5. ``abort`` (wake word while speaking) cancels the current response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from aiohttp import WSMsgType

from .asr import AsrError, AsrStream, TranscribeAsrStream
from .mcp import McpDeviceClient, McpToolError
from .opus_codec import (
    OpusDecoder,
    OpusEncoder,
    build_server_binary,
    parse_device_binary,
)
from .qwenpaw import QwenPawClient, QwenPawError
from .tts import TtsTurn

if TYPE_CHECKING:
    from aiohttp import web

    from .config import Config

logger = logging.getLogger("bridge.session")

# Sentence boundaries for TTS flushing.
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]*[。！？!?；;\n]+")
_HARD_FLUSH_LEN = 100


class DeviceRegistry:
    """Tracks currently connected device sessions."""

    def __init__(self) -> None:
        self._devices: dict[str, "DeviceSession"] = {}

    def add(self, session: "DeviceSession") -> None:
        self._devices[session.device_id] = session

    def remove(self, session: "DeviceSession") -> None:
        if self._devices.get(session.device_id) is session:
            del self._devices[session.device_id]

    def get(self, device_id: str) -> "DeviceSession | None":
        return self._devices.get(device_id)

    def list(self) -> list["DeviceSession"]:
        return list(self._devices.values())


class DeviceSession:
    def __init__(
        self,
        cfg: "Config",
        ws: "web.WebSocketResponse",
        device_id: str,
        client_id: str,
        protocol_version: int,
        qwenpaw: QwenPawClient,
        registry: DeviceRegistry,
    ) -> None:
        self.cfg = cfg
        self.ws = ws
        self.device_id = device_id
        self.client_id = client_id
        self.protocol_version = protocol_version
        self.qwenpaw = qwenpaw
        self.registry = registry
        self.loop = asyncio.get_running_loop()
        self.session_id = str(uuid.uuid4())
        # Persistent QwenPaw conversation per physical device.
        self.qwenpaw_session = f"xiaozhi-{device_id}"
        self.connected_at = time.time()

        self.decoder = OpusDecoder(16000)
        self.encoder = OpusEncoder(cfg.tts_sample_rate)
        # Single worker: keeps ASR feeds / TTS controls strictly ordered and
        # off the event loop.
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio")

        # ASR / utterance state
        self._asr: AsrStream | None = None
        self._asr_sentences: list[str] = []
        self._listen_mode = "auto"
        self._silence_task: asyncio.Task | None = None
        self._asr_error_spoken = False

        # Response state
        self._response_task: asyncio.Task | None = None

        # MCP
        self.mcp = McpDeviceClient(self._send_mcp_payload, cfg.mcp_timeout)

    # ------------------------------------------------------------------ info

    def describe(self) -> dict:
        return {
            "device_id": self.device_id,
            "client_id": self.client_id,
            "session_id": self.session_id,
            "protocol_version": self.protocol_version,
            "connected_at": self.connected_at,
            "mcp_tools": [t.get("name") for t in self.mcp.tools],
        }

    # ------------------------------------------------------------ main loop

    async def run(self) -> None:
        self.registry.add(self)
        logger.info(
            "Device connected: %s (session=%s, protocol v%d)",
            self.device_id,
            self.session_id,
            self.protocol_version,
        )
        try:
            async for msg in self.ws:
                if msg.type == WSMsgType.TEXT:
                    await self._on_text(msg.data)
                elif msg.type == WSMsgType.BINARY:
                    await self._on_binary(msg.data)
                elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE, WSMsgType.CLOSING):
                    break
        finally:
            self.registry.remove(self)
            await self._cancel_response()
            await self._stop_asr()
            self._executor.shutdown(wait=False)
            logger.info("Device disconnected: %s", self.device_id)

    # ------------------------------------------------------------ messaging

    async def _send_json(self, obj: dict) -> None:
        message = dict(obj)
        message.setdefault("session_id", self.session_id)
        try:
            await self.ws.send_str(json.dumps(message, ensure_ascii=False))
        except Exception as exc:  # noqa: BLE001
            logger.debug("send_str failed: %s", exc)

    async def send_audio(self, opus: bytes) -> None:
        frame = build_server_binary(opus, self.protocol_version, int(time.time() * 1000))
        try:
            await self.ws.send_bytes(frame)
        except Exception as exc:  # noqa: BLE001
            logger.debug("send_bytes failed: %s", exc)

    async def _send_mcp_payload(self, payload: dict) -> None:
        await self._send_json({"type": "mcp", "payload": payload})

    # --------------------------------------------------------- device -> us

    async def _on_text(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON from device")
            return
        mtype = message.get("type")
        if mtype == "hello":
            await self._on_hello(message)
        elif mtype == "listen":
            await self._on_listen(message)
        elif mtype == "abort":
            reason = message.get("reason", "")
            logger.info("Abort (reason=%s)", reason)
            await self._cancel_response()
        elif mtype == "mcp":
            await self.mcp.on_payload(message.get("payload") or {})
        elif mtype == "iot":
            logger.debug("Legacy iot message ignored")
        else:
            logger.debug("Unhandled message type: %s", mtype)

    async def _on_hello(self, message: dict) -> None:
        await self._send_json(
            {
                "type": "hello",
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": self.cfg.tts_sample_rate,
                    "channels": 1,
                    "frame_duration": 60,
                },
            }
        )
        features = message.get("features") or {}
        if features.get("mcp"):
            asyncio.create_task(self._setup_mcp())

    async def _setup_mcp(self) -> None:
        try:
            await self.mcp.setup()
        except (asyncio.TimeoutError, McpToolError) as exc:
            logger.warning("MCP setup failed: %s", exc)
        except Exception:  # noqa: BLE001
            logger.exception("MCP setup error")

    async def _on_binary(self, data: bytes) -> None:
        opus = parse_device_binary(data, self.protocol_version)
        if not opus or self._asr is None:
            return
        pcm = self.decoder.decode(opus)
        if not pcm:
            return
        await self.loop.run_in_executor(self._executor, self._asr.feed_sync, pcm)

    # ------------------------------------------------------------------ ASR

    async def _on_listen(self, message: dict) -> None:
        state = message.get("state")
        if state == "start":
            self._listen_mode = message.get("mode", "auto")
            await self._stop_asr()
            await self._start_asr()
        elif state == "stop":
            await self._finish_utterance()
        elif state == "detect":
            logger.info("Wake word detected: %s", message.get("text", ""))
        else:
            logger.debug("Unknown listen state: %s", state)

    async def _start_asr(self) -> None:
        self._asr_sentences = []
        try:
            if self.cfg.asr_provider == "transcribe":
                stream = TranscribeAsrStream(
                    self.cfg.asr_transcribe_url,
                    self.cfg.asr_transcribe_model,
                    self.cfg.asr_transcribe_key,
                    self._on_asr_sentence,
                )
            else:
                stream = AsrStream(self.cfg.asr_model, self._on_asr_sentence)
            await self.loop.run_in_executor(self._executor, stream.start_sync)
            self._asr = stream
        except AsrError as exc:
            logger.error("ASR start failed: %s", exc)
            if not self._asr_error_spoken:
                self._asr_error_spoken = True
                await self._speak_error("语音识别服务连接失败，请检查配置")

    def _on_asr_sentence(self, sentence: dict) -> None:
        """ASR sentence event (already on the event loop)."""
        text = sentence.get("text", "")
        is_end = sentence.get("is_end", False)
        if is_end and text:
            self._asr_sentences.append(text)
            if self._listen_mode != "manual":
                self._arm_silence_timer()
        # Show interim / final transcription on the device display.
        display = "".join(self._asr_sentences) + ("" if is_end else text)
        if display:
            asyncio.ensure_future(self._send_json({"type": "stt", "text": display}))

    def _arm_silence_timer(self) -> None:
        self._disarm_silence_timer()
        self._silence_task = asyncio.create_task(self._silence_timeout())

    def _disarm_silence_timer(self) -> None:
        """Cancel the silence timer, unless we are running inside it."""
        task = self._silence_task
        self._silence_task = None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _silence_timeout(self) -> None:
        try:
            await asyncio.sleep(self.cfg.utterance_silence)
        except asyncio.CancelledError:
            return
        await self._finish_utterance()

    async def _stop_asr(self) -> None:
        """Tear down the ASR stream without triggering a response."""
        self._disarm_silence_timer()
        stream = self._asr
        self._asr = None
        if stream is not None:
            await self.loop.run_in_executor(self._executor, stream.stop_sync)

    async def _finish_utterance(self) -> None:
        """Finalize the utterance and trigger the agent response."""
        self._disarm_silence_timer()
        stream = self._asr
        if stream is None:
            return
        self._asr = None
        await self.loop.run_in_executor(self._executor, stream.stop_sync)
        text = stream.final_text or "".join(self._asr_sentences).strip()
        if not text:
            logger.info("Empty utterance, ignored")
            return
        logger.info("Utterance from %s: %s", self.device_id, text)
        await self._send_json({"type": "stt", "text": text})
        await self._start_response(text)

    # -------------------------------------------------------------- response

    async def _start_response(self, text: str) -> None:
        await self._cancel_response()
        self._response_task = asyncio.create_task(self._respond(text))

    async def _cancel_response(self) -> None:
        task = self._response_task
        self._response_task = None
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                logger.exception("Response task error")

    async def _respond(self, text: str) -> None:
        turn = TtsTurn(self.cfg, self.encoder, self._executor, self._send_json, self.send_audio)
        try:
            await self._send_json({"type": "llm", "emotion": "happy"})
            buffer = ""
            has_reply = False
            async for delta in self.qwenpaw.chat_stream(text, self.qwenpaw_session, self.device_id):
                if not delta:
                    continue
                has_reply = True
                buffer += delta
                # Flush complete sentences to TTS as soon as possible.
                while True:
                    match = _SENTENCE_RE.search(buffer)
                    if not match:
                        break
                    sentence = buffer[: match.end()]
                    buffer = buffer[match.end():]
                    await turn.speak(sentence)
                if len(buffer) >= _HARD_FLUSH_LEN:
                    await turn.speak(buffer)
                    buffer = ""
            if buffer.strip():
                await turn.speak(buffer)
            if has_reply:
                await turn.finish()
            else:
                # No textual reply at all: still toggle tts so the device
                # leaves the speaking state.
                await self._send_json({"type": "tts", "state": "start"})
                await self._send_json({"type": "tts", "state": "stop"})
        except (QwenPawError, asyncio.TimeoutError) as exc:
            logger.error("Response failed: %s", exc)
            await turn.abort()
            await self._speak_error(str(exc))
        except asyncio.CancelledError:
            await turn.abort()
            raise
        except Exception:  # noqa: BLE001
            logger.exception("Unexpected error in response pipeline")
            await turn.abort()
            await self._speak_error("桥接服务内部错误，请查看日志")

    async def _speak_error(self, message: str) -> None:
        """Tell the user something went wrong, using a fresh TTS turn."""
        turn = TtsTurn(self.cfg, self.encoder, self._executor, self._send_json, self.send_audio)
        try:
            await turn.speak(message)
            await turn.finish()
        except Exception:  # noqa: BLE001
            logger.exception("Error while speaking error message")
