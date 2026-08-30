"""Mocked end-to-end test of the bridge pipeline.

Runs the bridge in-process with:
- a mock QwenPaw server (real SSE over HTTP),
- fake DashScope ASR/TTS classes (monkeypatched),
- a fake ESP32 device speaking the xiaozhi WebSocket protocol.

Usage:  python test_mock_e2e.py
"""

from __future__ import annotations

import asyncio
import json
import sys

import aiohttp
from aiohttp import web

import bridge.asr  # noqa: F401  (ensure import for patching)
import bridge.session as session_mod
import bridge.tts as tts_mod
from bridge.config import Config
from bridge.opus_codec import OpusEncoder, build_server_binary, parse_device_binary
from bridge.server import create_app

BRIDGE_PORT = 18000
QWENPAW_PORT = 18088
REPLY = "你好！我是QwenPaw。很高兴认识你。"
REASONING = "这是思考过程，不应该被播出来。"

# ---------------------------------------------------------------- mock ASR


class FakeAsrStream:
    """Simulates paraformer: partials per frame, one final sentence."""

    def __init__(self, model, on_sentence, sample_rate=16000):
        self._on_sentence = on_sentence
        self._loop = asyncio.get_running_loop()
        self._frames = 0
        self.final_text = "你好"

    def start_sync(self):
        pass

    def feed_sync(self, pcm):
        self._frames += 1
        if self._frames == 1:
            self._loop.call_soon_threadsafe(
                self._on_sentence, {"text": "你", "is_end": False}
            )
        elif self._frames >= 3:
            self._loop.call_soon_threadsafe(
                self._on_sentence, {"text": "你好", "is_end": True}
            )

    def stop_sync(self):
        pass


# ---------------------------------------------------------------- mock TTS


class FakeSynthesizer:
    """Simulates CosyVoice streaming synthesis: emits silence PCM."""

    def __init__(self, model, voice, format, callback):
        self._callback = callback
        self._loop = asyncio.get_running_loop()

    def streaming_call(self, text):
        pass

    def streaming_complete(self):
        pcm = b"\x00" * 2880 * 3  # three 60 ms frames @ 24 kHz
        self._loop.call_soon_threadsafe(self._callback.on_data, pcm)

    def streaming_cancel(self):
        pass


# ------------------------------------------------------------- mock QwenPaw


async def mock_qwenpaw_chat(request: web.Request) -> web.Response:
    body = await request.json()
    assert request.headers.get("X-Agent-Id") == "default"
    assert body["input"][0]["content"][0]["text"] == "你好", body
    assert body["channel"] == "console"
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    def sse(event: dict) -> bytes:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()

    resp_id = "response_test"
    reason_id = "msg_reasoning"
    reply_id = "msg_reply"
    seq = 0

    # Mirror the real AgentScope Runtime SSE shape: a reasoning message whose
    # text deltas must be filtered out, followed by the final message.
    seq += 1
    await resp.write(
        sse({"id": resp_id, "object": "response", "status": "created", "output": [], "sequence_number": seq})
    )
    seq += 1
    await resp.write(
        sse({"id": reason_id, "type": "reasoning", "role": "assistant", "content": [],
             "status": "in_progress", "object": "message", "sequence_number": seq})
    )
    for ch in REASONING:
        seq += 1
        await resp.write(
            sse({"type": "text", "delta": True, "index": 0, "object": "content",
                 "msg_id": reason_id, "text": ch, "sequence_number": seq})
        )
    seq += 1
    await resp.write(
        sse({"type": "text", "delta": False, "index": 0, "object": "content",
             "msg_id": reason_id, "text": REASONING, "sequence_number": seq})
    )
    seq += 1
    await resp.write(
        sse({"id": reason_id, "type": "reasoning", "role": "assistant",
             "content": [{"type": "text", "text": REASONING}],
             "status": "completed", "object": "message", "sequence_number": seq})
    )

    seq += 1
    await resp.write(
        sse({"id": reply_id, "type": "message", "role": "assistant", "content": [],
             "status": "in_progress", "object": "message", "sequence_number": seq})
    )
    for ch in REPLY:
        seq += 1
        await resp.write(
            sse({"type": "text", "delta": True, "index": 0, "object": "content",
                 "msg_id": reply_id, "text": ch, "sequence_number": seq})
        )
        await asyncio.sleep(0.005)
    seq += 1
    await resp.write(
        sse({"type": "text", "delta": False, "index": 0, "object": "content",
             "msg_id": reply_id, "text": REPLY, "sequence_number": seq})
    )
    seq += 1
    await resp.write(
        sse({"id": reply_id, "type": "message", "role": "assistant",
             "content": [{"type": "text", "text": REPLY}],
             "status": "completed", "object": "message", "sequence_number": seq})
    )
    seq += 1
    await resp.write(
        sse({"id": resp_id, "object": "response", "status": "completed",
             "output": [
                 {"id": reason_id, "type": "reasoning", "role": "assistant",
                  "content": [{"type": "text", "text": REASONING}]},
                 {"id": reply_id, "type": "message", "role": "assistant",
                  "content": [{"type": "text", "text": REPLY}]},
             ],
             "sequence_number": seq})
    )
    return resp


async def start_mock_qwenpaw() -> web.AppRunner:
    app = web.Application()
    app.router.add_post("/api/console/chat", mock_qwenpaw_chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", QWENPAW_PORT)
    await site.start()
    return runner


# --------------------------------------------------------------- fake device


class FakeDevice:
    def __init__(self, ws):
        self.ws = ws
        self.encoder = OpusEncoder(16000)
        self.got_hello = None
        self.stt_texts: list[str] = []
        self.tts_started = False
        self.tts_stopped = False
        self.sentence_starts: list[str] = []
        self.audio_frames = 0
        self.mcp_requests: dict[int, dict] = {}
        self._tts_stopped = asyncio.Event()
        self._reader: asyncio.Task | None = None

    async def run(self) -> None:
        await self.ws.send_json(
            {
                "type": "hello",
                "version": 3,
                "features": {"mcp": True},
                "transport": "websocket",
                "audio_params": {
                    "format": "opus",
                    "sample_rate": 16000,
                    "channels": 1,
                    "frame_duration": 60,
                },
            }
        )
        # Start listening and stream 3 opus frames of silence.
        await self.ws.send_json({"type": "listen", "state": "start", "mode": "auto"})
        for _ in range(3):
            opus = self.encoder.encode(b"\x00" * 1920)
            await self.ws.send_bytes(build_server_binary(opus, 3))

        # Keep reading in the background so MCP requests that arrive after
        # the tts stop (e.g. HTTP-triggered tools/call) are still answered.
        self._reader = asyncio.create_task(self._read_loop())
        await self._tts_stopped.wait()

    async def close(self) -> None:
        await self.ws.close()
        if self._reader is not None:
            try:
                await self._reader
            except Exception:  # noqa: BLE001
                pass

    async def _read_loop(self) -> None:
        async for msg in self.ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                await self._on_json(json.loads(msg.data))
            elif msg.type == aiohttp.WSMsgType.BINARY:
                payload = parse_device_binary(msg.data, 3)
                assert payload, "server sent empty/unframed audio"
                self.audio_frames += 1
            else:
                return

    async def _on_json(self, data: dict) -> None:
        mtype = data.get("type")
        if mtype == "hello":
            self.got_hello = data
        elif mtype == "stt":
            self.stt_texts.append(data.get("text", ""))
        elif mtype == "tts":
            state = data.get("state")
            if state == "start":
                self.tts_started = True
            elif state == "stop":
                self.tts_stopped = True
                self._tts_stopped.set()
            elif state == "sentence_start":
                self.sentence_starts.append(data.get("text", ""))
        elif mtype == "mcp":
            payload = data["payload"]
            await self._on_mcp(payload)

    async def _on_mcp(self, payload: dict) -> None:
        method = payload.get("method")
        if method is None:  # response to our call
            return
        request_id = payload.get("id")
        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake-device", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "self.get_device_status",
                        "description": "Get device status",
                        "inputSchema": {"type": "object", "properties": {}},
                    }
                ],
                "nextCursor": "",
            }
        elif method == "tools/call":
            result = {
                "content": [{"type": "text", "text": "ok"}],
                "isError": False,
            }
        else:
            result = {}
        await self.ws.send_json(
            {
                "type": "mcp",
                "payload": {"jsonrpc": "2.0", "id": request_id, "result": result},
            }
        )


# --------------------------------------------------------------------- test


async def main() -> int:
    # Patch in the fakes.
    session_mod.AsrStream = FakeAsrStream
    tts_mod.SpeechSynthesizer = FakeSynthesizer

    qwenpaw_runner = await start_mock_qwenpaw()

    cfg = Config(
        host="127.0.0.1",
        port=BRIDGE_PORT,
        token="",
        qwenpaw_base_url=f"http://127.0.0.1:{QWENPAW_PORT}",
        qwenpaw_agent_id="default",
        qwenpaw_api_token="",
        qwenpaw_channel="console",
        dashscope_api_key="sk-fake",
        asr_model="paraformer-realtime-v2",
        tts_model="cosyvoice-v2",
        tts_voice="longxiaochun",
        tts_sample_rate=24000,
        utterance_silence=0.3,
        mcp_timeout=10.0,
        log_level="WARNING",
    )
    app = create_app(cfg)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", BRIDGE_PORT)
    await site.start()

    failures: list[str] = []
    try:
        async with aiohttp.ClientSession() as http:
            async with http.ws_connect(
                f"http://127.0.0.1:{BRIDGE_PORT}/xiaozhi/v1/",
                headers={
                    "Device-Id": "test-device-01",
                    "Client-Id": "uuid-1234",
                    "Protocol-Version": "3",
                },
            ) as ws:
                device = FakeDevice(ws)
                await asyncio.wait_for(device.run(), timeout=15)

                # ---- assertions on the voice pipeline
                d = device
                if not d.got_hello or d.got_hello.get("transport") != "websocket":
                    failures.append(f"hello handshake failed: {d.got_hello}")
                if d.got_hello and d.got_hello["audio_params"]["sample_rate"] != 24000:
                    failures.append("server hello audio_params wrong")
                if "你好" not in d.stt_texts:
                    failures.append(f"final stt missing, got: {d.stt_texts}")
                if not d.tts_started or not d.tts_stopped:
                    failures.append("tts start/stop missing")
                if d.audio_frames < 1:
                    failures.append("no binary audio frames received")
                if not any("你好！" in s for s in d.sentence_starts):
                    failures.append(f"sentence_start missing, got: {d.sentence_starts}")

                # ---- MCP tool call over the HTTP management API
                # (must run while the device websocket is still open)
                async with aiohttp.ClientSession() as mgmt:
                    async with mgmt.get(
                        f"http://127.0.0.1:{BRIDGE_PORT}/devices"
                    ) as resp:
                        devices = await resp.json()
                        if not devices or devices[0]["device_id"] != "test-device-01":
                            failures.append(f"/devices wrong: {devices}")
                        elif not devices[0]["mcp_tools"]:
                            failures.append(f"device tools not discovered: {devices}")
                    async with mgmt.post(
                        f"http://127.0.0.1:{BRIDGE_PORT}/devices/test-device-01/tools/call",
                        json={"name": "self.get_device_status", "arguments": {}},
                    ) as resp:
                        result = await resp.json()
                        if resp.status != 200 or result.get("result", {}).get("isError"):
                            failures.append(f"tools/call failed: {resp.status} {result}")
                await device.close()
    finally:
        await runner.cleanup()
        await qwenpaw_runner.cleanup()

    if failures:
        print("E2E TEST FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("E2E TEST PASSED")
    print(f"  hello=ok stt={device.stt_texts!r}")
    print(f"  sentences={device.sentence_starts!r}")
    print(f"  audio_frames={device.audio_frames} tts_start/stop=ok")
    print("  mcp tools/call=ok")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
