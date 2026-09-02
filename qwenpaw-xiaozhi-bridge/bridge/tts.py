"""TTS backends wrapped into a xiaozhi TTS turn.

Two providers are supported:

- ``edge`` (default): Microsoft edge-tts. Free, no API key, no quota.
  Each sentence is synthesized into MP3 and decoded to PCM locally.
- ``dashscope``: DashScope CosyVoice streaming synthesis.

A :class:`TtsTurn` represents one assistant response:

- ``speak(text)`` submits a sentence (lazily sends the ``tts start`` message
  and starts the synthesizer on first use),
- ``finish()`` completes synthesis and waits until every Opus frame has been
  pushed to the device, then sends ``tts stop``,
- ``abort()`` cancels the turn (user interruption) and also sends ``tts stop``.

PCM is sliced into 60 ms frames, Opus-encoded and sent as binary WebSocket
frames. Downlink frames are paced to real time: the device has a small jitter
buffer, so dumping frames faster than they are played overflows it and the
audio turns into mush. The first ~8 frames are allowed to burst (natural
prebuffer); after that each frame is followed by a 60 ms pacing sleep.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable

from .config import Config
from .opus_codec import FRAME_DURATION_MS, OpusEncoder

logger = logging.getLogger("bridge.tts")

# DashScope import is lazy: the edge provider works without it.
try:  # pragma: no cover - import guard
    from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer
except ImportError:  # noqa: BLE001
    SpeechSynthesizer = None  # type: ignore[assignment]
    AudioFormat = None  # type: ignore[assignment]
    ResultCallback = object  # type: ignore[assignment]

_AUDIO_FORMATS = {
    16000: AudioFormat.PCM_16000HZ_MONO_16BIT if AudioFormat else None,
    24000: AudioFormat.PCM_24000HZ_MONO_16BIT if AudioFormat else None,
}

# JSON / audio senders provided by the session
SendJson = Callable[[dict], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

_SENTINEL = object()

# Pacing: allow this many frames of burst before throttling to real time.
_PREBUFFER_FRAMES = 8

# Strip common markdown noise so the model does not read it out loud.
_MD_RE = re.compile(r"[*#`_~>|]+")
_URL_RE = re.compile(r"https?://\S+")


def clean_for_tts(text: str) -> str:
    text = _URL_RE.sub("链接", text)
    text = _MD_RE.sub("", text)
    return text.strip()


async def _edge_tts_pcm(text: str, voice: str, sample_rate: int, retries: int = 2) -> bytes:
    """Synthesize one sentence via edge-tts and decode to PCM16 mono.

    edge-tts opens a fresh WebSocket to Microsoft per sentence; occasional
    transient failures (rate limiting / flaky network) manifest as "no audio".
    Retry a couple of times before giving up. Returns b"" when nothing could
    be produced after all attempts.
    """
    import edge_tts
    import miniaudio

    last_error: str | None = None
    for attempt in range(retries + 1):
        try:
            communicate = edge_tts.Communicate(text, voice)
            mp3 = bytearray()
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    data = chunk.get("data")
                    if data:
                        mp3.extend(data)
            if mp3:
                sound = miniaudio.decode(
                    bytes(mp3),
                    output_format=miniaudio.SampleFormat.SIGNED16,
                    nchannels=1,
                    sample_rate=sample_rate,
                )
                try:
                    return sound.samples.tobytes()
                except AttributeError:  # older miniaudio: array('h')
                    return bytes(memoryview(sound.samples))
            last_error = "未收到音频数据"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        # Small backoff before retrying; the first failure is often transient.
        if attempt < retries:
            await asyncio.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f"edge-tts 合成失败: {last_error}")


class _TtsCallback(ResultCallback):  # type: ignore[misc]
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_pcm: Callable[[bytes], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._loop = loop
        self._on_pcm = on_pcm
        self._on_error = on_error

    def on_open(self) -> None:
        logger.debug("TTS stream opened")

    def on_complete(self) -> None:
        logger.debug("TTS stream completed")

    def on_error(self, message) -> None:
        try:
            self._loop.call_soon_threadsafe(self._on_error, str(message))
        except RuntimeError:
            pass

    def on_event(self, message) -> None:
        logger.debug("TTS event: %s", message)

    def on_data(self, data: bytes) -> None:
        try:
            self._loop.call_soon_threadsafe(self._on_pcm, data)
        except RuntimeError:
            pass


class TtsTurn:
    """One speaking turn."""

    def __init__(
        self,
        cfg: Config,
        encoder: OpusEncoder,
        executor,
        send_json: SendJson,
        send_audio: SendAudio,
    ) -> None:
        self._cfg = cfg
        self._encoder = encoder
        self._executor = executor
        self._send_json = send_json
        self._send_audio = send_audio
        self._loop = asyncio.get_running_loop()
        self._synthesizer = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._sender_task: asyncio.Task | None = None
        self._started = False
        self._finished = False
        self._error: str | None = None

    # ------------------------------------------------------------------ API

    async def speak(self, text: str) -> None:
        """Speak one piece of text (usually a sentence)."""
        text = clean_for_tts(text)
        if not text:
            return
        if self._error or self._finished:
            return
        if not self._started:
            await self._start(text)
            return
        if self._cfg.tts_provider == "edge":
            await self._edge_speak(text)
        else:
            await self._loop.run_in_executor(
                self._executor, self._synthesizer.streaming_call, text
            )
            await self._send_json({"type": "tts", "state": "sentence_start", "text": text})

    async def finish(self) -> None:
        """Complete the turn and stop speaking."""
        self._finished = True
        if self._error:
            await self._send_json({"type": "tts", "state": "stop"})
            return
        if not self._started:
            # Nothing was spoken (empty reply): still toggle the state machine
            # so the device leaves the speaking state cleanly.
            await self._send_json({"type": "tts", "state": "start"})
            await self._send_json({"type": "tts", "state": "stop"})
            return
        if self._cfg.tts_provider != "edge":
            await self._loop.run_in_executor(
                self._executor, self._synthesizer.streaming_complete
            )
        await self._drain()
        await self._send_json({"type": "tts", "state": "stop"})

    async def abort(self) -> None:
        """Cancel the turn (user interrupt)."""
        self._finished = True
        if self._synthesizer is not None and self._error is None:
            try:
                await self._loop.run_in_executor(
                    self._executor, self._synthesizer.streaming_cancel
                )
            except Exception:  # noqa: BLE001
                pass
        if self._sender_task is not None:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None
        await self._send_json({"type": "tts", "state": "stop"})

    # -------------------------------------------------------------- internals

    async def _start(self, first_text: str) -> None:
        self._started = True
        await self._send_json({"type": "tts", "state": "start"})
        self._sender_task = asyncio.create_task(self._sender())
        if self._cfg.tts_provider == "edge":
            await self._edge_speak(first_text)
            return
        callback = _TtsCallback(self._loop, self._on_pcm, self._on_error)
        try:
            audio_format = _AUDIO_FORMATS[self._cfg.tts_sample_rate]
        except KeyError:
            audio_format = _AUDIO_FORMATS.get(24000)
        self._synthesizer = SpeechSynthesizer(
            model=self._cfg.tts_model,
            voice=self._cfg.tts_voice,
            format=audio_format,
            callback=callback,
        )
        await self._loop.run_in_executor(
            self._executor, self._synthesizer.streaming_call, first_text
        )
        await self._send_json({"type": "tts", "state": "sentence_start", "text": first_text})

    async def _edge_speak(self, text: str) -> None:
        """Synthesize one sentence with edge-tts and queue the PCM."""
        await self._send_json({"type": "tts", "state": "sentence_start", "text": text})
        try:
            pcm = await _edge_tts_pcm(text, self._cfg.edge_tts_voice, self._cfg.tts_sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.error("edge-tts synthesis failed: %s", exc)
            self._on_error(str(exc))
            return
        if pcm:
            self._on_pcm(pcm)

    def _on_pcm(self, data: bytes) -> None:
        # Runs on the event loop (scheduled from the SDK thread or directly).
        self._queue.put_nowait(data)

    def _on_error(self, message: str) -> None:
        # Runs on the event loop.
        if self._error is None:
            self._error = message
            logger.error("TTS error: %s", message)
        self._queue.put_nowait(_SENTINEL)

    async def _drain(self) -> None:
        self._queue.put_nowait(_SENTINEL)
        if self._sender_task is not None:
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None

    async def _sender(self) -> None:
        """Slice queued PCM into 60 ms frames, encode and send.

        Frames are paced to real time so the device playback buffer never
        overflows (this used to produce garbled / merged-up audio).
        """
        frame_bytes = self._encoder.frame_bytes
        frame_sec = FRAME_DURATION_MS / 1000.0
        buf = bytearray()
        finished = False
        sent_frames = 0
        next_send_at: float | None = None
        while True:
            if not finished:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    item = None
                if item is _SENTINEL:
                    finished = True
                    # Drain whatever was queued before the sentinel.
                    while not self._queue.empty():
                        extra = self._queue.get_nowait()
                        if extra is not _SENTINEL:
                            buf.extend(extra)
                elif item is not None:
                    buf.extend(item)
            # Emit all complete frames.
            while len(buf) >= frame_bytes:
                frame = bytes(buf[:frame_bytes])
                del buf[:frame_bytes]
                await self._send_audio(self._encoder.encode(frame))
                sent_frames += 1
                if sent_frames > _PREBUFFER_FRAMES:
                    now = self._loop.time()
                    if next_send_at is None:
                        next_send_at = now + frame_sec
                    else:
                        next_send_at = max(next_send_at, now - frame_sec) + frame_sec
                    delay = next_send_at - now
                    if delay > 0:
                        await asyncio.sleep(delay)
            if finished:
                if buf:
                    # Final partial frame padded with silence.
                    await self._send_audio(self._encoder.encode(bytes(buf)))
                return
