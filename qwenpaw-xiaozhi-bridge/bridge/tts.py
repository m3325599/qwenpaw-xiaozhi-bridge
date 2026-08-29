"""DashScope CosyVoice streaming TTS wrapped into a xiaozhi TTS turn.

A :class:`TtsTurn` represents one assistant response:

- ``speak(text)`` submits a sentence (lazily sends the ``tts start`` message
  and starts the synthesizer on first use),
- ``finish()`` completes synthesis and waits until every Opus frame has been
  pushed to the device, then sends ``tts stop``,
- ``abort()`` cancels the turn (user interruption) and also sends ``tts stop``.

PCM arrives on the DashScope SDK thread; it is queued to the event loop,
sliced into 60 ms frames, Opus-encoded and sent as binary WebSocket frames.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Awaitable, Callable

from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

from .config import Config
from .opus_codec import OpusEncoder

logger = logging.getLogger("bridge.tts")

_AUDIO_FORMATS = {
    16000: AudioFormat.PCM_16000HZ_MONO_16BIT,
    24000: AudioFormat.PCM_24000HZ_MONO_16BIT,
}

# JSON / audio senders provided by the session
SendJson = Callable[[dict], Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

_SENTINEL = object()

# Strip common markdown noise so the model does not read it out loud.
_MD_RE = re.compile(r"[*#`_~>|]+")
_URL_RE = re.compile(r"https?://\S+")


def clean_for_tts(text: str) -> str:
    text = _URL_RE.sub("链接", text)
    text = _MD_RE.sub("", text)
    return text.strip()


class _TtsCallback(ResultCallback):
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
        self._synthesizer: SpeechSynthesizer | None = None
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
        await self._loop.run_in_executor(self._executor, self._synthesizer.streaming_call, text)
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
        await self._loop.run_in_executor(self._executor, self._synthesizer.streaming_complete)
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
        callback = _TtsCallback(self._loop, self._on_pcm, self._on_error)
        try:
            audio_format = _AUDIO_FORMATS[self._cfg.tts_sample_rate]
        except KeyError:
            audio_format = AudioFormat.PCM_24000HZ_MONO_16BIT
        self._synthesizer = SpeechSynthesizer(
            model=self._cfg.tts_model,
            voice=self._cfg.tts_voice,
            format=audio_format,
            callback=callback,
        )
        self._sender_task = asyncio.create_task(self._sender())
        await self._loop.run_in_executor(self._executor, self._synthesizer.streaming_call, first_text)
        await self._send_json({"type": "tts", "state": "sentence_start", "text": first_text})

    def _on_pcm(self, data: bytes) -> None:
        # Runs on the event loop (scheduled from the SDK thread).
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
        """Slice queued PCM into 60 ms frames, encode and send."""
        frame_bytes = self._encoder.frame_bytes
        buf = bytearray()
        finished = False
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
            if finished:
                if buf:
                    # Final partial frame padded with silence.
                    await self._send_audio(self._encoder.encode(bytes(buf)))
                return
