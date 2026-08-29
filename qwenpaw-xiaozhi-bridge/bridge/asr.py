"""DashScope streaming ASR (paraformer-realtime) wrapper.

One :class:`AsrStream` instance corresponds to one utterance. Audio frames
are pushed from the asyncio event loop into the DashScope SDK (which runs
its own WebSocket thread); sentence events come back on the SDK thread and
are forwarded to the event loop via ``call_soon_threadsafe``.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Callable

from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

logger = logging.getLogger("bridge.asr")

# Sentence event handed to the event loop:
#   {"text": str, "is_end": bool, "final": str}
SentenceHandler = Callable[[dict], None]


class AsrError(RuntimeError):
    pass


class _Callback(RecognitionCallback):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_sentence: SentenceHandler,
        sentences: list[str],
        lock: threading.Lock,
    ) -> None:
        self._loop = loop
        self._on_sentence = on_sentence
        self._sentences = sentences
        self._lock = lock
        self.error: str | None = None

    def _schedule(self, event: dict) -> None:
        try:
            self._loop.call_soon_threadsafe(self._on_sentence, event)
        except RuntimeError:
            pass  # loop closed

    def on_open(self) -> None:
        logger.debug("ASR stream opened")

    def on_complete(self) -> None:
        logger.debug("ASR stream completed")

    def on_error(self, result: RecognitionResult) -> None:
        self.error = str(result)
        logger.error("ASR error: %s", result)

    def on_event(self, result: RecognitionResult) -> None:
        sentence = result.get_sentence()
        if not sentence:
            return
        text = sentence.get("text", "") or ""
        try:
            is_end = bool(RecognitionResult.is_sentence_end(sentence))
        except Exception:
            is_end = False
        if is_end and text:
            with self._lock:
                self._sentences.append(text)
        self._schedule({"text": text, "is_end": is_end})


class AsrStream:
    """A single streaming recognition session (one utterance)."""

    def __init__(
        self,
        model: str,
        on_sentence: SentenceHandler,
        sample_rate: int = 16000,
    ) -> None:
        self._loop = asyncio.get_running_loop()
        self._sentences: list[str] = []
        self._lock = threading.Lock()
        self._closed = False
        self._callback = _Callback(self._loop, on_sentence, self._sentences, self._lock)
        self._recognition = Recognition(
            model=model,
            format="pcm",
            sample_rate=sample_rate,
            callback=self._callback,
        )

    @property
    def final_text(self) -> str:
        """Complete recognized text; valid after :meth:`stop` returns."""
        with self._lock:
            return "".join(self._sentences).strip()

    def start_sync(self) -> None:
        try:
            response = self._recognition.start()
            status = getattr(response, "status_code", 200)
            if status is not None and status != 200:
                raise AsrError(f"ASR start failed, status={status}")
        except AsrError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise AsrError(f"ASR start failed: {exc}") from exc

    def feed_sync(self, pcm: bytes) -> None:
        if self._closed:
            return
        try:
            self._recognition.send_audio_frame(pcm)
        except Exception as exc:  # noqa: BLE001
            logger.debug("ASR feed failed: %s", exc)

    def stop_sync(self) -> None:
        """Finalize; blocks until remaining results are delivered."""
        if self._closed:
            return
        self._closed = True
        try:
            self._recognition.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ASR stop failed: %s", exc)

    @property
    def error(self) -> str | None:
        return self._callback.error
