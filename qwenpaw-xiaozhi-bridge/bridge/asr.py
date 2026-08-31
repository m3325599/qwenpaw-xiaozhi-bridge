"""ASR backends.

- :class:`AsrStream`: DashScope streaming ASR (paraformer-realtime).
- :class:`TranscribeAsrStream`: free OpenAI-compatible transcription API
  (default: SiliconFlow ``iic/SenseVoiceSmall``, free tier). Audio is
  buffered while the user speaks and POSTed as a WAV once the utterance ends.

Both expose the same interface used by the session:
``start_sync`` / ``feed_sync`` / ``stop_sync`` / ``final_text`` / ``error``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import threading
import urllib.request
import uuid
from typing import Callable

logger = logging.getLogger("bridge.asr")

# DashScope import is lazy: the transcribe backend works without it.
try:  # pragma: no cover - import guard
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
except ImportError:  # noqa: BLE001
    Recognition = None  # type: ignore[assignment]
    RecognitionResult = None  # type: ignore[assignment]

    class RecognitionCallback:  # type: ignore[no-redef]
        pass


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


def _wav_header(data_bytes: int, sample_rate: int) -> bytes:
    """Minimal RIFF/WAVE header for PCM16 mono."""
    return b"RIFF" + struct.pack("<I", 36 + data_bytes) + b"WAVEfmt " + struct.pack(
        "<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16
    ) + b"data" + struct.pack("<I", data_bytes)


class TranscribeAsrStream:
    """Buffered ASR via an OpenAI-compatible /audio/transcriptions endpoint.

    Audio is accumulated locally while the user speaks; ``stop_sync`` uploads
    the whole utterance as a WAV and returns the recognized text. Free
    providers such as SiliconFlow's SenseVoiceSmall fit this pattern.
    """

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str,
        on_sentence: SentenceHandler,
        sample_rate: int = 16000,
    ) -> None:
        self._url = url
        self._model = model
        self._api_key = api_key
        self._sample_rate = sample_rate
        self._on_sentence = on_sentence
        self._loop = asyncio.get_running_loop()
        self._pcm = bytearray()
        self._final_text = ""
        self.error: str | None = None
        self._closed = False

    @property
    def final_text(self) -> str:
        return self._final_text.strip()

    def start_sync(self) -> None:
        pass

    def feed_sync(self, pcm: bytes) -> None:
        if self._closed:
            return
        self._pcm.extend(pcm)

    def stop_sync(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._pcm:
            return
        try:
            self._final_text = self._transcribe(bytes(self._pcm))
        except Exception as exc:  # noqa: BLE001
            self.error = str(exc)
            logger.error("Transcribe failed: %s", exc)
            return
        if self._final_text:
            try:
                self._loop.call_soon_threadsafe(
                    self._on_sentence,
                    {"text": self._final_text, "is_end": True},
                )
            except RuntimeError:
                pass

    def _transcribe(self, pcm: bytes) -> str:
        wav = _wav_header(len(pcm), self._sample_rate) + pcm
        boundary = uuid.uuid4().hex
        model_field = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
            f"{self._model}\r\n"
        ).encode()
        file_field = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
            f'filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode() + wav + b"\r\n"
        body = model_field + file_field + f"--{boundary}--\r\n".encode()
        request = urllib.request.Request(
            self._url,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload.get("text", "") or ""
