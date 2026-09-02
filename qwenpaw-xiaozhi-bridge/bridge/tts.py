"""TTS backends wrapped into a xiaozhi TTS turn.

Configured via ``TTS_PROVIDER``:

- ``edge`` (default): Microsoft edge-tts. Free, no API key, no quota.
  Each sentence is synthesized into MP3 and decoded to PCM locally.
- ``siliconflow``: SiliconFlow CosyVoice2 (OpenAI-compatible speech API).
- ``piper``: local Piper voice (onnx), fully offline.
- ``melo``: local MeloTTS via sherpa-onnx, fully offline.
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
import os
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

# Providers that synthesize a full sentence to PCM before streaming.
_NON_STREAMING_PROVIDERS = {"edge", "siliconflow", "piper", "melo"}

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


async def _siliconflow_tts_pcm(text: str, voice: str, sample_rate: int, api_key: str) -> bytes:
    """Synthesize one sentence via SiliconFlow TTS (OpenAI-compatible speech API).

    Uses ``response_format=pcm`` and ``stream=false`` so the response body is a
    single raw PCM16 mono blob, matching the bridge's downstream expectations.
    """
    import aiohttp

    url = "https://api.siliconflow.cn/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "FunAudioLLM/CosyVoice2-0.5B",
        "input": text,
        "voice": voice,
        "response_format": "pcm",
        "sample_rate": sample_rate,
        "stream": False,
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"硅基流动 TTS 失败 ({resp.status}): {body}")
            data = await resp.read()
    if not data:
        raise RuntimeError("硅基流动 TTS 返回空音频")
    return data


# Local TTS models are loaded once and cached: loading a Piper voice or a
# sherpa-onnx OfflineTts is expensive (model file I/O + runtime init).
_PIPER_CACHE: dict[str, object] = {}
_MELO_CACHE: dict[str, object] = {}


def _resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolate 16-bit mono PCM to a different sample rate.

    Piper / MeloTTS synthesize at their native sample rate (typically 22050 Hz);
    the downlink Opus encoder expects ``TTS_SAMPLE_RATE``, so rescale here.
    """
    if src_rate == dst_rate:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float64)
    if samples.size == 0:
        return pcm
    n_out = max(1, int(round(samples.size * dst_rate / src_rate)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.int16).tobytes()


def _load_piper_voice(model_path: str):
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise RuntimeError(
            "未安装 piper-tts，请执行 `pip install piper-tts`（仅 TTS_PROVIDER=piper 需要）"
        ) from exc
    voice = _PIPER_CACHE.get(model_path)
    if voice is None:
        voice = PiperVoice.load(model_path)
        _PIPER_CACHE[model_path] = voice
    return voice


def _piper_tts_pcm(text: str, model_path: str, sample_rate: int) -> bytes:
    """Synthesize one sentence with a local Piper voice into PCM16 mono."""
    voice = _load_piper_voice(model_path)
    raw = bytearray()
    for chunk in voice.synthesize_stream_raw(text):
        raw.extend(chunk)
    if not raw:
        raise RuntimeError("Piper 返回空音频")
    return _resample_pcm(bytes(raw), int(voice.config.sample_rate), sample_rate)


def _load_melo_tts(model_dir: str):
    try:
        import sherpa_onnx
    except ImportError as exc:
        raise RuntimeError(
            "未安装 sherpa-onnx，请执行 `pip install sherpa-onnx`（仅 TTS_PROVIDER=melo 需要）"
        ) from exc
    tts = _MELO_CACHE.get(model_dir)
    if tts is not None:
        return tts

    model = os.path.join(model_dir, "model.onnx")
    tokens = os.path.join(model_dir, "tokens.txt")
    lexicon = os.path.join(model_dir, "lexicon.txt")
    dict_dir = os.path.join(model_dir, "dict")
    for path, name in (
        (model, "model.onnx"),
        (tokens, "tokens.txt"),
        (lexicon, "lexicon.txt"),
    ):
        if not os.path.isfile(path):
            raise RuntimeError(f"缺少 MeloTTS 模型文件 {name}: {path}")

    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=model, tokens=tokens, lexicon=lexicon, dict_dir=dict_dir
    )
    model_cfg = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits, num_threads=2, provider="cpu"
    )
    cfg_kwargs: dict = {"model": model_cfg, "max_num_sentences": 1}
    rule_fsts = [
        os.path.join(model_dir, name)
        for name in ("phone.fst", "date.fst", "number.fst")
        if os.path.isfile(os.path.join(model_dir, name))
    ]
    if rule_fsts:
        cfg_kwargs["rule_fsts"] = ",".join(rule_fsts)
    tts = sherpa_onnx.OfflineTts(config=sherpa_onnx.OfflineTtsConfig(**cfg_kwargs))
    _MELO_CACHE[model_dir] = tts
    return tts


def _melo_tts_pcm(
    text: str, model_dir: str, speaker_id: int, speed: float, sample_rate: int
) -> bytes:
    """Synthesize one sentence with a local MeloTTS (sherpa-onnx) model."""
    import numpy as np

    tts = _load_melo_tts(model_dir)
    audio = tts.generate(text, sid=speaker_id, speed=speed)
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError("MeloTTS 返回空音频")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
    return _resample_pcm(pcm, int(audio.sample_rate), sample_rate)


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
        if self._cfg.tts_provider in _NON_STREAMING_PROVIDERS:
            await self._speak_non_streaming(text)
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
        if self._cfg.tts_provider not in _NON_STREAMING_PROVIDERS:
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
        if self._cfg.tts_provider in _NON_STREAMING_PROVIDERS:
            await self._speak_non_streaming(first_text)
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

    async def _speak_non_streaming(self, text: str) -> None:
        """Synthesize one whole sentence, then queue its PCM for streaming."""
        await self._send_json({"type": "tts", "state": "sentence_start", "text": text})
        try:
            pcm = await self._synthesize_pcm(text)
        except Exception as exc:  # noqa: BLE001
            self._on_error(f"TTS 合成失败: {exc}")
            return
        if pcm:
            self._on_pcm(pcm)

    async def _synthesize_pcm(self, text: str) -> bytes:
        """Dispatch to the configured non-streaming TTS provider.

        Returns PCM16 mono already at ``tts_sample_rate``. ``edge`` additionally
        falls back to SiliconFlow TTS when Microsoft is unreachable, so one bad
        sentence no longer cuts the whole reply short.
        """
        provider = self._cfg.tts_provider
        sr = self._cfg.tts_sample_rate
        if provider == "edge":
            try:
                return await _edge_tts_pcm(text, self._cfg.edge_tts_voice, sr)
            except Exception as exc:  # noqa: BLE001
                logger.warning("edge-tts 失败，尝试回退 TTS: %s", exc)
                if (
                    self._cfg.tts_fallback_provider == "siliconflow"
                    and self._cfg.siliconflow_api_key
                ):
                    pcm = await _siliconflow_tts_pcm(
                        text,
                        self._cfg.siliconflow_tts_voice,
                        sr,
                        self._cfg.siliconflow_api_key,
                    )
                    logger.info("edge-tts 已回退到硅基流动 TTS")
                    return pcm
                raise
        if provider == "siliconflow":
            return await _siliconflow_tts_pcm(
                text, self._cfg.siliconflow_tts_voice, sr, self._cfg.siliconflow_api_key
            )
        if provider == "piper":
            return await self._loop.run_in_executor(
                self._executor, _piper_tts_pcm, text, self._cfg.piper_model_path, sr
            )
        if provider == "melo":
            return await self._loop.run_in_executor(
                self._executor,
                _melo_tts_pcm,
                text,
                self._cfg.melo_model_dir,
                self._cfg.melo_speaker_id,
                self._cfg.melo_speed,
                sr,
            )
        raise RuntimeError(f"未知的 TTS provider: {provider}")

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
