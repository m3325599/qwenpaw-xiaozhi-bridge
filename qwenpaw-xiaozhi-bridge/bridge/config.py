"""Configuration loading from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    raw = _get(key)
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    raw = _get(key)
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


def _normalize_voice(voice: str, model: str) -> str:
    """ cosyvoice-v2/v3 只接受带对应代号后缀的音色名，
    裸 v1 音色名会导致引擎报 418 InvalidParameter，这里自动补后缀。"""
    if model.startswith("cosyvoice-v"):
        version = model[len("cosyvoice-v"):]
        if version.isdigit():
            suffix = f"_v{version}"
            if not voice.endswith(suffix):
                return voice + suffix
    return voice


@dataclass
class Config:
    """Runtime configuration of the bridge."""

    host: str
    port: int
    token: str

    qwenpaw_base_url: str
    qwenpaw_agent_id: str
    qwenpaw_api_token: str
    qwenpaw_channel: str

    # ASR provider: "dashscope" | "transcribe"
    asr_provider: str
    asr_model: str
    asr_transcribe_url: str
    asr_transcribe_model: str
    asr_transcribe_key: str

    # TTS provider: "edge" (free) | "dashscope"
    tts_provider: str
    tts_model: str
    tts_voice: str
    edge_tts_voice: str
    tts_sample_rate: int

    dashscope_api_key: str

    utterance_silence: float
    mcp_timeout: float
    log_level: str

    @property
    def chat_url(self) -> str:
        return f"{self.qwenpaw_base_url.rstrip('/')}/api/console/chat"

    @classmethod
    def from_env(cls) -> "Config":
        # ASR provider resolution: "auto" uses the free OpenAI-compatible
        # transcribe endpoint (e.g. SiliconFlow SenseVoice) when a key is
        # configured, otherwise falls back to DashScope streaming ASR.
        asr_provider = _get("ASR_PROVIDER", "auto").lower()
        asr_transcribe_key = _get("ASR_TRANSCRIBE_KEY")
        if asr_provider == "auto":
            asr_provider = "transcribe" if asr_transcribe_key else "dashscope"

        tts_provider = _get("TTS_PROVIDER", "edge").lower()
        tts_model = _get("TTS_MODEL", "cosyvoice-v2")
        return cls(
            host=_get("BRIDGE_HOST", "0.0.0.0"),
            port=_get_int("BRIDGE_PORT", 8089),
            token=_get("BRIDGE_TOKEN"),
            qwenpaw_base_url=_get("QWENPAW_BASE_URL", "http://127.0.0.1:8088"),
            qwenpaw_agent_id=_get("QWENPAW_AGENT_ID", "default"),
            qwenpaw_api_token=_get("QWENPAW_API_TOKEN"),
            qwenpaw_channel=_get("QWENPAW_CHANNEL", "console"),
            asr_provider=asr_provider,
            asr_model=_get("ASR_MODEL", "paraformer-realtime-v2"),
            asr_transcribe_url=_get(
                "ASR_TRANSCRIBE_URL",
                "https://api.siliconflow.cn/v1/audio/transcriptions",
            ),
            asr_transcribe_model=_get(
                "ASR_TRANSCRIBE_MODEL", "FunAudioLLM/SenseVoiceSmall"
            ),
            asr_transcribe_key=asr_transcribe_key,
            tts_provider=tts_provider,
            tts_model=tts_model,
            tts_voice=_normalize_voice(_get("TTS_VOICE", "longxiaochun_v2"), tts_model),
            edge_tts_voice=_get("EDGE_TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
            tts_sample_rate=_get_int("TTS_SAMPLE_RATE", 24000),
            dashscope_api_key=_get("DASHSCOPE_API_KEY"),
            utterance_silence=_get_float("UTTERANCE_SILENCE", 0.8),
            mcp_timeout=_get_float("MCP_TIMEOUT", 30.0),
            log_level=_get("LOG_LEVEL", "INFO").upper(),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.asr_provider == "dashscope" and not self.dashscope_api_key:
            errors.append(
                "ASR 走 DashScope 但 DASHSCOPE_API_KEY 未配置"
                "（如需免费 ASR，在 .env 中配置 ASR_TRANSCRIBE_KEY）"
            )
        if self.asr_provider == "transcribe" and not self.asr_transcribe_key:
            errors.append("ASR_TRANSCRIBE_KEY 未配置")
        if self.asr_provider not in ("dashscope", "transcribe"):
            errors.append("ASR_PROVIDER 仅支持 dashscope / transcribe / auto")
        if self.tts_provider == "dashscope" and not self.dashscope_api_key:
            errors.append(
                "TTS 走 DashScope 但 DASHSCOPE_API_KEY 未配置"
                "（免费方案：TTS_PROVIDER=edge，无需任何 key）"
            )
        if self.tts_provider not in ("edge", "dashscope"):
            errors.append("TTS_PROVIDER 仅支持 edge / dashscope")
        if not self.qwenpaw_base_url.startswith(("http://", "https://")):
            errors.append("QWENPAW_BASE_URL 必须以 http:// 或 https:// 开头")
        if self.tts_sample_rate not in (16000, 24000):
            errors.append("TTS_SAMPLE_RATE 仅支持 16000 / 24000")
        if not 0.1 <= self.utterance_silence <= 5.0:
            errors.append("UTTERANCE_SILENCE 取值范围 0.1 ~ 5.0 秒")
        return errors
