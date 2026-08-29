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

    dashscope_api_key: str
    asr_model: str
    tts_model: str
    tts_voice: str
    tts_sample_rate: int

    utterance_silence: float
    mcp_timeout: float
    log_level: str

    @property
    def chat_url(self) -> str:
        return f"{self.qwenpaw_base_url.rstrip('/')}/api/console/chat"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            host=_get("BRIDGE_HOST", "0.0.0.0"),
            port=_get_int("BRIDGE_PORT", 8000),
            token=_get("BRIDGE_TOKEN"),
            qwenpaw_base_url=_get("QWENPAW_BASE_URL", "http://127.0.0.1:8088"),
            qwenpaw_agent_id=_get("QWENPAW_AGENT_ID", "default"),
            qwenpaw_api_token=_get("QWENPAW_API_TOKEN"),
            qwenpaw_channel=_get("QWENPAW_CHANNEL", "console"),
            dashscope_api_key=_get("DASHSCOPE_API_KEY"),
            asr_model=_get("ASR_MODEL", "paraformer-realtime-v2"),
            tts_model=_get("TTS_MODEL", "cosyvoice-v2"),
            tts_voice=_get("TTS_VOICE", "longxiaochun"),
            tts_sample_rate=_get_int("TTS_SAMPLE_RATE", 24000),
            utterance_silence=_get_float("UTTERANCE_SILENCE", 0.8),
            mcp_timeout=_get_float("MCP_TIMEOUT", 30.0),
            log_level=_get("LOG_LEVEL", "INFO").upper(),
        )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.dashscope_api_key:
            errors.append("DASHSCOPE_API_KEY 未配置（.env 中填写阿里云百炼 API Key）")
        if not self.qwenpaw_base_url.startswith(("http://", "https://")):
            errors.append("QWENPAW_BASE_URL 必须以 http:// 或 https:// 开头")
        if self.tts_sample_rate not in (16000, 24000):
            errors.append("TTS_SAMPLE_RATE 仅支持 16000 / 24000")
        if not 0.1 <= self.utterance_silence <= 5.0:
            errors.append("UTTERANCE_SILENCE 取值范围 0.1 ~ 5.0 秒")
        return errors
