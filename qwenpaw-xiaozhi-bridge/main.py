"""Entry point of the QwenPaw <-> Xiaozhi bridge."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

from aiohttp import web

from bridge.config import Config
from bridge.server import create_app

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "bridge.log")


def _setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # 控制台输出
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    # 每天零点滚动一个日志文件，保留最近 14 天
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", backupCount=14, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _silence_win_proactor_gc() -> None:
    """ Python 3.8 + Windows Proactor 事件循环关闭时，
    WebSocket 传输对象的 __del__ 会向已关闭的 loop 投递回调，
    在 stderr 打出大量 "Event loop is closed" 噪音，屏蔽掉。"""
    if sys.platform == "win32":
        try:
            from asyncio.proactor_events import _ProactorBasePipeTransport

            _ProactorBasePipeTransport.__del__ = lambda self: None  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    cfg = Config.from_env()

    _setup_logging(cfg.log_level)
    _silence_win_proactor_gc()

    errors = cfg.validate()
    if errors:
        for error in errors:
            print(f"配置错误: {error}", file=sys.stderr)
        sys.exit(1)

    if cfg.dashscope_api_key:
        # The dashscope SDK reads the key from the environment.
        os.environ["DASHSCOPE_API_KEY"] = cfg.dashscope_api_key

    app = create_app(cfg)

    if cfg.asr_provider == "transcribe":
        asr_desc = f"{cfg.asr_transcribe_model}（免费转写 API）"
    else:
        asr_desc = f"{cfg.asr_model}（DashScope）"
    if cfg.tts_provider == "edge":
        tts_desc = f"edge-tts/{cfg.edge_tts_voice}（免费）"
    elif cfg.tts_provider == "siliconflow":
        tts_desc = f"硅基流动/{cfg.siliconflow_tts_voice}（免费额度）"
    elif cfg.tts_provider == "piper":
        tts_desc = f"Piper 本地/{cfg.piper_model_path}"
    elif cfg.tts_provider == "melo":
        tts_desc = f"MeloTTS 本地/{cfg.melo_model_dir}（speaker {cfg.melo_speaker_id}）"
    else:
        tts_desc = f"{cfg.tts_model}({cfg.tts_voice})（DashScope）"

    print("=" * 60)
    print(" QwenPaw <-> 小智(ESP32) 桥接服务")
    print(f"   监听地址      : ws://{cfg.host}:{cfg.port}/xiaozhi/v1/")
    print(f"   QwenPaw       : {cfg.qwenpaw_base_url} (agent: {cfg.qwenpaw_agent_id})")
    print(f"   ASR           : {asr_desc}")
    print(f"   TTS           : {tts_desc}")
    print(f"   设备接入令牌  : {'已启用' if cfg.token else '未启用(不校验)'}")
    print("=" * 60)

    web.run_app(app, host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()
