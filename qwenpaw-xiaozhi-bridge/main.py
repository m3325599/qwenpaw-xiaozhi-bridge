"""Entry point of the QwenPaw <-> Xiaozhi bridge."""

from __future__ import annotations

import logging
import os
import sys

from aiohttp import web

from bridge.config import Config
from bridge.server import create_app


def main() -> None:
    cfg = Config.from_env()

    logging.basicConfig(
        level=getattr(logging, cfg.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    errors = cfg.validate()
    if errors:
        for error in errors:
            print(f"配置错误: {error}", file=sys.stderr)
        sys.exit(1)

    if cfg.dashscope_api_key:
        # The dashscope SDK reads the key from the environment.
        os.environ["DASHSCOPE_API_KEY"] = cfg.dashscope_api_key

    app = create_app(cfg)

    print("=" * 60)
    print(" QwenPaw <-> 小智(ESP32) 桥接服务")
    print(f"   监听地址      : ws://{cfg.host}:{cfg.port}/xiaozhi/v1/")
    print(f"   QwenPaw       : {cfg.qwenpaw_base_url} (agent: {cfg.qwenpaw_agent_id})")
    print(f"   ASR / TTS     : {cfg.asr_model} / {cfg.tts_model}({cfg.tts_voice})")
    print(f"   设备接入令牌  : {'已启用' if cfg.token else '未启用(不校验)'}")
    print("=" * 60)

    web.run_app(app, host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()
