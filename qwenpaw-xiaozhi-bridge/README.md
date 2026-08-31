# qwenpaw-xiaozhi-bridge

把小智（xiaozhi ESP32）硬件设备接入 **QwenPaw** 的桥接服务。

设备端无需改动原有小智固件的音频/显示/按键逻辑，桥接服务对外实现小智 WebSocket 协议，对内调用 QwenPaw 的对话接口，语音识别/合成使用阿里云百炼（DashScope）免费额度。

## 架构

```
ESP32-S3 (小智固件, Opus 16kHz)
    │  WebSocket (小智协议 v1/v2/v3)
    ▼
qwenpaw-xiaozhi-bridge  (本服务, aiohttp)
    │  ├─ ASR: DashScope paraformer-realtime-v2  (语音→文字)
    │  ├─ LLM: QwenPaw  POST /api/console/chat   (SSE 流式)
    │  └─ TTS: DashScope cosyvoice-v2            (文字→语音, 24kHz Opus 下行)
    │  └─ MCP: 设备工具发现/调用 (JSON-RPC over WebSocket)
    ▼
QwenPaw (自部署, 默认 http://127.0.0.1:8088)
```

一次完整语音交互流程：

1. 设备建立 WebSocket 连接，发送 `hello`（声明支持 MCP）→ 桥接回复 hello 并完成设备工具发现（`initialize` + `tools/list`）。
2. 设备发送 `listen start`，开始上行 Opus 音频帧 → 桥接实时喂给 DashScope ASR，识别中间结果通过 `stt` 消息回显到设备屏幕。
3. 自动模式下，一句话结束 + `UTTERANCE_SILENCE` 秒静音 → 判定说完了 → 最终文字发给 QwenPaw。
4. QwenPaw SSE 返回的回复按句子切流，逐句送 CosyVoice 合成，`tts start` → 二进制 Opus 帧 → `tts stop` 下行到设备播放。
5. 播报过程中设备可随时 `abort`（唤醒词打断）。

## 快速开始（Docker，推荐）

```bash
cd qwenpaw-xiaozhi-bridge
cp .env.example .env
# 编辑 .env：至少填 DASHSCOPE_API_KEY，其余按需调整
docker compose up -d --build

# 验证（千问 Paw 端口 8088 + 1 = 8089）
curl http://127.0.0.1:8089/healthz
```

QwenPaw 运行在宿主机上时，docker-compose 已把 `QWENPAW_BASE_URL` 自动指向 `http://host.docker.internal:8088`。

## 手动运行（Python 3.10+）

需要系统安装 `libopus`（apt: `libopus0` / brew: `opus`）：

```bash
cd qwenpaw-xiaozhi-bridge
pip install -r requirements.txt
cp .env.example .env   # 编辑配置
python main.py
```

## 配置说明

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `0.0.0.0` / `8089` | 监听地址与端口（千问 Paw 8088 + 1） |
| `BRIDGE_TOKEN` | 空 | 设备接入令牌，留空不校验；需与固件 menuconfig 中一致 |
| `QWENPAW_BASE_URL` | `http://127.0.0.1:8088` | QwenPaw 地址（本机 localhost 免鉴权） |
| `QWENPAW_AGENT_ID` | `default` | 控制台左上角选择的智能体 ID |
| `QWENPAW_API_TOKEN` | 空 | QwenPaw 开启 Web 登录且跨机访问时填写 |
| `QWENPAW_CHANNEL` | `console` | 对话使用的通道 |
| `DASHSCOPE_API_KEY` | 必填 | 阿里云百炼 API Key |
| `ASR_MODEL` | `paraformer-realtime-v2` | 流式识别模型 |
| `TTS_MODEL` / `TTS_VOICE` | `cosyvoice-v2` / `longxiaochun` | 合成模型与音色 |
| `TTS_SAMPLE_RATE` | `24000` | 下行采样率（16000/24000） |
| `UTTERANCE_SILENCE` | `0.8` | 自动模式句尾静音判定时长（秒） |
| `MCP_TIMEOUT` | `30` | 设备 MCP 工具调用超时（秒） |

## ESP32 固件配置

固件侧已支持编译期直连（基于小智固件源码新增了 `QwenPaw Bridge Server` 菜单）：

```bash
idf.py menuconfig
# Xiaozhi Assistant → QwenPaw Bridge Server
#   QWENPAW_WS_URL            ws://<服务器IP>:8089/xiaozhi/v1/
#   QWENPAW_WS_TOKEN          与 BRIDGE_TOKEN 一致（可留空）
#   QWENPAW_WS_PROTOCOL_VERSION  3
idf.py build flash monitor
```

配置后固件跳过官方 OTA 检查/激活流程，开机直接连接桥接服务。

> 说明：运行期也可用 `xiaozhi://server/ws/<host>:<port>` 配网方式覆盖 URL；Kconfig 值仅在 Settings 中无值时生效。

## HTTP 管理接口

| 接口 | 说明 |
| --- | --- |
| `GET /` | 服务信息 |
| `GET /healthz` | 健康检查 |
| `GET /devices` | 在线设备列表（含已发现的 MCP 工具名） |
| `GET /devices/{id}/tools` | 某设备全部工具定义 |
| `POST /devices/{id}/tools/call` | 调用设备工具，body: `{"name": "...", "arguments": {...}}` |

示例——查看在线设备并调用设备状态工具：

```bash
curl http://127.0.0.1:8089/devices
curl -X POST http://127.0.0.1:8089/devices/<device-id>/tools/call \
     -H 'Content-Type: application/json' \
     -d '{"name": "self.get_device_status", "arguments": {}}'
```

## 测试

内置一个不依赖外网的全链路模拟测试（mock QwenPaw + mock DashScope + 模拟小智设备）：

```bash
python test_mock_e2e.py
# 预期输出: E2E TEST PASSED
```

覆盖：hello 握手、listen/Opus 上行、ASR 中间/最终结果、QwenPaw SSE 增量解析、分句 TTS、tts start/stop、二进制音频下行、MCP 工具发现与 HTTP 调用。

## 常见问题

- **设备连不上**：先 `curl http://<ip>:8089/healthz` 确认服务可达；固件 URL 必须带路径 `/xiaozhi/v1/`；若设置了 `BRIDGE_TOKEN`，固件侧 token 不一致会 401。
- **ASR/TTS 报错**：检查 `DASHSCOPE_API_KEY` 是否有效、百炼控制台是否开通了对应模型的免费额度。
- **QwenPaw 返回 401/403**：本机访问用 `http://127.0.0.1:8088` 免鉴权；跨机访问需在 QwenPaw 开启 API 访问并填写 `QWENPAW_API_TOKEN`。
- **打断无效/回声**：确认固件使用服务端 AEC 关闭（`CONFIG_USE_SERVER_AEC` 未启用）且唤醒词检测正常，打断依赖设备端发送 `abort`。
- **想换音色/语速**：改 `.env` 中 `TTS_VOICE`（如 `longwan`、`longcheng`）后重启容器。

## 目录结构

```
qwenpaw-xiaozhi-bridge/
├── main.py              # 入口
├── bridge/
│   ├── config.py        # 配置加载
│   ├── server.py        # aiohttp 应用 + WS 端点 + 管理 API
│   ├── session.py       # 每设备会话状态机（协议核心）
│   ├── opus_codec.py    # Opus 编解码 + 二进制帧封装
│   ├── asr.py           # DashScope 流式 ASR
│   ├── tts.py           # DashScope CosyVoice 流式 TTS
│   ├── qwenpaw.py       # QwenPaw REST/SSE 客户端
│   └── mcp.py           # 设备 MCP 客户端（JSON-RPC）
├── test_mock_e2e.py     # 全链路模拟测试
├── Dockerfile / docker-compose.yml
└── .env.example
```
