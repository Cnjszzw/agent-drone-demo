# 🚁 AI Agent 无人机操控 Demo

基于 **LangChain + DeepSeek** 的自然语言无人机操控原型验证项目。用户在命令行或通过 HTTP API 输入自然语言指令，Agent 自动拆解为飞行/拍摄/返航等操作步骤，经过安全校验后模拟 MQTT 下发执行。

> 技术预研项目，用于验证"自然语言 → LLM 规划 → 工具调用编排"核心链路的可行性。

## 技术栈

| 层 | 技术 | 说明 |
|---|------|------|
| Agent 框架 | LangChain | `@tool` 装饰器 + `AgentExecutor` ReAct 编排 |
| LLM | DeepSeek | OpenAI 兼容接口 |
| API 服务 | FastAPI + Uvicorn | SSE 流式 + Swagger 自动文档 |
| MCP 外部工具 | langchain-mcp-adapters | 高德地图地理编码（地名→坐标） |
| 语言 | Python 3.11+ | |

## 启动（三步）

```bash
# 1. 进入项目 + 激活虚拟环境
cd /Users/zhaozhiwen/CodeRepo/WVP/agent-demo
source .venv/bin/activate

# 2. 启动服务
uvicorn app:app --reload --port 8000

# 3. 浏览器打开
open http://localhost:8000
```

### 首次安装

```bash
cd /Users/zhaozhiwen/CodeRepo/WVP/agent-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env 填入 DEEPSEEK_API_KEY 和 AMAP_API_KEY
```

### .env 配置

```bash
DEEPSEEK_API_KEY=sk-xxxxxxxx   # 必填，注册: https://platform.deepseek.com
AMAP_API_KEY=xxxxxxxx          # 可选，地名→坐标，不填则仅支持 GPS 坐标输入
```

### PyCharm 设置

`Settings → Project → Python Interpreter → 齿轮 → Add → Existing → 选 agent-demo/.venv/bin/python3.11`

## 使用方式

### 前端聊天框（推荐）

打开 http://localhost:8000，在聊天框中输入指令。Agent 执行过程通过 **SSE（Server-Sent Events）流式推送** 实时展示每个工具调用和结果。

快捷按钮预设了常见指令，点击即可发送。

### API 接口

**SSE 流式聊天（推荐）**

```bash
curl -X POST http://localhost:8000/api/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "飞到 (31.03, 121.44) 高度 80m，拍照 3 张"}' \
  --no-buffer
```

SSE 事件流：
```
data: {"type":"action","tool":"fly_to_point","input":"..."}
data: {"type":"tool_result","output":"✅ 已到达目标点..."}
data: {"type":"action","tool":"take_photo","input":"..."}
data: {"type":"tool_result","output":"✅ 连拍完成，共 3 张照片"}
data: {"type":"finish","output":"任务完成！..."}
```

**非流式聊天（兼容）**

```bash
curl -X POST http://localhost:8000/api/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查询当前状态"}'
```

**状态查询**

```bash
curl http://localhost:8000/api/agent/status
```

## 支持的指令

| 指令示例 | 调用的工具 |
|---------|-----------|
| `飞到 (31.03, 121.44) 高度 80m` | `fly_to_point` |
| `拍照 3 张` | `take_photo` |
| `开始录像 30 秒` | `start_recording` |
| `停止录像` | `stop_recording` |
| `返航` | `return_home` |
| `云台向下` / `云台回中` | `gimbal_control` |
| `飞机现在什么状态` | `get_drone_status` |
| `飞到 (31.03, 121.44) 高度 80m，录像 60s，返航` | 复合指令，LLM 自动拆解编排 |

## 项目结构

```
agent-demo/
├── app.py               # FastAPI 服务（SSE 流式 + API + 静态文件）
├── static/index.html    # 前端聊天框（单页 HTML，SSE 流式展示）
├── agent.py             # Agent 工厂
├── tools.py             # 7 个 LangChain @tool 无人机工具
├── safety.py            # SafetyGate 硬编码安全校验
├── executor.py          # MockExecutor 模拟 MQTT 下发
├── config.py            # 配置加载（.env → 全局变量）
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板
├── agent-demo-plan.md   # 完整设计文档（含面试叙事）
└── README.md            # 本文件
```

## 工具与 WVP 功能对应

| 工具函数 | WVP Java 实现 | MQTT Topic |
|---------|-------------|------------|
| `fly_to_point` | `DrcController` | `dji/device/{sn}/control/fly` |
| `start_recording` | `CameraRecordingStartImpl` | `dji/device/{sn}/camera/record/start` |
| `stop_recording` | `CameraRecordingStopImpl` | `dji/device/{sn}/camera/record/stop` |
| `take_photo` | `CameraPhotoTakeImpl` | `dji/device/{sn}/camera/photo` |
| `return_home` | `DockController` | `dji/device/{sn}/control/return_home` |
| `gimbal_control` | `GimbalResetImpl` | `dji/device/{sn}/gimbal/control` |
| `get_drone_status` | manage 模块 OSD 遥测 | — |

## 架构说明

```
用户输入（自然语言）
    │
    ▼
┌─────────────────────────┐
│  FastAPI / CLI          │  ← 接口层
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  LangChain AgentExecutor│  ← Agent 编排层
│  · ReAct 循环            │     LLM 意图理解 → Tool Calling
│  · 自动 Schema 生成      │     → 结果回传 → 下一步决策
│  · 容错重试              │
└───────────┬─────────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌──────────┐  ┌──────────────┐
│ SafetyGate│  │ MockExecutor │  ← 业务层
│ 硬编码规则 │  │ 模拟 MQTT 下发│     安全不经过 LLM
└──────────┘  └──────────────┘
```

核心原则：**LLM 负责意图理解，SafetyGate 负责安全决策。两个角色绝不混淆。**

## 常见问题

**Q: 提示 `❌ 执行失败` 怎么办？**

检查 `.env` 中 `DEEPSEEK_API_KEY` 是否正确，网络是否可达 `api.deepseek.com`。

**Q: 如何调试 Agent 推理过程？**

打开 `agent.py`，将 `create_agent(verbose=False)` 改为 `verbose=True`，终端会打印完整的 LLM 推理链。

**Q: 飞行指令为什么需要按 y/n 确认？**

这是 Human-in-the-loop 安全机制。真实无人机系统中，起飞/降落等高风险操作必须在 LLM 规划后、实际执行前插入人工确认。CLI 模式用终端交互，API 模式（Demo）自动确认并记录警告日志——生产环境应改为独立的确认流程（前端弹出确认卡片 → 用户确认 → 回调确认接口）。

## 相关文档

- [完整设计文档 & 面试叙事](agent-demo-plan.md)
- [DeepSeek API 文档](https://platform.deepseek.com/api-docs)
- [LangChain 文档](https://python.langchain.com/docs/introduction/)
