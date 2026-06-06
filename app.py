#!/usr/bin/env python3
"""
AI Agent 无人机操控 Demo —— FastAPI 入口

启动: uvicorn app:app --reload --port 8000

API:
  POST /api/agent/chat       聊天接口（自然语言 → Agent 执行 → 返回结果）
  GET  /api/agent/status     查询无人机当前状态
  GET  /api/agent/health     健康检查

生产化路径（面试用）:
此 Demo 验证可行性后，FastAPI 服务独立部署（Python 进程），
通过 requests 调 wvp-server (Java 8) 的 REST 接口下发 MQTT 指令。
两层之间走内部 HTTP，单机部署两个进程，JDK 版本问题不存在。

面试话术:
"不是微服务架构，而是 JDK 8 硬约束下的务实选择。
Python Agent 层做 LLM 编排，Java 控制层做设备通信，两层 REST 解耦。"
"""
import time
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import llm_config, drone_config
from agent import create_agent
from tools import set_confirm_handler, set_notify_handler, executor

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-api")

# ── FastAPI 应用 ──────────────────────────────────

app = FastAPI(
    title="AI Agent 无人机操控",
    description="LangChain + DeepSeek Agent，自然语言控制无人机",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API 模式下自动确认（Demo 简化，生产应有独立确认流程）──

set_confirm_handler(
    lambda prompt: True  # API 模式自动确认，生产环境需改为独立确认流程
)
set_notify_handler(
    lambda event_type, data: logger.info(
        "📢 [WS通知] event=%s data=%s (生产环境: Python→Java HTTP→Java WS→前端)", event_type, data
    )
)
logger.warning("⚠️ API 模式：高风险操作将自动确认（Demo 行为，生产需改为独立确认流程）")

# ── 全局 Agent 实例（启动时创建，复用） ──────────

agent = create_agent()
logger.info("✅ Agent 已就绪: %s @ %s", llm_config.model, llm_config.base_url)


# ── 数据模型 ──────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="自然语言指令",
        examples=["飞到 (31.03, 121.44) 高度 80m，拍照 3 张，然后返航"],
    )


class ChatResponse(BaseModel):
    success: bool
    output: str
    elapsed_seconds: float


class StatusResponse(BaseModel):
    device_id: str
    status: str


# ── API 路由 ──────────────────────────────────────

@app.get("/api/agent/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "llm_model": llm_config.model,
        "device_id": drone_config.device_id,
    }


@app.get("/api/agent/status", response_model=StatusResponse)
def get_status():
    """查询无人机当前状态"""
    return StatusResponse(
        device_id=drone_config.device_id,
        status=executor.get_status(),
    )


@app.post("/api/agent/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    自然语言操控无人机。

    示例请求体:
    {
      "message": "飞到 (31.03, 121.44) 高度 80m，拍照 3 张，然后返航"
    }

    Agent 自动将自然语言拆解为有序操作步骤，
    逐条经过 SafetyGate 校验后模拟 MQTT 下发执行。
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="指令不能为空")

    logger.info("💬 收到指令: %s", request.message)

    try:
        start = time.time()
        result = agent.invoke({"input": request.message})
        elapsed = time.time() - start

        logger.info("✅ 执行完成: %.1fs", elapsed)
        return ChatResponse(
            success=True,
            output=result["output"],
            elapsed_seconds=round(elapsed, 1),
        )

    except Exception as e:
        logger.error("❌ 执行失败: %s", e)
        raise HTTPException(status_code=500, detail=f"Agent 执行异常: {str(e)}")


# ── 启动入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"""
╔════════════════════════════════════════════════╗
║  🚁 AI Agent 无人机操控 — FastAPI 模式         ║
║                                                ║
║  Swagger 文档: http://localhost:8000/docs       ║
║  POST /api/agent/chat  — 自然语言操控无人机     ║
║  GET  /api/agent/status — 查询设备状态          ║
║  GET  /api/agent/health — 健康检查              ║
╚════════════════════════════════════════════════╝
设备: {drone_config.device_id} | LLM: {llm_config.model}
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)
