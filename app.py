#!/usr/bin/env python3
"""
AI Agent 无人机操控 Demo —— FastAPI 入口（SSE 流式 + 前端聊天框）

启动: uvicorn app:app --reload --port 8000
打开: http://localhost:8000

API:
  POST /api/agent/chat/stream  SSE 流式聊天（实时推送工具调用进度）
  GET  /api/agent/status       查询无人机当前状态
  GET  /api/agent/health       健康检查
"""
import json
import queue
import time
import logging
import asyncio
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler

from config import llm_config, drone_config
from agent import create_agent
from tools import set_confirm_handler, set_notify_handler, executor

# ── 日志 ──────────────────────────────────────────

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

# ── 确认 & 通知回调（API 模式） ──────────────────

set_confirm_handler(lambda prompt: True)
set_notify_handler(
    lambda event_type, data: logger.info(
        "📢 [通知前端] %s: %s", event_type, json.dumps(data, ensure_ascii=False)
    )
)
logger.warning("⚠️ API 模式：高风险操作自动确认（生产环境需独立确认流程）")

# ── Agent 实例 ────────────────────────────────────

agent = create_agent()
logger.info("✅ Agent 就绪: %s @ %s", llm_config.model, llm_config.base_url)


# ── SSE 回调（桥接 LangChain 事件 → SSE 流） ──────

class SSEQueueCallback(BaseCallbackHandler):
    """将 LangChain Agent 的执行事件推入线程安全队列，供 SSE 端点消费。"""

    def __init__(self, q: queue.Queue):
        self.q = q

    def on_agent_action(self, action, **kwargs):
        self.q.put({
            "type": "action",
            "tool": action.tool,
            "input": str(action.tool_input),
        })

    def on_tool_end(self, output, **kwargs):
        self.q.put({
            "type": "tool_result",
            "output": str(output),
        })

    def on_agent_finish(self, finish, **kwargs):
        self.q.put({
            "type": "finish",
            "output": str(finish.return_values.get("output", "")),
        })

    def on_tool_error(self, error, **kwargs):
        self.q.put({
            "type": "error",
            "message": str(error),
        })

    def on_llm_error(self, error, **kwargs):
        self.q.put({
            "type": "error",
            "message": f"LLM 调用异常: {error}",
        })


# ── 数据模型 ──────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(
        ..., description="自然语言指令",
        examples=["飞到 (31.03, 121.44) 高度 80m，录制 60 秒视频，然后返航"],
    )


class StatusResponse(BaseModel):
    device_id: str
    status: str


# ── API 路由 ──────────────────────────────────────

@app.get("/api/agent/health")
def health():
    return {
        "status": "ok",
        "llm_model": llm_config.model,
        "device_id": drone_config.device_id,
    }


@app.get("/api/agent/status", response_model=StatusResponse)
def get_status():
    return StatusResponse(
        device_id=drone_config.device_id,
        status=executor.get_status(),
    )


@app.post("/api/agent/chat")
def chat(request: ChatRequest):
    """
    非流式聊天（向后兼容，前端优先用 /chat/stream）。
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="指令不能为空")

    try:
        start = time.time()
        result = agent.invoke({"input": request.message})
        elapsed = time.time() - start
        return {
            "success": True,
            "output": result["output"],
            "elapsed_seconds": round(elapsed, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/agent/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    SSE 流式聊天 —— 实时推送 Agent 执行的每个步骤。

    事件类型:
      action:     Agent 决定调用工具  {"type":"action","tool":"fly_to_point","input":"..."}
      tool_result: 工具执行完成       {"type":"tool_result","output":"✅ 已到达..."}
      finish:     任务完成            {"type":"finish","output":"任务总结..."}
      error:      执行异常            {"type":"error","message":"..."}
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="指令不能为空")

    event_queue: queue.Queue = queue.Queue()
    callback = SSEQueueCallback(event_queue)

    def run_agent():
        try:
            agent.invoke(
                {"input": request.message},
                {"callbacks": [callback]},
            )
        except Exception as e:
            event_queue.put({"type": "error", "message": str(e)})
        finally:
            event_queue.put(None)  # 哨兵：流结束

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    async def generate():
        while True:
            try:
                event = await asyncio.wait_for(
                    asyncio.to_thread(lambda: event_queue.get(timeout=0.5)),
                    timeout=1.0,
                )
            except (asyncio.TimeoutError, queue.Empty):
                # 心跳注释，保持 SSE 连接活跃（浏览器不触发事件）
                yield ": heartbeat\n\n"
                continue

            if event is None:
                break

            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── 静态文件（前端聊天框） ────────────────────────
# 放在最后，确保 API 路由优先匹配

app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ── 启动入口 ──────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print(f"""
╔════════════════════════════════════════════════╗
║  🚁 AI Agent 无人机操控 Demo                   ║
║                                                ║
║  前端聊天框: http://localhost:8000              ║
║  Swagger  : http://localhost:8000/docs          ║
║  SSE 流式 : POST /api/agent/chat/stream         ║
╚════════════════════════════════════════════════╝
设备: {drone_config.device_id} | LLM: {llm_config.model}
""")
    uvicorn.run(app, host="0.0.0.0", port=8000)
