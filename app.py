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
import os
import json
import queue
import time
import logging
import asyncio
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.callbacks import BaseCallbackHandler

from config import llm_config, drone_config
from agent import create_agent
from tools import set_confirm_handler, set_notify_handler, executor, ALL_TOOLS
from executor import MockExecutor

# ── 日志 ──────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-api")

# ── 全局状态 ──────────────────────────────────────

agent = None          # 启动时异步初始化
mcp_tools_loaded = 0  # 已加载的 MCP 工具数
_mcp_client = None    # MCP 客户端引用，用于关闭


async def _init_agent():
    """启动时异步初始化 Agent（含 MCP 工具加载）"""
    global agent, mcp_tools_loaded, _mcp_client

    # 尝试加载高德 MCP 工具
    amap_key = os.getenv("AMAP_API_KEY", "")
    if amap_key:
        try:
            from mcp_tools import load_amap_tools
            mcp_tools, _mcp_client = await load_amap_tools(amap_key)
            ALL_TOOLS.extend(mcp_tools)
            mcp_tools_loaded = len(mcp_tools)
            logger.info("✅ 高德 MCP 工具已集成: %d 个", mcp_tools_loaded)
        except Exception as e:
            logger.warning("⚠️ MCP 工具加载失败（将仅用坐标模式）: %s", e)
    else:
        logger.info("ℹ️ 未设置 AMAP_API_KEY，使用坐标模式（用户需提供 GPS 坐标）")

    agent = create_agent()
    logger.info("✅ Agent 就绪: %s @ %s | 工具: %d 个",
                llm_config.model, llm_config.base_url, len(ALL_TOOLS))


async def _shutdown():
    """关闭 MCP 客户端连接"""
    global _mcp_client
    if _mcp_client:
        try:
            await _mcp_client.__aexit__(None, None, None)
        except Exception:
            pass


# ── FastAPI 应用（lifespan 管理启动/关闭） ────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 服务启动中...")
    await _init_agent()
    logger.info("✅ 服务就绪")
    yield
    logger.info("🛑 服务关闭中...")
    await _shutdown()

app = FastAPI(
    title="AI Agent 无人机操控",
    description="LangChain + DeepSeek Agent，自然语言控制无人机",
    version="1.0.0",
    lifespan=lifespan,
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


@app.post("/api/agent/stop")
def emergency_stop():
    """
    紧急停止 —— 不经过 Agent、不经过 LLM。

    生产环境实际链路:
      前端急停按钮 → Java /api/drone/emergency_stop → MQTT 直达无人机
      同时前端关闭当前 SSE 连接，旧 Agent 会话自然终止。

    此端点仅用于 Demo 中通知执行器停止正在进行的模拟任务。
    """
    MockExecutor.trigger_emergency_stop()
    logger.warning("⛔ 紧急停止已触发：无人机原地悬停，所有任务终止")
    return {
        "success": True,
        "message": "紧急停止：无人机已原地悬停，所有任务终止",
    }


@app.post("/api/agent/reset")
def reset_stop():
    """重置紧急停止标志（新会话开始时调用）"""
    MockExecutor.reset_emergency_stop()
    return {"success": True, "message": "停止标志已重置"}


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
        loop = asyncio.get_event_loop()
        while True:
            try:
                # 短 timeout（100ms），事件到达立即推送
                event = await loop.run_in_executor(
                    None, lambda: event_queue.get(timeout=0.1)
                )
            except queue.Empty:
                # 无事件时发心跳保持连接
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


# ── 规划 & 执行（DJI Copilot 风格三段式） ──────────

from plan_executor import generate_plan, execute_plan_stream


@app.post("/api/agent/plan")
def create_plan(request: ChatRequest):
    """
    Phase 1: 生成任务规划（DJI Copilot 风格）。

    返回结构化 JSON: { objective, steps, preflight }
    前端展示"规划确认卡片"——用户审核步骤和飞前检查项后手动点击执行。
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="指令不能为空")

    logger.info("📋 生成规划: %s", request.message)
    try:
        plan = generate_plan(request.message)
        return {"success": True, "plan": plan}
    except Exception as e:
        logger.error("规划失败: %s", e)
        raise HTTPException(status_code=500, detail=f"规划生成失败: {str(e)}")


@app.post("/api/agent/execute")
async def execute_plan(request: ChatRequest):
    """
    Phase 3: 用户确认后，逐步骤执行规划（SSE 流式进度）。

    事件类型:
      step_start:  { type:"step_start", index:0, total:6, description:"...", tool:"..." }
      step_done:   { type:"step_done",  index:0, result:"✅ ..." }
      step_error:  { type:"step_error", index:0, error:"..." }
      all_done:    { type:"all_done",  summary:"...", results:[...] }
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="plan 不能为空")

    try:
        plan = json.loads(request.message)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="plan 必须是有效 JSON")

    if not plan.get("steps"):
        raise HTTPException(status_code=400, detail="plan.steps 为空")

    logger.info("▶ 执行规划: %d 步", len(plan["steps"]))

    # 无线程——在当前 async loop 上直接执行，MCP session 不会丢失
    async def generate():
        from plan_executor import _find_tool, _extract_coords_from_geo_result, _clean_tool_args

        steps = plan.get("steps", [])
        results = []

        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")
            tool_args = dict(step.get("tool_args", {}))

            yield f"data: {json.dumps({'type': 'step_start', 'index': i, 'total': len(steps), 'description': step.get('description', tool_name), 'tool': tool_name}, ensure_ascii=False)}\n\n"

            tool_func = _find_tool(tool_name)
            if tool_func is None:
                yield f"data: {json.dumps({'type': 'step_error', 'index': i, 'error': f'未找到工具: {tool_name}'}, ensure_ascii=False)}\n\n"
                results.append(f"❌ 未知工具 {tool_name}")
                break

            try:
                # 清理 LLM 参数（height: "100m" → 100）
                tool_args = _clean_tool_args(tool_name, tool_args)

                # maps_geo → fly_to_point 坐标自动填充
                if tool_name == "fly_to_point" and i > 0:
                    prev = results[-1] if results else ""
                    coords = _extract_coords_from_geo_result(prev)
                    if coords and tool_args.get("lat") is None:
                        tool_args["lat"] = coords["lat"]
                        tool_args["lng"] = coords["lng"]
                        logger.info("  📍 maps_geo→fly_to_point: %.4f, %.4f",
                                    coords["lat"], coords["lng"])

                # 统一用 ainvoke（MCP 工具需要）
                if hasattr(tool_func, 'ainvoke'):
                    result = await tool_func.ainvoke(tool_args)
                else:
                    result = tool_func.invoke(tool_args)

                results.append(str(result))
                yield f"data: {json.dumps({'type': 'step_done', 'index': i, 'result': str(result)}, ensure_ascii=False)}\n\n"

            except Exception as e:
                logger.error("步骤执行失败 [%s]: %s", tool_name, e)
                yield f"data: {json.dumps({'type': 'step_error', 'index': i, 'error': str(e)}, ensure_ascii=False)}\n\n"
                results.append(f"❌ {tool_name}: {e}")
                break

        ok = "✅"; fail = "❌"
        summary_lines = []
        for idx, (s, r) in enumerate(zip(steps, results)):
            icon = fail if r.startswith("❌") else ok
            summary_lines.append(f"  {icon} [{idx+1}] {s.get('description', '')}")
        summary = "\n".join(summary_lines) if summary_lines else "无步骤"

        yield f"data: {json.dumps({'type': 'all_done', 'summary': summary, 'results': results}, ensure_ascii=False)}\n\n"

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
