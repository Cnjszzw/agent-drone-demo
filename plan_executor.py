"""
任务规划 & 步骤执行 —— DJI Copilot 风格

三段式流程:
  1. Plan（规划）: LLM 输出结构化执行计划
  2. Confirm（确认）: 前端展示计划卡片，用户确认/取消
  3. Execute（执行）: SSE 流式推送每步执行状态

DJI Copilot 的 UX:
- 规划确认卡片: 任务目标 + 执行动作列表 + 飞前检查
- 执行中: ✓已完成  ◐进行中  ○等待中
- 完成后: 全部打勾

面试话术:
  "我们参照 DJI Copilot 的交互模式，设计了三段式流程。
  LLM 先生成结构化任务计划（JSON），用户在前端确认卡片中审核后手动执行。
  执行过程通过 SSE 实时推送每步状态——进行中(转圈)、已完成(打勾)、
  等待中(空圈)。这个设计对应了 Human-in-the-loop 原则——
  高风险操作必须经过人工确认才进入执行阶段。"
"""
import json
import logging
from langchain_openai import ChatOpenAI

from config import llm_config
from tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# ── 规划 Prompt ──────────────────────────────────

PLAN_SYSTEM_PROMPT = """你是一个无人机任务规划器。分析用户指令，生成结构化的 JSON 执行计划。

可用工具（只列出飞行/拍摄类核心工具，MCP 地图工具按需调用）:

飞行类:
- fly_to_point(lat, lng, height) — 飞向指定坐标
- return_home() — 返航降落

相机类:
- record_for_duration(duration_seconds) — 录制指定时长视频
- take_photo(count) — 拍照
- panorama_photo() — 全景拍照
- set_zoom(factor) — 变焦 1x-56x
- switch_lens(mode) — 切换镜头 wide/zoom/ir
- gimbal_control(mode) — 云台 center/down

MCP 地图类（用户提到地名时自动调用）:
- maps_geo(address) — 地名→坐标

输出严格的 JSON 格式，不要额外文本:

{
  "objective": "用户指令的一句话概述",
  "steps": [
    {
      "description": "步骤的中文描述（面向用户展示）",
      "tool": "工具名称",
      "tool_args": { "参数名": 参数值 }
    }
  ],
  "preflight": {
    "return_altitude": "100m",
    "lost_action": "返航"
  }
}

规则:
1. 如果用户提到地名（如"陆家嘴"），第一步必须是 maps_geo 获取坐标
2. 涉及飞行的步骤放在最前面，拍摄类步骤居中，返航必须最后
3. 每个步骤的 description 要用中文，简明扼要（6-15 字）
4. tool_args 中不要放从 LLM 推理出的值——坐标必须来自 maps_geo 的结果
5. 如果用户提到地理编码后的坐标，在 maps_geo 步骤后标注：
   "lat": null, "lng": null  // 由上一步 maps_geo 结果填充
6. 不要编造 GPS 坐标
"""


def generate_plan(user_message: str) -> dict:
    """
    Phase 1: 生成任务规划。

    调用 LLM 输出结构化 JSON 计划，不执行任何工具。
    返回给前端展示"规划确认卡片"。
    """
    llm = ChatOpenAI(
        model=llm_config.model,
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.base_url,
        temperature=0.1,
        max_tokens=2048,
        timeout=llm_config.timeout,
    )

    response = llm.invoke([
        {"role": "system", "content": PLAN_SYSTEM_PROMPT},
        {"role": "user", "content": f"请为以下指令生成执行计划:\n{user_message}"},
    ])

    raw = response.content.strip()
    # 去掉可能的 markdown 包裹
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Plan JSON 解析失败，尝试修复: %s", raw[:200])
        # 尝试提取 JSON 块
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            plan = json.loads(match.group())
        else:
            raise ValueError(f"无法解析规划结果: {raw[:300]}")

    logger.info("📋 规划完成: %d 步", len(plan.get("steps", [])))
    for i, s in enumerate(plan.get("steps", [])):
        logger.info("  [%d] %s → %s(%s)",
                    i + 1, s.get("description", "?"),
                    s.get("tool", "?"), json.dumps(s.get("tool_args", {}), ensure_ascii=False))

    return plan


# ── 步骤执行 SSE 事件发送 ──────────────────────────────────

def execute_plan_stream(plan: dict, agent, event_queue):
    """
    Phase 3: 逐步骤执行规划，通过 queue 发送 SSE 进度事件。

    事件类型:
      step_start:  { type: "step_start",  index: 0, total: 6, description: "..." }
      step_done:   { type: "step_done",   index: 0, result: "✅ ..." }
      step_error:  { type: "step_error",  index: 0, error: "..." }
      all_done:    { type: "all_done",    summary: "..." }
    """
    import threading
    import asyncio

    steps = plan.get("steps", [])
    total = len(steps)
    results = []

    async def _call_tool(tool_func, args):
        """MCP 工具仅支持 ainvoke，统一用异步调用"""
        if hasattr(tool_func, 'ainvoke'):
            return await tool_func.ainvoke(args)
        return tool_func.invoke(args)

    async def _execute_all():
        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")
            tool_args = dict(step.get("tool_args", {}))

            event_queue.put({
                "type": "step_start",
                "index": i, "total": total,
                "description": step.get("description", tool_name),
                "tool": tool_name,
            })

            tool_func = _find_tool(tool_name)
            if tool_func is None:
                event_queue.put({
                    "type": "step_error", "index": i,
                    "error": f"未找到工具: {tool_name}",
                })
                results.append(f"❌ 未知工具 {tool_name}")
                break  # 中止

            try:
                # maps_geo → fly_to_point 坐标自动填充
                if tool_name == "fly_to_point" and i > 0:
                    prev = results[-1] if results else ""
                    coords = _extract_coords_from_geo_result(prev)
                    if coords and tool_args.get("lat") is None:
                        tool_args["lat"] = coords["lat"]
                        tool_args["lng"] = coords["lng"]
                        logger.info("  📍 maps_geo→fly_to_point: %.4f, %.4f",
                                    coords["lat"], coords["lng"])

                result = await _call_tool(tool_func, tool_args)
                results.append(str(result))

                event_queue.put({
                    "type": "step_done", "index": i,
                    "result": str(result),
                })

            except Exception as e:
                logger.error("步骤执行失败 [%s]: %s", tool_name, e)
                event_queue.put({
                    "type": "step_error", "index": i,
                    "error": str(e),
                })
                results.append(f"❌ {tool_name}: {e}")
                break  # 失败立即中止

        event_queue.put({
            "type": "all_done",
            "summary": "\n".join(
                f"  {'✅' if not r.startswith('❌') else '❌'} [{i+1}] {s.get('description', '')}"
                for i, (s, r) in enumerate(zip(steps, results))
            ) if results else "无步骤",
            "results": results,
        })

    def _run_sync():
        asyncio.run(_execute_all())

    t = threading.Thread(target=_run_sync, daemon=True)
    t.start()


def _find_tool(tool_name: str):
    """在 ALL_TOOLS（含 MCP 工具）中查找指定工具"""
    for t in ALL_TOOLS:
        if t.name == tool_name:
            return t
    return None


def _extract_coords_from_geo_result(result_str: str) -> dict | None:
    """从 maps_geo 返回的 JSON 中提取第一个候选坐标"""
    import re
    # maps_geo 返回: {"results":[{"location":"121.50,31.23",...}]}
    match = re.search(r'"location"\s*:\s*"([\d.]+),([\d.]+)"', result_str)
    if match:
        return {"lng": float(match.group(1)), "lat": float(match.group(2))}
    return None
