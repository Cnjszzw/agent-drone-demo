#!/usr/bin/env python3
"""
LangGraph 环绕飞行子图 — 概念演示

对标 DJI 司空 2 Copilot 场景四："飞到信号塔，环绕并录像"
https://github.com/Cnjszzw/agent-drone-demo

运行: python langgraph_orbit.py

此文件不依赖 langgraph 库，通过模拟执行演示 LangGraph 子图的核心概念。
生产化时替换为: from langgraph.graph import StateGraph

核心对比:
  AgentExecutor: LLM 控制"环绕到哪了、该不该停" → 不可靠
  LangGraph:     图的条件边控制"θ >= 360° 就停" → 确定性的

面试话术:
  "环绕飞行天然是状态机。我用 LangGraph 子图建模了进入环绕→每30°拍照→
   条件边(θ>=360°退出)→停止录像的完整流程。图的边决定执行周期，LLM 不参与
   时机决策。主链路（指点飞行、拍照、返航）继续用 AgentExecutor 避免过度设计。"
"""

import time
import random
from typing import TypedDict


# ═══════════════════════════════════════════════════════════
# 状态定义（LangGraph 的 State / TypedDict）
# ═══════════════════════════════════════════════════════════

class OrbitState(TypedDict):
    """环绕飞行子图的状态机。每个节点读写此状态。"""
    target_name: str          # 目标名称，如"信号塔"
    target_lat: float          # 目标纬度
    target_lng: float          # 目标经度
    radius: float              # 环绕半径（米）
    current_angle: float       # 当前角度 0-360（通过 Redis 轮询获取）
    photos_taken: int          # 已拍照片数
    recording: bool            # 是否正在录像
    error: str                 # 异常信息


# ═══════════════════════════════════════════════════════════
# 图节点定义（每个节点 = LangGraph 的一个 node）
# ═══════════════════════════════════════════════════════════

def node_approach(state: OrbitState) -> OrbitState:
    """
    节点: approach — 飞向目标点

    生产环境: 复用 fly_to_point(lat, lng, height)
    AgentExecutor 也可以路由到这个子图——LLM 理解了"飞到信号塔环绕"的意图
    后，不调 fly_to_point，而是把 state 传给这个子图。
    """
    print(f"\n🛫 [approach] 飞向 {state['target_name']} "
          f"({state['target_lat']:.4f}, {state['target_lng']:.4f})")
    _simulate("飞行中", 2)
    print(f"  ✅ 已到达 {state['target_name']}，高度 80m")

    # 生产环境: Redis 轮询等待到达 → completed 后返回
    return state


def node_start_record(state: OrbitState) -> OrbitState:
    """
    节点: start_record — 到达后开始录像
    环什么时候结束这个节点决定不了，但"开始录像"永远在进入环绕之前。
    """
    print("\n🎥 [start_record] 开始录像（持续模式）")
    state["recording"] = True
    _simulate("", 0.5)
    print("  ✅ 录像已开始")
    return state


def node_enter_orbit(state: OrbitState) -> OrbitState:
    """
    节点: enter_orbit — 锁定兴趣点(POI)，进入环绕轨道

    生产环境: POST /api/drone/orbit/start
              camera 锁定目标坐标
    """
    print(f"\n🔄 [enter_orbit] 锁定 {state['target_name']}，"
          f"进入环绕（半径 {state['radius']}m）")
    _simulate("", 1)
    state["current_angle"] = 0
    print("  ✅ 已进入环绕轨道，相机锁定目标")
    return state


def node_orbit_step(state: OrbitState) -> OrbitState:
    """
    节点: orbit_step — 前进 30° + 拍一张照片

    这是 LangGraph 子图的核心——每一步 30°，图的条件边判断要不要继续。

    生产环境:
      - Redis 轮询当前角度: redis.get(f"drone:orbit:{task_id}:angle")
      - 到达 30° 的整数倍时触发拍照
      - 角度数据来自嵌入式 → MQTT → Java → Redis
    """
    state["current_angle"] += 30
    state["photos_taken"] += 1

    print(f"\n📍 [orbit_step] 角度: {state['current_angle']}°")

    # 拍照
    print(f"  📸 拍照 {state['photos_taken']}/12 "
          f"(角度 {state['current_angle']}°)")
    _simulate("", 0.3)

    # 模拟进度（生产环境: 真实飞行需要更长时间）
    bar_len = state["current_angle"] // 30
    bar = "◉" * bar_len + "○" * (12 - bar_len)
    print(f"  环绕进度: [{bar}] {state['current_angle']}°/360°")

    # 模拟偶发故障
    if random.random() < 0.05:
        state["error"] = "GPS 信号短暂丢失，环绕暂停 1s 后恢复"
        print(f"  ⚠️ {state['error']}")
        _simulate("", 0.5)

    return state


def node_exit_orbit(state: OrbitState) -> OrbitState:
    """节点: exit_orbit — 环绕完成，退出轨道"""
    print(f"\n✅ [exit_orbit] 360° 环绕完成，共拍照 {state['photos_taken']} 张")
    _simulate("", 0.5)
    return state


def node_stop_record(state: OrbitState) -> OrbitState:
    """节点: stop_record — 停止录像"""
    print("\n⏹️ [stop_record] 停止录像")
    state["recording"] = False
    _simulate("", 0.5)
    print(f"  ✅ 录像已停止，视频: ORBIT_{int(time.time()) % 100000}.mp4")
    return state


# ═══════════════════════════════════════════════════════════
# 条件边（LangGraph: add_conditional_edges）
# ═══════════════════════════════════════════════════════════

def should_continue_orbit(state: OrbitState) -> str:
    """
    LangGraph 条件边: 图决定何时退出环绕，不是 LLM。

    生产环境 LangGraph 代码:
      graph.add_conditional_edges(
          "orbit_step",
          should_continue_orbit,
          {"orbit_step": "orbit_step", "exit_orbit": "exit_orbit"}
      )
    """
    if state["current_angle"] >= 360:
        return "exit_orbit"
    return "orbit_step"


# ═══════════════════════════════════════════════════════════
# 图执行（模拟 LangGraph 的 graph.compile().invoke()）
# ═══════════════════════════════════════════════════════════

def run_orbit_graph(state: OrbitState):
    """
    模拟 LangGraph 图执行。

    生产环境:
      graph = build_orbit_graph().compile()
      result = graph.invoke(initial_state)
    """
    # 固定节点序列: approach → start_record → enter_orbit
    state = node_approach(state)
    state = node_start_record(state)
    state = node_enter_orbit(state)

    # 循环节点: orbit_step ←──┐
    #          条件边: θ >= 360°? │
    #                  ├─ 是 → exit_orbit → stop_record
    #                  └─ 否 → orbit_step ─┘
    while True:
        state = node_orbit_step(state)
        next_node = should_continue_orbit(state)
        if next_node == "exit_orbit":
            break

    state = node_exit_orbit(state)
    state = node_stop_record(state)

    # 汇总
    print("\n" + "=" * 55)
    print(f"🎉 环绕巡查完成!")
    print(f"   目标: {state['target_name']}")
    print(f"   环绕: 360° / {state['radius']}m 半径")
    print(f"   拍照: {state['photos_taken']} 张")
    print(f"   录像: 已保存")
    if state["error"]:
        print(f"   告警: {state['error']}")
    print("=" * 55)


# ═══════════════════════════════════════════════════════════
# 与 AgentExecutor 的对比
# ═══════════════════════════════════════════════════════════

def run_with_agentexecutor(state: OrbitState):
    """
    模拟如果用 AgentExecutor 处理环绕飞行会发生什么。

    AgentExecutor 的 ReAct 循环:
      LLM → tool_call → tool_result → LLM → tool_call → ...

    问题: LLM 不知道"绕了多少度、还要不要继续"。
    """
    print("\n\n" + "─" * 55)
    print("❌ AgentExecutor 处理环绕飞行的不可靠性演示")
    print("─" * 55)

    # LLM 需要自己判断"绕得差不多了"
    actions = [
        ("fly_to_point", "飞到信号塔"),
        ("start_recording", "开始录像"),
        ("enter_orbit", "进入环绕"),
    ]

    for tool, desc in actions:
        print(f"\n🤖 LLM: 我决定调用 {tool}")
        print(f"   执行: {desc} ✅")
        _simulate("", 0.3)

    # LLM 猜时机...
    for i in range(random.randint(8, 20)):  # 可能 8-20 张，不可控
        print(f"\n🤖 LLM: 再拍一张（第 {i+1} 次）")
        print(f"   📸 take_photo ✅")
        _simulate("", 0.1)

    # LLM 可能早停或晚停
    if random.random() < 0.4:
        print("\n🤖 LLM: 应该差不多了？停止录像")
        print("   ⚠️ 实际只绕了 240°——少了 120°")
    else:
        print("\n🤖 LLM: 继续绕...继续绕...")
        print("   ⚠️ 绕了 3 圈才停——多耗了 67% 电量")

    print("\n📊 AgentExecutor vs LangGraph:")
    print("   AgentExecutor: 拍照数量不可控，停止时机依赖 LLM 判断")
    print("   LangGraph:     12 张/圈 确定，θ>=360° 停止确定")


# ═══════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════

def _simulate(label, seconds):
    """模拟执行延迟"""
    if label:
        for i in range(int(seconds * 5)):
            time.sleep(0.2)
            dots = "." * ((i % 3) + 1)
            print(f"\r  ⏳ {label}{dots}", end="", flush=True)
        print()
    else:
        time.sleep(seconds)


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════╗
║  LangGraph 环绕飞行子图 — 概念演示             ║
║  对标 DJI 司空 2 Copilot 场景四                ║
║  "飞到信号塔，环绕并录像"                       ║
╚═══════════════════════════════════════════════╝
""")

    # 初始状态——由 LLM 解析意图后填充
    initial_state: OrbitState = {
        "target_name": "信号塔",
        "target_lat": 31.031,
        "target_lng": 121.445,
        "radius": 80,
        "current_angle": 0,
        "photos_taken": 0,
        "recording": False,
        "error": "",
    }

    print("\n▶ LangGraph 子图执行:")
    run_orbit_graph(initial_state)

    run_with_agentexecutor(initial_state)

    print("\n\n📖 参考文档:")
    print("   CODE_WALKTHROUGH.md §7.6 — LangGraph 接入点分析")
    print("   agent-demo-plan.md        — 完整设计决策")
