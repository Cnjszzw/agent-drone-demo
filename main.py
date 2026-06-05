#!/usr/bin/env python3
"""
AI Agent 无人机操控 Demo —— CLI 入口

运行方式:
  1. cp .env.example .env  → 填入 DeepSeek API Key
  2. pip install -r requirements.txt
  3. python main.py

此后进入 CLI 交互界面，用自然语言给无人机下达指令。
Agent 自动调用工具执行，终端打印完整 MQTT 下发链路。

设计决策见 agent-demo-plan.md
"""
import time
from config import llm_config, drone_config
from agent import create_agent


def print_banner():
    print("""
╔════════════════════════════════════════════════╗
║     🚁 AI Agent 无人机操控 Demo (CLI)          ║
║                                                ║
║  技术栈: LangChain + DeepSeek (OpenAI 兼容)    ║
║  场景: 消防应急指挥 —— 自然语言无人机操控      ║
║                                                ║
║  FastAPI 模式: uvicorn app:app --reload        ║
╚════════════════════════════════════════════════╝
""")
    print(f"设备: {drone_config.device_id} | 电量: {drone_config.battery}% | 返航点: ({drone_config.home_lat}, {drone_config.home_lng})")
    print(f"LLM: {llm_config.model} @ {llm_config.base_url}")


def main():
    # CLI 模式使用终端 input 确认（默认行为，无需设置 confirm handler）
    print_banner()
    agent = create_agent()

    while True:
        try:
            user_input = input("\n💬 请输入指令 (quit 退出):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        print("\n🤖 Agent 正在分析指令...\n")
        try:
            start = time.time()
            result = agent.invoke({"input": user_input})
            elapsed = time.time() - start

            print("\n" + "─" * 55)
            print(result["output"])
            print("─" * 55)
            print(f"⏱ 总耗时: {elapsed:.1f}s")

        except Exception as e:
            print(f"\n❌ 执行失败: {e}")
            print("💡 提示: 请检查 DeepSeek API Key 是否正确，以及网络是否可达 api.deepseek.com")


if __name__ == "__main__":
    main()
