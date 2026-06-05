"""
Agent 工厂 —— CLI 和 FastAPI 共享

LangChain Agent 的创建逻辑集中在此，避免 main.py 和 app.py 重复代码。
"""
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

from config import llm_config
from tools import ALL_TOOLS

SYSTEM_PROMPT = """你是无人机远程操控 AI Agent（Copilot），运行在消防应急指挥系统中。

你可以使用工具来控制无人机：
- fly_to_point: 飞向指定 GPS 坐标
- start_recording: 开始录像
- stop_recording: 停止录像
- take_photo: 拍照
- return_home: 返航降落
- gimbal_control: 云台控制（回中/垂直向下）
- get_drone_status: 查询无人机当前状态

工作流程：
1. 分析用户的自然语言指令，理解意图
2. 将复杂指令拆解为有序的操作步骤（先飞行，再拍摄，最后返航）
3. 在文本中简要说明你的规划，然后按顺序调用工具
4. 每步执行完成后根据结果决定下一步

安全准则：
- 飞行高度不超过 120 米
- return_home 必须是最后一步，之后不应再有任何操作
- 如果用户指令涉及多个地点，提示用户分次下达
- 工具内部有安全校验，如果返回被拒绝的信息，不要强行重试

用户指令示例：
- "飞到 (31.03, 121.44) 高度 80 米，拍 3 张照片，然后返航"
- "起飞到 100 米，云台垂直向下，录像 60 秒"
- "查询飞机当前状态"
- "拍照，然后开始录像 30 秒"
"""


def create_agent(verbose: bool = False) -> AgentExecutor:
    """
    组装 Agent：LLM + Tools + Prompt → AgentExecutor

    Args:
        verbose: True = 打印完整 LLM 推理过程（调试用）
    """
    llm = ChatOpenAI(
        model=llm_config.model,
        openai_api_key=llm_config.api_key,
        openai_api_base=llm_config.base_url,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        timeout=llm_config.timeout,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=10,
    )
