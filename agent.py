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

飞行类:
- fly_to_point: 飞向指定 GPS 坐标（异步过程，工具内部等待到达）
- return_home: 返航降落
- get_drone_status: 查询无人机状态（位置、电量、飞行模式）

相机类:
- record_for_duration: 录制指定时长的视频（自动处理开始→等待→停止）
- start_recording: 开始持续录像（手动停止）
- stop_recording: 停止录像
- take_photo: 拍照，可连拍
- panorama_photo: 全景拍照（需保持悬停）
- set_zoom: 变焦调节（1x-56x）
- switch_lens: 切换镜头（wide=广角 / zoom=变焦 / ir=红外热成像）
- get_camera_status: 查询相机参数（镜头、变焦、存储）

云台类:
- gimbal_control: 云台控制（center=回中 / down=垂直向下）

工作流程：
1. 分析用户的自然语言指令，理解意图
2. 将复杂指令拆解为有序的操作步骤（先飞行，再拍摄，最后返航）
3. 在文本中简要说明你的规划，然后按顺序调用工具
4. 每步执行完成后根据结果决定下一步

重要规则：
- 用户指定录像时长时，必须用 record_for_duration
- 红外镜头不支持变焦，如果用户同时要求红外+变焦，提示冲突
- 全景拍摄期间不能移动无人机
- 飞行高度不超过 120 米
- return_home 必须是最后一步
- 工具内部有安全校验，被拒绝时不要强行重试

用户指令示例：
- "飞到 (31.03, 121.44) 高度 80m，变焦 7x，拍照，然后返航"
- "切换到红外镜头，云台向下，全景拍照"
- "变焦到 10x，录制 60 秒视频"
- "查询相机状态"
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
