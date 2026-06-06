"""
MCP 工具集成 —— 高德地图 MCP Server

通过 langchain-mcp-adapters 连接高德地图 MCP Server，
将地理编码（地址→坐标）、POI 搜索等能力以 LangChain Tool 形式暴露给 Agent。

高德 MCP Server: https://mcp.amap.com/mcp?key=YOUR_KEY
MCP 协议: Model Context Protocol (Anthropic 提出，JSON-RPC 2.0 over StreamableHTTP)

面试话术:
  "用户常说'飞到紫竹高新区5号楼'这种自然语言地名，LLM 不知道坐标。
  我们用 langchain-mcp-adapters 接了高德的 MCP Server——
  Agent 先调 MCP 地理编码把地名转为 GPS 坐标，再调 fly_to_point 执行飞行。
  MCP 让外部工具集成标准化了——以前要自己写 HTTP 调用和 Schema 映射，
  现在一行 MultiServerMCPClient 搞定。"
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False
    logger.warning(
        "langchain-mcp-adapters 未安装。安装: pip install langchain-mcp-adapters"
    )

AMAP_MCP_URL = "https://mcp.amap.com/mcp"


async def load_amap_tools(api_key: Optional[str] = None):
    """
    连接高德 MCP Server，获取地理编码等工具，转换为 LangChain BaseTool。

    MCP Server 提供的工具体系:
    - 地理编码（地址→坐标）: "紫竹高新区5号楼" → (31.03, 121.44)
    - 逆地理编码（坐标→地址）: (31.03, 121.44) → "上海市闵行区..."
    - POI 搜索: "附近的消防栓"、"陆家嘴附近的派出所"
    - 路径规划: 驾车/步行/骑行路线
    - 周边搜索: 指定位置周边设施

    使用方式:
      tools, client = await load_amap_tools("your-key")
      ALL_TOOLS.extend(tools)  # 注册到 Agent

    Args:
        api_key: 高德开放平台 API Key（或设置环境变量 AMAP_API_KEY）
    Returns:
        (工具列表, MCP 客户端实例)
    """
    if not _MCP_AVAILABLE:
        raise RuntimeError("langchain-mcp-adapters 未安装")

    key = api_key or os.getenv("AMAP_API_KEY", "")
    if not key:
        raise ValueError(
            "高德 API Key 未设置。设置环境变量 AMAP_API_KEY 或传入 api_key。"
            "申请地址: https://console.amap.com/dev/key/app"
        )

    client = MultiServerMCPClient({
        "amap": {
            "transport": "streamable_http",
            "url": f"{AMAP_MCP_URL}?key={key}",
        }
    })

    tools = client.get_tools()

    logger.info("📡 高德 MCP 工具: %d 个", len(tools))
    for t in tools:
        logger.info("  🔧 %s: %s", t.name,
                    (t.description or "")[:80])

    return tools, client
