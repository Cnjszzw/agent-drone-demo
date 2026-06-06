"""
MCP 工具集成 —— 高德地图 MCP Server

通过 MCP StreamableHTTP 协议连接高德地图服务，将地理编码（地址→坐标）、
POI 搜索等能力以 LangChain Tool 的形式暴露给 Agent。

协议: MCP (Model Context Protocol) — Anthropic 提出的开放标准
传输: StreamableHTTP (JSON-RPC 2.0)
端点: https://mcp.amap.com/mcp?key=YOUR_KEY

不依赖 langchain-mcp-adapters，直接实现 MCP 协议层。
这样做的好处:
1. 避免依赖版本冲突
2. 面试时能讲清楚 MCP initialize → tools/list → tools/call 的完整握手流程
3. 工具 Schema 的转换逻辑完全自主可控

面试话术:
  "我们通过 MCP 协议接了高德的地理编码服务。MCP 是 Anthropic 提出的
  开放标准，握手流程是 initialize → tools/list → tools/call。
  我们在 Python 侧实现了一个轻量 MCP 客户端，通过 StreamableHTTP
  连接高德 MCP Server，返回的工具 Schema 自动转换为 LangChain
  StructuredTool。这样 Agent 就能处理'飞到紫竹高新区5号楼'这种
  自然语言地名了。"
"""
import json
import logging
import os
from typing import Optional

import httpx
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

AMAP_MCP_URL = "https://mcp.amap.com/mcp"


class MCPClient:
    """
    轻量 MCP StreamableHTTP 客户端。

    MCP 协议握手（JSON-RPC 2.0）:
      1. initialize  — 建立会话，获取 server 能力声明
      2. tools/list  — 获取可用工具列表
      3. tools/call  — 调用指定工具

    每次请求携带相同的 URL（含 key 参数），服务端通过 key 识别身份。
    """

    def __init__(self, url: str):
        self.url = url
        self.session_id: Optional[str] = None
        self._tools: list[dict] = []

    async def initialize(self) -> dict:
        """MCP 握手: initialize"""
        result = await self._rpc("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {
                "name": "agent-drone-demo",
                "version": "1.0.0"
            }
        })
        self.session_id = result.get("sessionId")
        logger.info("✅ MCP 握手完成 | session: %s", self.session_id)
        return result

    async def list_tools(self) -> list[dict]:
        """MCP: tools/list → 获取可用工具"""
        result = await self._rpc("tools/list", {})
        self._tools = result.get("tools", [])
        logger.info("📡 高德 MCP 工具: %d 个", len(self._tools))
        for t in self._tools:
            logger.info("  🔧 %s: %s", t["name"],
                        t.get("description", "")[:80])
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """MCP: tools/call → 调用工具"""
        result = await self._rpc("tools/call", {
            "name": name,
            "arguments": arguments,
        })
        # MCP 返回 content 列表，取第一项的 text
        content = result.get("content", [])
        if content and isinstance(content[0], dict):
            return content[0].get("text", json.dumps(result))
        return json.dumps(result, ensure_ascii=False)

    async def close(self):
        """关闭连接"""
        self.session_id = None
        self._tools = []

    async def _rpc(self, method: str, params: dict) -> dict:
        """发送 JSON-RPC 2.0 请求"""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params,
        }
        headers = {"Content-Type": "application/json"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(self.url, json=payload, headers=headers)

            # 从响应头获取 session ID
            sid = resp.headers.get("Mcp-Session-Id")
            if sid:
                self.session_id = sid

            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"MCP 错误 [{method}]: {data['error']}")
            return data.get("result", {})


def _build_tool_func(tool_def: dict, mcp_url: str):
    """
    将 MCP 工具定义转换为 LangChain Tool 的调用闭包。

    MCP 工具 Schema → LangChain Tool:
      - MCP tool.name        → LangChain tool name
      - MCP tool.description → LangChain tool description
      - MCP tool.inputSchema → Pydantic args_schema (动态创建)
      - tools/call result    → sync callable 返回值
    """
    import asyncio
    from pydantic import create_model, Field

    # 动态创建 Pydantic 参数模型（从 MCP inputSchema 推导）
    input_schema = tool_def.get("inputSchema", {})
    props = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    fields = {}
    for name, schema in props.items():
        ptype = str  # 简化: 所有 MCP 参数用 str 接收
        desc = schema.get("description", "")
        default = ... if name in required else None
        fields[name] = (ptype, Field(default, description=desc))

    ArgModel = create_model(
        f"{tool_def['name']}_args", **fields
    ) if fields else None

    async def _call_amap(**kwargs):
        client = MCPClient(mcp_url)
        await client.initialize()
        result = await client.call_tool(tool_def["name"], kwargs)
        await client.close()
        return result

    def _sync_wrapper(**kwargs):
        """同步包装器——LangChain Tool 要求同步 callable"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_call_amap(**kwargs))
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _call_amap(**kwargs))
            return future.result(timeout=30)

    if ArgModel:
        return StructuredTool.from_function(
            func=_sync_wrapper,
            name=tool_def["name"],
            description=f"[高德地图] {tool_def.get('description', '')}",
            args_schema=ArgModel,
        )
    else:
        return StructuredTool.from_function(
            func=_sync_wrapper,
            name=tool_def["name"],
            description=f"[高德地图] {tool_def.get('description', '')}",
        )


async def load_amap_tools(api_key: str) -> list:
    """
    连接高德 MCP Server，获取地理编码等工具，转换为 LangChain Tool。

    Args:
        api_key: 高德开放平台 API Key
    Returns:
        LangChain StructuredTool 列表
    """
    mcp_url = f"{AMAP_MCP_URL}?key={api_key}"

    client = MCPClient(mcp_url)
    await client.initialize()
    raw_tools = await client.list_tools()
    await client.close()

    langchain_tools = []
    for tool_def in raw_tools:
        tool = _build_tool_func(tool_def, mcp_url)
        langchain_tools.append(tool)
        logger.info("  🔧 %s", tool_def["name"])

    return langchain_tools
