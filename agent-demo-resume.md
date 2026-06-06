# AI Agent 无人机操控原型 — 简历项目描述

## 项目概述

面向消防应急指挥场景，设计并实现基于 LLM Agent 的无人机自然语言操控原型系统。用户通过自然语言输入（如"飞到 (31.03, 121.44) 高度 80m，录像 60s，然后返航"），Agent 自动完成意图理解、任务规划、安全校验、指令编排与模拟执行的完整链路。

- **角色**：独立完成
- **时间**：2025/2026
- **GitHub**：https://github.com/Cnjszzw/agent-drone-demo

## 技术栈

Python、LangChain、DeepSeek API、FastAPI、MCP、LangGraph、Redis

## 项目实现

### 1. 架构选型与设计

背景：公司 WVP 可视化视频调度平台已具备无人机手动控制能力（MQTT 指令下发、DRC 遥控模式、ZLM 视频推流），业务方对标 DJI 司空 2 Copilot 提出自然语言操控无人机的需求。

- **技术选型**：Java 侧 LangChain4j 要求 JDK 17，而 wvp-server 为 JDK 8（国内 toG/私有化部署标配），升级风险不可控；Python 侧 LangChain 社区成熟、文档完善。最终确定 Python（Agent 编排层）+ Java（设备控制层）两层架构，通过 REST 通信解耦。
- **三层架构**：LLM 意图理解层（Prompt + Tool Schema）→ Agent 调度层（LangChain AgentExecutor ReAct 循环）→ 工具执行层（SafetyGate 校验 + HTTP 调 Java 接口下发指令）。
- **MCP 外部工具集成**：通过 MCP 协议（StreamableHTTP，JSON-RPC 2.0）集成高德地图 MCP Server，langchain-mcp-adapters 的 MultiServerMCPClient 一行完成 initialize→tools/list 握手，15 个地图工具自动注册。地理编码返回多个候选项时，将候选列表回传 LLM 根据上下文选择最佳匹配（如"陆家嘴"6 个候选——云南/湖北/江西/江苏/上海，LLM 判定选上海·浦东）。

### 2. Plan → Confirm → Execute 三段式流程

对标 DJI Copilot 交互模式：Phase 1 LLM 输出结构化 JSON 任务计划（目标+步骤列表+飞前检查），Phase 2 前端渲染规划确认卡片等待用户审核执行，Phase 3 SSE 流式推送每步状态（○→◐→✓/✗）。任何步骤返回 ❌ 或 ⛔ 时立即 break 中止后续执行，防止"无人机已悬停但 Agent 继续执行变焦拍照"的级联错误。

### 3. 安全层设计

- 核心原则：LLM 负责意图理解，安全决策由硬编码规则执行，两者绝不混淆。
- 实现规则：飞行高度限制（10-120m）、GPS 坐标中国境内范围校验（防 LLM 幻觉编造境外坐标）、电量预估（球面余弦公式 + 3% 耗电率）。
- Human-in-the-loop：高风险操作（起飞、降落）执行前强制人工确认。

### 4. 异步进度感知

无人机操作是分钟级的异步过程（飞行到目标点需 2-3 分钟），而 LLM 是同步推理的。核心设计：**工具函数内部封装异步等待，对 LLM 暴露同步接口**。

- 飞行进度感知：Python 通过 redis-py 直连 localhost Redis，每秒 GET `drone:task:{id}:status` 查询任务状态（<1ms），直到 completed/failed/超时后返回结果给 LLM。中间进度（30%→60%→90%）仅推前端展示，LLM 不感知。
- 方案对比：评估了 HTTP 轮询 Java、Python 直连 MQTT、Python↔Java WS 长连接、Java HTTP 回调四种方案。Redis 直连被选中——localhost 访问 <1ms、零新依赖、无连接管理、1s 轮询对 2-3 分钟飞行可忽略。
- 录像控制：硬件只有 start/stop 原子指令，工具层封装 `record_for_duration` 复合工具（start → 倒计时 → stop），LLM 只看到业务语义。

此设计模式与 Anthropic Computer Use、DJI 司空 2 Copilot 的 tool-use 架构一致——将异步等待封装在工具内部、对 LLM 暴露同步函数，是 Agent 工程化的行业共识。

## 难点与解决

### 难点一：LLM 幻觉导致前端通知不可靠

**问题**：最初将"通知前端画飞行预览线"设计为独立的 `notify_frontend` Tool，由 LLM 自主决定何时调用。实际测试发现 LLM 偶发漏调（跳过通知直接飞行），导致前端没有预览线，指挥人员无法确认目标位置。

**根因**：`notify_frontend` 不是意图理解问题——飞行前一定需要预览通知。把它交给概率模型（LLM）决策，本质上把确定性规则变成了概率性行为。

**解决**：将通知逻辑从 Tool 下沉到 `fly_to_point` 工具函数内部，作为 SafetyGate 后的第一个硬编码步骤——安全校验通过后自动调 Java ws-server 的 WebSocket 接口推送预览数据到前端，再等待用户确认。遵循原则：**能硬编码的规则不交给 LLM 决策**。这一思路贯穿了 SafetyGate、预览通知、确认流程等所有关键节点。

### 难点二：同步 LLM vs 异步现实世界的 gap

**问题**：LLM 是同步推理的（秒级），无人机操作是异步的（飞行 2-3 分钟、录像 60s）。LLM 调 `fly_to_point()` 后收到返回码就以为完成了，实际飞机还在路上。同理，LLM 无法感知实时进度，无法判断何时可以执行下一步。

**解决**：Agent 工具函数内部封装轮询等待逻辑。`fly_to_point` 下发指令后以 1s 间隔轮询任务状态（通过后端接口或 Redis），到达/失败后返回确定性结果给 LLM。录像同理——底层只有 start/stop 两个原子指令，在工具层封装 `record_for_duration` 复合工具（start → 倒计时等待 → stop）。LLM 只看到工具返回的最终结果，不感知中间的异步等待过程。

### 难点三：工具粒度的取舍

**问题**：工具定义太细（把 start_recording 和 stop_recording 分别暴露给 LLM），LLM 需要自己编排"开始→等待→停止"的时序——这对 LLM 来说极不可靠，因为它没有时间感知能力。

**解决**：将工具按业务语义封装，而非按硬件原语暴露。`record_for_duration(60)` 替代了 start_recording + wait + stop_recording 的三步编排。原则：**Agent 工具暴露的是业务能力，不是硬件接口**。跟操作系统封装 read/write 为 fopen/fclose 一个道理。

## 技术收获

- 深入理解了 LLM Function Calling 机制及其工程局限性（JSON 输出不稳定、幻觉、无法感知时间）
- 掌握了 Agent 架构中"确定性规则 vs 概率性决策"的职责划分原则
- 积累了异步系统（无人机）与同步推理（LLM）之间的状态同步设计经验
