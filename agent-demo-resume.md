# AI Agent 无人机操控原型 — 简历项目描述

## 项目概述

面向消防应急指挥场景，设计并实现基于 LLM Agent 的无人机自然语言操控原型系统。用户通过自然语言输入（如"飞到 (31.03, 121.44) 高度 80m，录像 60s，然后返航"），Agent 自动完成意图理解、任务规划、安全校验、指令编排与模拟执行的完整链路。

- **角色**：独立完成
- **时间**：2025/2026
- **GitHub**：https://github.com/Cnjszzw/agent-drone-demo

## 技术栈

Python、LangChain、DeepSeek API（OpenAI 兼容）、FastAPI、Function Calling / Tool Use

## 项目实现

### 1. Agent 框架选型与架构设计

背景：公司 WVP 可视化视频调度平台已具备无人机手动控制能力（MQTT 指令下发、DRC 遥控模式、ZLM 视频推流），业务方对标 DJI 司空 2 Copilot 提出自然语言操控无人机的需求。

- 技术选型对比：Java 侧 LangChain4j 要求 JDK 17，而 wvp-server 为 JDK 8（国内 toG/私有化部署标配），升级风险不可控；Python 侧 LangChain 社区成熟、文档完善，且 DeepSeek 提供 OpenAI 兼容接口。最终确定 Python（Agent 编排层）+ Java（设备控制层）两层架构，通过 REST 通信解耦。
- 三层架构：LLM 意图理解层（Prompt + Tool Schema）→ Agent 调度层（LangChain AgentExecutor ReAct 循环）→ 工具执行层（SafetyGate 校验 + 模拟 MQTT 下发）。

### 2. 工具定义与 Function Calling 实现

- 基于 LangChain `@tool` 装饰器定义 7 个无人机工具函数（fly_to_point / start_recording / stop_recording / take_photo / return_home / gimbal_control / get_drone_status），函数的 docstring + 类型注解自动生成 OpenAI Function Calling Schema，无需手动编写 JSON Schema。
- 工具设计对应 WVP 生产环境真实控制链路：每个工具函数内部模拟完整的 MQTT Topic + JSON Payload 构建与下发过程，生产化时只需替换 Executor 实现类即可对接真实 MQTT 通道。

### 3. 安全校验层设计（SafetyGate）

- 核心原则：LLM 只负责意图理解，安全决策必须由硬编码规则执行，两者绝不混淆。
- 实现规则：飞行高度限制（10-120m）、GPS 坐标中国境内范围校验（防 LLM 幻觉编造境外坐标）、电量预估（球面余弦公式计算飞行距离 × 3% 耗电率 + 10% 余量）。
- Human-in-the-loop：高风险操作（起飞、降落）执行前强制人工确认（CLI 模式交互式确认，API 模式预留 pending→confirmed 两阶段提交接口）。

### 4. 双模式接口

- CLI 模式：基于 Python 标准输入输出，适合本地调试与功能验证。
- FastAPI 模式：提供 RESTful API（POST /api/agent/chat、GET /api/agent/status），Swagger 自动文档，支持前端集成。
- 两种模式共享同一套 Agent 工厂和工具定义，确认机制通过回调注入切换。

## 项目亮点

- **安全设计**：明确划分 LLM 意图理解与硬编码安全校验的职责边界，所有飞行安全规则 100% 确定性执行，不经过 LLM 判断。
- **工程化思考**：考虑到 JDK 8 私有化部署的现实约束，选择 Python Agent + Java 控制的两层架构，而非强行统一技术栈。
- **框架能力**：熟练运用 LangChain 的 @tool 装饰器、AgentExecutor ReAct 编排、ChatPromptTemplate 等核心组件，理解 Agent 底层 Function Calling 机制。
