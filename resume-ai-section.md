二、AI Agent 无人机自然语言操控（技术预研）
上海寰创通信股份有限公司
2025/04 -- 2026/03
技术栈：Python、LangChain、DeepSeek API（OpenAI 兼容 Function Calling）、FastAPI、SSE、Redis

背景与诉求：
平台已具备无人机手动控制能力（MQTT 指令下发、DRC 遥控、ZLM 视频推流），业务方对标 DJI 司空 2 Copilot 提出自然语言操控无人机的需求。用户希望用一句话完成"飞到目标点、拍摄 N 秒、返航降落"的复合指令，替代"逐个点按钮"的手动操作流程。独立负责技术预研与原型验证。

技术选型与架构设计：
- 对比 Java 侧 LangChain4j（要求 JDK 17）与 Python 侧 LangChain，公司项目为 JDK 8（toG 私有化部署标配），升级 JDK 不可行。同时 Python 侧 Agent 生态（LangChain、AutoGen）远成熟于 Java 侧。最终确定 Python（Agent 编排层）+ Java（设备控制层）两层架构，通过 REST 通信解耦。
- 三层架构：LLM 意图理解层（System Prompt + Tool Schema）→ Agent 调度层（LangChain AgentExecutor ReAct 循环）→ 工具执行层（16 个 @tool 函数，SafetyGate 校验 + 模拟 MQTT 下发）。进度反馈复用 Java 侧现有 WebSocket 通道推至前端。

工具定义与 Function Calling：
- 基于 LangChain @tool 装饰器定义 16 个无人机工具函数，覆盖飞行控制（指点飞行、返航）、相机拍摄（录像、拍照、全景）、相机参数（变焦、镜头切换、曝光模式、ISO、快门、曝光补偿）、云台控制、状态查询。函数的 docstring + 类型注解自动生成 OpenAI Function Calling Schema。
- 工具设计对应 WVP 生产环境真实控制链路（DrcController、CameraRecordingStartImpl 等），MockExecutor 模拟完整 MQTT 下发过程，生产化只需替换 Executor 实现类。
- 遵循"工具暴露业务语义非硬件原语"原则：硬件仅有的 start/stop 录像原子指令，在工具层封装 record_for_duration(60) 复合工具（内部 start→倒计时→stop），LLM 不感知时序编排。

安全校验与 Human-in-the-loop：
- SafetyGate 层硬编码安全规则：高度限制（10-120m）、GPS 坐标中国境内范围校验（防 LLM 幻觉编造境外坐标）、电量预估（球面余弦公式 + 3% 耗电率）。原则：LLM 负责意图理解，安全决策由确定性的硬编码规则执行，两者绝不混淆。
- 高风险操作（起飞、降落）执行前强制人工确认（Human-in-the-loop），前端弹出确认卡片后等待用户确认才执行。

关键技术难点与解决：
- 难点一：LLM 幻觉导致确定性规则被遗漏。曾将"飞行前通知前端画预览线"设计为独立 Tool 由 LLM 自主调用，但 LLM 偶发跳过（10% 概率）。根因：飞行前通知是确定性规则，不应交给概率模型。解决：将通知逻辑从 Tool 层下沉到 fly_to_point 函数内部硬编码调用，遵循"能硬编码的规则不交给 LLM 决策"原则。该原则贯穿 SafetyGate、确认流程、前端通知等全部关键节点。
- 难点二：同步 LLM vs 异步无人机的 gap。LLM 是秒级同步推理的，无人机飞行是分钟级异步过程（2-3 分钟到达目标点）。解决：工具函数内部封装 Redis 轮询等待——Python redis-py 直连 localhost Redis（<1ms/次），每 1s 查询任务状态，completed/failed/超时才返回给 LLM。中间进度（30%→60%→90%）仅推前端展示，LLM 不感知。评估了 HTTP 轮询、直接订阅 MQTT、WebSocket 长连接、HTTP 回调等四种替代方案后选 Redis 直连——无新依赖、无连接管理、1s 轮询对 2-3 分钟飞行可忽略。
- 难点三：工具粒度取舍。把 start_recording / stop_recording 原子指令直接暴露给 LLM 时，LLM 无法可靠编排"开始→等待 60s→停止"的时序（LLM 无时间感知能力）。解决：封装 record_for_duration 复合工具，内部实现完整时序，LLM 只看到业务语义。

急停与 SSE 流式交互：
- 急停按钮常驻前端聊天框，点击后走独立 MQTT 通道直达无人机（不经过 Agent/LLM），前端同时关闭 SSE 连接终止当前 Agent 会话。新会话自动重建，旧会话自然消亡。设计原则：停止权握在确定性代码手里，不交给 LLM。
- 基于 FastAPI + SSE 实现流式交互：Agent 执行过程通过 Server-Sent Events 实时推送工具调用卡片和结果到前端聊天框，每个工具调用独立渲染为进度卡片，用户可观察完整执行链。
- 前端聊天框基于单页 HTML + SSE EventSource 实现暗色主题交互界面，预设快捷指令。

项目产出：
1. 独立完成从需求分析、技术选型、架构设计到原型编码的端到端闭环
2. 定义 16 个无人机控制工具函数，覆盖飞行/相机/云台/状态四大类
3. 沉淀 Agent 工程化经验：LLM 与确定性规则的职责划分、异步操作的工具层封装、复合工具的业务语义设计
4. GitHub：https://github.com/Cnjszzw/agent-drone-demo
