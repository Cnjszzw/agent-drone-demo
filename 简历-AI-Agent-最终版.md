AI Agent 无人机自然语言操控系统
上海寰创通信股份有限公司
2025/04 — 2026/03
技术栈：Python、LangChain、DeepSeek API、FastAPI、SSE、LangGraph、Redis

项目概述：
面向消防应急指挥场景，设计并实现基于 LLM Agent 的无人机自然语言操控系统。用户通过自然语言输入（如"飞到信号塔，环绕一圈并录像，然后返航"），Agent 自动完成意图理解、任务规划、安全校验与指令编排执行，替代传统的逐按钮手动操作流程。对标 DJI 司空 2 Copilot，独立完成从需求分析、技术选型到架构设计、编码联调的全流程闭环。

技术选型与架构设计：
- 评估 LangChain4j（Java）与 LangChain（Python）两种方案。LangChain4j 要求 JDK 17，与 WVP 主服务的 JDK 8 不兼容，且 Java Agent 生态远不如 Python 成熟。确定 Python 负责 LLM 编排与工具调用，Java 负责底层设备通信（MQTT 指令下发、DRC 权限管理），两层通过 REST 通信解耦。
- 三层架构：LLM 意图理解层（System Prompt + Function Calling Schema）→ Agent 调度层（LangChain AgentExecutor ReAct 循环）→ 工具执行层（SafetyGate 安全校验 + 16 个 @tool 函数）。
- Agent 工具函数对应底层 Java 控制类的真实接口（DrcController、CameraRecordingStartImpl、CameraFocalLengthSetImpl 等），工具层只做编排，不做控制，保持职责清晰。

工具定义与 Function Calling：
- 基于 LangChain @tool 装饰器定义 16 个无人机工具函数，覆盖飞行控制（指点飞行、返航、急停）、相机拍摄（录像、拍照、全景）、相机参数（变焦、镜头切换、曝光模式、ISO、快门、曝光补偿）、云台控制、状态查询。函数的 docstring + 类型注解自动生成 OpenAI Function Calling Schema。
- 遵循"工具暴露业务语义而非硬件原语"原则：底层硬件仅有 start/stop 录像原子指令，工具层封装 record_for_duration(60) 复合工具（start→倒计时→stop），LLM 不感知时序编排细节。
- 前端联动通过 Python HTTP 调 Java WebSocket 接口实现——飞行前自动推预览线和目标点到 Cesium 地图，进度实时更新复用现有 WS 通道。

安全校验与 Human-in-the-Loop：
- SafetyGate 层硬编码安全规则：飞行高度限制（10-120m）、GPS 坐标中国境内范围校验（防 LLM 编造坐标）、电量预估（球面余弦公式 + 3% 耗电率）。原则：LLM 负责意图理解，安全决策由确定性代码执行，两者绝不混淆。
- 高风险操作（起飞、降落）执行前强制人工确认——Agent 生成规划后挂起等待，用户在前端确认卡片中点确认后才进入实际执行。

关键技术难点：

难点一：LLM 幻觉导致确定性行为遗漏
最初将"飞行前通知前端画预览线"设计为独立 Tool 由 LLM 自主调用，上线测试发现 LLM 偶发跳过（~10% 概率），导致前端无预览线。根因：飞行前通知是确定性规则，不应交给概率模型决策。将通知逻辑从 Tool 层下沉到 fly_to_point 函数内部作为硬编码步骤——SafetyGate 校验通过后自动推预览线。原则延伸至全部关键节点：能硬编码的规则不交给 LLM。

难点二：同步 LLM 与异步飞行的状态同步
LLM 是秒级同步推理，无人机飞行是分钟级异步过程（2-3 分钟到达目标点）。工具函数内部封装 Redis 轮询等待——Python redis-py 直连 localhost Redis，每秒 GET 任务状态（<1ms/次），completed/failed/超时后返回 LLM。中间进度（30%→60%→90%）仅推前端展示，LLM 不感知。评估了 HTTP 轮询、MQTT 直连、WebSocket 长连接、HTTP 回调四种替代方案，Redis 直连方案零新依赖、零连接管理、1s 轮询对 2-3 分钟飞行完全可接受。

难点三：工具粒度——业务语义与硬件原语的冲突
硬件录像只有 start/stop 两个原子指令。直接暴露给 LLM 时，LLM 无法可靠编排"开始→等待 60s→停止"的时序（LLM 无时间感知能力，可能立即 stop 或忘了 stop）。将原子指令封装为 record_for_duration 复合工具，内部实现完整时序，LLM 只看到业务语义。类比操作系统将 read/write 封装为 fopen/fclose。

LangGraph 环绕飞行子图：
环绕飞行天然是状态机——"飞到目标→开始录像→进入环绕→每 30° 拍照→360° 完成→停止录像"。AgentExecutor 的隐式 ReAct 循环依赖 LLM 自主判断"还要不要继续绕"，拍照数量和停止时机不可控。引入 LangGraph 将环绕流程建模为显式子图：enter_orbit → orbit_loop（条件边: θ >= 360° 退出）→ exit_orbit。图的边决定执行周期和拍照间隔，LLM 不参与时机决策。主链路（指点飞行、拍照、返航）继续用 AgentExecutor——不是全量替换，是状态机子任务用 LangGraph、线性任务用 AgentExecutor 的务实混用。

急停机制：
急停按钮常驻前端聊天框，走独立 MQTT 通道直达无人机硬件（毫秒级响应），不经过 Agent 或 LLM 处理。用户点击后前端同时关闭当前 SSE 连接终止 Agent 会话，聊天框插入停止通知。停止权握在确定性代码手里——与 SafetyGate 同一原则。

SSE 流式交互：
基于 FastAPI + SSE 实现 Agent 执行过程的实时推送。LangChain callback 桥接工具调用事件（on_agent_action / on_tool_end / on_agent_finish）到 SSE 流，前端聊天框实时渲染工具调用卡片和进度。每次工具执行独立展示为进度卡片（"🔧 fly_to_point — 飞行中 60% — ✅ 已到达"），用户可观察完整执行链。

项目成果：
- 支持 16 项无人机控制能力，覆盖飞行、相机、云台、状态四大类
- 复合指令端到端成功率 95%+，单次复合指令（飞→拍→返）平均执行时间 2-3 分钟
- 急停响应 <100ms（MQTT 硬件通道），Agent 感知 <1s（Redis 轮询间隔）
