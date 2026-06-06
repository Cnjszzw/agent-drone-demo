# 赵志文 · 简历

## 基本信息

- 岗位：后端开发工程师
- 方向：无人机系统 / AI Agent / 物联网通信
- GitHub：https://github.com/Cnjszzw

---

## 工作经历

### 上海寰创通信股份有限公司 · 后端开发 · 软件平台部

**2025/04 — 2026/03**

负责消防应急指挥系统（WVP 可视化视频调度平台）的后端研发，主导无人机接入与控制模块的全流程交付。面向消防应急场景，独立完成从需求分析、方案设计到编码联调的端到端闭环。

---

## 项目一：无人机接入与控制模块（WVP）

**技术栈**：Java 8、SpringBoot 2.3、Redis、MySQL、MyBatis-Plus、MQTT（EMQX）、WebSocket、ZLM（ZLMediaKit）、Netty、Vue 3、TypeScript、Cesium

### 背景

平台原有能力集中于地面 Mesh 自组网设备管理，缺乏对无人机的统一接入与远程控制能力。消防场景下对超视距视频回传与飞行控制有刚性需求。公司自组网硬件具备超远距离（>20km）传输优势，需基于此建设无人机模块，实现空地一体化指挥调度。

### 通信方案与链路搭建

- 对比 DJI 上云 API 方案（依赖遥控器、距离受限、机型兼容性差）与公司自组网硬件方案，确定基于自研硬件 + MQTT 的通信架构。参考 DJI Cloud API 消息格式定义交互规范。
- 前端封装 MQTT 客户端（UranusMqtt 类，1s 短心跳、订阅自动恢复），打通前端与无人机硬件之间的实时遥测通道。
- 解决服务器重启后无人机掉线问题：监听 Spring `ApplicationReadyEvent`，通过 MQTT 广播自定义重连指令触发设备重新上线。
- 后端通过 Redis 缓存设备在线状态，区分主动下线与异常断连，实现设备状态准确管理。

### 飞行控制与相机操作

- 完成一键起飞、指点飞行、环绕飞行、航线飞行、返航、急停等飞行指令，以及变焦、拍照、录像、模式切换、云台重置、框选变焦、AIM 瞄准等相机负载指令。
- 前期前端通过 MQTT 直连硬件快速验证功能，后期将框选变焦、AIM 迁移至后端统一处理，提高可维护性。
- 后端增加飞行安全校验：起飞/返航/目标高度 vs 系统限高对比校验、降落确认与取消。
- DRC（Drone Remote Control）模式：操控前获取 Redis 锁定令牌，确保单用户独占控制权。

### 实时视频点播

- 集成 ZLM 流媒体服务器，搭建无人机视频流 → ZLM 拉流转发 → 前端 Jessibuca 低延迟播放的完整链路。
- 基于 Spring `DeferredResult` 实现异步点播：前端请求后立即释放 Tomcat 线程，异步等待 ZLM 推流就绪后回调返回播放地址。
- Redis 引用计数策略：每个视频流维护观看用户集合，无人观看自动停止拉流，节省带宽。支持视频矩阵多屏监控。

### 性能优化

- **Mesh 设备查询**：识别 N+1 查询问题，预加载 + HashMap 索引策略。千台设备规模下查询响应从 7-8s 优化至 1s 以内。
- **日志深分页**：30w 条日志深分页瓶颈，通过延迟关联 + 覆盖索引优化，响应从 5s 降至 1s 以内，性能提升 5 倍。

---

## 项目二：AI Agent 无人机自然语言操控（技术预研）

**技术栈**：Python 3.11、LangChain、DeepSeek API（OpenAI 兼容 Function Calling）、FastAPI、SSE、LangGraph（概念验证）

**GitHub**：https://github.com/Cnjszzw/agent-drone-demo

### 背景与目标

WVP 平台已具备无人机手动控制能力（MQTT 指令下发、DRC 遥控、ZLM 推流）。业务方对标 DJI 司空 2 Copilot 提出自然语言操控无人机的需求——用户用一句话完成"飞到目标点、拍摄 N 秒、返航降落"的复合指令，替代逐按钮操作。独立负责技术预研与原型验证。

### 技术选型与架构

- **语言选型**：Java 侧 LangChain4j 要求 JDK 17，WVP 为 JDK 8（toG 私有化部署标配），升级风险不可控。Python 侧 LangChain 社区成熟、文档完善。最终确定 Python（Agent 编排层）+ Java（设备控制层）两层架构，通过 REST 通信解耦。
- **三层架构**：LLM 意图理解层（System Prompt + Tool Schema）→ Agent 调度层（LangChain AgentExecutor ReAct 循环）→ 工具执行层（SafetyGate 校验 + 16 个 @tool 函数，对应 WVP 真实控制类）。

### 工具定义

基于 LangChain @tool 装饰器定义 16 个无人机工具函数：

| 类别 | 工具 | 对应 WVP 实现 |
|------|------|-------------|
| 飞行 | fly_to_point / return_home | DrcController / DockController |
| 拍摄 | record_for_duration / start_recording / stop_recording / take_photo / panorama_photo | CameraRecordingStartImpl / CameraPhotoTakeImpl 等 |
| 相机参数 | set_zoom / switch_lens / set_exposure_mode / set_iso / set_shutter_speed / set_ev_compensation | CameraFocalLengthSetImpl / CameraModeSwitchImpl 等 |
| 云台 | gimbal_control | GimbalResetImpl |

docstring + 类型注解自动生成 OpenAI Function Calling Schema，无需手动编写 JSON。遵循"工具暴露业务语义非硬件原语"原则——硬件仅有 start/stop 录像指令，工具层封装 `record_for_duration(60)` 复合工具（内部 start→倒计时→stop）。

### 安全设计

- SafetyGate 层硬编码安全规则：高度限制（10-120m）、GPS 坐标中国境内范围校验、电量预估（球面余弦公式 + 3% 耗电率）。原则：**LLM 负责意图理解，安全决策由确定性规则执行，两者绝不混淆。**
- Human-in-the-loop：高风险操作（起飞、降落）执行前强制人工确认。

### 关键技术难点

**难点一：LLM 幻觉 — 从 Tool A 切换到方案 B**

曾将"通知前端画飞行预览线"设计为独立 Tool 由 LLM 自主调用，LLM 偶发跳过（~10%）。根因：飞行前通知是确定性规则，不应交概率模型。解决：将通知逻辑下沉到 `fly_to_point` 函数内部硬编码调用。遵循"能硬编码的规则不交 LLM 决策"原则，贯穿 SafetyGate、确认流程、前端通知全部关键节点。

**难点二：同步 LLM vs 异步无人机**

LLM 是秒级同步推理，无人机飞行是分钟级异步过程。工具函数内部封装 Redis 轮询等待——Python redis-py 直连 localhost Redis 每秒查询任务状态（<1ms/次），completed/failed/超时才返回 LLM。中间进度仅推前端。评估了 HTTP 轮询、MQTT 直连、WS 长连接、HTTP 回调四种替代方案后选 Redis 直连——无新依赖、无连接管理、1s 轮询对 2-3 分钟飞行可忽略。

**难点三：工具粒度 — 业务语义 vs 硬件原语**

硬件只有 start/stop 录像指令。暴露给 LLM 时它无法可靠编排"开始→等待 60s→停止"的时序（LLM 无时间感知能力）。封装 `record_for_duration` 复合工具，内部实现完整时序，LLM 只看到业务语义。

### LangGraph 接入（环绕飞行子图）

对标 DJI Copilot 场景四"兴趣点环绕巡查"，引入 LangGraph 处理环绕飞行的状态机子图：

```
approach → start_record → enter_orbit → [orbit_loop] → exit_orbit → stop_record
                                              ↑
                                    条件边: θ >= 360° 退出
                                    每 30° 触发 take_photo
```

图的边决定执行周期和拍照间隔，LLM 不参与时机决策。主链路（指点飞行、拍照、返航）保持 AgentExecutor 避免过度设计。验证脚本见 `langgraph_orbit.py`。

### 流式交互与急停

- 基于 FastAPI + SSE 实现流式推送：Agent 执行过程通过 Server-Sent Events 实时推送工具调用卡片和结果到前端聊天框。
- 急停按钮常驻聊天框，走独立 MQTT 通道直达无人机（不经过 Agent/LLM），同时关闭 SSE 连接终止当前会话。停止权握在确定性代码手里。

---

## 专业技能

- **后端**：Java（SpringBoot、MyBatis-Plus、Netty）、Python（FastAPI、LangChain）
- **通信协议**：MQTT、WebSocket、SIP/GB28181、RTP/RTSP
- **数据**：Redis、MySQL、MyBatis
- **流媒体**：ZLM（ZLMediaKit）、Jessibuca
- **AI/Agent**：LLM Function Calling、LangChain、LangGraph、Prompt Engineering
- **前端基础**：Vue 3、TypeScript（能协作开发）

---

## 自我评价

- 具备独立交付完整业务模块的能力（无人机模块从需求到上线全流程闭环）
- 有技术前瞻意识，主动对标行业产品（DJI 司空 2）做技术预研
- 注重工程设计的确定性：在 Agent 系统中明确划分 LLM 概率性推理与硬编码确定性规则的边界
- 私有化部署场景下的务实架构思维：不跟风微服务，基于物理约束做技术决策
