# AI Agent 无人机操控 Demo — 代码详解

> 本文档面向完全没接触过 LangChain 的开发者，逐文件解释每一个概念和代码逻辑。

---

## 目录

1. [核心概念速览](#1-核心概念速览)
2. [文件地图：从启动到执行，代码怎么走](#2-文件地图从启动到执行代码怎么走)
3. [逐文件详解](#3-逐文件详解)
   - [config.py — 配置加载](#configpy)
   - [safety.py — 安全校验层](#safetypy)
   - [executor.py — 模拟执行器](#executorpy)
   - [tools.py — 工具定义](#toolspy)
   - [agent.py — Agent 工厂](#agentpy)
   - [main.py — CLI 入口](#mainpy)
   - [app.py — FastAPI 入口](#apppy)
4. [完整执行流程推演](#4-完整执行流程推演)
5. [LangChain 核心概念解读](#5-langchain-核心概念解读)
6. [常见疑问解答](#6-常见疑问解答)

---

## 1. 核心概念速览

在逐文件阅读之前，先理解几个关键概念：

### LLM 是什么

LLM（Large Language Model，大语言模型）就是一个能跟你对话的 AI，比如 ChatGPT、DeepSeek。你发一段文字给它，它回复一段文字。

### Function Calling 是什么

普通的 LLM 只能"说话"，不能执行动作。Function Calling 是让 LLM 能够"调用函数"的能力。

```
普通对话:  你: "飞到上海"  →  LLM: "好的，正在飞往上海..."（只是文字，无人机没动）

Function Calling:  你: "飞到 (31.03, 121.44)"  
                   →  LLM: 我决定调用 fly_to_point(lat=31.03, lng=121.44)
                   →  程序真的调用了 fly_to_point 函数
                   →  无人机真的飞过去了
```

关键点：**LLM 不执行函数，它只是"决定"调用哪个函数以及传什么参数。真正的执行是程序代码完成的。**

### @tool 装饰器是什么

LangChain 的 `@tool` 是给函数贴一个标签，告诉 LangChain："这个函数可以给 LLM 调用"。LangChain 会自动读取函数名、docstring（函数的说明文档）、参数类型，生成一份"工具说明书"（JSON Schema）发给 LLM。

```python
@tool
def fly_to_point(lat: float, lng: float, height: float) -> str:
    """控制无人机飞向指定的 GPS 坐标位置。"""
    # ... 实际执行逻辑
```

LangChain 自动生成发给 LLM 的工具说明：
```json
{
  "name": "fly_to_point",
  "description": "控制无人机飞向指定的 GPS 坐标位置。",
  "parameters": {
    "lat": { "type": "number", "description": "..." },
    "lng": { "type": "number", "description": "..." },
    "height": { "type": "number", "description": "..." }
  }
}
```

### AgentExecutor 是什么

AgentExecutor 是 LangChain 的核心调度器。它的工作循环（ReAct 模式）：

```
┌────────────────────────────────────────────┐
│                                             │
│  1. 把用户输入 + 工具列表 发给 LLM           │
│                    ↓                        │
│  2. LLM 返回: "我要调用 fly_to_point"       │
│                    ↓                        │
│  3. AgentExecutor 调 fly_to_point 函数      │
│                    ↓                        │
│  4. 函数返回: "✅ 已到达目标点"              │
│                    ↓                        │
│  5. 把结果回传给 LLM，LLM 决定下一步         │
│                    ↓                        │
│  6. LLM 返回: "我要调用 start_recording"    │
│                    ↓                        │
│  ... 循环直到 LLM 返回纯文本（不再调用工具）  │
│                                             │
└────────────────────────────────────────────┘
```

一句话总结：**AgentExecutor 是"LLM 和工具函数之间的调度员"**。

---

## 2. 文件地图：从启动到执行，代码怎么走

以 CLI 模式为例，用户输入 `"飞到 (31.03, 121.44) 高度 80m"` 后发生了什么：

```
main.py                           agent.py                     tools.py
───────                           ────────                     ────────
                                                               (启动时 import)
1. 用户输入                       
   "飞到 (31.03, 121.44)..."                                    

2. agent.invoke({"input": ...})  ───→  3. AgentExecutor 把
   调用 AgentExecutor                   用户输入 + System Prompt
                                        + 工具列表发给 DeepSeek
                                       
                                       4. DeepSeek 返回:
                                       tool_calls: [
                                         {name: "fly_to_point",
                                          args: {lat: 31.03,
                                                 lng: 121.44,
                                                 height: 80}}
                                       ]
                                       
                                       5. AgentExecutor ────────→ 6. fly_to_point() 被调用
                                          自动匹配工具函数              │
                                                                     ├─ 7. SafetyGate 校验坐标/高度
                                                                     ├─ 8. 用户确认 (y/n)
                                                                     └─ 9. MockExecutor 模拟执行
                                                                          打印 MQTT 日志
                                                                          返回 "✅ 已到达..."
                                       
                                       10. 执行结果回传 LLM
                                       
                                       11. LLM 返回纯文本:
                                       "已到达目标点，任务完成"
                                       
12. 打印结果给用户                   
```

**关键理解**：LLM（DeepSeek）只参与第 4 步（决定调用哪个工具）和第 11 步（生成最终回复）。第 6-9 步是纯粹的本地 Python 代码执行，LLM 完全不参与。

---

## 3. 逐文件详解

### config.py

**职责**：从环境变量/`.env` 文件加载配置。

```python
from dotenv import load_dotenv
load_dotenv()  # 自动读取 .env 文件中的 KEY=VALUE

class LLMConfig:
    api_key = os.getenv("DEEPSEEK_API_KEY", "默认值")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    # temperature=0.1 表示 LLM 输出更"确定"（不随机），适合指令解析
    temperature = 0.1
    max_tokens = 2048    # LLM 单次回复最大长度
    timeout = 60         # 网络请求超时
```

**为什么 temperature=0.1**：temperature 控制 LLM 输出的随机性。0.1 表示极其稳定——同样的输入几乎总是同样的输出。解析用户指令（"飞到 X 拍照再返航"）需要的是准确性和一致性，而不是创意。

---

### safety.py

**职责**：飞行安全校验。**这是整个项目最重要的设计决策之一——安全校验规则是硬编码的 Python 逻辑，绝不依赖 LLM 判断。**

```python
class SafetyGate:
    MIN_LAT, MAX_LAT = 18.0, 54.0   # 中国纬度范围
    MIN_LNG, MAX_LNG = 73.0, 135.0  # 中国经度范围
    MAX_HEIGHT = 120.0               # 法规限高
    MIN_HEIGHT = 10.0                # 最低安全高度

    def validate_fly(self, lat, lng, height):
        # 规则 1: 坐标必须在境内 → 防止 LLM 凭空编造一个 GPS 坐标
        if lat < 18.0 or lat > 54.0: return reject
        if lng < 73.0 or lng > 135.0: return reject

        # 规则 2: 高度必须合法 → 120m 是法规硬约束
        if height > 120: return reject

        # 规则 3: 电量预估 → LLM 不擅长数学，必须自己算
        distance = 球面余弦公式(lat, lng, home_lat, home_lng)
        consumption = distance * 3% + 10%余量
        if consumption > 70%: return warning
```

**为什么这些规则不能交给 LLM 判断**：

1. LLM 的输出是**概率性**的。你问两次"高度 150m 能飞吗"，LLM 可能一次说能一次说不能。
2. LLM 可能产生**幻觉**。比如伪造一个不存在的坐标，或者误判安全规则。
3. 安全规则需要**确定性**和**可审计性**。每次拒绝都必须有明确的原因记录。
4. LLM **不擅长数学**。电量预估需要精确计算，LLM 算不准。

**面试核心论点**："LLM 负责理解'飞到5号楼'应该变成什么 GPS 坐标；SafetyGate 负责验证这个坐标能不能飞。两个角色绝不混淆。"

---

### executor.py

**职责**：模拟 MQTT 指令下发——这是整个项目中唯一"假装执行"的部分。

```python
class MockExecutor:
    def fly_to_point(self, lat, lng, height):
        # 1. 构建 MQTT 消息
        topic = "dji/device/DJISN001/control/fly"
        payload = {"lat": 31.03, "lng": 121.44, "height": 80}
        
        # 2. 打印到终端（真实场景是 publish 到 MQTT broker）
        print("📤 [MQTT] Topic: dji/device/DJISN001/control/fly")
        
        # 3. 模拟飞行时间（sleep 几秒 + 进度条）
        sleep(3)
        
        # 4. 更新模拟状态（位置变化、电量消耗）
        self.current_lat = lat
        self.battery -= 5
        
        # 5. 返回结果
        return "✅ 已到达目标点 (31.030000, 121.440000)，电量 78%"
```

返回的字符串会通过 AgentExecutor 回传给 LLM，LLM 据此决定下一步。

**MockExecutor 的意义**：它让你在没有无人机、没有 EMQX、没有 WVP 的情况下，能完整跑通 Agent 的全部逻辑。而且它的方法签名和真实 MQTT 执行器完全一致——替换就是换一个实现类。

**面试时**："Demo 阶段 MockExecutor 验证工具链路正确性。验证通过后，替换为 MqttExecutor（调 Java wvp-server 的 REST 接口发 MQTT），工具函数签名零改动。"

---

### tools.py

**职责**：定义所有可供 LLM 调用的工具函数。这是整个项目的"能力清单"。

#### 确认机制

```python
# 默认确认方式：终端交互
_confirm_handler = lambda prompt: input(prompt).lower() == "y"

def set_confirm_handler(handler):
    """注入不同的确认方式。CLI 用 input()，API 用自动确认。"""
    global _confirm_handler
    _confirm_handler = handler
```

为什么需要这个设计：CLI 模式下用户在终端按 `y/n` 确认；FastAPI 模式下没有终端，需要不同的确认方式。这是一个**依赖注入**的简化版。

#### 工具函数解析（以 fly_to_point 为例）

```python
@tool  # ← LangChain 装饰器：标记这个函数为"可被 LLM 调用的工具"
def fly_to_point(lat: float, lng: float, height: float) -> str:
    """
    控制无人机飞向指定的 GPS 坐标。       ← docstring 变成工具的 description
    
    Args:
        lat: 目标纬度，范围 18-54         ← 参数说明帮助 LLM 理解参数含义
        lng: 目标经度
        height: 飞行高度（米），10-120
    """
    logger.info("🛫 飞行指令: ...")     # 日志

    # 步骤 1：安全检查（硬编码 Python 逻辑）
    result = safety_gate.validate_fly(lat, lng, height)
    if not result.passed:
        return "❌ 飞行指令被拒绝: " + result.reason
    # ↑ 返回的字符串会回传给 LLM，LLM 会理解"被拒绝了"并告知用户

    # 步骤 2：人工确认（高风险操作）
    if not _confirm("确认飞至 XXX? (y/n): "):
        return "❌ 用户取消飞行"

    # 步骤 3：执行
    return executor.fly_to_point(lat, lng, height)
    # ↑ 返回值会回传给 LLM
```

**三个关键理解**：

1. **@tool 的 magic**：LangChain 自动提取函数名、docstring、参数类型注解，生成 Function Calling Schema 发给 LLM。你不需要手写 JSON。

2. **返回值就是 LLM 的输入**：工具函数的 return 值会被 AgentExecutor 回传给 LLM。所以返回值要包含足够的信息让 LLM 做出下一步决策。

3. **LLM 不调用工具，AgentExecutor 调用**：LLM 只输出 `{"tool": "fly_to_point", "args": {...}}`。AgentExecutor 解析这个 JSON，找到对应的函数，执行它。

#### 七个工具一览

| 工具函数 | 功能 | 是否高风险 | LLM 何时调用 |
|---------|------|-----------|------------|
| `fly_to_point` | 飞向 GPS 坐标 | 是（需确认） | 用户说"飞到 X" |
| `start_recording` | 开始录像 | 否 | 用户说"录像 X 秒" |
| `stop_recording` | 停止录像 | 否 | 用户说"停止录像" |
| `take_photo` | 拍照 | 否 | 用户说"拍照" |
| `return_home` | 返航降落 | 是（需确认） | 用户说"返航" |
| `gimbal_control` | 云台控制 | 否 | 用户说"云台向下" |
| `get_drone_status` | 查询状态 | 否 | 用户说"现在什么状态" |

```python
ALL_TOOLS = [fly_to_point, start_recording, ...]  # 所有工具注册到列表
```

---

### agent.py

**职责**：创建 Agent。这里是 LangChain 框架的核心组装点。

```python
def create_agent(verbose=False):
    # 1. 创建 LLM 客户端
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_base="https://api.deepseek.com",  # DeepSeek 兼容 OpenAI 接口
        temperature=0.1,
    )

    # 2. 构建 Prompt 模板
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),       # 系统指令：定义 Agent 的行为
        ("human", "{input}"),            # 用户输入：运行时替换
        ("placeholder", "{agent_scratchpad}"),  # 框架内部使用：存放工具调用历史
    ])

    # 3. 创建 Agent
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    # 4. 包装为 AgentExecutor
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        max_iterations=10,            # 最多 10 轮工具调用
        handle_parsing_errors=True,   # LLM 输出格式错误时自动重试
    )
```

#### System Prompt 的作用

`SYSTEM_PROMPT` 是发给 LLM 的"系统指令"，定义了 Agent 的角色、能力、行为规范和安全准则。每次 LLM 调用都会携带这段 prompt。

```
"你是无人机远程操控 AI Agent（Copilot）..."
```

**System Prompt 怎么写**（这是 Agent 工程的核心技能）：

1. **角色定义**："你是无人机操控 Agent"——告诉 LLM 它是什么
2. **能力清单**："你可以使用以下工具..."——告诉 LLM 它能做什么
3. **工作流程**："先分析指令，再拆解步骤..."——告诉 LLM 怎么做
4. **安全准则**："高度不超 120m，返航必须是最后一步..."——约束 LLM 的行为
5. **示例**："飞到 X，拍照片，然后返航"——给 LLM 参考

#### ChatPromptTemplate 的三个消息槽位

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),        # 角色指令，每次都不变
    ("human", "{input}"),             # 用户输入，运行时替换
    ("placeholder", "{agent_scratchpad}"),  # LangChain 内部占位符
])
```

- `system`：系统指令，始终存在
- `human`：用户的当前输入
- `{agent_scratchpad}`：LangChain 内部使用的占位符，用于存放工具调用的中间结果。**你不用管它填什么，框架自动处理。**

#### AgentExecutor 的关键参数

- `max_iterations=10`：最多执行 10 轮工具调用。比如用户说"飞到 A，拍照，返航"，LLM 会依次调用 3 个工具，每调一次算一轮。10 轮足够绝大多数场景。这个参数防止 LLM 陷入死循环。
- `handle_parsing_errors=True`：LLM 有时候返回格式不对的 JSON。这个参数让框架自动把错误信息回传给 LLM，让它重新输出一次。
- `verbose=False`：生产环境关闭（只显示结果），调试时改为 `True`（打印 LLM 完整的推理过程）。

---

### main.py

**职责**：CLI 交互模式入口。代码非常简单——组装 Agent，然后进入"等待用户输入 → 调用 Agent → 打印结果"的循环。

```python
def main():
    print_banner()
    agent = create_agent()    # 创建 Agent（一次性）

    while True:
        user_input = input("💬 请输入指令: > ")
        if user_input == "quit": break

        result = agent.invoke({"input": user_input})
        # ↑ 这一行触发了整个 Agent 流程：
        #   用户输入 → LLM 推理 → 工具调用 → 回传结果 → ... → 最终回复

        print(result["output"])  # LLM 的最终文本回复
```

**agent.invoke() 做了什么**：这是整个项目最核心的一行代码。它触发了：

1. 把 `{"input": "飞到 X 拍照再返航"}` 填入 Prompt 模板
2. 发给 DeepSeek
3. DeepSeek 返回 `tool_calls: [fly_to_point(...)]`
4. 框架自动调用 `fly_to_point()` 函数
5. 函数返回 `"✅ 已到达目标点"`
6. 把 `"✅ 已到达目标点"` 回传给 DeepSeek
7. DeepSeek 返回 `tool_calls: [take_photo(...)]`
8. ...
9. 直到 DeepSeek 返回纯文本（不再有 tool_calls）
10. 返回 `{"output": "任务完成！..."}`

---

### app.py

**职责**：FastAPI HTTP 服务模式入口。和 main.py 的区别只是"用户输入从终端变成了 HTTP 请求"。

```python
app = FastAPI()

# API 模式：自动确认飞行操作（Demo 简化）
set_confirm_handler(lambda prompt: True)

# 启动时创建 Agent 实例（复用，不需要每次请求都创建）
agent = create_agent()

@app.post("/api/agent/chat")
def chat(request: ChatRequest):
    result = agent.invoke({"input": request.message})
    return ChatResponse(
        success=True,
        output=result["output"],
        elapsed_seconds=...
    )
```

**三个值得注意的设计细节**：

1. **确认回调切换**：`set_confirm_handler(lambda prompt: True)`。CLI 模式用户手动按 y/n，API 模式自动确认（Demo 行为）。生产环境应改为独立的两阶段确认流程。

2. **Agent 全局复用**：`agent = create_agent()` 在模块加载时执行一次，后续所有 HTTP 请求共享同一个 Agent 实例。这是因为 Agent 本身是无状态的（状态在 MockExecutor 里）。

3. **线程安全问题**：当前 MockExecutor 的状态（current_lat、battery 等）是实例变量。如果多用户并发调用，状态会互相覆盖。生产环境需要为每个会话创建独立的 Executor 实例——这是 Demo 简化处理的已知局限。

---

## 4. 完整执行流程推演

以用户输入 **"飞到 (31.03, 121.44) 高度 80m，拍照 3 张，然后返航"** 为例，从头到尾推演一次。

### 阶段 1：Agent 启动

```
main.py: agent = create_agent()
  → ChatOpenAI 客户端创建（配置 API Key、base_url 等）
  → AgentExecutor 创建（绑定 LLM + 7 个工具 + System Prompt）
```

### 阶段 2：用户输入

```
💬 请输入指令: > 飞到 (31.03, 121.44) 高度 80m，拍照 3 张，然后返航
```

### 阶段 3：第一轮 LLM 调用

```
agent.invoke({"input": "飞到 (31.03, 121.44) 高度 80m，拍照 3 张，然后返航"})

→ AgentExecutor 构建完整的 messages 发给 DeepSeek:

  [system] 你是无人机远程操控 AI Agent...
  [human]  飞到 (31.03, 121.44) 高度 80m，拍照 3 张，然后返航
  [tools]  [fly_to_point, start_recording, take_photo, return_home, ...]

→ DeepSeek 返回:

  tool_calls: [
    {
      "id": "call_001",
      "function": {
        "name": "fly_to_point",
        "arguments": '{"lat": 31.03, "lng": 121.44, "height": 80}'
      }
    }
  ]

→ AgentExecutor 解析 tool_calls → 匹配到 tools.py 中的 fly_to_point 函数
```

### 阶段 4：工具执行（fly_to_point）

```
AgentExecutor 调用 fly_to_point(lat=31.03, lng=121.44, height=80)

  → 日志: 🛫 飞行指令: (31.030000, 121.440000) 高度 80m

  → safety_gate.validate_fly(31.03, 121.44, 80)
      · 纬度 31.03 在 [18, 54] ✓
      · 经度 121.44 在 [73, 135] ✓
      · 高度 80 在 [10, 120] ✓
      · 距离: 2.3km，预估耗电 17% → 正常
      · 打印: ✅ 安全校验通过 | 距离: 2.3km | 预估耗电: 17%

  → 用户确认: ⚠️ 确认飞至 (31.030000, 121.440000) 高度 80m? (y/n): y

  → executor.fly_to_point(31.03, 121.44, 80)
      · 打印 MQTT: Topic: dji/device/DJISN001/control/fly
      · 打印 MQTT: Payload: {"lat":31.03,"lng":121.44,"height":80}
      · 模拟飞行 3-5 秒（进度条）
      · 更新位置: current_lat=31.03, current_lng=121.44, current_height=80
      · 消耗电量: battery -= 8

  → 返回: "✅ 已到达目标点 (31.030000, 121.440000)，当前高度 80.0m，电量 77%"
```

### 阶段 5：第二轮 LLM 调用

```
→ "✅ 已到达目标点" 作为 tool result 回传 DeepSeek

→ DeepSeek 分析: 第一步完成了，用户还要求"拍照 3 张"和"返航"

→ DeepSeek 返回:

  tool_calls: [
    {
      "function": {
        "name": "take_photo",
        "arguments": '{"count": 3}'
      }
    }
  ]

→ AgentExecutor 调用 take_photo(count=3)
  → 打印 3 次 MQTT 拍照消息（连拍 3 张）
  → 返回: "✅ 连拍完成，共 3 张照片"
```

### 阶段 6：第三轮 LLM 调用

```
→ "✅ 连拍完成，共 3 张照片" 回传 DeepSeek

→ DeepSeek 分析: 拍照完成了，还剩"返航"

→ DeepSeek 返回:

  tool_calls: [
    {
      "function": {
        "name": "return_home",
        "arguments": '{}'
      }
    }
  ]

→ AgentExecutor 调用 return_home()
  → 用户确认: ⚠️ 确认返航降落? (y/n): y
  → executor.return_home()
  → 返回: "✅ 已返航降落至起飞点"
```

### 阶段 7：最终回复

```
→ "✅ 已返航降落至起飞点" 回传 DeepSeek

→ DeepSeek 判断: 所有步骤完成，不再需要调用工具

→ DeepSeek 返回纯文本（无 tool_calls）:

  "任务完成！已飞至目标点 (31.03, 121.44) 高度 80m，连拍 3 张照片，已返航降落。所有步骤执行成功。"

→ agent.invoke() 返回 {"output": "任务完成！..."}

→ main.py 打印结果给用户
```

### 时间线总结

```
用户输入
    │
    ├─ [0.0s] agent.invoke() 开始
    │
    ├─ [0.5s] 第一轮 LLM: 决定飞向目标点
    ├─ [3.5s] 飞行模拟 (3s) + 安全校验 + 确认
    │
    ├─ [4.0s] 第二轮 LLM: 决定拍照
    ├─ [7.0s] 拍照模拟 (3×1s)
    │
    ├─ [7.5s] 第三轮 LLM: 决定返航
    ├─[11.0s] 返航模拟 (3s) + 确认
    │
    ├─[11.5s] 最终 LLM 总结
    │
    └─[12.0s] 完成，返回结果给用户
```

---

## 5. LangChain 核心概念解读

### @tool 装饰器

```python
# 这是你写的
@tool
def fly_to_point(lat: float, lng: float, height: float) -> str:
    """控制无人机飞向指定的 GPS 坐标。"""
    ...

# LangChain 内部自动做的事：
# 1. 提取函数名: "fly_to_point"
# 2. 提取描述: "控制无人机飞向指定的 GPS 坐标。"
# 3. 提取参数类型: lat: float, lng: float, height: float
# 4. 生成 JSON Schema:
#    {
#      "type": "function",
#      "function": {
#        "name": "fly_to_point",
#        "description": "控制无人机飞向指定的 GPS 坐标。",
#        "parameters": {
#          "type": "object",
#          "properties": {
#            "lat": {"type": "number", "description": "..."},
#            ...
#          }
#        }
#      }
#    }
# 5. 把这个 Schema 塞到 LLM 请求的 tools 参数里
```

### ChatPromptTemplate

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是无人机 Agent..."),     # 角色指令
    ("human", "{input}"),                   # 用户输入（占位符）
    ("placeholder", "{agent_scratchpad}"),  # 工具调用历史（框架管理）
])

# 当 agent.invoke({"input": "飞到上海"}) 时：
# "{input}" 被替换为 "飞到上海"
# "{agent_scratchpad}" 由框架自动填充（包含之前的 tool_call 和 tool_result）
```

### AgentExecutor 的 max_iterations

```python
AgentExecutor(..., max_iterations=10)
```

每一次 LLM 调用 + 工具执行算一轮。如果 LLM 连续 10 轮都在调工具（不返回纯文本），AgentExecutor 会强制停止。

防止的场景：
```
LLM: 我要拍照       → 执行 → "拍完了"
LLM: 我要再拍照     → 执行 → "拍完了"
LLM: 我要再拍照     → ... （LLM 陷入循环）
```

### handle_parsing_errors

```python
AgentExecutor(..., handle_parsing_errors=True)
```

LLM 有时返回格式不对的 tool_call（比如 JSON 多了个逗号）。开启后，AgentExecutor 会把错误信息回传给 LLM：

```
AgentExecutor: "你上次的输出格式错误: JSON parse error at line 3。请重新输出。"
LLM: "抱歉，修正后: ..."
```

---

## 6. 常见疑问解答

### Q: LLM 怎么知道"拍照 3 张"要调用 take_photo(count=3) 而不是 start_recording？

A: 三个因素共同作用：

1. **函数命名**：`take_photo` 这个名字暗示了功能是拍照
2. **docstring 描述**：`"拍照，可指定连拍张数"` 明确说明了这是拍照工具
3. **System Prompt**：告诉了 LLM 每个工具的用途

LLM 在训练时见过海量的"工具选择"任务，它理解如何根据用户意图匹配最合适的工具。

### Q: LLM 会不会编造 GPS 坐标？

A: 会。这就是为什么必须有 SafetyGate。LLM 可能输出 `fly_to_point(lat=99.99, lng=99.99)`——它不理解 99.99 不在中国境内。SafetyGate 的坐标范围校验就是拦截这类情况。

### Q: MockExecutor 返回的字符串用在哪？

A: 返回给 LLM。比如 `"✅ 已到达目标点，电量 78%"`——LLM 读到这个结果后判断"飞行完成了，下一步应该拍照"。

### Q: 为什么 app.py 和 main.py 里的 Agent 创建方式一样？

A: 它们共用 `agent.py` 中的 `create_agent()` 函数。唯一的区别是确认回调：CLI 用 `input()`，API 用自动确认。这个设计避免了重复代码。

### Q: 如果用户输入"飞到上海"但没有给 GPS 坐标怎么办？

A: 当前 Demo 不支持地名解析。LLM 会要求用户提供具体坐标，或者 LLM 自己尝试猜一个坐标（然后被 SafetyGate 拦截或放行——取决于坐标是否在合理范围）。生产环境需要集成地图 API（高德/天地图）将地名转为 GPS。

### Q: AgentExecutor 是线程安全的吗？

A: 当前不是。MockExecutor 的状态（位置、电量）是实例变量，多用户并发会互相影响。这是 Demo 的已知局限。生产环境每个用户会话需要独立的 Executor 实例。
