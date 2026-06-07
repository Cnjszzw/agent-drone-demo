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

---

## 7. 真实工程的挑战（Demo → 生产化的差距）

> 此章节是面试核心。Demo 跑通的是"理想链路"，以下三个问题是真实工程中绕不过去的坑。
> 每个问题都遵循"最初方案 → 暴露问题 → 根因分析 → 最终方案"的叙事结构。

### 7.1 难点一：LLM 幻觉 — 从 Tool A 切换到方案 B

**背景**：人工操控无人机前，前端会在 Cesium 地图上画一条预览线（从当前位置到目标点）和一个目标点图标。这是 JS 调 Cesium API 实现的——Agent（Python 进程）没法直接画。

**最初方案（方案 A）**：把"通知前端画预览线"做成一个独立的 `notify_frontend` Tool，注册在 ALL_TOOLS 里，让 LLM 在调用 `fly_to_point` 之前调用。

```
LLM 推理流程:
  1. LLM 决定调 notify_frontend(type="flight_preview", ...)
  2. notify_frontend → POST /api/agent/notify → Java WS → 前端画预览线
  3. LLM 决定调 fly_to_point(lat, lng, height)
```

**暴露的问题**：测试中 LLM 偶发漏调 `notify_frontend`——直接跳过前端通知调了 `fly_to_point`。概率约 10%，受 temperature、system prompt 措辞、输入表达方式等因素影响。问题是它的不稳定性——你没法保证 100% 每次都会调。

**根因分析**：`notify_frontend` 不是意图理解问题。飞行前一定需要预览通知，这是一个**确定性规则**，不应该由概率模型（LLM）来决策。把确定性规则包装成 Tool 交给 LLM，本质上混淆了两个不同性质的决策域。

**最终方案（方案 B）**：将通知逻辑从 Tool 下沉到 `fly_to_point` 工具函数内部，作为 SafetyGate 校验通过后的第一个硬编码步骤：

```python
@tool
def fly_to_point(lat, lng, height):
    # 1. SafetyGate 校验（硬编码）
    result = safety_gate.validate_fly(lat, lng, height)
    
    # 2. 通知前端画预览线（硬编码 — 不交给 LLM 决策）
    requests.post(WVP_BASE + "/api/agent/notify", {
        "type": "flight_preview",
        "lat": lat, "lng": lng, "height": height
    })
    # Java 收到后通过现有 WebSocket 推送到前端
    # 前端 on("agent.flight.preview") → Cesium 画线
    
    # 3. Human-in-the-loop 确认
    confirm = ...
    
    # 4. 执行飞行
    executor.fly_to_point(lat, lng, height)
```

**为什么走 Java WebSocket 中转而不用 Python 直推前端**：前端跟 Java 之间已有 WebSocket 长连接——鉴权、心跳、Session 管理全在这条通道上。Python 再开一条 WebSocket 需要重复实现鉴权和 Session 管理。单机部署下一次 HTTP 调用（Python → Java）延迟 <5ms，代价可忽略。

**面试话术**：

> "最初我把前端通知设计成了独立的 notify_frontend Tool，让 LLM 自主决定是否调用。结果 LLM 偶尔跳过——不是每次都漏，但你不能接受一个 10% 机率不出预览线的系统。根因是：飞行前必须通知前端是确定性规则，不是意图理解问题。我把这个逻辑从 Tool 层移到了 fly_to_point 函数内部——SafetyGate 校验通过后自动通知，不走 LLM 决策路径。这个原则贯穿了整个设计：SafetyGate、确认流程、前端通知，凡是'一定做'的事都不要交给概率模型。"

**核心原则**：**能硬编码的规则不交给 LLM 决策。** 这是区分"调库工程师"和"Agent 工程化"的分水岭。

---

### 7.2 难点二：同步 LLM vs 异步无人机 — 如何感知进度

**问题**：LLM 是同步推理的（一次 tool call 秒级返回），无人机操作是异步的（飞行 2-3 分钟、录像 60s）。LLM 调 `fly_to_point()` 收到返回码就以为完成了——实际飞机还在半路。

**真实生产链路**：

```
嵌入式 → MQTT {status:"in_progress", progress:60%}
    → Java wvp-server 消费
    → 写入 Redis: drone:task:T001:status
    → Python redis-py: GET drone:task:T001:status（<1ms, localhost）
    → 进度仅推前端展示，LLM 不感知
    → 直到 status=completed/failed 才返回给 LLM
```

**进度感知方案对比**（面试时这张表直接抛出来）：

| 方案 | 做法 | 否决原因 |
|------|------|---------|
| HTTP 轮询 Java | Python GET /api/task/status | HTTP 序列化开销不必要 |
| Python 直连 MQTT | paho-mqtt 订阅 topic | 需重复实现消息解析逻辑，违背分层原则 |
| Python ↔ Java WS | WS 推送，Python 本地缓存 | 连接管理复杂，收益被 1s 轮询抵消 |
| Java HTTP 回调 Python | Java 收到 MQTT 后 POST Python | 飞行状态变化多次，回调只能触发一次 |
| ✅ **Redis 直连** | Python redis-py GET | **选中：<1ms、零新依赖、零连接管理** |

**为什么 Redis 直连是最优解**：
1. Python 只是"观察者"（只读），不写入，不破坏数据一致性
2. localhost 下一次 GET <1ms，60-180 次/分钟对 Redis 零负载
3. 飞行耗时 2-3 分钟，1s 轮询延迟 0.8%，完全可忽略
4. 无新增依赖（redis-py 是 Python 标配库）
5. 无连接管理、无心跳维护、无状态同步问题

**核心原则**：进度只推前端，不推 LLM。LLM 不需要也不应该知道"飞了 60%"，它只关心终点：到达/失败/超时。

**面试话术**：

> "无人机飞行是分钟级异步过程，LLM 是秒级同步推理。弥合这个 gap 的关键是——工具函数内部封装轮询等待，LLM 不感知异步过程。Python 通过 redis-py 直连 localhost Redis，每秒 GET 一次任务状态——单机 <1ms，对 Redis 零负载。中间进度只推前端展示，LLM 只在到达/失败时得到最终结果。
>
> 方案选型上，我们评估了五种方案。Python 直连 MQTT 需要重复实现 Java 已有的消息解析逻辑；WS 长连接引入连接管理复杂度；HTTP 回调只能触发一次。Redis 直连是最简单的正确解——无新依赖、无连接管理、1s 轮询对 2-3 分钟飞行完全可接受。"

---

### 7.3 难点三：工具粒度 — 暴露业务能力还是硬件原语

**问题**：无人机硬件只提供 `start_recording` 和 `stop_recording` 两个原子指令。"录制 60 秒视频"需要调 start → 等待 60s → 调 stop。如果把两个原子指令都暴露给 LLM，LLM 需要自己编排时序——但 LLM 没有时间感知能力，它无法可靠地"等 60 秒再调 stop"。

```
把原子指令暴露给 LLM:
  1. LLM 调 start_recording(0)
  2. 返回 "录像已开始"
  3. LLM 想: 用户说录像 60 秒，我已经开始了，下一步是什么？
  4. LLM 可能立即调 stop_recording → 录了 0 秒
  5. 或者 LLM 直接忘了调 stop → 录到没电
```

**解决**：Agent 工具层做复合封装（Composite Tool），暴露业务语义而非硬件原语。

```python
# 不暴露给 LLM 的底层原子指令（私有函数）
def _start_recording():
    """硬件原语：开始录像"""
    backend.post("/drone/camera/record/start")

def _stop_recording():
    """硬件原语：停止录像"""
    backend.post("/drone/camera/record/stop")

# 暴露给 LLM 的业务工具（复合语义）
@tool
def record_for_duration(duration_seconds: int):
    """
    录制指定时长的视频。
    底层自动处理 start → 等待 → stop 流程。
    """
    _start_recording()     # 1. 开始录像
    
    for i in range(duration_seconds):
        sleep(1)            # 2. 等待指定时长
        # 同时检查：录像有没有被意外中断？
        if _get_recording_status() == "error":
            _stop_recording()
            return "❌ 录像异常中断: 存储卡已满"
    
    _stop_recording()       # 3. 停止录像
    return f"✅ 录像完成，时长 {duration_seconds}s，文件已保存"
```

**原则**：Agent 工具暴露的是业务能力，不是硬件接口。就像操作系统提供 `read/write`（硬件原语），上层库封装出 `fopen/fclose`（业务语义）。Agent 需要的是 `record_for_duration(60)`，不是 start + sleep + stop。

**面试话术**：

> "硬件只有 start 和 stop 两个原子指令。如果直接暴露给 LLM，LLM 需要自己编排'开始→等待→停止'的时序。但 LLM 没有时间感知能力——它可能开始录像后立即停止，或者忘了停止。我们做了工具层的复合封装——`record_for_duration(60)` 内部自动处理 start→倒计时→stop，LLM 只看到业务语义。原则是：Agent 工具暴露的是业务能力，不是硬件接口。"

---

### 7.4 三个难点的共同逻辑

| 难点 | 最初方案 | 问题根因 | 最终方案 | 核心原则 |
|------|---------|---------|---------|---------|
| 前端预览通知 | Tool A: notify_frontend 交给 LLM | LLM 偶发漏调，确定性规则交给了概率模型 | Tool 内嵌：fly_to_point 内部硬编码通知 | 能硬编码的规则不交给 LLM |
| 异步进度感知 | Demo 同步 sleep | 真实操作是分钟级异步过程 | 工具层轮询等待，LLM 不感知 | 工具函数是同步接口，异步在内部 |
| 工具粒度 | 暴露 start/stop 原子指令 | LLM 无时间感知，无法可靠编排时序 | 复合封装 record_for_duration | 工具暴露业务语义，非硬件原语 |

**一条线串起来**：这三个问题的本质都是"LLM 能做什么 vs 不该做什么"的边界划分。Agent 工程化的核心能力不是写 System Prompt，而是识别哪些逻辑应该留在 Tool 里（确定性、安全性、时序控制），哪些可以交给 LLM（意图理解、语义拆解、自然语言生成）。

---

### 7.5 行业验证：这个设计模式是不是编造的

> 面试时可能会被问"别人也这么做吗"或"为什么不用更实时的方式"。以下是行业证据。

#### 核心模式：把异步等待封装在工具函数内部

你的设计核心不是 Redis 轮询这个具体技术选择，而是一个更根本的架构决策：

> **工具函数内部封装异步等待，对 LLM 暴露同步接口。**
> **LLM 不需要知道"进行中"，它只关心终点：完成了还是失败了。**

这个模式在以下场景中都是标准实践：

#### Anthropic Claude Computer Use（2024.10）

```
用户: "提交这个表单"
Claude: computer_use.click(x, y)
  → 工具内部: 点击按钮 → 截图检测页面变化 → URL 没变？重试 → 页面跳转了 → 返回
  → 对 Claude 来说: 一次函数调用。Claude 不感知"页面正在加载 30%"
```

Anthropic 没让 Claude 轮询"页面加载百分比"，它把"点击 + 等待页面跳转"封装在工具里。跟你把"下发指令 + 等待飞行到达"封装在 `fly_to_point` 里是同一个模式。

#### DJI 司空 2 Copilot（你们对标的产品）

从 Copilot 的文档可以看到它的执行流程：
- "飞到公园大门口" → 用户看到进度卡片 → Copilot 内部等待 → 到达后执行下一步
- Copilot 不会在飞了 30% 的时候回来问 LLM"还要继续吗"
- 进度只推给用户看（前端卡片），不回传给 LLM 做决策

跟你的设计完全一致。

#### 机器人操控领域

```
"拿起杯子"
→ 工具函数: 机械臂下降 → 力传感器检测到阻力 → 夹爪闭合 → 抬升 → 检测重量确认抓取 → 返回
```

机器人学领域几十年来一直是这个模式：**planning 层做规划，execution 层阻塞等待，planning 层只看最终结果。** Action Server 封装了执行细节，Planner 只关心高层次的完成/失败信号。

#### OpenAI / Anthropic 的 Function Calling API 本身

两者的 API 设计都没有提供"异步回调"的 Function Calling 机制。不是他们没做——是这个模式不需要。Function Calling 就是同步的：LLM 调用工具，工具返回结果，LLM 基于结果继续推理。

#### 面试防御话术

如果面试官追问"轮询是不是不太优雅"：

> "你说得对，最理想的方案是 Java 收到 MQTT 完成事件后通过消息队列通知 Python。但在私有化部署、单机单 Agent 的场景下，引入消息队列的复杂度远超 1s 轮询的代价。这是务实的工程选择，不是不知道有更好的方案。把异步等待封装在工具函数内部的模式本身是行业共识——Anthropic Computer Use、DJI Copilot、机器人操控都是这么做的。"

如果追问"飞行 30 分钟，轮询 1800 次怎么办"：

> "飞行超过 10 分钟的场景，轮询间隔可以动态调整——前 30 秒 1s 间隔保证响应性，之后降到 5-10s。这不是架构问题，是轮询策略配置。实际上无人机单次航线一般在 15 分钟以内，DJI Matrice 系列最长悬停也就 35 分钟。"

---

### 7.6 LangGraph 接入点：环绕飞行子图

> 面试时被问"用过 LangGraph 吗"——以下内容就是你的回答。

#### 现状痛点：AgentExecutor 处理不了环绕飞行

DJI Copilot 的场景四——"飞到信号塔，环绕并录像"：

```
执行流程: 飞到目标 → 开始录像 → 环绕一圈（每30°拍照）→ 停止录像 → 展示结果
```

AgentExecutor 的问题是——环绕一圈要等多久、怎么知道绕完了——这些都依赖 LLM 判断：

```
用 AgentExecutor 处理:
  LLM: 开始录像 → LLM: 启动环绕 → 
  LLM: （等了片刻）拍一张 → LLM: （又等了片刻）再拍一张 →
  LLM: 应该绕得差不多了吧？停止录像
  ↑ LLM 没有时钟，完全在猜时机
```

#### 方案：环绕飞行子图用 LangGraph，主链路保持 AgentExecutor

不是全量替换——主链路（指点飞行、拍照、返航）继续用 AgentExecutor。只在环绕飞行这种"有明确状态机"的子任务上引入 LangGraph。

```python
# langgraph_orbit.py —— 环绕巡查子图（伪代码，演示 LangGraph 概念）

# ── 状态定义 ──
class OrbitState(TypedDict):
    target: tuple[float, float]     # 目标坐标
    radius: float                   # 环绕半径
    current_angle: float            # 当前角度 0-360
    photos_taken: int               # 已拍张数
    recording: bool

def build_orbit_graph():
    graph = StateGraph(OrbitState)

    # ── 节点 ──
    graph.add_node("approach",      fly_to_target)       # 飞向目标
    graph.add_node("start_record",  start_recording)     # 开始录像
    graph.add_node("enter_orbit",   lock_poi)            # 锁定兴趣点
    graph.add_node("orbit_step",    orbit_one_step)      # 前进 30° + 拍照
    graph.add_node("exit_orbit",    exit_orbit)          # 退出环绕
    graph.add_node("stop_record",   stop_recording)      # 停止录像

    # ── 边 ──
    graph.add_edge(START, "approach")
    graph.add_edge("approach", "start_record")
    graph.add_edge("start_record", "enter_orbit")
    graph.add_edge("enter_orbit", "orbit_step")

    # 条件边: 图决定何时绕完，不是 LLM
    graph.add_conditional_edges(
        "orbit_step",
        lambda state: "exit_orbit" if state["current_angle"] >= 360 else "orbit_step",
        # θ < 360°: 继续绕          θ >= 360°: 退出
    )

    graph.add_edge("exit_orbit", "stop_record")
    graph.add_edge("stop_record", END)

    return graph.compile()
```

#### 三个核心差异

**① 环绕何时结束 — 图的条件边，不是 LLM 猜**

```
AgentExecutor: LLM 自己判断"应该绕完了"
LangGraph:     orbit_step → 条件边: θ >= 360°? → exit_orbit
               飞行进度通过 Redis 轮询拿当前角度
               图的边在运行时判断，不经过 LLM
```

**② 拍照时机 — 图节点内置的逻辑**

```
AgentExecutor: LLM 决定"现在拍一张"
LangGraph:     orbit_step 每前进 30° 内置一次 take_photo
               一圈自动 12 张，均匀覆盖 360°
```

**③ 录像的起止 — 帧定在图的边上**

```
AgentExecutor: LLM 可能忘了关录像
LangGraph:     approach → start_record 固定边
               exit_orbit → stop_record 固定边
               录像时长 = 精确等于一圈环绕时间
```

#### 角度怎么感知、每 30 度怎么触发

面试时一定会被追问细节。核心机制很简单：

**角度计算**：无人机当前位置 (lat1, lng1) 和圆心（信号塔）(lat0, lng0)，用 atan2 计算方位角（0-360 度）。

**每 30 度触发**：`orbit_loop` 节点内部每秒从 Redis OSD 轮询一次无人机位置，实时计算角度。发现跨过 30 度的整数倍（如 29° -> 31°）就调用一次 `take_photo`。满 360 度时触发条件边退出。

```
orbit_loop 内部逻辑:
  1. redis.get(osd_position)   # 每秒查一次
  2. angle = atan2(lat-lat0, lng-lng0)
  3. if angle - last_trigger_angle >= 30:
         take_photo()
         last_trigger_angle = angle
  4. if angle >= 360:
         return → 条件边路由到 exit_orbit
  5. sleep(1) → 回到步骤 1
```

无人机环绕速度由硬件控制（如 3m/s），绕 80m 半径圆大约 167 秒一圈。每 30 度约 14 秒。1 秒轮询一次的粒度足够精确。

#### 为什么不能靠 LLM 判断拍照时机

不只是"LLM 没有时钟"的问题，更致命的是延迟：

- LLM 推理一次 1-3 秒。无人机环绕速度约 3m/s，3 秒飞出 9 米。
- 30 度的弧段（80m 半径）约 42 米。等 LLM 返回"该拍照了"，飞机已经错过最佳拍摄点 3-9 米。
- 12 次拍照就是 12 次 LLM 调用，每次都有这个延迟。累积误差让一圈拍下来位置全乱。

**确定性代码检查 `if angle >= 30:` 微秒级完成。LLM 延迟让实时控制物理上不可行。** 这不是"不够好"，是根本不该用 LLM 的场景。

**LangGraph 在这里的价值**：不是能做 while 循环做不到的事，而是当系统有多种子流程（环绕/航线/返航），各自有分支和条件时，用图表达比 if-else 堆在工具函数里更清晰。加上 `interrupt()` 机制可以在急停时把整个子图挂起——普通 while 循环做不到。

#### 为什么不全换 LangGraph

```
主链路（指点飞行→拍照→返航）是线性任务链，AgentExecutor 够用。
环绕飞行是有状态机的子任务，LangGraph 合适。
混用 = 务实，全换 = 过度设计。
```

#### 面试话术

> "我评估了 LangGraph 作为补充编排框架的可行性。LangChain AgentExecutor 对线性任务链（飞到→拍照→返航）足够简洁，但对于有明确状态机的子任务——比如 DJI Copilot 的'兴趣点环绕巡查'——AgentExecutor 依赖 LLM 自主判断时机，这不可靠。
>
> 环绕飞行天然是状态机：进入环绕 → 每 30° 拍照 → θ >= 360° 退出。我用 LangGraph 把它建模为子图，执行周期由图的边决定而非 LLM 判断。LangGraph 的 interrupt() 也解决了环绕过程中的急停——图挂起后所有并行分支一次性暂停。
>
> 这不是'全换 LangGraph'，是'主链路 AgentExecutor + 状态机子图 LangGraph'的务实混用。"

#### 参考代码

验证脚本见 `langgraph_orbit.py`——无 langgraph 依赖，通过注释和伪代码演示完整子图结构。面试时可以打开 IDE 展示。

---

### 7.7 MCP 集成：高德地图地理编码

#### 痛点：用户说地名，LLM 不知道坐标

```
用户: "飞到紫竹高新区5号楼，拍照"

AgentExecutor 面临的问题:
  fly_to_point 需要 (lat, lng, height) 三个参数
  LLM 不知道"紫竹高新区5号楼"的坐标
  → 如果 LLM 编造坐标 → SafetyGate 可能拦截或放行错误的点
  → 用户体验差：不能说地名，必须手动输入 GPS 数字
```

#### 方案：MCP 协议集成高德地图 Server

MCP（Model Context Protocol）是 Anthropic 提出的开放标准——让 LLM 能安全地调用外部工具和数据源。高德地图提供了官方 MCP Server，通过 StreamableHTTP 暴露 15 个地图工具。

接入方式：

```python
# mcp_tools.py —— 核心就是这两行
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "amap": {
        "transport": "streamable_http",
        "url": "https://mcp.amap.com/mcp?key=YOUR_KEY",
    }
})
await client.__aenter__()
tools = client.get_tools()  # 返回 15 个标准 LangChain BaseTool
ALL_TOOLS.extend(tools)     # 注册到 Agent
```

MCP 握手流程（框架自动完成）：

```
Python Agent                       高德 MCP Server
    │                                    │
    │── POST initialize ────────────────→│  JSON-RPC 2.0
    │←── sessionId + capabilities ──────│  协议版本: 2025-03-26
    │                                    │
    │── POST tools/list ────────────────→│
    │←── 15 个工具定义 (含 Schema) ──────│
    │                                    │
    │── POST tools/call {maps_geo} ─────→│  运行时调用
    │←── {"location":"121.50,31.23"} ───│
```

#### 15 个高德工具

| 工具 | 功能 | Agent 何时调用 |
|------|------|-------------|
| `maps_geo` | 地址→坐标（地理编码） | 用户说"飞到陆家嘴" |
| `maps_regeocode` | 坐标→地址（逆编码） | 报告无人机当前位置 |
| `maps_text_search` | POI 关键词搜索 | 搜索"最近的医院" |
| `maps_around_search` | 周边搜索 | "信号塔附近有什么" |
| `maps_direction_driving` | 驾车路径规划 | 规划地面救援路线 |
| `maps_distance` | 坐标间距计算 | 估算飞行距离 |
| `maps_weather` | 天气查询 | 飞行前检查天气条件 |

#### 完整调用链

```
用户: "飞到陆家嘴，拍照"

1. maps_geo("上海市浦东新区陆家嘴")
   → {"location": "121.507513,31.234295"}

2. fly_to_point(lat=31.23, lng=121.50, height=80)
   → 内部轮询 Redis → "✅ 已到达"

3. take_photo(count=1)
   → "✅ 拍照完成"
```

#### 面试话术

> "用户习惯说地名而不是 GPS 坐标。我们通过 MCP 协议接入了高德地图的官方 Server——`langchain-mcp-adapters` 一行 `MultiServerMCPClient` 完成握手，15 个地图工具自动注册为 LangChain Tool。Agent 在处理'飞到陆家嘴'这类指令时，会先调 `maps_geo` 做地理编码，拿到坐标后再调 `fly_to_point`。这比硬编码坐标列表或让 LLM 猜坐标都可靠。"

#### #### MCP 关键难点：地名歧义——多个候选坐标如何选

maps_geo 查询"陆家嘴"返回了 6 个结果（云南·昆明、湖北·武汉、江西·鹰潭、江苏·南通、江苏·昆山、上海·浦东）。代码不能硬编码"取第一个"或"取最后一个"——高德 API 的排序逻辑在不同版本和地区可能不同。

解决方案：将候选列表回传给 LLM，让 LLM 根据上下文语义选择：

```
maps_geo("陆家嘴") → 6 个候选
  → 格式化为: "[0] 城市:昆明 区:西山 (102.69,25.00)
                [1] 城市:武汉 区:武昌 (114.29,30.51)
                ...
                [5] 城市:上海 区:浦东 (121.50,31.23)"
  → LLM 收到 prompt: "用户想飞到「陆家嘴」，高德返回了以下候选项，请选择最匹配的一个"
  → LLM 输出: "5"
  → 取候选[5]坐标填入 fly_to_point
```

**面试话术**：

> "MCP 调用不是无脑的——地理编码返回多个候选项时，你不能硬编码取第几个。我们把候选列表格式化后回传给 LLM，让 LLM 根据上下文语义选择最佳匹配。比如'陆家嘴'有 6 个候选，LLM 理解用户大概率指的是上海浦东，选第 5 个。这比硬编码规则更可靠。"

---

### 7.8 Plan → Confirm → Execute 三段式流程

> 面试时被问"Agent 和普通对话机器人有什么区别"——这就是关键差异。

#### 设计：参照 DJI Copilot 交互模式

DJI Copilot 的交互不是"用户说一句话→Agent 直接执行"——而是先生成任务规划卡片，用户审核步骤和参数后手动确认执行。高风险操作必须有人工确认环节。设计为三段式：

**Phase 1: Plan（规划）**

```
POST /api/agent/plan

LLM 收到用户指令 → 输出结构化 JSON:
{
  "objective": "飞往陆家嘴，变焦7倍后拍摄照片并返航",
  "steps": [
    {"description": "获取陆家嘴坐标", "tool": "maps_geo", "tool_args": {"address": "陆家嘴"}},
    {"description": "飞向陆家嘴上空100m", "tool": "fly_to_point", "tool_args": {"lat": null, "lng": null, "height": 100}},
    {"description": "设置7倍变焦", "tool": "set_zoom", "tool_args": {"factor": 7}},
    {"description": "拍摄照片", "tool": "take_photo", "tool_args": {"count": 1}},
    {"description": "返航降落", "tool": "return_home", "tool_args": {}}
  ],
  "preflight": {"return_altitude": "100m", "lost_action": "返航"}
}
```

注意 `fly_to_point` 的坐标是 `null`——因为还未知，需要上一步 `maps_geo` 执行后填入。

**Phase 2: Confirm（确认）**

前端渲染规划确认卡片：任务目标 + 步骤列表（○所有步骤等待中）+ 飞前检查 + 两个按钮（取消 / 立即执行）。用户审核所有步骤和飞前检查项，确认无误后手动点击"立即执行"。

**Phase 3: Execute（执行）**

```
POST /api/agent/execute  (SSE 流式)
  → step_start(0): maps_geo("陆家嘴")
  → step_done(0):  6 个候选 → LLM 选上海 → 坐标(121.50,31.23)
  → step_start(1): fly_to_point(121.50,31.23,100)
  → step_done(1):  ✅ 已到达（轮询 Redis 直到 completed）
  → step_start(2): set_zoom(7)
  → step_done(2):  ✅ 变焦完成
  → ...
  → all_done
```

#### 执行异常中止

任何步骤失败后立即 `break`，不执行后续步骤：

```python
# app.py — execute 端点内
try:
    result = await tool_func.ainvoke(tool_args)
    results.append(str(result))
    yield step_done_event

    # 工具返回 ❌ 或 ⛔ → 立即中止
    if str(result).startswith(("❌", "⛔")):
        break

except Exception as e:
    yield step_error_event
    break  # 框架异常也中止
```

防止的场景：
```
fly_to_point 失败 → 无人机已悬停
  → 如果不 break，后续会执行 set_zoom → take_photo → return_home
  → 无人机已悬停时再发指令可能撞到障碍物
```

#### 面试话术

> "我们参照 DJI Copilot，设计了 Plan → Confirm → Execute 三段式流程。LLM 先生成结构化任务计划——不是直接执行，而是让用户在确认卡片中审核。只有用户手动点击'执行'后，SSE 才开始推送每步状态。这个设计天然对应了 Human-in-the-loop 原则——高风险操作必须经过人工确认。
>
> 执行过程中任何步骤失败都立即中止后续——不是因为代码写得保守，而是无人机场景下'成功了继续、失败了也要继续'是危险的。悬停后的无人机再收指令可能撞到东西。"

参考代码见 `plan_executor.py` + `app.py` execute 端点。
