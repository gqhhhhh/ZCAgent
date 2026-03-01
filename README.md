# ZCAgent

**基于 LLM + RAG 的智能座舱一体化语义 Agent 系统**

ZCAgent 是一个面向汽车智能座舱的多 Agent 语义理解与任务执行框架。它将大语言模型（LLM）、检索增强生成（RAG）、多层记忆系统和任务调度引擎整合为一体，支持导航、音乐、电话、车辆控制等复杂场景下的自然语言交互，并提供与 **LangChain**、**LangGraph**、**MCP**、**AutoGen** 等主流框架的集成适配。

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| **多 Agent 协作** | CoT（Chain-of-Thought）深度推理 + Plan-and-Execute 任务规划双路径架构 |
| **混合 RAG 检索** | BM25 稀疏检索 + MMR 多样性检索 + ColBERT 重排序三级管线 |
| **三层记忆系统** | 工作记忆 / 短期记忆（TTL 过期） / 长期记忆（重要性评估 + 冲突消解） |
| **任务 DAG 调度** | 基于有向无环图的任务依赖管理，支持并行执行与安全优先级抢占 |
| **安全检查** | 驾驶状态感知，行驶中自动阻止危险操作，高速场景确认机制 |
| **外部 API 工具** | 高德地图（POI 搜索、地理编码、路线规划）、网页搜索 |
| **框架集成** | LangChain / LangGraph / MCP / AutoGen 适配器，可直接接入现有生态 |

---

## 📁 项目结构

```
ZCAgent/
├── config/
│   └── config.yaml              # 全局配置（LLM、RAG、记忆、任务、安全）
├── examples/
│   └── langchain_multi_agent_demo.py  # LangChain 多 Agent 协同完整演示
├── src/
│   ├── agent/                   # 多 Agent 核心
│   │   ├── base_agent.py        # Agent 基类
│   │   ├── cot_agent.py         # Chain-of-Thought 推理 Agent
│   │   ├── plan_execute_agent.py# Plan-and-Execute 任务 Agent
│   │   └── dispatcher.py        # 中央调度器（快速路径 / 深度路径）
│   ├── cockpit/                 # 座舱语义层
│   │   ├── domains.py           # 领域 & 意图类型定义
│   │   ├── intent_parser.py     # 关键词 + LLM 意图解析
│   │   └── safety_checker.py    # 安全规则引擎
│   ├── llm/
│   │   └── llm_client.py        # LLM API 抽象层（支持 Mock）
│   ├── memory/                  # 三层记忆系统
│   │   ├── working_memory.py    # 工作记忆（容量受限，重要性淘汰）
│   │   ├── short_term_memory.py # 短期记忆（TTL 自动过期）
│   │   ├── long_term_memory.py  # 长期记忆（偏好 / 事实 / 冲突消解）
│   │   └── memory_manager.py    # 统一记忆管理器
│   ├── rag/                     # 检索增强生成
│   │   ├── bm25_retriever.py    # BM25 稀疏检索
│   │   ├── mmr_retriever.py     # MMR 多样性检索
│   │   ├── colbert_reranker.py  # ColBERT 风格重排序
│   │   └── hybrid_retriever.py  # 混合检索管线
│   ├── task/                    # 任务系统
│   │   ├── task_graph.py        # 任务 DAG（依赖 + 并行）
│   │   ├── task_executor.py     # 任务执行器
│   │   └── task_scheduler.py    # 优先级调度器
│   ├── tools/                   # 外部 API 工具
│   │   ├── base_tool.py         # 工具基类
│   │   ├── amap_tool.py         # 高德地图 API（POI / 地理编码 / 路线）
│   │   └── web_search_tool.py   # 网页搜索 API
│   └── integrations/            # 框架集成适配器
│       ├── langchain_adapter.py # LangChain Tool 适配（支持真实 BaseTool）
│       ├── langgraph_adapter.py # LangGraph 状态图工作流
│       ├── mcp_adapter.py       # MCP (Model Context Protocol) 服务器
│       └── autogen_adapter.py   # AutoGen AssistantAgent 适配
└── tests/                       # 单元测试（105 个测试用例）
```

---

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/gqhhhhh/ZCAgent.git
cd ZCAgent

# 安装基础依赖
pip install -e .
```

### 基本使用

```python
from src.agent.dispatcher import AgentDispatcher

dispatcher = AgentDispatcher()

# 导航场景
response = dispatcher.process("导航到天安门")
print(response.content)  # "导航到: 天安门"

# 多任务场景
response = dispatcher.process("导航到天安门，顺便放首爵士乐")

# 安全检查（行驶中阻止危险操作）
response = dispatcher.process("看视频", driving_state="driving")
print(response.content)  # "操作被阻止: ..."
```

### 使用外部 API 工具

```python
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

# 高德地图 POI 搜索（设置 AMAP_API_KEY 环境变量使用真实 API，否则返回模拟数据）
amap = AmapTool()
result = amap.run(action="poi_search", keywords="加油站", city="北京")
print(result.data)

# 网页搜索（设置 WEB_SEARCH_API_KEY 环境变量使用真实 API）
search = WebSearchTool()
result = search.run(query="今天北京天气")
print(result.data)
```

---

## 🔗 真实 LangChain 工具集成

ZCAgent 的 `langchain_adapter` 在检测到 `langchain-core` 已安装时，会让工具类**真正继承** `langchain_core.tools.BaseTool`，使其成为标准的 LangChain 工具，可以直接传入 `create_react_agent`、`AgentExecutor` 等 LangChain 原生组件。

### 第一步：安装 LangChain 依赖

```bash
# 核心依赖（使用无已知漏洞的版本）
pip install "langchain>=0.3.25" "langchain-core>=0.3.81" "langchain-openai>=0.3.0"

# 可选：langgraph（增强状态图能力）
pip install "langgraph>=0.2.0"
```

### 第二步：配置 API Key

```bash
export OPENAI_API_KEY="sk-..."          # OpenAI 或兼容 API Key（必须）
export OPENAI_BASE_URL="https://..."    # 可选，自定义 API 端点（如 Azure、本地模型）
export AMAP_API_KEY="..."               # 可选，高德地图 Web 服务 Key
export WEB_SEARCH_API_KEY="..."         # 可选，Bing Web Search API Key
```

> 未配置地图/搜索 Key 时，工具自动返回**模拟数据**，不影响 Agent 运行。

### 第三步：使用 `@tool` 装饰器定义 LangChain 工具

LangChain 推荐用 `@tool` 装饰器快速定义工具，函数的 docstring 会成为工具描述，LLM 据此决定何时调用：

```python
from langchain_core.tools import tool
from src.agent.dispatcher import AgentDispatcher
from src.tools.amap_tool import AmapTool

dispatcher = AgentDispatcher()
amap = AmapTool()

@tool
def cockpit_command(command: str) -> str:
    """执行智能座舱指令，包括导航、音乐播放、电话、车辆控制等。
    输入自然语言指令，如"导航到天安门"、"播放爵士乐"。"""
    return dispatcher.process(command).content

@tool
def map_poi_search(keywords: str, city: str = "") -> str:
    """用高德地图搜索兴趣点（POI），如加油站、餐厅、停车场等。"""
    import json
    result = amap.run(action="poi_search", keywords=keywords, city=city)
    return json.dumps(result.data, ensure_ascii=False) if result.success else result.to_text()
```

### 第四步：创建 ReAct Agent 并调用

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [cockpit_command, map_poi_search]

# 从 LangChain Hub 获取标准 ReAct Prompt
prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)

result = executor.invoke({"input": "导航到最近的加油站，同时播放一首轻松的音乐"})
print(result["output"])
```

### 第五步：使用内置工厂函数（推荐）

```python
from langchain_openai import ChatOpenAI
from src.integrations.langchain_adapter import create_react_agent_executor

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 一行代码创建包含所有 ZCAgent 工具的 ReAct Agent Executor
executor = create_react_agent_executor(llm=llm, verbose=True)

result = executor.invoke({"input": "导航到天安门，顺便查一下今天北京的天气"})
print(result["output"])
```

---

## 🤝 多 Agent 协同工作原理

ZCAgent 实现了三种多 Agent 协同模式，从简单到复杂逐步递进。

### 模式一：ZCAgent 内置双路径 Agent（无需 LangChain）

ZCAgent 核心调度器（`AgentDispatcher`）本身就是一个协调多个专业 Agent 的系统：

```
用户输入
    │
    ▼
┌─────────────────────────────────────────┐
│           AgentDispatcher（调度器）      │
│                                         │
│  Step 1: IntentParser 解析意图           │
│          关键词匹配 + LLM 双重策略        │
│                │                        │
│  Step 2: SafetyChecker 安全检查          │
│          驾驶状态 × 操作危险等级          │
│                │                        │
│         ┌──────┴──────┐                 │
│    置信度≥0.6      置信度<0.6            │
│         │              │                │
│    快速路径        深度路径              │
│         │         CoTAgent              │
│         │    链式思维逐步推理             │
│         │              │                │
│         └──────┬───────┘                │
│                │                        │
│  Step 3: PlanExecuteAgent 任务规划        │
│          构建 DAG → 按依赖波次执行         │
│                │                        │
│  Step 4: MemoryManager 记忆存储           │
│          工作记忆 / 短期 / 长期持久化      │
└─────────────────────────────────────────┘
```

**各 Agent 职责分工：**

| Agent | 职责 | 触发条件 |
|-------|------|---------|
| `IntentParser` | 将自然语言解析为结构化意图 | 每次请求必经 |
| `SafetyChecker` | 阻止危险操作（驾驶中看视频等） | 每次请求必经 |
| `CoTAgent` | 链式思维深度推理，处理复杂/模糊意图 | 意图置信度 < 0.6 |
| `PlanExecuteAgent` | 将多意图拆解为任务 DAG 并执行 | 每次非安全拦截请求 |
| `MemoryManager` | 管理三层记忆，提供上下文 | 每次请求读写 |

**代码示例：**

```python
from src.agent.dispatcher import AgentDispatcher

dispatcher = AgentDispatcher()

# 复合请求：CoT 推理 → PlanExecute 并行执行两个任务
response = dispatcher.process("导航到天安门，顺便放首爵士乐")
# AgentDispatcher 内部流程：
# 1. IntentParser: 解析出 navigate_to + play_music 两个意图
# 2. SafetyChecker: 驻车状态，两个操作均安全
# 3. PlanExecuteAgent: 构建包含两个任务的 DAG，并发执行
# 4. MemoryManager: 保存本次交互
print(response.content)  # "导航到: 天安门；正在播放: 爵士乐"
```

### 模式二：Supervisor 多 Agent 模式（LangChain LCEL）

通过 LangChain Expression Language（LCEL），实现专业 Agent 分工 + Supervisor 协调的模式：

```
用户请求
    │
    ▼
┌──────────────────────────────────────┐
│        Supervisor Agent（协调者）     │
│  分析任务 → 路由给专业 Agent           │
│  汇总结果 → 生成最终回答              │
└──────────┬──────────────┬────────────┘
           │              │
    ┌──────▼──────┐  ┌────▼──────────────┐
    │Cockpit Agent│  │  Research Agent   │
    │ 座舱控制专家 │  │   信息检索专家    │
    │             │  │                   │
    │ 工具：       │  │ 工具：             │
    │ cockpit_cmd │  │ map_poi_search    │
    │             │  │ web_search        │
    └─────────────┘  └───────────────────┘
```

**代码示例：**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_react_agent, AgentExecutor
from src.agent.dispatcher import AgentDispatcher
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
dispatcher = AgentDispatcher()
amap = AmapTool()

# 定义专业工具
@tool
def cockpit_command(command: str) -> str:
    """执行导航、音乐、电话等座舱控制指令。"""
    return dispatcher.process(command).content

@tool
def map_search(keywords: str) -> str:
    """搜索地图POI，如加油站、停车场、餐厅。"""
    import json
    r = amap.run(action="poi_search", keywords=keywords)
    return json.dumps(r.data, ensure_ascii=False)

# Cockpit Agent：只处理座舱指令
cockpit_executor = AgentExecutor(
    agent=create_react_agent(llm, [cockpit_command], cockpit_prompt),
    tools=[cockpit_command],
    max_iterations=3,
)

# Research Agent：只处理信息检索
research_executor = AgentExecutor(
    agent=create_react_agent(llm, [map_search], research_prompt),
    tools=[map_search],
    max_iterations=3,
)

# Supervisor：聚合两个 Agent 的结果
def supervisor(user_input: str) -> str:
    cockpit_result = cockpit_executor.invoke({"input": user_input})["output"]
    research_result = research_executor.invoke({"input": user_input})["output"]
    
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是结果聚合者，整合座舱和检索结果给出最终回答。\n"
                   "座舱执行结果: {cockpit}\n检索结果: {research}"),
        ("human", "用户请求: {input}"),
    ])
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input, "cockpit": cockpit_result, "research": research_result})

# 运行
result = supervisor("帮我导航到天安门，顺便查最近加油站")
print(result)
```

### 模式三：LangGraph 状态图工作流

ZCAgent 内置 LangGraph 风格状态图，将 Agent 处理流程建模为有向图：

```
意图解析 → 安全检查 → [安全通过?]
                             │
                   ┌─────────┤─────────┐
                   │                   │
             置信度≥0.6          置信度<0.6
                   │                   │
            工具增强节点         CoT推理节点
                   │                   │
                   └─────────┬─────────┘
                             │
                       任务执行节点
```

**代码示例：**

```python
from src.integrations.langgraph_adapter import create_langgraph_workflow

workflow = create_langgraph_workflow()

# 简单导航（高置信度 → 快速路径 + 工具增强）
state = workflow.invoke({"user_input": "导航到天安门"})
print(state.final_response)   # "导航到: 天安门"
print(state.tool_results)     # {"amap": {"pois": [...]}}

# 复合请求（低置信度 → CoT 推理）
state = workflow.invoke({"user_input": "导航到天安门，顺便放首歌"})
print(state.intent)           # {"type": "navigate_to", "confidence": 0.64, ...}
print(state.cot_result)       # 推理过程

# 安全拦截（行驶中）
state = workflow.invoke({"user_input": "看视频", "driving_state": "driving"})
print(state.final_response)   # "操作被阻止: ..."
```

### 完整演示脚本

以上三种模式的可运行完整代码在 `examples/langchain_multi_agent_demo.py`：

```bash
# 无需 LangChain 即可运行（演示 LangGraph 工作流）
python examples/langchain_multi_agent_demo.py

# 安装 LangChain 后可体验 ReAct Agent 和多 Agent 协同
pip install langchain langchain-openai langchain-core>=0.3.81
export OPENAI_API_KEY="sk-..."
python examples/langchain_multi_agent_demo.py
```

---

## 🔗 框架集成

### LangChain

```python
from src.integrations.langchain_adapter import ZCAgentLangChainTool, create_langchain_agent

# 作为单个 Tool 使用（LangChain 安装后自动继承 BaseTool）
tool = ZCAgentLangChainTool()
result = tool.run("导航到天安门")

# 创建包含座舱 + 地图 + 搜索的 Agent 工具集
agent_config = create_langchain_agent(llm=your_llm)
# agent_config["tools"] 可直接传入 initialize_agent() 或 AgentExecutor

# 创建开箱即用的 ReAct AgentExecutor（需要 langchain 已安装）
from src.integrations.langchain_adapter import create_react_agent_executor
executor = create_react_agent_executor(llm=your_llm)
result = executor.invoke({"input": "导航到天安门"})
```

### LangGraph

```python
from src.integrations.langgraph_adapter import create_langgraph_workflow

workflow = create_langgraph_workflow()
state = workflow.invoke({"user_input": "导航到天安门，顺便放首歌"})
print(state.final_response)
print(state.tool_results)  # 外部 API 调用结果
```

工作流节点：`意图解析 → 安全检查 → [CoT 推理 | 快速路径] → 工具增强 → 任务执行`

### MCP (Model Context Protocol)

```python
from src.integrations.mcp_adapter import ZCAgentMCPServer

server = ZCAgentMCPServer()

# 程序化调用
tools = server.list_tools()          # 列出可用工具
result = server.call_tool(           # 调用工具
    "cockpit_command",
    {"command": "导航到天安门"}
)

# 启动 stdio 服务（可集成到 Claude Desktop 等 MCP 客户端）
# server.run_stdio()
```

### AutoGen

```python
from src.integrations.autogen_adapter import ZCAgentAssistant

assistant = ZCAgentAssistant(name="cockpit")

# 生成回复
reply = assistant.generate_reply(
    messages=[{"role": "user", "content": "导航到天安门"}]
)

# 获取函数定义（用于 OpenAI function calling）
tool_defs = assistant.get_tool_definitions()
```

---

## ⚙️ 配置

编辑 `config/config.yaml`：

```yaml
llm:
  model: "gpt-4"
  temperature: 0.3
  max_tokens: 2048
  api_base: ""       # 自定义 API 端点
  api_key: ""        # 或通过 OPENAI_API_KEY 环境变量设置

rag:
  bm25_weight: 0.4
  mmr_weight: 0.6
  top_k: 5

memory:
  working_memory_capacity: 10
  short_term_ttl_seconds: 300
  long_term_importance_threshold: 0.7

safety:
  blocked_while_driving:
    - "watch_video"
    - "browse_web"
  require_confirmation:
    - "emergency_call"
    - "open_window_highway"
```

### 环境变量

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI / 兼容 API 密钥 |
| `OPENAI_BASE_URL` | 自定义 API 端点 |
| `AMAP_API_KEY` | 高德地图 Web 服务 API Key |
| `WEB_SEARCH_API_KEY` | Bing Web Search API Key |
| `WEB_SEARCH_ENDPOINT` | 自定义搜索 API 端点 |

> 未配置 API Key 时，工具会返回**模拟数据**，方便开发和测试。

---

## 🧪 测试

```bash
pip install pytest
python -m pytest tests/ -v
```

当前共 **105 个测试用例**，覆盖：
- 意图解析与安全检查
- CoT / Plan-Execute Agent
- 三层记忆系统
- 混合 RAG 检索
- 任务图与调度器
- 外部 API 工具
- 框架集成适配器

---

## 🏗️ 系统架构

```
用户输入
  │
  ▼
┌─────────────────┐
│  IntentParser    │──── 关键词匹配 + LLM 解析
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SafetyChecker   │──── 驾驶状态安全校验
└────────┬────────┘
         │
    ┌────┴────┐
    │ 置信度  │
    │ ≥ 0.6? │
    └────┬────┘
     是 /   \ 否
       /     \
      ▼       ▼
  快速路径  深度路径
      │    ┌──────────┐
      │    │ CoTAgent │── Chain-of-Thought 推理
      │    └────┬─────┘
      │         │
      ▼         ▼
┌─────────────────────┐
│  外部工具增强         │──── 高德地图 / 网页搜索
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ PlanExecuteAgent     │──── 任务 DAG 构建 + 执行
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  MemoryManager       │──── 上下文持久化
└─────────────────────┘
```

---

## 📜 License

MIT
