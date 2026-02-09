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
│       ├── langchain_adapter.py # LangChain Tool 适配
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

# 安装依赖
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

## 🔗 框架集成

### LangChain

```python
from src.integrations.langchain_adapter import ZCAgentLangChainTool, create_langchain_agent

# 作为单个 Tool 使用
tool = ZCAgentLangChainTool()
result = tool.run("导航到天安门")

# 创建包含座舱 + 地图 + 搜索的 Agent 工具集
agent_config = create_langchain_agent(llm=your_llm)
# agent_config["tools"] 可直接传入 initialize_agent() 或 AgentExecutor
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
| `OPENAI_API_BASE` | 自定义 API 端点 |
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
