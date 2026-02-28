# Integrations 模块 —— 框架集成适配器

## 概述

`integrations/` 提供与主流 AI Agent 框架的适配层，使 ZCAgent 能无缝接入现有生态。所有适配器 **不强依赖** 第三方框架——未安装时仍可独立运行。

## 文件说明

| 文件 | 框架 | 作用 |
|------|------|------|
| `langchain_adapter.py` | LangChain | 将座舱指令、地图搜索、网页搜索包装为 LangChain Tool |
| `langgraph_adapter.py` | LangGraph | 基于状态图的工作流引擎，内置轻量 `StateGraph` 实现 |
| `mcp_adapter.py` | MCP | Model Context Protocol 服务器，可对接 Claude Desktop 等客户端 |
| `autogen_adapter.py` | AutoGen | AutoGen 兼容的 AssistantAgent，支持 function calling |

## 技术栈

- **适配器模式** — 每个适配器封装 `AgentDispatcher` + 工具类，对外暴露框架原生接口
- **轻量替代** — `langgraph_adapter.py` 自带 `StateGraph` / `CompiledWorkflow`，无需安装 LangGraph
- **JSON-RPC** — MCP 适配器实现标准 JSON-RPC 2.0 协议（tools/list, tools/call）
- **OpenAI Function Calling** — AutoGen 适配器导出标准 tool definitions

## 快速使用

### LangChain

```python
from src.integrations.langchain_adapter import ZCAgentLangChainTool, create_langchain_agent

tool = ZCAgentLangChainTool()
result = tool.run("导航到天安门")

# 创建完整工具集
agent_config = create_langchain_agent(llm=your_llm)
```

### LangGraph

```python
from src.integrations.langgraph_adapter import create_langgraph_workflow

workflow = create_langgraph_workflow()
state = workflow.invoke({"user_input": "导航到天安门"})
print(state.final_response)
```

工作流节点：`意图解析 → 安全检查 → [CoT推理 | 快速路径] → 工具增强 → 任务执行`

### MCP (Model Context Protocol)

```python
from src.integrations.mcp_adapter import ZCAgentMCPServer

server = ZCAgentMCPServer()
tools = server.list_tools()
result = server.call_tool("cockpit_command", {"command": "导航到天安门"})

# 启动 stdio 服务（对接 Claude Desktop）
# server.run_stdio()
```

### AutoGen

```python
from src.integrations.autogen_adapter import ZCAgentAssistant

assistant = ZCAgentAssistant(name="cockpit")
reply = assistant.generate_reply(
    messages=[{"role": "user", "content": "导航到天安门"}]
)
```

## 扩展方式

实现新的适配器只需包装 `AgentDispatcher` 并暴露目标框架的接口。
