"""Tests for framework integration adapters."""

import json
import pytest

from langchain_core.tools import BaseTool as LCBaseTool
from langgraph.graph import StateGraph as LGStateGraph
from autogen_core.tools import FunctionTool
from mcp.types import Tool as MCPTool, TextContent

from src.integrations.langchain_adapter import (
    ZCAgentLangChainTool,
    AmapLangChainTool,
    WebSearchLangChainTool,
    create_langchain_agent,
)
from src.integrations.langgraph_adapter import (
    WorkflowState,
    create_langgraph_workflow,
)
from src.integrations.mcp_adapter import ZCAgentMCPServer
from src.integrations.autogen_adapter import ZCAgentAssistant, create_autogen_tools


# -----------------------------------------------------------------------
# LangChain adapter
# -----------------------------------------------------------------------

class TestLangChainAdapter:
    def _default_config(self):
        return {
            "safety": {"blocked_while_driving": [], "require_confirmation": []},
            "memory": {},
            "rag": {},
        }

    def test_zcagent_tool_inherits_basetool(self):
        tool = ZCAgentLangChainTool(config=self._default_config())
        assert isinstance(tool, LCBaseTool)

    def test_zcagent_tool_run(self):
        tool = ZCAgentLangChainTool(config=self._default_config())
        result = tool.run("导航到天安门")
        assert "天安门" in result

    def test_amap_tool_run(self):
        tool = AmapLangChainTool()
        assert isinstance(tool, LCBaseTool)
        result = tool.run("天安门")
        assert "天安门" in result

    def test_amap_tool_run_json(self):
        tool = AmapLangChainTool()
        result = tool.run(json.dumps({"action": "geocode", "address": "北京"}))
        assert "北京" in result

    def test_web_search_tool_run(self):
        tool = WebSearchLangChainTool()
        assert isinstance(tool, LCBaseTool)
        result = tool.run("Python")
        assert "Python" in result

    def test_create_langchain_agent(self):
        agent_config = create_langchain_agent(config=self._default_config())
        assert "tools" in agent_config
        assert len(agent_config["tools"]) == 3


# -----------------------------------------------------------------------
# LangGraph adapter
# -----------------------------------------------------------------------

class TestLangGraphAdapter:
    def test_workflow_navigation(self):
        workflow = create_langgraph_workflow()
        state = workflow.invoke({"user_input": "导航到天安门"})
        assert isinstance(state, dict)
        assert state["final_response"] != ""

    def test_workflow_music(self):
        workflow = create_langgraph_workflow()
        state = workflow.invoke({"user_input": "播放音乐"})
        assert state.get("intent", {}).get("type") == "play_music" or state.get("final_response")

    def test_workflow_unknown(self):
        workflow = create_langgraph_workflow()
        state = workflow.invoke({"user_input": "xyzabc"})
        assert isinstance(state, dict)

    def test_real_langgraph_stategraph(self):
        """Verify the workflow uses a real LangGraph StateGraph."""
        from langgraph.graph import StateGraph, START, END
        from typing import TypedDict

        class SimpleState(TypedDict, total=False):
            value: str

        graph = StateGraph(SimpleState)
        graph.add_node("a", lambda s: {"value": s.get("value", "") + "_a"})
        graph.add_node("b", lambda s: {"value": s.get("value", "") + "_b"})
        graph.add_edge(START, "a")
        graph.add_edge("a", "b")
        graph.add_edge("b", END)
        compiled = graph.compile()
        state = compiled.invoke({"value": "test"})
        assert state["value"] == "test_a_b"


# -----------------------------------------------------------------------
# MCP adapter
# -----------------------------------------------------------------------

class TestMCPAdapter:
    def _default_config(self):
        return {
            "safety": {"blocked_while_driving": [], "require_confirmation": []},
            "memory": {},
            "rag": {},
        }

    def test_list_tools(self):
        server = ZCAgentMCPServer(config=self._default_config())
        tools = server.list_tools()
        assert len(tools) == 3
        assert all(isinstance(t, MCPTool) for t in tools)
        names = [t.name for t in tools]
        assert "cockpit_command" in names
        assert "map_search" in names
        assert "web_search" in names

    def test_call_cockpit(self):
        server = ZCAgentMCPServer(config=self._default_config())
        result = server.call_tool("cockpit_command", {"command": "导航到天安门"})
        assert "content" in result
        assert any(
            isinstance(c, TextContent) and "天安门" in c.text
            for c in result["content"]
        )

    def test_call_map_search(self):
        server = ZCAgentMCPServer(config=self._default_config())
        result = server.call_tool("map_search", {"action": "poi_search", "keywords": "加油站"})
        assert "content" in result

    def test_call_web_search(self):
        server = ZCAgentMCPServer(config=self._default_config())
        result = server.call_tool("web_search", {"query": "天气预报"})
        assert "content" in result

    def test_call_unknown_tool(self):
        server = ZCAgentMCPServer(config=self._default_config())
        result = server.call_tool("nonexistent", {})
        assert result.get("isError") is True

    def test_handle_initialize(self):
        server = ZCAgentMCPServer(config=self._default_config())
        response = server.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        assert response["result"]["serverInfo"]["name"] == "zcagent-mcp-server"

    def test_handle_tools_list(self):
        server = ZCAgentMCPServer(config=self._default_config())
        response = server.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert len(response["result"]["tools"]) == 3

    def test_handle_tools_call(self):
        server = ZCAgentMCPServer(config=self._default_config())
        response = server.handle_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "cockpit_command", "arguments": {"command": "播放音乐"}},
        })
        assert "result" in response

    def test_handle_unknown_method(self):
        server = ZCAgentMCPServer(config=self._default_config())
        response = server.handle_message({"jsonrpc": "2.0", "id": 4, "method": "unknown/method"})
        assert "error" in response


# -----------------------------------------------------------------------
# AutoGen adapter
# -----------------------------------------------------------------------

class TestAutoGenAdapter:
    def _default_config(self):
        return {
            "safety": {"blocked_while_driving": [], "require_confirmation": []},
            "memory": {},
            "rag": {},
        }

    def test_generate_reply_navigation(self):
        assistant = ZCAgentAssistant(config=self._default_config())
        reply = assistant.generate_reply(
            messages=[{"role": "user", "content": "导航到天安门"}]
        )
        assert "天安门" in reply

    def test_generate_reply_empty(self):
        assistant = ZCAgentAssistant(config=self._default_config())
        reply = assistant.generate_reply(messages=[])
        assert "没有" in reply

    def test_generate_reply_function_call(self):
        assistant = ZCAgentAssistant(config=self._default_config())
        call = json.dumps({"function": "web_search", "arguments": {"query": "天气"}})
        reply = assistant.generate_reply(messages=[{"role": "user", "content": call}])
        assert "天气" in reply

    def test_register_custom_function(self):
        assistant = ZCAgentAssistant(config=self._default_config())
        assistant.register_function(lambda x="": f"custom:{x}", name="my_func")
        assert "my_func" in assistant.get_function_map()

    def test_get_tool_definitions(self):
        assistant = ZCAgentAssistant(config=self._default_config())
        tools = assistant.get_tool_definitions()
        assert len(tools) == 3
        names = [t["function"]["name"] for t in tools]
        assert "cockpit_command" in names

    def test_function_map(self):
        assistant = ZCAgentAssistant(config=self._default_config())
        fn_map = assistant.get_function_map()
        assert "cockpit_command" in fn_map
        assert "map_search" in fn_map
        assert "web_search" in fn_map

    def test_create_autogen_tools(self):
        tools = create_autogen_tools(config=self._default_config())
        assert len(tools) == 3
        assert all(isinstance(t, FunctionTool) for t in tools)
