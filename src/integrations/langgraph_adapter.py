"""LangGraph integration adapter for ZCAgent.

Provides a graph-based workflow that mirrors the ZCAgent two-stage pipeline
(intent parsing → CoT reasoning / fast-path → plan-and-execute) as a
real LangGraph state graph.

Uses the real ``langgraph`` library to build and compile a ``StateGraph``
with typed state, conditional edges, and automatic state merging.

Example usage::

    from src.integrations.langgraph_adapter import create_langgraph_workflow

    workflow = create_langgraph_workflow()
    result = workflow.invoke({"user_input": "导航到天安门，顺便放首歌"})
    print(result["final_response"])
"""

from __future__ import annotations

import logging
from typing import Literal, TypedDict

from langgraph.graph import StateGraph, END, START  # type: ignore[import]

from src.agent.dispatcher import DEFAULT_FAST_PATH_THRESHOLD
from src.cockpit.intent_parser import IntentParser
from src.cockpit.safety_checker import SafetyChecker
from src.agent.cot_agent import CoTAgent
from src.agent.plan_execute_agent import PlanExecuteAgent
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# State definition (TypedDict for LangGraph)
# -----------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    """Typed state flowing through the LangGraph workflow."""
    user_input: str
    driving_state: str
    intent: dict
    safety_result: dict
    cot_result: dict
    plan_result: dict
    tool_results: dict
    final_response: str
    metadata: dict


# -----------------------------------------------------------------------
# Node implementations (return partial dicts for LangGraph state merging)
# -----------------------------------------------------------------------

def _parse_intent_node(state: WorkflowState) -> dict:
    parser = IntentParser()
    intent = parser.parse(state.get("user_input", ""))
    return {
        "intent": {
            "type": intent.intent_type.value,
            "domain": intent.domain.value,
            "confidence": intent.confidence,
            "slots": intent.slots,
        },
    }


def _safety_check_node(state: WorkflowState) -> dict:
    from src.cockpit.domains import IntentType, DomainType, ParsedIntent, INTENT_DOMAIN_MAP
    checker = SafetyChecker()
    intent = state.get("intent", {})
    try:
        intent_type = IntentType(intent.get("type", "unknown"))
    except ValueError:
        intent_type = IntentType.UNKNOWN
    domain = INTENT_DOMAIN_MAP.get(intent_type, DomainType.GENERAL)
    parsed = ParsedIntent(intent_type=intent_type, domain=domain,
                          confidence=intent.get("confidence", 0),
                          slots=intent.get("slots", {}))
    result = checker.check(parsed, state.get("driving_state", "parked"))
    return {
        "safety_result": {
            "is_safe": result.is_safe,
            "requires_confirmation": result.requires_confirmation,
            "blocked_reason": result.blocked_reason,
        },
    }


def _route_decision(state: WorkflowState) -> Literal["blocked", "tool_augment", "cot_reasoning"]:
    """根据安全状态和置信度决定走快速路径还是深度推理路径。"""
    if not state.get("safety_result", {}).get("is_safe", True):
        return "blocked"
    if state.get("intent", {}).get("confidence", 0) >= DEFAULT_FAST_PATH_THRESHOLD:
        return "tool_augment"
    return "cot_reasoning"


def _cot_node(state: WorkflowState) -> dict:
    agent = CoTAgent()
    response = agent.process(state.get("user_input", ""))
    result = {
        "cot_result": {
            "content": response.content,
            "intents": response.intent_results,
            "confidence": response.confidence,
        },
    }
    if response.intent_results:
        result["intent"] = response.intent_results[0]
    return result


def _plan_execute_node(state: WorkflowState) -> dict:
    agent = PlanExecuteAgent()
    intent_results = state.get("cot_result", {}).get("intents") or [state.get("intent", {})]
    context = {"intent_results": intent_results}
    response = agent.process(state.get("user_input", ""), context)
    return {
        "plan_result": {
            "content": response.content,
            "task_results": response.task_results,
            "confidence": response.confidence,
        },
        "final_response": response.content,
    }


def _tool_augment_node(state: WorkflowState) -> dict:
    """Optionally call external tools (Amap, web search) to enrich results."""
    intent = state.get("intent", {})
    domain = intent.get("domain", "")
    slots = intent.get("slots", {})
    tool_results = dict(state.get("tool_results", {}))

    if domain == "navigation" and slots.get("destination"):
        amap = AmapTool()
        result = amap.run(action="poi_search", keywords=slots["destination"])
        tool_results["amap"] = result.data

    if intent.get("type") in ("query", "chat", "unknown"):
        search = WebSearchTool()
        result = search.run(query=state.get("user_input", ""))
        tool_results["web_search"] = result.data

    return {"tool_results": tool_results}


def _blocked_node(state: WorkflowState) -> dict:
    reason = state.get("safety_result", {}).get("blocked_reason", "操作被阻止")
    return {"final_response": f"操作被阻止: {reason}"}


# -----------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------

def create_langgraph_workflow(config: dict | None = None):
    """Build and compile a real LangGraph workflow for ZCAgent.

    Returns a compiled LangGraph ``CompiledStateGraph`` whose ``invoke``
    method accepts a dict with ``user_input`` (and optional
    ``driving_state``) and returns a ``WorkflowState`` dict with the full
    processing results.
    """
    graph = StateGraph(WorkflowState)

    graph.add_node("parse_intent", _parse_intent_node)
    graph.add_node("safety_check", _safety_check_node)
    graph.add_node("cot_reasoning", _cot_node)
    graph.add_node("tool_augment", _tool_augment_node)
    graph.add_node("plan_execute", _plan_execute_node)
    graph.add_node("blocked", _blocked_node)

    graph.add_edge(START, "parse_intent")
    graph.add_edge("parse_intent", "safety_check")
    graph.add_conditional_edges("safety_check", _route_decision)
    graph.add_edge("blocked", END)
    graph.add_edge("cot_reasoning", "tool_augment")
    graph.add_edge("tool_augment", "plan_execute")
    graph.add_edge("plan_execute", END)

    return graph.compile()
