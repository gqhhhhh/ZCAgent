"""LangGraph integration adapter for ZCAgent.

Provides a graph-based workflow that mirrors the ZCAgent two-stage pipeline
(intent parsing → CoT reasoning / fast-path → plan-and-execute) as a
LangGraph-style state graph.

The adapter works **without** LangGraph installed by providing a lightweight
``StateGraph`` implementation.  When LangGraph is available, callers can
convert the workflow into a native LangGraph ``CompiledGraph``.

Example usage::

    from src.integrations.langgraph_adapter import create_langgraph_workflow

    workflow = create_langgraph_workflow()
    result = workflow.invoke({"user_input": "导航到天安门，顺便放首歌"})
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agent.dispatcher import AgentDispatcher
from src.agent.dispatcher import DEFAULT_FAST_PATH_THRESHOLD
from src.cockpit.intent_parser import IntentParser
from src.cockpit.safety_checker import SafetyChecker
from src.agent.cot_agent import CoTAgent
from src.agent.plan_execute_agent import PlanExecuteAgent
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Lightweight state-graph abstraction (works without langgraph installed)
# -----------------------------------------------------------------------

@dataclass
class WorkflowState:
    """Mutable state flowing through the graph."""
    user_input: str = ""
    driving_state: str = "parked"
    intent: dict = field(default_factory=dict)
    safety_result: dict = field(default_factory=dict)
    cot_result: dict = field(default_factory=dict)
    plan_result: dict = field(default_factory=dict)
    tool_results: dict = field(default_factory=dict)
    final_response: str = ""
    metadata: dict = field(default_factory=dict)


class StateGraph:
    """Minimal directed graph for workflow orchestration.

    Nodes are ``(name, fn)`` pairs where *fn* accepts and returns a
    ``WorkflowState``.  Edges can be unconditional or conditional.
    """

    def __init__(self):
        self._nodes: dict[str, Callable[[WorkflowState], WorkflowState]] = {}
        self._edges: dict[str, str | Callable[[WorkflowState], str]] = {}
        self._entry: str | None = None

    def add_node(self, name: str, fn: Callable[[WorkflowState], WorkflowState]):
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str):
        self._edges[src] = dst

    def add_conditional_edge(self, src: str, condition: Callable[[WorkflowState], str]):
        self._edges[src] = condition

    def set_entry_point(self, name: str):
        self._entry = name

    def compile(self) -> "CompiledWorkflow":
        return CompiledWorkflow(self)


class CompiledWorkflow:
    """Executable compiled form of a ``StateGraph``."""

    def __init__(self, graph: StateGraph):
        self._graph = graph

    def invoke(self, inputs: dict | WorkflowState) -> WorkflowState:
        if isinstance(inputs, dict):
            state = WorkflowState(**{k: v for k, v in inputs.items()
                                     if hasattr(WorkflowState, k)})
        else:
            state = inputs

        current = self._graph._entry
        visited: set[str] = set()

        while current and current not in visited:
            visited.add(current)
            fn = self._graph._nodes.get(current)
            if fn is None:
                break
            state = fn(state)

            edge = self._graph._edges.get(current)
            if edge is None:
                break
            if callable(edge):
                current = edge(state)
            else:
                current = edge

        return state


# -----------------------------------------------------------------------
# Node implementations
# -----------------------------------------------------------------------

def _parse_intent_node(state: WorkflowState) -> WorkflowState:
    parser = IntentParser()
    intent = parser.parse(state.user_input)
    state.intent = {
        "type": intent.intent_type.value,
        "domain": intent.domain.value,
        "confidence": intent.confidence,
        "slots": intent.slots,
    }
    return state


def _safety_check_node(state: WorkflowState) -> WorkflowState:
    from src.cockpit.domains import IntentType, DomainType, ParsedIntent, INTENT_DOMAIN_MAP
    checker = SafetyChecker()
    try:
        intent_type = IntentType(state.intent.get("type", "unknown"))
    except ValueError:
        intent_type = IntentType.UNKNOWN
    domain = INTENT_DOMAIN_MAP.get(intent_type, DomainType.GENERAL)
    parsed = ParsedIntent(intent_type=intent_type, domain=domain,
                          confidence=state.intent.get("confidence", 0),
                          slots=state.intent.get("slots", {}))
    result = checker.check(parsed, state.driving_state)
    state.safety_result = {
        "is_safe": result.is_safe,
        "requires_confirmation": result.requires_confirmation,
        "blocked_reason": result.blocked_reason,
    }
    return state


def _route_decision(state: WorkflowState) -> str:
    """根据安全状态和置信度决定走快速路径还是深度推理路径。"""
    if not state.safety_result.get("is_safe", True):
        return "blocked"
    # 与 dispatcher 保持一致的快速路径阈值
    if state.intent.get("confidence", 0) >= DEFAULT_FAST_PATH_THRESHOLD:
        return "tool_augment"
    return "cot_reasoning"


def _cot_node(state: WorkflowState) -> WorkflowState:
    agent = CoTAgent()
    response = agent.process(state.user_input)
    state.cot_result = {
        "content": response.content,
        "intents": response.intent_results,
        "confidence": response.confidence,
    }
    # Propagate refined intents for the plan step
    if response.intent_results:
        state.intent = response.intent_results[0]
    return state


def _plan_execute_node(state: WorkflowState) -> WorkflowState:
    agent = PlanExecuteAgent()
    intent_results = state.cot_result.get("intents") or [state.intent]
    context = {"intent_results": intent_results}
    response = agent.process(state.user_input, context)
    state.plan_result = {
        "content": response.content,
        "task_results": response.task_results,
        "confidence": response.confidence,
    }
    state.final_response = response.content
    return state


def _tool_augment_node(state: WorkflowState) -> WorkflowState:
    """Optionally call external tools (Amap, web search) to enrich results."""
    domain = state.intent.get("domain", "")
    slots = state.intent.get("slots", {})

    if domain == "navigation" and slots.get("destination"):
        amap = AmapTool()
        result = amap.run(action="poi_search", keywords=slots["destination"])
        state.tool_results["amap"] = result.data

    if state.intent.get("type") in ("query", "chat", "unknown"):
        search = WebSearchTool()
        result = search.run(query=state.user_input)
        state.tool_results["web_search"] = result.data

    return state


def _blocked_node(state: WorkflowState) -> WorkflowState:
    reason = state.safety_result.get("blocked_reason", "操作被阻止")
    state.final_response = f"操作被阻止: {reason}"
    return state


# -----------------------------------------------------------------------
# Public factory
# -----------------------------------------------------------------------

def create_langgraph_workflow(config: dict | None = None) -> CompiledWorkflow:
    """Build and compile a LangGraph-style workflow for ZCAgent.

    Returns a ``CompiledWorkflow`` whose ``invoke`` method accepts a dict
    with ``user_input`` (and optional ``driving_state``) and returns a
    ``WorkflowState`` with the full processing results.
    """
    graph = StateGraph()

    graph.add_node("parse_intent", _parse_intent_node)
    graph.add_node("safety_check", _safety_check_node)
    graph.add_node("cot_reasoning", _cot_node)
    graph.add_node("tool_augment", _tool_augment_node)
    graph.add_node("plan_execute", _plan_execute_node)
    graph.add_node("blocked", _blocked_node)

    graph.set_entry_point("parse_intent")
    graph.add_edge("parse_intent", "safety_check")
    graph.add_conditional_edge("safety_check", _route_decision)
    graph.add_edge("blocked", "__end__")
    graph.add_edge("cot_reasoning", "tool_augment")
    graph.add_edge("tool_augment", "plan_execute")

    return graph.compile()
