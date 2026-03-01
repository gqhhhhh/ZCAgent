"""LangChain integration adapter for ZCAgent.

Provides a LangChain-compatible tool wrapper and a helper to build a
LangChain agent that uses the ZCAgent cockpit capabilities.

When ``langchain-core`` is installed the tool classes inherit from the real
``langchain_core.tools.BaseTool``, making them first-class LangChain tools
that can be passed directly to ``create_react_agent`` / ``AgentExecutor``.
When LangChain is *not* installed they fall back to a lightweight shim so the
rest of the codebase keeps working without the extra dependency.

Example usage (LangChain installed)::

    from langchain_openai import ChatOpenAI
    from src.integrations.langchain_adapter import create_react_agent_executor

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    executor = create_react_agent_executor(llm=llm)
    result = executor.invoke({"input": "导航到天安门"})
    print(result["output"])

Example usage (no LangChain)::

    from src.integrations.langchain_adapter import create_langchain_agent

    agent_config = create_langchain_agent()
    tool = agent_config["tools"][0]
    print(tool.run("导航到天安门"))
"""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool as _LCBaseTool  # type: ignore[import]

from src.agent.dispatcher import AgentDispatcher
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Concrete tool implementations
# ---------------------------------------------------------------------------

class ZCAgentLangChainTool(_LCBaseTool):
    """LangChain tool wrapping the full ZCAgent cockpit pipeline.

    Inherits from ``langchain_core.tools.BaseTool`` when LangChain is
    installed, so instances can be passed directly to LangChain agents::

        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini")
        tool = ZCAgentLangChainTool()
        agent = create_react_agent(llm, [tool], prompt)
        executor = AgentExecutor(agent=agent, tools=[tool])
    """

    name: str = "zcagent_cockpit"
    description: str = (
        "智能座舱Agent工具，支持导航、音乐、电话、车辆控制等座舱功能。"
        "输入自然语言指令，返回执行结果。"
    )

    def __init__(self, config: dict | None = None, llm_client: Any = None, **kwargs):
        super().__init__(**kwargs)
        self._dispatcher = AgentDispatcher(config=config, llm_client=llm_client)

    def _run(self, query: str, run_manager=None, **kwargs) -> str:
        """Execute the tool (LangChain ``_run`` interface)."""
        response = self._dispatcher.process(query)
        return response.content

    async def _arun(self, query: str, run_manager=None, **kwargs) -> str:
        return self._run(query)


class AmapLangChainTool(_LCBaseTool):
    """LangChain tool wrapping the Amap (高德地图) API.

    Accepts either a plain keyword string or a JSON object string with an
    ``action`` key (``poi_search``, ``geocode``, or ``route``).
    """

    name: str = "amap_map"
    description: str = (
        "高德地图工具，支持POI搜索、地理编码、路径规划。"
        "可传入关键词字符串，或 JSON 格式参数如 "
        '{"action": "poi_search", "keywords": "加油站", "city": "北京"}。'
    )

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._amap = AmapTool(api_key=api_key)

    def _run(self, query: str, run_manager=None, **kwargs) -> str:
        try:
            params = json.loads(query)
        except (json.JSONDecodeError, TypeError):
            params = {"action": "poi_search", "keywords": query}
        result = self._amap.run(**params)
        return result.to_text() if not result.success else json.dumps(result.data, ensure_ascii=False)

    async def _arun(self, query: str, run_manager=None, **kwargs) -> str:
        return self._run(query)


class WebSearchLangChainTool(_LCBaseTool):
    """LangChain tool wrapping the web-search API."""

    name: str = "web_search"
    description: str = "网页搜索工具，输入关键词返回最新搜索结果。"

    def __init__(self, api_key: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._search = WebSearchTool(api_key=api_key)

    def _run(self, query: str, run_manager=None, **kwargs) -> str:
        result = self._search.run(query=query)
        return result.to_text() if not result.success else json.dumps(result.data, ensure_ascii=False)

    async def _arun(self, query: str, run_manager=None, **kwargs) -> str:
        return self._run(query)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_langchain_agent(
    llm: Any = None,
    config: dict | None = None,
    llm_client: Any = None,
) -> dict:
    """Return a configuration dict with all ZCAgent LangChain tools.

    The returned dict contains a ``tools`` list that can be passed to
    ``initialize_agent`` or ``AgentExecutor`` when LangChain is installed,
    or iterated directly when it is not.

    Args:
        llm: A LangChain-compatible LLM instance (optional).
        config: ZCAgent configuration dict.
        llm_client: ZCAgent ``LLMClient`` instance.

    Returns:
        Dict with keys ``tools``, ``description``, and ``llm``.
    """
    tools = [
        ZCAgentLangChainTool(config=config, llm_client=llm_client),
        AmapLangChainTool(),
        WebSearchLangChainTool(),
    ]
    return {
        "tools": tools,
        "description": "ZCAgent LangChain integration with cockpit, map, and search tools",
        "llm": llm,
    }


def create_react_agent_executor(
    llm: Any = None,
    config: dict | None = None,
    llm_client: Any = None,
    verbose: bool = False,
) -> Any:
    """Build a ready-to-use LangChain ``AgentExecutor`` using the ReAct pattern.

    Requires ``langchain``, ``langchain-core``, and a compatible LLM package
    (e.g. ``langchain-openai``) to be installed.  Raises ``ImportError`` when
    they are missing.

    Args:
        llm: A LangChain chat model (e.g. ``ChatOpenAI``).  When *None* a
             ``ChatOpenAI(model="gpt-4o-mini")`` is instantiated automatically
             using the ``OPENAI_API_KEY`` environment variable.
        config: Optional ZCAgent configuration dict passed to the cockpit tool.
        llm_client: Optional ZCAgent ``LLMClient`` passed to the cockpit tool.
        verbose: Whether to enable verbose logging in ``AgentExecutor``.

    Returns:
        A LangChain ``AgentExecutor`` that can be called with
        ``executor.invoke({"input": "..."})`` or streamed with
        ``executor.stream(...)``.

    Example::

        from langchain_openai import ChatOpenAI
        from src.integrations.langchain_adapter import create_react_agent_executor

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        executor = create_react_agent_executor(llm=llm)
        result = executor.invoke({"input": "导航到天安门，顺便查一下天气"})
        print(result["output"])
    """
    from langchain.agents import create_react_agent, AgentExecutor  # type: ignore[import]
    from langchain import hub  # type: ignore[import]

    if llm is None:
        from langchain_openai import ChatOpenAI  # type: ignore[import]
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [
        ZCAgentLangChainTool(config=config, llm_client=llm_client),
        AmapLangChainTool(),
        WebSearchLangChainTool(),
    ]

    # Pull the standard ReAct prompt from LangChain Hub (requires network) or
    # fall back to a minimal built-in prompt.
    try:
        prompt = hub.pull("hwchase17/react")
    except Exception:
        from langchain_core.prompts import PromptTemplate  # type: ignore[import]
        prompt = PromptTemplate.from_template(
            "你是智能座舱助手，请使用以下工具完成用户请求。\n\n"
            "可用工具:\n{tools}\n\n"
            "工具名称: {tool_names}\n\n"
            "请按照以下格式思考并回答:\n"
            "Thought: 我需要做什么\n"
            "Action: 工具名称\n"
            "Action Input: 工具输入\n"
            "Observation: 工具返回结果\n"
            "... (可重复 Thought/Action/Observation)\n"
            "Thought: 我已经有了答案\n"
            "Final Answer: 最终回答\n\n"
            "历史对话:\n{chat_history}\n\n"
            "用户: {input}\n"
            "{agent_scratchpad}"
        )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=5,
    )
