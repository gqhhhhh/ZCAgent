"""LangChain integration adapter for ZCAgent.

Provides a LangChain-compatible tool wrapper and a helper to build a
LangChain agent that uses the ZCAgent cockpit capabilities.

Example usage::

    from src.integrations.langchain_adapter import create_langchain_agent

    agent_executor = create_langchain_agent(llm=your_llm)
    result = agent_executor.invoke({"input": "导航到天安门"})
"""

import logging
from typing import Any

from src.agent.dispatcher import AgentDispatcher
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ZCAgentLangChainTool:
    """LangChain-compatible tool wrapping the full ZCAgent pipeline.

    Implements the interface expected by ``langchain_core.tools.BaseTool``
    (``name``, ``description``, ``_run``) without requiring LangChain as a
    hard dependency.  When LangChain is installed you can register an
    instance directly::

        from langchain.agents import initialize_agent, AgentType
        tool = ZCAgentLangChainTool()
        agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    """

    name: str = "zcagent_cockpit"
    description: str = (
        "智能座舱Agent工具，支持导航、音乐、电话、车辆控制等座舱功能。"
        "输入自然语言指令，返回执行结果。"
    )

    def __init__(self, config: dict | None = None, llm_client=None):
        self.dispatcher = AgentDispatcher(config=config, llm_client=llm_client)

    def _run(self, query: str) -> str:
        """Execute the tool (LangChain interface)."""
        response = self.dispatcher.process(query)
        return response.content

    # Alias so plain Python callers can use ``tool.run(...)`` as well.
    def run(self, query: str) -> str:  # noqa: D401
        return self._run(query)

    async def _arun(self, query: str) -> str:
        """Async version – falls back to sync execution."""
        return self._run(query)


class AmapLangChainTool:
    """LangChain-compatible tool wrapping the Amap API."""

    name: str = "amap_map"
    description: str = "高德地图工具，支持POI搜索、地理编码、路径规划。输入JSON字符串。"

    def __init__(self, api_key: str | None = None):
        self.amap = AmapTool(api_key=api_key)

    def _run(self, query: str) -> str:
        import json
        try:
            params = json.loads(query)
        except (json.JSONDecodeError, TypeError):
            params = {"action": "poi_search", "keywords": query}
        result = self.amap.run(**params)
        return result.to_text() if not result.success else json.dumps(result.data, ensure_ascii=False)

    def run(self, query: str) -> str:  # noqa: D401
        return self._run(query)

    async def _arun(self, query: str) -> str:
        return self._run(query)


class WebSearchLangChainTool:
    """LangChain-compatible tool wrapping web search."""

    name: str = "web_search"
    description: str = "网页搜索工具，输入关键词返回搜索结果。"

    def __init__(self, api_key: str | None = None):
        self.search = WebSearchTool(api_key=api_key)

    def _run(self, query: str) -> str:
        import json
        result = self.search.run(query=query)
        return result.to_text() if not result.success else json.dumps(result.data, ensure_ascii=False)

    def run(self, query: str) -> str:  # noqa: D401
        return self._run(query)

    async def _arun(self, query: str) -> str:
        return self._run(query)


def create_langchain_agent(
    llm: Any = None,
    config: dict | None = None,
    llm_client: Any = None,
) -> dict:
    """Create a LangChain-compatible agent configuration.

    Returns a dict describing the tools and suggested setup so that callers
    can plug them into ``initialize_agent`` or ``AgentExecutor`` when
    LangChain is installed.

    Args:
        llm: A LangChain-compatible LLM instance (optional).
        config: ZCAgent configuration dict.
        llm_client: ZCAgent ``LLMClient`` instance.

    Returns:
        Dict with ``tools`` list and ``description``.
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
