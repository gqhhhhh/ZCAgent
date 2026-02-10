"""AutoGen integration adapter for ZCAgent.

Provides an AutoGen-compatible assistant agent wrapper so that ZCAgent
capabilities can participate in AutoGen multi-agent conversations.

The adapter works **without** the ``autogen`` / ``pyautogen`` package
installed.  When AutoGen is available, :class:`ZCAgentAssistant` can be
used as a drop-in ``AssistantAgent`` replacement.

Example usage::

    from src.integrations.autogen_adapter import ZCAgentAssistant

    assistant = ZCAgentAssistant(name="cockpit_assistant")
    reply = assistant.generate_reply(
        messages=[{"role": "user", "content": "导航到天安门"}]
    )
"""

from __future__ import annotations

import json
import logging
from typing import Any

from src.agent.dispatcher import AgentDispatcher
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ZCAgentAssistant:
    """AutoGen-compatible assistant wrapping ZCAgent functionality.

    Mirrors the ``AssistantAgent`` interface (``generate_reply``,
    ``register_function``) so it can participate in AutoGen group chats
    or two-agent conversations.
    """

    def __init__(
        self,
        name: str = "zcagent_assistant",
        config: dict | None = None,
        llm_client: Any = None,
    ):
        self.name = name
        self.dispatcher = AgentDispatcher(config=config, llm_client=llm_client)
        self.amap = AmapTool()
        self.web_search = WebSearchTool()
        self._functions: dict[str, Any] = {}
        self._register_default_functions()

    # ------------------------------------------------------------------
    # AutoGen-compatible interface
    # ------------------------------------------------------------------

    def generate_reply(
        self,
        messages: list[dict] | None = None,
        sender: Any = None,
        **kwargs,
    ) -> str:
        """Generate a reply given conversation messages.

        Inspects the last user message for actionable cockpit commands
        and returns the execution result.

        Args:
            messages: Conversation history (list of ``{"role", "content"}``).
            sender: The sending agent (ignored in standalone mode).

        Returns:
            Reply string.
        """
        if not messages:
            return "没有收到消息"

        last_message = messages[-1].get("content", "")

        # Check for explicit function calls
        if last_message.startswith("{"):
            try:
                call = json.loads(last_message)
                fn_name = call.get("function", "")
                fn_args = call.get("arguments", {})
                if fn_name in self._functions:
                    result = self._functions[fn_name](**fn_args)
                    return json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else str(result)
            except (json.JSONDecodeError, TypeError):
                pass

        # Default: process as cockpit command
        response = self.dispatcher.process(last_message)
        return response.content

    def register_function(self, fn: Any, *, name: str | None = None,
                          description: str = "") -> None:
        """Register an external function for tool use.

        Args:
            fn: Callable to register.
            name: Function name (defaults to ``fn.__name__``).
            description: Human-readable description.
        """
        fn_name = name or getattr(fn, "__name__", str(fn))
        self._functions[fn_name] = fn

    # ------------------------------------------------------------------
    # Function declarations (AutoGen function-calling style)
    # ------------------------------------------------------------------

    def get_function_map(self) -> dict[str, Any]:
        """Return registered functions as a function-map dict."""
        return dict(self._functions)

    def get_tool_definitions(self) -> list[dict]:
        """Return OpenAI-style tool definitions for function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "cockpit_command",
                    "description": "执行智能座舱指令",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "自然语言指令"},
                        },
                        "required": ["command"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "map_search",
                    "description": "高德地图搜索",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string"},
                            "keywords": {"type": "string"},
                        },
                        "required": ["action"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "网页搜索",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _register_default_functions(self):
        """Register built-in tool functions."""
        self._functions["cockpit_command"] = self._fn_cockpit
        self._functions["map_search"] = self._fn_map_search
        self._functions["web_search"] = self._fn_web_search

    def _fn_cockpit(self, command: str = "", driving_state: str = "parked") -> dict:
        response = self.dispatcher.process(command, driving_state=driving_state)
        return {"content": response.content, "confidence": response.confidence}

    def _fn_map_search(self, action: str = "poi_search", **kwargs) -> dict:
        result = self.amap.run(action=action, **kwargs)
        return result.data

    def _fn_web_search(self, query: str = "", count: int = 5) -> dict:
        result = self.web_search.run(query=query, count=count)
        return result.data
