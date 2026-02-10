"""MCP (Model Context Protocol) server adapter for ZCAgent.

Exposes ZCAgent capabilities as an MCP-compatible server that LLM clients
(e.g. Claude Desktop, Cursor) can connect to.

The adapter works **without** the ``mcp`` package installed by providing
a lightweight protocol implementation.  When the official ``mcp`` SDK is
available it can be used as a drop-in replacement.

Example usage::

    from src.integrations.mcp_adapter import ZCAgentMCPServer

    server = ZCAgentMCPServer()
    # Start via stdio (for Claude Desktop integration)
    server.run_stdio()
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any

from src.agent.dispatcher import AgentDispatcher
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool

logger = logging.getLogger(__name__)


class ZCAgentMCPServer:
    """MCP server exposing ZCAgent tools to LLM clients.

    Implements the *tools/list* and *tools/call* methods of the MCP
    specification so that an LLM client can discover and invoke the
    cockpit agent, map search, and web search capabilities.
    """

    SERVER_NAME = "zcagent-mcp-server"
    SERVER_VERSION = "0.1.0"

    def __init__(self, config: dict | None = None, llm_client: Any = None):
        self.dispatcher = AgentDispatcher(config=config, llm_client=llm_client)
        self.amap = AmapTool()
        self.web_search = WebSearchTool()

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict]:
        """Return MCP tool definitions."""
        return [
            {
                "name": "cockpit_command",
                "description": "智能座舱语音指令处理，支持导航、音乐、电话、车辆控制等",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "自然语言指令",
                        },
                        "driving_state": {
                            "type": "string",
                            "enum": ["parked", "driving", "highway"],
                            "default": "parked",
                        },
                    },
                    "required": ["command"],
                },
            },
            {
                "name": "map_search",
                "description": "高德地图搜索，支持POI搜索、地理编码、路径规划",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["poi_search", "geocode", "route"],
                        },
                        "keywords": {"type": "string"},
                        "address": {"type": "string"},
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                    },
                    "required": ["action"],
                },
            },
            {
                "name": "web_search",
                "description": "网页搜索，输入关键词返回搜索结果",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索关键词"},
                        "count": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict) -> dict:
        """Execute a tool by name and return the result.

        Args:
            name: Tool name from ``list_tools``.
            arguments: Tool-specific arguments.

        Returns:
            Dict with ``content`` list following the MCP response format.
        """
        try:
            if name == "cockpit_command":
                return self._handle_cockpit(arguments)
            if name == "map_search":
                return self._handle_map(arguments)
            if name == "web_search":
                return self._handle_web_search(arguments)
            return self._error_response(f"Unknown tool: {name}")
        except Exception as exc:
            logger.error("MCP tool call error (%s): %s", name, exc)
            return self._error_response(str(exc))

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _handle_cockpit(self, args: dict) -> dict:
        command = args.get("command", "")
        driving_state = args.get("driving_state", "parked")
        response = self.dispatcher.process(command, driving_state=driving_state)
        return {
            "content": [
                {"type": "text", "text": response.content},
            ],
            "metadata": {
                "confidence": response.confidence,
                "task_results": response.task_results,
            },
        }

    def _handle_map(self, args: dict) -> dict:
        action = args.pop("action", "poi_search")
        result = self.amap.run(action=action, **args)
        return {
            "content": [
                {"type": "text", "text": json.dumps(result.data, ensure_ascii=False)},
            ],
        }

    def _handle_web_search(self, args: dict) -> dict:
        result = self.web_search.run(**args)
        return {
            "content": [
                {"type": "text", "text": json.dumps(result.data, ensure_ascii=False)},
            ],
        }

    @staticmethod
    def _error_response(message: str) -> dict:
        return {
            "content": [{"type": "text", "text": f"Error: {message}"}],
            "isError": True,
        }

    # ------------------------------------------------------------------
    # JSON-RPC message handling (stdio transport)
    # ------------------------------------------------------------------

    def handle_message(self, message: dict) -> dict | None:
        """Handle a single JSON-RPC message and return the response."""
        method = message.get("method", "")
        msg_id = message.get("id")

        if method == "initialize":
            return self._jsonrpc_response(msg_id, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": self.SERVER_NAME, "version": self.SERVER_VERSION},
                "capabilities": {"tools": {}},
            })

        if method == "tools/list":
            return self._jsonrpc_response(msg_id, {"tools": self.list_tools()})

        if method == "tools/call":
            params = message.get("params", {})
            result = self.call_tool(params.get("name", ""), params.get("arguments", {}))
            return self._jsonrpc_response(msg_id, result)

        if method == "notifications/initialized":
            return None  # No response for notifications

        return self._jsonrpc_response(msg_id, None, error={
            "code": -32601,
            "message": f"Method not found: {method}",
        })

    def run_stdio(self):
        """Run the MCP server over stdin/stdout (blocking)."""
        logger.info("Starting ZCAgent MCP server (stdio)")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            response = self.handle_message(message)
            if response is not None:
                sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                sys.stdout.flush()

    @staticmethod
    def _jsonrpc_response(msg_id: Any, result: Any = None,
                          error: dict | None = None) -> dict:
        resp: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id}
        if error:
            resp["error"] = error
        else:
            resp["result"] = result
        return resp
