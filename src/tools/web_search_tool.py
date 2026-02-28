"""Web search tool for information retrieval.

网页搜索工具：基于 Bing Web Search API 进行实时信息检索。
未配置 WEB_SEARCH_API_KEY 时返回模拟结果，方便开发测试。
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from src.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# HTTP 请求超时秒数
_HTTP_TIMEOUT_SECONDS = 10


class WebSearchTool(BaseTool):
    """Web search integration using a configurable search API.

    Supports Bing Web Search API by default.  Set ``WEB_SEARCH_API_KEY``
    and optionally ``WEB_SEARCH_ENDPOINT`` in the environment.  When no
    key is available the tool returns simulated results for testing.
    """

    name = "web_search"
    description = "网页搜索工具，支持实时信息检索"

    _DEFAULT_ENDPOINT = "https://api.bing.microsoft.com/v7.0/search"

    def __init__(self, api_key: str | None = None, endpoint: str | None = None):
        self.api_key = api_key or os.environ.get("WEB_SEARCH_API_KEY", "")
        self.endpoint = (
            endpoint
            or os.environ.get("WEB_SEARCH_ENDPOINT", "")
            or self._DEFAULT_ENDPOINT
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, query: str = "", count: int = 5, **_kwargs) -> ToolResult:
        """Search the web for *query* and return up to *count* results.

        Args:
            query: The search query string.
            count: Maximum number of results to return.

        Returns:
            ToolResult containing a list of search results.
        """
        if not query:
            return ToolResult(success=False, error="缺少搜索关键词")

        try:
            if not self.api_key:
                return self._simulate_search(query, count)
            return self._bing_search(query, count)
        except Exception as exc:
            logger.error("Web search error: %s", exc)
            return ToolResult(success=False, error=str(exc))

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词",
                },
                "count": {
                    "type": "integer",
                    "description": "返回结果数量",
                    "default": 5,
                },
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bing_search(self, query: str, count: int) -> ToolResult:
        """Execute a Bing Web Search API request."""
        params = urllib.parse.urlencode({"q": query, "count": count})
        url = f"{self.endpoint}?{params}"

        req = urllib.request.Request(url, headers={
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        results = []
        for item in data.get("webPages", {}).get("value", [])[:count]:
            results.append({
                "title": item.get("name", ""),
                "url": item.get("url", ""),
                "snippet": item.get("snippet", ""),
            })

        return ToolResult(success=True, data={
            "results": results,
            "count": len(results),
        })

    # ------------------------------------------------------------------
    # Simulation fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_search(query: str, count: int) -> ToolResult:
        results = []
        for i in range(min(count, 3)):
            results.append({
                "title": f"{query} - 搜索结果{i + 1}(模拟)",
                "url": f"https://example.com/search?q={urllib.parse.quote(query)}&p={i}",
                "snippet": f"这是关于「{query}」的模拟搜索结果摘要。",
            })
        return ToolResult(success=True, data={
            "results": results,
            "count": len(results),
            "simulated": True,
        })
