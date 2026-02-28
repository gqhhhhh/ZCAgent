"""Amap (Gaode Maps) API tool for navigation and POI search.

高德地图 API 封装：支持 POI 搜索、地理编码和驾车路线规划。
未配置 AMAP_API_KEY 时返回结构一致的模拟数据，保证开发流程不中断。
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

from src.tools.base_tool import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# HTTP 请求超时秒数（避免网络异常时长时间阻塞）
_HTTP_TIMEOUT_SECONDS = 10


class AmapTool(BaseTool):
    """Amap (高德地图) API integration for geocoding, POI search, and route planning.

    Requires an Amap Web Service API key set via ``AMAP_API_KEY`` environment
    variable or passed directly.  When no key is available the tool returns a
    simulated result so that the rest of the pipeline can still be tested.
    """

    name = "amap"
    description = "高德地图API工具，支持地理编码、POI搜索和路径规划"

    _BASE_URL = "https://restapi.amap.com/v3"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("AMAP_API_KEY", "")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, action: str = "poi_search", **kwargs) -> ToolResult:
        """Execute an Amap action.

        Args:
            action: One of ``"poi_search"``, ``"geocode"``, ``"route"``.
            **kwargs: Action-specific parameters.

        Returns:
            ToolResult with response data.
        """
        actions = {
            "poi_search": self._poi_search,
            "geocode": self._geocode,
            "route": self._route_plan,
        }
        handler = actions.get(action)
        if handler is None:
            return ToolResult(success=False, error=f"未知操作: {action}")

        try:
            return handler(**kwargs)
        except Exception as exc:
            logger.error("Amap tool error (%s): %s", action, exc)
            return ToolResult(success=False, error=str(exc))

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "action": {
                    "type": "string",
                    "enum": ["poi_search", "geocode", "route"],
                    "description": "要执行的地图操作",
                },
                "keywords": {
                    "type": "string",
                    "description": "搜索关键词（poi_search时使用）",
                },
                "address": {
                    "type": "string",
                    "description": "地址文本（geocode时使用）",
                },
                "origin": {
                    "type": "string",
                    "description": "起点经纬度，格式 lng,lat（route时使用）",
                },
                "destination": {
                    "type": "string",
                    "description": "终点经纬度，格式 lng,lat（route时使用）",
                },
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _request(self, path: str, params: dict) -> dict:
        """Send an HTTP GET request to the Amap API."""
        if not self.api_key:
            raise RuntimeError("AMAP_API_KEY is not configured")

        params["key"] = self.api_key
        params["output"] = "json"
        url = f"{self._BASE_URL}{path}?{urllib.parse.urlencode(params)}"
        logger.debug("Amap request: %s", url)

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("status") != "1":
            raise RuntimeError(f"Amap API error: {data.get('info', 'unknown')}")
        return data

    # ------------------------------------------------------------------

    def _poi_search(self, keywords: str = "", city: str = "", **_kwargs) -> ToolResult:
        """Search for points of interest."""
        if not keywords:
            return ToolResult(success=False, error="缺少搜索关键词")

        if not self.api_key:
            return self._simulate_poi(keywords, city)

        params: dict[str, str] = {"keywords": keywords}
        if city:
            params["city"] = city
        data = self._request("/place/text", params)

        pois = []
        for poi in data.get("pois", [])[:5]:
            pois.append({
                "name": poi.get("name", ""),
                "address": poi.get("address", ""),
                "location": poi.get("location", ""),
                "type": poi.get("type", ""),
            })

        return ToolResult(success=True, data={"pois": pois, "count": len(pois)})

    def _geocode(self, address: str = "", **_kwargs) -> ToolResult:
        """Geocode an address to coordinates."""
        if not address:
            return ToolResult(success=False, error="缺少地址")

        if not self.api_key:
            return self._simulate_geocode(address)

        data = self._request("/geocode/geo", {"address": address})
        geocodes = data.get("geocodes", [])
        if not geocodes:
            return ToolResult(success=False, error="未找到该地址")

        geo = geocodes[0]
        return ToolResult(success=True, data={
            "formatted_address": geo.get("formatted_address", ""),
            "location": geo.get("location", ""),
            "level": geo.get("level", ""),
        })

    def _route_plan(self, origin: str = "", destination: str = "",
                    strategy: int = 0, **_kwargs) -> ToolResult:
        """Plan a driving route."""
        if not origin or not destination:
            return ToolResult(success=False, error="缺少起点或终点")

        if not self.api_key:
            return self._simulate_route(origin, destination)

        data = self._request("/direction/driving", {
            "origin": origin,
            "destination": destination,
            "strategy": str(strategy),
        })

        paths = data.get("route", {}).get("paths", [])
        if not paths:
            return ToolResult(success=False, error="未找到路线")

        path = paths[0]
        return ToolResult(success=True, data={
            "distance": path.get("distance", ""),
            "duration": path.get("duration", ""),
            "strategy": path.get("strategy", ""),
        })

    # ------------------------------------------------------------------
    # Simulation fallbacks (when no API key is configured)
    # ------------------------------------------------------------------

    @staticmethod
    def _simulate_poi(keywords: str, city: str) -> ToolResult:
        return ToolResult(success=True, data={
            "pois": [
                {"name": f"{keywords}(模拟)", "address": f"{city or '北京'}市模拟路1号",
                 "location": "116.397428,39.90923", "type": "模拟POI"},
            ],
            "count": 1,
            "simulated": True,
        })

    @staticmethod
    def _simulate_geocode(address: str) -> ToolResult:
        return ToolResult(success=True, data={
            "formatted_address": address,
            "location": "116.397428,39.90923",
            "level": "模拟",
            "simulated": True,
        })

    @staticmethod
    def _simulate_route(origin: str, destination: str) -> ToolResult:
        return ToolResult(success=True, data={
            "distance": "12000",
            "duration": "1800",
            "strategy": "模拟路线",
            "simulated": True,
        })
