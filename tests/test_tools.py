"""Tests for external API tools."""

import pytest

from src.tools.base_tool import BaseTool, ToolResult
from src.tools.amap_tool import AmapTool
from src.tools.web_search_tool import WebSearchTool


class TestToolResult:
    def test_success_to_text(self):
        result = ToolResult(success=True, data={"key": "value"})
        assert "key" in result.to_text()

    def test_failure_to_text(self):
        result = ToolResult(success=False, error="test error")
        assert "失败" in result.to_text()
        assert "test error" in result.to_text()


class TestAmapTool:
    def test_simulate_poi_search(self):
        tool = AmapTool()  # No API key → simulated
        result = tool.run(action="poi_search", keywords="天安门")
        assert result.success
        assert result.data.get("simulated") is True
        assert len(result.data.get("pois", [])) > 0
        assert "天安门" in result.data["pois"][0]["name"]

    def test_simulate_geocode(self):
        tool = AmapTool()
        result = tool.run(action="geocode", address="北京市天安门")
        assert result.success
        assert result.data.get("simulated") is True
        assert "location" in result.data

    def test_simulate_route(self):
        tool = AmapTool()
        result = tool.run(action="route", origin="116.3,39.9", destination="116.4,40.0")
        assert result.success
        assert result.data.get("simulated") is True
        assert "distance" in result.data

    def test_missing_keywords(self):
        tool = AmapTool()
        result = tool.run(action="poi_search")
        assert not result.success
        assert "关键词" in result.error

    def test_missing_address(self):
        tool = AmapTool()
        result = tool.run(action="geocode")
        assert not result.success

    def test_missing_route_params(self):
        tool = AmapTool()
        result = tool.run(action="route")
        assert not result.success

    def test_unknown_action(self):
        tool = AmapTool()
        result = tool.run(action="unknown_action")
        assert not result.success
        assert "未知" in result.error

    def test_get_schema(self):
        tool = AmapTool()
        schema = tool.get_schema()
        assert schema["name"] == "amap"
        assert "parameters" in schema


class TestWebSearchTool:
    def test_simulate_search(self):
        tool = WebSearchTool()  # No API key → simulated
        result = tool.run(query="Python教程")
        assert result.success
        assert result.data.get("simulated") is True
        assert len(result.data.get("results", [])) > 0

    def test_missing_query(self):
        tool = WebSearchTool()
        result = tool.run()
        assert not result.success
        assert "关键词" in result.error

    def test_result_count(self):
        tool = WebSearchTool()
        result = tool.run(query="test", count=2)
        assert result.success
        assert len(result.data.get("results", [])) <= 2

    def test_get_schema(self):
        tool = WebSearchTool()
        schema = tool.get_schema()
        assert schema["name"] == "web_search"
