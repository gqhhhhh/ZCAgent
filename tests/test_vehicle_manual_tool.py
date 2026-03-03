"""Tests for Vehicle Manual RAG tool."""

import pytest

from src.rag.bm25_retriever import Document
from src.tools.vehicle_manual_tool import VehicleManualTool


class TestVehicleManualTool:
    def test_query_safety(self):
        tool = VehicleManualTool()
        result = tool.run(query="安全带怎么使用")
        assert result.success
        assert result.data["count"] > 0
        assert len(result.data["passages"]) > 0
        assert result.data["answer"]  # non-empty answer

    def test_query_ac(self):
        tool = VehicleManualTool()
        result = tool.run(query="空调温度怎么调")
        assert result.success
        passages = result.data["passages"]
        assert any("空调" in p["content"] or "温度" in p["content"] for p in passages)

    def test_query_navigation(self):
        tool = VehicleManualTool()
        result = tool.run(query="导航怎么设置目的地")
        assert result.success
        passages = result.data["passages"]
        assert any("导航" in p["content"] for p in passages)

    def test_query_charging(self):
        tool = VehicleManualTool()
        result = tool.run(query="电池充电注意事项")
        assert result.success
        passages = result.data["passages"]
        assert any("充电" in p["content"] or "电池" in p["content"] for p in passages)

    def test_missing_query(self):
        tool = VehicleManualTool()
        result = tool.run()
        assert not result.success
        assert "查询" in result.error

    def test_top_k(self):
        tool = VehicleManualTool()
        result = tool.run(query="车辆安全功能", top_k=2)
        assert result.success
        assert result.data["count"] <= 2

    def test_passage_structure(self):
        tool = VehicleManualTool()
        result = tool.run(query="胎压监测")
        assert result.success
        for passage in result.data["passages"]:
            assert "doc_id" in passage
            assert "content" in passage
            assert "category" in passage
            assert "score" in passage

    def test_get_schema(self):
        tool = VehicleManualTool()
        schema = tool.get_schema()
        assert schema["name"] == "vehicle_manual"
        assert "parameters" in schema
        assert "query" in schema["parameters"]

    def test_custom_documents(self):
        custom_docs = [
            Document("c1", "自定义手册内容：测试专用文档", {"category": "测试"}),
        ]
        tool = VehicleManualTool(documents=custom_docs)
        result = tool.run(query="测试专用")
        assert result.success
        assert result.data["count"] > 0
        assert "测试" in result.data["passages"][0]["content"]

    def test_empty_documents(self):
        tool = VehicleManualTool(documents=[])
        result = tool.run(query="任意查询")
        assert result.success
        assert result.data["count"] == 0

    def test_tool_attributes(self):
        tool = VehicleManualTool()
        assert tool.name == "vehicle_manual"
        assert tool.description
