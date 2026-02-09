"""Tests for the hybrid RAG retrieval system."""

import pytest

from src.rag.bm25_retriever import BM25Retriever, Document
from src.rag.mmr_retriever import MMRRetriever
from src.rag.colbert_reranker import ColBERTReranker
from src.rag.hybrid_retriever import HybridRetriever


def _sample_docs():
    return [
        Document("d1", "当车速超过120km/h时，禁止打开车窗以确保安全"),
        Document("d2", "导航系统支持语音输入目的地，也可以手动输入地址"),
        Document("d3", "音乐播放器支持在线和本地音乐播放，可调节音量"),
        Document("d4", "空调系统可以设置温度范围16-32度，支持自动模式"),
        Document("d5", "紧急呼叫功能在任何驾驶状态下都可以使用"),
    ]


class TestBM25Retriever:
    def test_retrieve_basic(self):
        retriever = BM25Retriever()
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("导航", top_k=3)
        assert len(results) > 0
        assert any("导航" in r.content for r in results)

    def test_retrieve_empty(self):
        retriever = BM25Retriever()
        results = retriever.retrieve("test")
        assert results == []

    def test_retrieve_top_k(self):
        retriever = BM25Retriever()
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("安全", top_k=2)
        assert len(results) <= 2

    def test_relevance_ordering(self):
        retriever = BM25Retriever()
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("温度空调", top_k=5)
        # First result should be about temperature/AC
        assert "温度" in results[0].content or "空调" in results[0].content


class TestMMRRetriever:
    def test_retrieve_basic(self):
        retriever = MMRRetriever()
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("导航目的地", top_k=3)
        assert len(results) == 3

    def test_diversity(self):
        retriever = MMRRetriever(lambda_param=0.3)  # More diversity
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("系统功能", top_k=3)
        # Results should cover different topics
        doc_ids = [r.doc_id for r in results]
        assert len(set(doc_ids)) == 3  # All unique


class TestColBERTReranker:
    def test_rerank(self):
        reranker = ColBERTReranker()
        docs = _sample_docs()[:3]
        results = reranker.rerank("导航设置目的地", docs, top_k=2)
        assert len(results) == 2
        assert all(r.score >= 0 for r in results)

    def test_rerank_empty(self):
        reranker = ColBERTReranker()
        results = reranker.rerank("test", [], top_k=3)
        assert results == []

    def test_rerank_preserves_content(self):
        reranker = ColBERTReranker()
        docs = [Document("d1", "导航到北京"), Document("d2", "播放音乐")]
        results = reranker.rerank("导航", docs, top_k=2)
        contents = [r.content for r in results]
        assert "导航到北京" in contents
        assert "播放音乐" in contents


class TestHybridRetriever:
    def test_retrieve(self):
        retriever = HybridRetriever()
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("导航语音输入")
        assert len(results) > 0

    def test_retrieve_empty(self):
        retriever = HybridRetriever()
        results = retriever.retrieve("test")
        assert results == []

    def test_rerank_applied(self):
        retriever = HybridRetriever({"rerank_top_k": 2})
        retriever.add_documents(_sample_docs())
        results = retriever.retrieve("安全功能", rerank_top_k=2)
        assert len(results) <= 2
