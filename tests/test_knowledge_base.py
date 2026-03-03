"""Tests for PDF knowledge base: loader, chunker, vector store, and KnowledgeBase."""

import os
import shutil
import tempfile

import pytest

from src.rag.bm25_retriever import Document
from src.rag.chunker import TextChunker
from src.rag.knowledge_base import KnowledgeBase
from src.rag.pdf_loader import PDFLoader
from src.rag.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TEST_PDF = os.path.join(FIXTURES_DIR, "test_manual.pdf")


def _sample_docs():
    return [
        Document("d1", "空调系统可以设置温度范围16-32度，支持AUTO自动模式"),
        Document("d2", "导航系统支持语音输入目的地，也可以手动输入地址"),
        Document("d3", "紧急呼叫功能在任何驾驶状态下都可以使用"),
        Document("d4", "音乐播放器支持蓝牙音频和USB音源播放"),
        Document("d5", "胎压监测系统实时显示四个轮胎的气压和温度"),
    ]


# ---------------------------------------------------------------------------
# PDFLoader
# ---------------------------------------------------------------------------

class TestPDFLoader:
    def test_load_pdf(self):
        loader = PDFLoader()
        docs = loader.load(TEST_PDF)
        assert len(docs) == 3  # 3 pages
        assert all(isinstance(d, Document) for d in docs)
        assert docs[0].metadata["page"] == 1
        assert docs[0].metadata["filename"] == "test_manual.pdf"

    def test_load_pdf_not_found(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path.pdf")

    def test_load_pdf_not_pdf(self):
        loader = PDFLoader()
        with pytest.raises(ValueError):
            loader.load(__file__)  # .py file

    def test_load_directory(self):
        loader = PDFLoader()
        docs = loader.load_directory(FIXTURES_DIR)
        assert len(docs) >= 3  # At least our 3-page test PDF

    def test_load_directory_not_found(self):
        loader = PDFLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory("/nonexistent/dir")


# ---------------------------------------------------------------------------
# TextChunker
# ---------------------------------------------------------------------------

class TestTextChunker:
    def test_chunk_short_text(self):
        chunker = TextChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_text("这是一段短文本。")
        assert len(chunks) == 1
        assert chunks[0] == "这是一段短文本。"

    def test_chunk_long_text(self):
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = "第一段内容。" * 5 + "\n\n" + "第二段内容。" * 5
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_empty_text(self):
        chunker = TextChunker()
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text("   ") == []

    def test_chunk_overlap_validation(self):
        with pytest.raises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_documents(self):
        chunker = TextChunker(chunk_size=30, chunk_overlap=5)
        docs = [Document("d1", "这是第一个文档的内容，需要被切分成多个片段。")]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) >= 1
        assert all("original_doc_id" in c.metadata for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_chunk_preserves_metadata(self):
        chunker = TextChunker(chunk_size=500)
        docs = [Document("d1", "短文本", {"source": "test.pdf", "page": 1})]
        chunks = chunker.chunk_documents(docs)
        assert len(chunks) == 1
        assert chunks[0].metadata["source"] == "test.pdf"
        assert chunks[0].metadata["page"] == 1


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_add_and_search(self):
        store = VectorStore()
        store.add_documents(_sample_docs())
        results = store.search("空调温度", top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, Document) for r in results)

    def test_search_empty(self):
        store = VectorStore()
        results = store.search("test")
        assert results == []

    def test_size(self):
        store = VectorStore()
        assert store.size == 0
        store.add_documents(_sample_docs()[:2])
        assert store.size == 2

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        try:
            # Save
            store = VectorStore()
            store.add_documents(_sample_docs())
            store.save(tmpdir)

            # Load
            store2 = VectorStore()
            store2.load(tmpdir)
            assert store2.size == 5

            # Search still works after load
            results = store2.search("导航", top_k=2)
            assert len(results) == 2
        finally:
            shutil.rmtree(tmpdir)

    def test_load_not_found(self):
        store = VectorStore()
        with pytest.raises(FileNotFoundError):
            store.load("/nonexistent/dir")


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class TestKnowledgeBase:
    def test_add_documents_and_search(self):
        kb = KnowledgeBase()
        kb.add_documents(_sample_docs())
        results = kb.search("空调温度")
        assert len(results) > 0
        assert all(isinstance(r, Document) for r in results)

    def test_add_pdf(self):
        kb = KnowledgeBase()
        kb.add_pdf(TEST_PDF)
        assert kb.chunk_count > 0
        results = kb.search("navigation")
        assert len(results) > 0

    def test_search_empty(self):
        kb = KnowledgeBase()
        results = kb.search("test")
        assert results == []

    def test_save_and_load(self):
        tmpdir = tempfile.mkdtemp()
        try:
            kb = KnowledgeBase()
            kb.add_documents(_sample_docs())
            kb.save(tmpdir)

            kb2 = KnowledgeBase()
            kb2.load(tmpdir)
            assert kb2.chunk_count == kb.chunk_count

            results = kb2.search("导航")
            assert len(results) > 0
        finally:
            shutil.rmtree(tmpdir)

    def test_rerank_top_k(self):
        kb = KnowledgeBase({"rerank_top_k": 2})
        kb.add_documents(_sample_docs())
        results = kb.search("系统功能", rerank_top_k=2)
        assert len(results) <= 2
