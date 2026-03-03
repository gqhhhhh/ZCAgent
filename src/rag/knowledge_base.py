"""Knowledge base manager for PDF-based RAG retrieval.

知识库管理器：将 PDF → 切分 → 向量化 → 混合检索 整合为一体化管线。
支持从 PDF 文件或目录构建知识库，持久化到磁盘，以及混合 RAG 检索。
"""

import logging
import os
from pathlib import Path

from src.rag.bm25_retriever import BM25Retriever, Document
from src.rag.chunker import TextChunker
from src.rag.colbert_reranker import ColBERTReranker
from src.rag.mmr_retriever import MMRRetriever
from src.rag.pdf_loader import PDFLoader
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """Unified knowledge base with PDF ingestion and hybrid retrieval.

    Provides an end-to-end pipeline:

    1. **Ingest** — Load PDF files, split text into chunks, and store
       embeddings in the vector store plus BM25 index.
    2. **Retrieve** — Run hybrid BM25 + vector search, merge results
       with configurable weights, and rerank with ColBERT.
    3. **Persist** — Save / load the entire knowledge base to / from disk
       so that re-indexing is not required on every startup.

    Example::

        kb = KnowledgeBase()
        kb.add_pdf("docs/vehicle_manual.pdf")
        kb.save("data/kb_store")

        # Later …
        kb2 = KnowledgeBase()
        kb2.load("data/kb_store")
        results = kb2.search("如何设置空调温度")
    """

    def __init__(self, config: dict | None = None):
        config = config or {}

        # Chunker settings
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 64)

        # Retrieval weights
        self.bm25_weight = config.get("bm25_weight", 0.4)
        self.vector_weight = config.get("vector_weight", 0.6)
        self.top_k = config.get("top_k", 5)
        self.rerank_top_k = config.get("rerank_top_k", 3)

        # Sub-components
        self._loader = PDFLoader()
        self._chunker = TextChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self._bm25 = BM25Retriever()
        self._vector_store = VectorStore()
        self._reranker = ColBERTReranker()

        # Track all chunks for potential re-use
        self._chunks: list[Document] = []

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_pdf(self, pdf_path: str):
        """Load a PDF, chunk it, and index all chunks.

        Args:
            pdf_path: Path to a PDF file.
        """
        pages = self._loader.load(pdf_path)
        self._index_documents(pages)

    def add_pdf_directory(self, dir_path: str):
        """Load all PDFs from a directory, chunk and index them.

        Args:
            dir_path: Path to a directory of PDF files.
        """
        pages = self._loader.load_directory(dir_path)
        self._index_documents(pages)

    def add_documents(self, documents: list[Document]):
        """Directly add pre-built Document objects (non-PDF source).

        Args:
            documents: List of Document objects to index.
        """
        self._index_documents(documents)

    def _index_documents(self, documents: list[Document]):
        """Chunk and index a list of raw Documents."""
        chunks = self._chunker.chunk_documents(documents)
        self._chunks.extend(chunks)
        self._bm25.add_documents(chunks)
        self._vector_store.add_documents(chunks)
        logger.info(
            "Indexed %d chunks from %d source documents",
            len(chunks), len(documents),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        rerank_top_k: int | None = None,
    ) -> list[Document]:
        """Hybrid search: BM25 + Vector + ColBERT reranking.

        Args:
            query: Search query text.
            top_k: Number of candidates per retriever.
            rerank_top_k: Final number of results after reranking.

        Returns:
            Reranked list of Document objects.
        """
        top_k = top_k or self.top_k
        rerank_top_k = rerank_top_k or self.rerank_top_k

        # Stage 1: Parallel retrieval
        bm25_results = self._bm25.retrieve(query, top_k=top_k)
        vector_results = self._vector_store.search(query, top_k=top_k)

        # Stage 2: Weighted merge & deduplication
        merged: dict[str, Document] = {}

        for doc in bm25_results:
            merged[doc.doc_id] = Document(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata.copy(),
                score=doc.score * self.bm25_weight,
            )

        for doc in vector_results:
            if doc.doc_id in merged:
                merged[doc.doc_id].score += doc.score * self.vector_weight
            else:
                merged[doc.doc_id] = Document(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    score=doc.score * self.vector_weight,
                )

        candidates = sorted(merged.values(), key=lambda d: d.score, reverse=True)

        # Stage 3: ColBERT reranking
        if candidates:
            return self._reranker.rerank(
                query, candidates[: top_k * 2], top_k=rerank_top_k,
            )
        return []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str):
        """Save the knowledge base to disk for later reuse.

        Args:
            directory: Target directory (created if necessary).
        """
        self._vector_store.save(directory)
        logger.info("Knowledge base saved to %s", directory)

    def load(self, directory: str):
        """Load a previously saved knowledge base.

        After loading, the BM25 index is rebuilt from the stored documents.

        Args:
            directory: Directory created by a prior :meth:`save` call.
        """
        self._vector_store.load(directory)
        # Rebuild BM25 index from the loaded documents
        self._chunks = list(self._vector_store.documents)
        self._bm25 = BM25Retriever()
        self._bm25.add_documents(self._chunks)
        logger.info(
            "Knowledge base loaded from %s (%d chunks)", directory, len(self._chunks),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def chunk_count(self) -> int:
        """Total number of indexed chunks."""
        return len(self._chunks)
