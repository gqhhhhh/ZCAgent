"""Hybrid retriever combining BM25 + MMR with ColBERT reranking.

三级混合检索管线：BM25 稀疏检索和 MMR 密集检索并行取候选，
按权重加权融合后交给 ColBERT 重排序器做最终排序。
支持从 PDF 文件构建外部知识库并进行混合检索。
"""

import logging

from src.rag.bm25_retriever import BM25Retriever, Document
from src.rag.mmr_retriever import MMRRetriever
from src.rag.colbert_reranker import ColBERTReranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 sparse and MMR dense retrieval.

    Uses BM25 for keyword-level matching (good for rules and exact terms),
    MMR for semantic diversity, then ColBERT for fine-grained reranking
    of the merged candidate set.

    Supports loading external PDF documents as a knowledge base via
    :meth:`load_pdf` and :meth:`load_pdf_directory`.
    """

    def __init__(self, config: dict | None = None):
        config = config or {}
        self.bm25_weight = config.get("bm25_weight", 0.4)
        self.mmr_weight = config.get("mmr_weight", 0.6)
        self.top_k = config.get("top_k", 5)
        self.rerank_top_k = config.get("rerank_top_k", 3)

        self.bm25 = BM25Retriever()
        self.mmr = MMRRetriever(
            lambda_param=config.get("mmr_lambda", 0.7)
        )
        self.reranker = ColBERTReranker()

    def add_documents(self, documents: list[Document]):
        """Add documents to both retrievers."""
        self.bm25.add_documents(documents)
        self.mmr.add_documents(documents)

    def load_pdf(self, pdf_path: str,
                 chunk_size: int = 512, chunk_overlap: int = 64):
        """Load a PDF file, chunk it, and add chunks to both retrievers.

        Args:
            pdf_path: Path to the PDF file.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        from src.rag.pdf_loader import PDFLoader
        from src.rag.chunker import TextChunker

        pages = PDFLoader().load(pdf_path)
        chunks = TextChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        ).chunk_documents(pages)
        self.add_documents(chunks)
        logger.info("Loaded %d chunks from PDF: %s", len(chunks), pdf_path)

    def load_pdf_directory(self, dir_path: str,
                           chunk_size: int = 512, chunk_overlap: int = 64):
        """Load all PDFs from a directory, chunk and index them.

        Args:
            dir_path: Path to a directory of PDF files.
            chunk_size: Maximum characters per chunk.
            chunk_overlap: Overlap between consecutive chunks.
        """
        from src.rag.pdf_loader import PDFLoader
        from src.rag.chunker import TextChunker

        pages = PDFLoader().load_directory(dir_path)
        chunks = TextChunker(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        ).chunk_documents(pages)
        self.add_documents(chunks)
        logger.info(
            "Loaded %d chunks from PDF directory: %s", len(chunks), dir_path,
        )

    def retrieve(self, query: str, top_k: int | None = None,
                 rerank_top_k: int | None = None) -> list[Document]:
        """Retrieve documents using hybrid BM25+MMR, then rerank with ColBERT.

        Args:
            query: The search query.
            top_k: Number of candidates from each retriever.
            rerank_top_k: Final number of documents after reranking.

        Returns:
            Reranked list of Document objects.
        """
        top_k = top_k or self.top_k
        rerank_top_k = rerank_top_k or self.rerank_top_k

        # Retrieve from both sources
        bm25_results = self.bm25.retrieve(query, top_k=top_k)
        mmr_results = self.mmr.retrieve(query, top_k=top_k)

        # Merge and deduplicate with weighted scoring
        merged: dict[str, Document] = {}

        for doc in bm25_results:
            merged[doc.doc_id] = Document(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata.copy(),
                score=doc.score * self.bm25_weight,
            )

        for doc in mmr_results:
            if doc.doc_id in merged:
                merged[doc.doc_id].score += doc.score * self.mmr_weight
            else:
                merged[doc.doc_id] = Document(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    score=doc.score * self.mmr_weight,
                )

        # Sort merged candidates by combined score
        candidates = sorted(merged.values(), key=lambda d: d.score, reverse=True)

        # Pass 2x candidates to reranker for better recall before final selection
        if candidates:
            return self.reranker.rerank(query, candidates[:top_k * 2],
                                        top_k=rerank_top_k)

        return []
