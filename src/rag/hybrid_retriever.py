"""Hybrid retriever combining BM25 + MMR with ColBERT reranking."""

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
