"""ColBERT-style token-level reranker for fine-grained relevance scoring.

ColBERT 延迟交互重排序：对查询和文档在 Token 级别计算最大相似度（MaxSim），
实现比全文匹配更细粒度的相关性评分。当前使用哈希模拟 embedding，
生产环境应替换为真实 BERT/Transformer embedding。
"""

import hashlib
import logging

import numpy as np

from src.rag.bm25_retriever import Document

logger = logging.getLogger(__name__)


class ColBERTReranker:
    """ColBERT-style late interaction reranker.

    Implements a simplified ColBERT scoring mechanism using token-level
    maximum similarity (MaxSim) between query and document tokens.
    In production, this would use actual BERT embeddings.
    """

    def __init__(self, token_embedding_fn=None, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.token_embedding_fn = token_embedding_fn or self._default_token_embedding

    def _default_token_embedding(self, token: str) -> np.ndarray:
        """Simple deterministic hash-based token embedding for fallback."""
        embedding = np.zeros(self.embedding_dim)
        for i, char in enumerate(token):
            h = hashlib.md5((char + str(i)).encode()).hexdigest()
            idx = int(h, 16) % self.embedding_dim
            embedding[idx] = 1.0
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for Chinese and English text."""
        tokens = []
        current = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            elif char.isalnum():
                current.append(char)
            else:
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    def _maxsim_score(self, query_tokens: list[str],
                      doc_tokens: list[str]) -> float:
        """Compute ColBERT MaxSim score between query and document.

        For each query token, find its maximum similarity to any document
        token, then sum these maximum similarities.
        """
        if not query_tokens or not doc_tokens:
            return 0.0

        query_embeddings = [self.token_embedding_fn(t) for t in query_tokens]
        doc_embeddings = [self.token_embedding_fn(t) for t in doc_tokens]

        total_score = 0.0
        for q_emb in query_embeddings:
            max_sim = max(
                float(np.dot(q_emb, d_emb))
                for d_emb in doc_embeddings
            )
            total_score += max_sim

        return total_score / max(len(query_tokens), 1)

    def rerank(self, query: str, documents: list[Document],
               top_k: int = 3) -> list[Document]:
        """Rerank documents using ColBERT-style MaxSim scoring.

        Args:
            query: The search query.
            documents: Candidate documents to rerank.
            top_k: Number of top documents to return.

        Returns:
            Reranked list of Document objects.
        """
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        scored_docs = []

        for doc in documents:
            doc_tokens = self._tokenize(doc.content)
            score = self._maxsim_score(query_tokens, doc_tokens)
            reranked_doc = Document(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata.copy(),
                score=score,
            )
            scored_docs.append(reranked_doc)

        scored_docs.sort(key=lambda d: d.score, reverse=True)
        return scored_docs[:top_k]
