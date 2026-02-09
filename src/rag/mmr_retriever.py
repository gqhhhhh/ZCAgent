"""MMR-based retrieval for balancing relevance and diversity."""

import logging
from dataclasses import dataclass, field

import numpy as np

from src.rag.bm25_retriever import Document

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MMRRetriever:
    """Maximal Marginal Relevance retriever.

    Balances relevance to the query with diversity among results,
    avoiding redundant information in the retrieved set.
    """

    def __init__(self, lambda_param: float = 0.7, embedding_fn=None):
        self.lambda_param = lambda_param
        self.embedding_fn = embedding_fn or self._default_embedding
        self._documents: list[Document] = []
        self._embeddings: list[np.ndarray] = []

    def _default_embedding(self, text: str) -> np.ndarray:
        """Simple character-level hash embedding for fallback.

        This is a basic fallback; in production, use a proper embedding model.
        """
        dim = 128
        embedding = np.zeros(dim)
        for i, char in enumerate(text):
            idx = hash(char) % dim
            embedding[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def add_documents(self, documents: list[Document]):
        """Add documents and compute their embeddings."""
        for doc in documents:
            self._documents.append(doc)
            self._embeddings.append(self.embedding_fn(doc.content))

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve documents using MMR selection.

        Args:
            query: The search query.
            top_k: Number of documents to return.

        Returns:
            List of Document objects selected by MMR.
        """
        if not self._documents:
            return []

        query_embedding = self.embedding_fn(query)

        # Compute relevance scores
        relevance_scores = [
            cosine_similarity(query_embedding, emb)
            for emb in self._embeddings
        ]

        selected_indices: list[int] = []
        candidate_indices = list(range(len(self._documents)))

        for _ in range(min(top_k, len(self._documents))):
            if not candidate_indices:
                break

            best_idx = -1
            best_mmr = float("-inf")

            for idx in candidate_indices:
                relevance = relevance_scores[idx]

                # Max similarity to already selected documents
                if selected_indices:
                    max_sim = max(
                        cosine_similarity(self._embeddings[idx], self._embeddings[s])
                        for s in selected_indices
                    )
                else:
                    max_sim = 0.0

                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_sim

                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                candidate_indices.remove(best_idx)

        results = []
        for idx in selected_indices:
            doc = Document(
                doc_id=self._documents[idx].doc_id,
                content=self._documents[idx].content,
                metadata=self._documents[idx].metadata.copy(),
                score=relevance_scores[idx],
            )
            results.append(doc)
        return results
