"""Lightweight vector store for document embeddings.

向量存储：基于 NumPy 的轻量级向量库，支持余弦相似度检索和磁盘持久化。
提供 add / search / save / load 接口，可作为 RAG 检索的密集检索后端。
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from src.rag.bm25_retriever import Document

logger = logging.getLogger(__name__)


class VectorStore:
    """NumPy-based vector store with cosine similarity search.

    Stores document embeddings as NumPy arrays and supports:
    - Adding documents with automatic embedding computation
    - Cosine-similarity based nearest-neighbour search
    - Persistent save/load to disk (``*.npz`` + ``*.json``)
    """

    def __init__(self, embedding_fn=None, embedding_dim: int = 128):
        """
        Args:
            embedding_fn: A callable ``(str) -> np.ndarray`` that converts
                text to a fixed-dimension vector.  Falls back to a simple
                hash-based embedding if *None*.
            embedding_dim: Dimension of each embedding vector.  Only used
                when relying on the default hash-based embedding.
        """
        self.embedding_fn = embedding_fn or self._default_embedding
        self.embedding_dim = embedding_dim
        self._documents: list[Document] = []
        self._embeddings: np.ndarray | None = None  # shape (N, dim)

    # ------------------------------------------------------------------
    # Default embedding (hash-based fallback)
    # ------------------------------------------------------------------

    def _default_embedding(self, text: str) -> np.ndarray:
        """Character-level hash embedding for fallback.

        In production, replace with a real model (e.g. text-embedding-ada-002
        or bge-large-zh).
        """
        dim = self.embedding_dim
        embedding = np.zeros(dim)
        for i, char in enumerate(text):
            idx = hash(char) % dim
            embedding[idx] += 1.0 / (i + 1)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]):
        """Add documents and compute their embeddings.

        Args:
            documents: Documents to add.
        """
        new_embeddings = np.array(
            [self.embedding_fn(doc.content) for doc in documents]
        )
        self._documents.extend(documents)

        if self._embeddings is None or len(self._embeddings) == 0:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])

        logger.info(
            "Added %d documents to vector store (total: %d)",
            len(documents), len(self._documents),
        )

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Find the top-k most similar documents to *query*.

        Args:
            query: Search query text.
            top_k: Number of results to return.

        Returns:
            List of Document objects sorted by descending cosine similarity.
        """
        if not self._documents or self._embeddings is None:
            return []

        query_emb = self.embedding_fn(query)
        # Cosine similarity via normalised dot product
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normed = self._embeddings / norms

        q_norm = np.linalg.norm(query_emb)
        if q_norm > 0:
            query_emb = query_emb / q_norm

        scores = normed @ query_emb  # shape (N,)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results: list[Document] = []
        for idx in top_indices:
            doc = self._documents[idx]
            results.append(
                Document(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata.copy(),
                    score=float(scores[idx]),
                )
            )
        return results

    @property
    def size(self) -> int:
        """Number of documents in the store."""
        return len(self._documents)

    @property
    def documents(self) -> list[Document]:
        """Return a copy of all stored documents."""
        return list(self._documents)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str):
        """Save the vector store to disk.

        Creates two files in *directory*:
        - ``embeddings.npz`` — the embedding matrix
        - ``documents.json`` — document metadata

        Args:
            directory: Target directory (created if necessary).
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        if self._embeddings is not None and len(self._embeddings) > 0:
            np.savez_compressed(str(path / "embeddings.npz"), embeddings=self._embeddings)

        # Save documents
        doc_dicts = [
            {
                "doc_id": d.doc_id,
                "content": d.content,
                "metadata": d.metadata,
            }
            for d in self._documents
        ]
        with open(str(path / "documents.json"), "w", encoding="utf-8") as f:
            json.dump(doc_dicts, f, ensure_ascii=False, indent=2)

        logger.info("Saved vector store (%d docs) to %s", len(self._documents), directory)

    def load(self, directory: str):
        """Load a previously saved vector store from disk.

        Args:
            directory: Directory containing ``embeddings.npz`` and
                ``documents.json``.

        Raises:
            FileNotFoundError: If the directory or required files do not exist.
        """
        path = Path(directory)
        docs_file = path / "documents.json"
        emb_file = path / "embeddings.npz"

        if not docs_file.exists():
            raise FileNotFoundError(f"documents.json not found in {directory}")

        with open(str(docs_file), "r", encoding="utf-8") as f:
            doc_dicts = json.load(f)

        self._documents = [
            Document(
                doc_id=d["doc_id"],
                content=d["content"],
                metadata=d.get("metadata", {}),
            )
            for d in doc_dicts
        ]

        if emb_file.exists():
            data = np.load(str(emb_file))
            self._embeddings = data["embeddings"]
        else:
            # Recompute embeddings if npz missing
            self._embeddings = np.array(
                [self.embedding_fn(doc.content) for doc in self._documents]
            )

        logger.info("Loaded vector store (%d docs) from %s", len(self._documents), directory)
