"""BM25 sparse retrieval for rule-based and keyword matching.

Okapi BM25 稀疏检索实现：基于词频（TF）和逆文档频率（IDF）计算文档相关性。
擅长精确关键词匹配场景，支持 jieba 中文分词（可选）。
"""

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document in the RAG knowledge base."""
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0


class BM25Retriever:
    """BM25-based sparse retriever for keyword matching.

    Implements the Okapi BM25 algorithm for scoring document relevance
    based on term frequency and inverse document frequency.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._documents: list[Document] = []
        self._tokenized_docs: list[list[str]] = []
        self._avg_dl: float = 0.0
        self._doc_freqs: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._indexed = False

    def add_documents(self, documents: list[Document]):
        """Add documents to the index."""
        self._documents.extend(documents)
        self._indexed = False

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into terms. Supports Chinese via jieba if available."""
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            # Fall back to simple whitespace + character tokenization
            tokens = []
            current = []
            for char in text:
                if char.isalnum() or '\u4e00' <= char <= '\u9fff':
                    current.append(char)
                else:
                    if current:
                        tokens.append("".join(current))
                        current = []
            if current:
                tokens.append("".join(current))
            return tokens

    def _build_index(self):
        """Build BM25 index from documents."""
        self._tokenized_docs = [
            self._tokenize(doc.content) for doc in self._documents
        ]
        self._avg_dl = (
            sum(len(d) for d in self._tokenized_docs) / max(len(self._tokenized_docs), 1)
        )

        # Calculate document frequencies
        self._doc_freqs = {}
        for doc_tokens in self._tokenized_docs:
            unique_tokens = set(doc_tokens)
            for token in unique_tokens:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        # Calculate IDF
        n = len(self._documents)
        self._idf = {}
        for token, df in self._doc_freqs.items():
            self._idf[token] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

        self._indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        """Retrieve top-k documents matching the query.

        Args:
            query: The search query.
            top_k: Number of documents to return.

        Returns:
            List of Document objects sorted by BM25 score.
        """
        if not self._indexed:
            self._build_index()

        if not self._documents:
            return []

        query_tokens = self._tokenize(query)
        scores = []

        for i, doc_tokens in enumerate(self._tokenized_docs):
            score = 0.0
            dl = len(doc_tokens)
            tf_map: dict[str, int] = {}
            for t in doc_tokens:
                tf_map[t] = tf_map.get(t, 0) + 1

            for qt in query_tokens:
                if qt not in self._idf:
                    continue
                tf = tf_map.get(qt, 0)
                idf = self._idf[qt]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / max(self._avg_dl, 1))
                score += idf * numerator / denominator

            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i, s in scores[:top_k]:
            doc = Document(
                doc_id=self._documents[i].doc_id,
                content=self._documents[i].content,
                metadata=self._documents[i].metadata.copy(),
                score=s,
            )
            results.append(doc)
        return results
