"""Text chunker for splitting documents into smaller segments.

文档切分器：将长文本拆分为较小的片段，支持递归字符切分策略。
保留上下文重叠区域，自动继承原始文档的元数据并追加 chunk 索引信息。
"""

import logging

from src.rag.bm25_retriever import Document

logger = logging.getLogger(__name__)


class TextChunker:
    """Split documents into smaller chunks with configurable overlap.

    Implements a recursive character splitting strategy similar to
    LangChain's ``RecursiveCharacterTextSplitter``.  Text is split
    using a hierarchy of separators (paragraphs → sentences → characters)
    to produce chunks that are as semantically coherent as possible.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", " "]

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: list[str] | None = None,
    ):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between
                consecutive chunks for context preservation.
            separators: Ordered list of split separators.  Defaults to
                paragraph → sentence → character boundaries.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str) -> list[str]:
        """Split a text string into overlapping chunks.

        Args:
            text: The text to split.

        Returns:
            List of text chunks.
        """
        if not text or not text.strip():
            return []
        return self._recursive_split(text, self.separators)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """Split a list of Documents into smaller chunk Documents.

        Each resulting Document inherits the original metadata and adds
        ``chunk_index`` and ``original_doc_id`` fields.

        Args:
            documents: Source documents to split.

        Returns:
            List of chunk Documents.
        """
        chunks: list[Document] = []

        for doc in documents:
            text_chunks = self.chunk_text(doc.content)
            for i, chunk_text in enumerate(text_chunks):
                chunk_doc = Document(
                    doc_id=f"{doc.doc_id}#chunk{i}",
                    content=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index": i,
                        "total_chunks": len(text_chunks),
                        "original_doc_id": doc.doc_id,
                    },
                )
                chunks.append(chunk_doc)

        logger.info(
            "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
            len(documents), len(chunks), self.chunk_size, self.chunk_overlap,
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split *text* using *separators* in order."""
        if len(text) <= self.chunk_size:
            stripped = text.strip()
            return [stripped] if stripped else []

        # Try each separator in order
        for sep in separators:
            if sep in text:
                parts = text.split(sep)
                return self._merge_parts(parts, sep, separators)

        # Fallback: hard split at chunk_size
        return self._hard_split(text)

    def _merge_parts(
        self, parts: list[str], sep: str, separators: list[str],
    ) -> list[str]:
        """Merge split *parts* back into chunks ≤ ``chunk_size``."""
        chunks: list[str] = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    trimmed = current.strip()
                    if trimmed:
                        chunks.append(trimmed)
                    # Start next chunk with overlap from end of current
                    if self.chunk_overlap > 0 and current:
                        overlap = current[-self.chunk_overlap:]
                        current = overlap + sep + part
                    else:
                        current = part
                else:
                    # Single part exceeds chunk_size — recurse deeper
                    remaining_seps = separators[separators.index(sep) + 1:]
                    if remaining_seps:
                        sub_chunks = self._recursive_split(part, remaining_seps)
                    else:
                        sub_chunks = self._hard_split(part)
                    chunks.extend(sub_chunks)
                    current = ""

        if current and current.strip():
            chunks.append(current.strip())

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Split text at fixed character boundaries as a last resort."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        return chunks
