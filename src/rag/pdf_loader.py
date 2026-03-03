"""PDF document loader for extracting text from PDF files.

PDF 文档加载器：从 PDF 文件中提取文本内容，保留页码等元数据。
支持单文件和目录批量加载，自动跳过无文本页面。
"""

import logging
import os
from pathlib import Path

from src.rag.bm25_retriever import Document

logger = logging.getLogger(__name__)


class PDFLoader:
    """Load and extract text from PDF files.

    Extracts text page-by-page from PDF files using ``pypdf``.
    Each page becomes a :class:`Document` with metadata including
    the source filename and page number.
    """

    def load(self, pdf_path: str) -> list[Document]:
        """Load a single PDF file and return a list of page-level Documents.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of Document objects, one per non-empty page.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is not a PDF.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Not a PDF file: {pdf_path}")

        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required for PDF loading. "
                "Install it with: pip install pypdf"
            )

        documents: list[Document] = []
        reader = PdfReader(str(path))
        filename = path.name

        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if not text:
                continue

            doc = Document(
                doc_id=f"{filename}:p{page_num}",
                content=text,
                metadata={
                    "source": str(path),
                    "filename": filename,
                    "page": page_num,
                    "total_pages": len(reader.pages),
                },
            )
            documents.append(doc)

        logger.info("Loaded %d pages from %s", len(documents), pdf_path)
        return documents

    def load_directory(self, dir_path: str) -> list[Document]:
        """Load all PDF files from a directory.

        Args:
            dir_path: Path to the directory containing PDF files.

        Returns:
            List of Document objects from all PDF files.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        documents: list[Document] = []
        pdf_files = sorted(path.glob("*.pdf"))

        for pdf_file in pdf_files:
            try:
                docs = self.load(str(pdf_file))
                documents.extend(docs)
            except Exception as exc:
                logger.warning("Failed to load %s: %s", pdf_file, exc)

        logger.info(
            "Loaded %d documents from %d PDF files in %s",
            len(documents), len(pdf_files), dir_path,
        )
        return documents
