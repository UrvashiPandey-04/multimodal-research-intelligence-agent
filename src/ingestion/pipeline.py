"""
Ingestion Pipeline module for processing user-uploaded PDFs and sample papers.
Provides PDF magic-byte validation, modular ingestion reporting, and per-session index management.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import pymupdf as fitz

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.ingestion.table_parser import TableParser
from src.ingestion.figure_parser import PyMuPDFFigureExtractor
from src.models.embeddings import get_embedding_model
from src.retrieval.vector_store import VectorStore
from src.retrieval.table_repository import TableRepository
from src.retrieval.table_retriever import TableRetriever
from src.retrieval.figure_repository import FigureRepository
from src.retrieval.figure_retriever import FigureRetriever


class PDFValidationError(Exception):
    """Raised when PDF fails validation checks (magic bytes, size, page count, encryption)."""
    pass


@dataclass
class IngestionReport:
    """Structured report of document ingestion metrics, warnings, and status."""
    paper_name: str
    pages: int = 0
    chunks: int = 0
    tables: int = 0
    figures: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_name": self.paper_name,
            "pages": self.pages,
            "chunks": self.chunks,
            "tables": self.tables,
            "figures": self.figures,
            "warnings": self.warnings,
            "errors": self.errors,
            "success": self.success,
        }


def validate_pdf_bytes(
    pdf_bytes: bytes,
    filename: str,
    max_size_mb: float = 20.0,
    max_pages: int = 50,
) -> int:
    """
    Validate PDF raw bytes before processing.

    Args:
        pdf_bytes: Raw binary content of the PDF file.
        filename: Original filename.
        max_size_mb: Maximum allowed file size in MB (default: 20.0 MB).
        max_pages: Maximum allowed page count (default: 50 pages).

    Returns:
        Total page count of the validated PDF.

    Raises:
        PDFValidationError: If any validation rule fails.
    """
    # 1. Magic bytes check (%PDF)
    if not pdf_bytes or len(pdf_bytes) < 4 or not pdf_bytes.startswith(b"%PDF"):
        raise PDFValidationError(f"File '{filename}' is not a valid PDF document (missing %PDF header).")

    # 2. File size check
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise PDFValidationError(
            f"File '{filename}' exceeds maximum allowed size of {max_size_mb} MB (file size: {size_mb:.1f} MB)."
        )

    # 3. PyMuPDF parsing & encryption check
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise PDFValidationError(f"File '{filename}' is corrupted and cannot be parsed as a PDF: {e}")

    try:
        if doc.is_encrypted:
            if not doc.authenticate(""):
                raise PDFValidationError(f"File '{filename}' is encrypted and password-protected.")

        page_count = doc.page_count
        if page_count == 0:
            raise PDFValidationError(f"File '{filename}' contains 0 pages.")

        if page_count > max_pages:
            raise PDFValidationError(
                f"File '{filename}' contains {page_count} pages, which exceeds the limit of {max_pages} pages."
            )

        return page_count
    finally:
        doc.close()


def ingest_pdf(
    pdf_path: str,
    vector_store: VectorStore,
    table_repo: Optional[TableRepository] = None,
    table_retriever: Optional[TableRetriever] = None,
    fig_extractor: Optional[PyMuPDFFigureExtractor] = None,
    fig_repo: Optional[FigureRepository] = None,
    figure_retriever: Optional[FigureRetriever] = None,
    paper_name: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> IngestionReport:
    """
    Ingest a single PDF paper into vector stores and repositories.

    Args:
        pdf_path: Absolute or relative file path to the PDF.
        vector_store: Target VectorStore instance for text chunks.
        table_repo: Optional TableRepository.
        table_retriever: Optional TableRetriever.
        fig_extractor: Optional PyMuPDFFigureExtractor.
        fig_repo: Optional FigureRepository.
        figure_retriever: Optional FigureRetriever.
        paper_name: Optional display filename override.
        progress_callback: Optional callback for status notifications.

    Returns:
        IngestionReport summarizing page, chunk, table, and figure counts.
    """
    path = Path(pdf_path)
    fname = paper_name or path.name
    report = IngestionReport(paper_name=fname)

    def notify(msg: str):
        if progress_callback:
            progress_callback(msg)

    try:
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        report.pages = validate_pdf_bytes(pdf_bytes, fname)
    except PDFValidationError as ve:
        report.errors.append(str(ve))
        report.success = False
        return report
    except Exception as e:
        report.errors.append(f"Failed to read file: {e}")
        report.success = False
        return report

    # 1. Parsing text
    notify("Parsing text")
    parser = PDFParser()
    try:
        norm_doc = parser.parse_pdf(path)
        if fname:
            norm_doc.source_filename = fname
            norm_doc.doc_id = Path(fname).stem
            for p in norm_doc.pages:
                p.source_filename = fname
                p.doc_id = Path(fname).stem
    except Exception as e:
        report.errors.append(f"Text parsing failed: {e}")
        report.success = False
        return report


    has_text = any(len(page.text.strip()) > 0 for page in norm_doc.pages)
    if not has_text:
        report.warnings.append("No extractable text found in PDF page contents. OCR may be unavailable or page images unreadable.")

    # 2. Extracting tables
    notify("Extracting tables")
    try:
        t_parser = TableParser()
        tables = t_parser.extract_tables_from_pdf(str(path))
        if fname:
            for t in tables:
                t.source_filename = fname
        report.tables = len(tables)
        if table_repo:
            table_repo.add_tables(tables)
        if table_retriever:
            table_retriever.add_tables(tables)
    except Exception as e:
        report.warnings.append(f"Table extraction encountered issue: {e}")

    # 3. Extracting figures
    notify("Extracting figures")
    try:
        if fig_extractor:
            figs = fig_extractor.extract_figures(str(path), norm_doc.doc_id)
            if fname:
                for f in figs:
                    f.source_filename = fname
            report.figures = len(figs)
            if fig_repo:
                fig_repo.add_figures(figs)
            if figure_retriever:
                figure_retriever.add_figures(figs)
    except Exception as e:
        report.warnings.append(f"Figure extraction encountered issue: {e}")


    # 4. Creating chunks
    notify("Creating chunks")
    chunker = DocumentChunker()
    chunks = chunker.chunk_document(norm_doc)
    report.chunks = len(chunks)

    if len(chunks) == 0:
        if not report.warnings:
            report.warnings.append("Document produced 0 text chunks.")
        notify("Done")
        return report

    # 5. Generating embeddings
    notify("Generating embeddings")
    emb_model = get_embedding_model()
    chunk_texts = [c.text for c in chunks]
    embeddings = emb_model.embed_batch(chunk_texts)

    # 6. Indexing document
    notify("Indexing document")
    vector_store.add_chunks(chunks, embeddings)

    notify("Done")
    return report
