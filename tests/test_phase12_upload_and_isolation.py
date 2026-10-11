"""
Phase 12 Tests: PDF Upload Validation, Ingestion Pipeline, Session Isolation, and Paper Filtering.
"""

import os
import uuid
import tempfile
import pytest
import pymupdf as fitz

from src.ingestion.pipeline import (
    validate_pdf_bytes,
    ingest_pdf,
    PDFValidationError,
    IngestionReport,
)
from src.retrieval.vector_store import VectorStore
from src.retrieval.text_retriever import TextRetriever
from src.agent.nodes import ResearchAgentNodes
from src.agent.graph import ResearchAgent


def create_simple_pdf_bytes(text: str = "Sample research page content", page_count: int = 1) -> bytes:
    """Helper to generate minimal valid PDF bytes."""
    doc = fitz.open()
    for i in range(page_count):
        page = doc.new_page()
        page.insert_text((50, 50), f"{text} - Page {i+1}")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_pdf_magic_byte_validation():
    """Verify non-PDF bytes are rejected by magic byte check."""
    invalid_bytes = b"This is not a PDF file header."
    with pytest.raises(PDFValidationError, match="not a valid PDF document"):
        validate_pdf_bytes(invalid_bytes, "invalid.txt")


def test_pdf_file_size_validation():
    """Verify oversized PDFs (>20MB) are rejected."""
    pdf_bytes = create_simple_pdf_bytes("Large file test")
    with pytest.raises(PDFValidationError, match="exceeds maximum allowed size"):
        validate_pdf_bytes(pdf_bytes, "large.pdf", max_size_mb=0.00001)

    # Test default 20 MB boundary message
    dummy_large = b"%PDF" + b"0" * (21 * 1024 * 1024)
    with pytest.raises(PDFValidationError, match="exceeds maximum allowed size of 20 MB"):
        validate_pdf_bytes(dummy_large, "oversized_21mb.pdf")


def test_pdf_page_count_validation():
    """Verify PDFs with >50 pages are rejected while <=50 pages are accepted."""
    # 50 pages: accepted
    pdf_50 = create_simple_pdf_bytes("Page 50 test", page_count=50)
    assert validate_pdf_bytes(pdf_50, "fifty_pages.pdf") == 50

    # 51 pages: rejected by default limit of 50
    pdf_51 = create_simple_pdf_bytes("Page 51 test", page_count=51)
    with pytest.raises(PDFValidationError, match="exceeds the limit of 50 pages"):
        validate_pdf_bytes(pdf_51, "fifty_one_pages.pdf")


def test_pdf_ingestion_report():
    """Verify ingest_pdf produces accurate page, chunk, and status counts."""
    pdf_bytes = create_simple_pdf_bytes("Transformer attention mechanisms are scalable.", page_count=2)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        vstore = VectorStore(in_memory=True, collection_name="test_ingest_report")
        report: IngestionReport = ingest_pdf(tmp_path, vector_store=vstore, paper_name="transformer.pdf")

        assert report.success is True
        assert report.paper_name == "transformer.pdf"
        assert report.pages == 2
        assert report.chunks >= 1
        assert vstore.count() >= 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_session_isolation_prevention_of_data_leakage():
    """
    Verify that User A's session collection is isolated from User B's session collection.
    User B cannot retrieve User A's uploaded document chunks.
    """
    session_a_id = f"session_{uuid.uuid4().hex[:8]}"
    session_b_id = f"session_{uuid.uuid4().hex[:8]}"

    vstore_a = VectorStore(in_memory=True, collection_name=session_a_id)
    vstore_b = VectorStore(in_memory=True, collection_name=session_b_id)

    # User A uploads confidential paper
    pdf_a = create_simple_pdf_bytes("User A Confidential Alpha Secret Project Details", page_count=1)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_a:
        tmp_a.write(pdf_a)
        tmp_a_path = tmp_a.name

    try:
        report_a = ingest_pdf(tmp_a_path, vector_store=vstore_a, paper_name="user_a_secret.pdf")
        assert report_a.success is True

        # User A's retriever should find the document
        retriever_a = TextRetriever(vector_store=vstore_a)
        results_a = retriever_a.search("Confidential Alpha Secret")
        assert len(results_a) > 0
        assert "User A Confidential" in results_a[0].text

        # User B's retriever must NOT find User A's document
        retriever_b = TextRetriever(vector_store=vstore_b)
        results_b = retriever_b.search("Confidential Alpha Secret")
        assert len(results_b) == 0

    finally:
        if os.path.exists(tmp_a_path):
            os.remove(tmp_a_path)


def test_paper_filtering():
    """Verify that specifying paper_filter restricts retrieval to selected documents."""
    vstore = VectorStore(in_memory=True, collection_name="test_paper_filter")

    pdf1 = create_simple_pdf_bytes("ConvLSTM precipitation nowcasting model details.", page_count=1)
    pdf2 = create_simple_pdf_bytes("Transformer attention mechanism details.", page_count=1)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp1, tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp2:
        tmp1.write(pdf1)
        tmp2.write(pdf2)
        p1_path, p2_path = tmp1.name, tmp2.name

    try:
        ingest_pdf(p1_path, vector_store=vstore, paper_name="convlstm.pdf")
        ingest_pdf(p2_path, vector_store=vstore, paper_name="transformer.pdf")

        text_retriever = TextRetriever(vector_store=vstore)
        from unittest.mock import MagicMock
        mock_llm = MagicMock()
        mock_llm.is_configured.return_value = True
        mock_llm.generate_text.return_value = "ConvLSTM details [E1]."
        nodes = ResearchAgentNodes(text_retriever=text_retriever, llm_client=mock_llm)
        agent = ResearchAgent(nodes=nodes)

        # Query with paper_filter="convlstm.pdf"
        state = agent.run("details", paper_filter="convlstm.pdf")
        chunks = state.get("tool_results", {}).get("text_chunks", [])

        assert len(chunks) > 0
        for c in chunks:
            assert c.source_filename == "convlstm.pdf"

    finally:
        for p in [p1_path, p2_path]:
            if os.path.exists(p):
                os.remove(p)
