"""
Unit tests for Phase 6: Scientific Table Extraction, Structured Markdown Representation, Table Repository, and Table Retrieval.
"""

import pytest
import pymupdf as fitz
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.table_parser import TableParser, ExtractedTable
from src.retrieval.table_repository import TableRepository
from src.retrieval.table_retriever import TableRetriever, TableSearchResult
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_validator import CitationValidator
from src.models.embeddings import BGEEmbeddings


@pytest.fixture
def synthetic_table_pdf(tmp_path: Path) -> Path:
    """Fixture creating a PDF file containing formatted vector tables for pdfplumber testing."""
    pdf_file = tmp_path / "table_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Draw table grid lines and text so pdfplumber detects grid cells
    # Draw headers
    page.insert_text((50, 50), "Model")
    page.insert_text((150, 50), "MSE")
    page.insert_text((250, 50), "CSI")

    # Draw Row 1
    page.insert_text((50, 80), "ConvLSTM")
    page.insert_text((150, 80), "0.09")
    page.insert_text((250, 80), "0.76")

    # Draw Row 2
    page.insert_text((50, 110), "FC-LSTM")
    page.insert_text((150, 110), "0.14")
    page.insert_text((250, 110), "0.68")

    # Draw horizontal & vertical table border lines
    page.draw_line((40, 35), (320, 35))
    page.draw_line((40, 60), (320, 60))
    page.draw_line((40, 95), (320, 95))
    page.draw_line((40, 125), (320, 125))

    page.draw_line((40, 35), (40, 125))
    page.draw_line((130, 35), (130, 125))
    page.draw_line((230, 35), (230, 125))
    page.draw_line((320, 35), (320, 125))

    doc.save(str(pdf_file))
    doc.close()
    return pdf_file


def test_table_parser_markdown_generation():
    """Test conservative cleaning and Markdown table string generation."""
    parser = TableParser()
    headers = ["Model", "Accuracy (%)", "F1 Score"]
    rows = [
        ["ConvLSTM-5x5", "95.1", "0.94"],
        ["FC-LSTM", "91.2", "0.89"],
        [None, "  88.5  ", ""],  # Handles None and whitespace
    ]

    md_text = parser._generate_markdown(headers, rows)

    assert "| Model | Accuracy (%) | F1 Score |" in md_text
    assert "| --- | --- | --- |" in md_text
    assert "| ConvLSTM-5x5 | 95.1 | 0.94 |" in md_text
    assert "| FC-LSTM | 91.2 | 0.89 |" in md_text
    assert "|  | 88.5 |  |" in md_text


def test_table_parser_extraction_and_determinism(synthetic_table_pdf: Path):
    """Test pdfplumber table extraction, header/row detection, deterministic IDs, and page provenance."""
    parser = TableParser()
    tables = parser.extract_tables_from_pdf(synthetic_table_pdf)

    assert len(tables) >= 1
    table = tables[0]

    assert isinstance(table, ExtractedTable)
    assert table.doc_id == "table_paper"
    assert table.source_filename == "table_paper.pdf"
    assert table.page_number == 1
    assert table.table_index == 0
    assert table.table_id == "table_paper_p001_t00"

    assert len(table.headers) >= 3
    assert "Model" in table.headers[0]
    assert len(table.rows) >= 2
    assert "ConvLSTM" in table.rows[0][0]

    # Verify determinism on re-running extraction
    tables_retry = parser.extract_tables_from_pdf(synthetic_table_pdf)
    assert [t.table_id for t in tables] == [t.table_id for t in tables_retry]


def test_table_repository():
    """Test TableRepository storing, retrieving by ID, doc_id, and page_number."""
    repo = TableRepository()
    assert repo.count() == 0

    t1 = ExtractedTable(
        table_id="docA_p002_t00",
        doc_id="docA",
        source_filename="docA.pdf",
        page_number=2,
        table_index=0,
        headers=["Model", "RMSE"],
        rows=[["ConvLSTM", "0.09"]],
        markdown_text="| Model | RMSE |\n|---|---|\n| ConvLSTM | 0.09 |",
    )
    t2 = ExtractedTable(
        table_id="docA_p002_t01",
        doc_id="docA",
        source_filename="docA.pdf",
        page_number=2,
        table_index=1,
        headers=["Param", "Value"],
        rows=[["LR", "0.001"]],
        markdown_text="| Param | Value |\n|---|---|\n| LR | 0.001 |",
    )

    repo.add_tables([t1, t2])
    assert repo.count() == 2
    assert repo.get_table("docA_p002_t00") == t1
    assert len(repo.get_tables_by_doc("docA")) == 2
    assert len(repo.get_tables_by_page("docA", 2)) == 2


def test_table_retriever_vector_search():
    """Test isolated ChromaDB table vector insertion and semantic search."""
    retriever = TableRetriever(in_memory=True, collection_name="test_table_collection")

    t1 = ExtractedTable(
        table_id="conv_p006_t00",
        doc_id="convlstm",
        source_filename="convlstm.pdf",
        page_number=6,
        table_index=0,
        headers=["Model", "Rainfall-MSE", "CSI"],
        rows=[["ConvLSTM", "0.089", "0.78"]],
        markdown_text="| Model | Rainfall-MSE | CSI |\n|---|---|---|\n| ConvLSTM | 0.089 | 0.78 |",
    )
    t2 = ExtractedTable(
        table_id="optical_p010_t00",
        doc_id="rover",
        source_filename="rover.pdf",
        page_number=10,
        table_index=0,
        headers=["Parameter", "Meaning", "Value"],
        rows=[["Lm", "Lagrange multiplier", "0.5"]],
        markdown_text="| Parameter | Meaning | Value |\n|---|---|---|\n| Lm | Lagrange multiplier | 0.5 |",
    )

    mock_embeddings = MagicMock(spec=BGEEmbeddings)
    # Mock query close to t1
    mock_embeddings.embed_batch.return_value = [[1.0] + [0.0] * 383, [0.0] * 383 + [1.0]]
    mock_embeddings.embed_text.return_value = [0.95] + [0.0] * 383
    retriever.embedding_model = mock_embeddings

    retriever.add_tables([t1, t2])
    results = retriever.retrieve_tables("Which model has lowest Rainfall-MSE score?", top_k=2)

    assert len(results) == 2
    top_res = results[0]
    assert top_res.table_id == "conv_p006_t00"
    assert top_res.doc_id == "convlstm"
    assert top_res.page_number == 6
    assert "Rainfall-MSE" in top_res.markdown_text


def test_table_evidence_registration_and_citation_validation():
    """Test registering table evidence in EvidenceRegistry and validating table citations."""
    reg = EvidenceRegistry()
    table_res = TableSearchResult(
        table_id="convlstm_p008_t00",
        doc_id="convlstm",
        source_filename="convlstm_precipitation_nowcasting.pdf",
        page_number=8,
        table_index=0,
        headers=["Model", "MSE"],
        rows=[["ConvLSTM", "0.089"]],
        markdown_text="| Model | MSE |\n|---|---|\n| ConvLSTM | 0.089 |",
        score=0.88,
        distance=0.24,
        metadata={},
    )

    ev_item = reg.register_table(table_res)

    assert ev_item.evidence_id == "E1"
    assert ev_item.evidence_type == "table"
    assert ev_item.table_id == "convlstm_p008_t00"
    assert ev_item.page_number == 8

    # Validate table citation
    validator = CitationValidator()
    val_result = validator.validate_citations("ConvLSTM achieves lowest MSE [E1].", reg)

    assert val_result.validation_status == "valid"
    assert len(val_result.citations) == 1
    cit = val_result.citations[0]
    assert cit.evidence_id == "E1"
    assert cit.formatted_citation == "[E1] convlstm_precipitation_nowcasting.pdf, Table (p. 8)"
