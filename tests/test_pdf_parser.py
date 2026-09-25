"""
Unit tests for PyMuPDF-based PDF Parser with Normalized Data Models.
"""

import pytest
import pymupdf as fitz
from pathlib import Path
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.schema import NormalizedDocument, NormalizedPage


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Fixture creating a temporary 2-page PDF file for testing."""
    pdf_file = tmp_path / "sample_paper.pdf"
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    page1.insert_text((50, 50), "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.")
    page1.insert_text((50, 70), "Abstract: We propose ConvLSTM for spatio-temporal sequence forecasting in hydro-meteorological modeling.")
    page1.insert_text((50, 90), "Our model handles complex non-linear spatial relationships and temporal dependencies effectively.")

    # Page 2
    page2 = doc.new_page()
    page2.insert_text((50, 50), "Methodology and Experimental Results for Flood Inundation Analysis.")
    page2.insert_text((50, 70), "The proposed deep neural network model achieves superior RMSE performance compared to baselines.")
    page2.insert_text((50, 90), "Experiments demonstrate robust generalization across diverse catchment basins and rainfall events.")

    doc.save(str(pdf_file))
    doc.close()
    return pdf_file


def test_parse_valid_pdf(sample_pdf: Path):
    """Test parsing a valid 2-page PDF file."""
    parser = PDFParser(ocr_enabled=False)
    doc_data = parser.parse_pdf(sample_pdf)

    assert isinstance(doc_data, NormalizedDocument)
    assert doc_data.doc_id == "sample_paper"
    assert doc_data.source_filename == "sample_paper.pdf"
    assert doc_data.total_pages == 2
    assert len(doc_data.pages) == 2

    # Verify Page 1
    page1 = doc_data.pages[0]
    assert isinstance(page1, NormalizedPage)
    assert page1.doc_id == "sample_paper"
    assert page1.source_filename == "sample_paper.pdf"
    assert page1.page_number == 1
    assert "Convolutional LSTM Network" in page1.text
    assert page1.quality_metrics.word_count > 0
    assert page1.extraction_method == "native"
    assert page1.quality_metrics.is_usable is True

    # Verify Page 2
    page2 = doc_data.pages[1]
    assert page2.page_number == 2
    assert "superior RMSE performance" in page2.text
    assert page2.quality_metrics.word_count > 0

    # Verify serialization
    p1_dict = page1.to_dict()
    assert p1_dict["doc_id"] == "sample_paper"
    assert p1_dict["source_filename"] == "sample_paper.pdf"
    assert p1_dict["page_number"] == 1
    assert p1_dict["extraction_method"] == "native"


def test_parse_blank_page_pdf(tmp_path: Path):
    """Test parsing a PDF page with zero text to verify OCR fallback trigger logic."""
    pdf_file = tmp_path / "scanned_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()  # empty page
    doc.save(str(pdf_file))
    doc.close()

    parser = PDFParser(min_words_per_page=15, ocr_enabled=False)
    doc_data = parser.parse_pdf(pdf_file)

    assert doc_data.total_pages == 1
    page1 = doc_data.pages[0]
    assert page1.quality_metrics.word_count == 0
    assert page1.text == ""
    assert page1.quality_metrics.is_usable is False


def test_parse_non_existent_file():
    """Test parsing a file path that does not exist."""
    parser = PDFParser()
    with pytest.raises(FileNotFoundError):
        parser.parse_pdf("non_existent_file_12345.pdf")


def test_parse_invalid_pdf(tmp_path: Path):
    """Test parsing an invalid non-PDF file."""
    invalid_file = tmp_path / "corrupt.pdf"
    invalid_file.write_text("This is not a valid PDF binary content.")

    parser = PDFParser()
    with pytest.raises(ValueError, match="Failed to open PDF file"):
        parser.parse_pdf(invalid_file)
