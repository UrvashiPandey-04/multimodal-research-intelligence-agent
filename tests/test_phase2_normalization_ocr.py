"""
Unit tests for Phase 2: Text Quality Assessment, Conservative Normalization, and OCR Fallback.
"""

import pytest
import pymupdf as fitz
from pathlib import Path
from unittest.mock import MagicMock

from src.ingestion.schema import NormalizedDocument, NormalizedPage, QualityMetrics
from src.ingestion.text_quality import TextQualityAssessor, assess_text_quality
from src.ingestion.normalizer import TextNormalizer, normalize_text
from src.ingestion.ocr import OCREngine, OCRNotAvailableError
from src.ingestion.pdf_parser import PDFParser


def test_text_quality_assessor_good_text():
    """Test quality assessor with good natural language text."""
    assessor = TextQualityAssessor(min_words=10)
    text = "Convolutional LSTM Network is a machine learning model designed for precipitation nowcasting and spatio-temporal forecasting."
    metrics = assessor.evaluate(text)

    assert metrics.is_usable is True
    assert metrics.word_count > 10
    assert metrics.printable_ratio > 0.95
    assert len(metrics.failure_reasons) == 0


def test_text_quality_assessor_poor_text():
    """Test quality assessor with insufficient / low word count text."""
    assessor = TextQualityAssessor(min_words=15)
    text = "Short header"
    metrics = assessor.evaluate(text)

    assert metrics.is_usable is False
    assert metrics.word_count == 2
    assert "insufficient_word_count" in metrics.failure_reasons[0]


def test_text_normalizer_conservative():
    """Test conservative text normalizer preserves numbers, equations, and citations."""
    raw_text = (
        "   Model   accuracy is  95.4%   (Smith et al., 2021).  \r\n"
        "Equation:   E = mc^2 + \\sigma_0   \n\n\n"
        "Result: 1.5e-4 RMSE value.   "
    )
    normalized = normalize_text(raw_text)

    # Check excessive space removal
    assert "Model accuracy is 95.4% (Smith et al., 2021)." in normalized
    # Check newline normalization (collapsed 3 newlines to 2)
    assert "\n\nResult: 1.5e-4 RMSE value." in normalized
    # Verify numbers, scientific notation, and citations preserved
    assert "95.4%" in normalized
    assert "(Smith et al., 2021)" in normalized
    assert "E = mc^2 + \\sigma_0" in normalized
    assert "1.5e-4" in normalized


def test_pdf_parser_native_extraction_retained(tmp_path: Path):
    """Test that a PDF with good native text retains 'native' extraction method."""
    pdf_file = tmp_path / "native_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, 550, 400)
    text_content = (
        "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.\n"
        "Abstract: Spatio-temporal sequence forecasting is a key problem in meteorological prediction.\n"
        "We formulate precipitation nowcasting as a spatio-temporal sequence forecasting problem where both input and prediction target are image sequences."
    )
    page.insert_textbox(rect, text_content)
    doc.save(str(pdf_file))
    doc.close()

    parser = PDFParser(ocr_enabled=False)
    norm_doc = parser.parse_pdf(pdf_file)

    assert isinstance(norm_doc, NormalizedDocument)
    assert norm_doc.doc_id == "native_paper"
    assert norm_doc.source_filename == "native_paper.pdf"
    assert norm_doc.total_pages == 1
    assert norm_doc.native_page_count == 1
    assert norm_doc.ocr_page_count == 0

    page1 = norm_doc.pages[0]
    assert page1.extraction_method == "native"
    assert page1.page_number == 1
    assert "Convolutional LSTM" in page1.text
    assert page1.quality_metrics.is_usable is True


def test_pdf_parser_ocr_fallback_triggered(tmp_path: Path):
    """Test that a PDF page with poor/empty native text triggers OCR fallback via mock OCREngine."""
    pdf_file = tmp_path / "scanned_paper.pdf"
    doc = fitz.open()
    page = doc.new_page()  # Blank page with no text (simulates scanned image)
    doc.save(str(pdf_file))
    doc.close()

    # Mock OCREngine to return simulated OCR text
    mock_ocr = MagicMock(spec=OCREngine)
    mock_ocr.is_available.return_value = True
    mock_ocr.process_pymupdf_page.return_value = (
        "Extracted text via Tesseract OCR for scanned flood inundation map figure and tables. "
        "The model achieves an RMSE of 0.12 meters across test catchment basins."
    )

    parser = PDFParser(min_words_per_page=15, ocr_enabled=True, ocr_engine=mock_ocr)
    norm_doc = parser.parse_pdf(pdf_file)

    assert norm_doc.total_pages == 1
    assert norm_doc.native_page_count == 0
    assert norm_doc.ocr_page_count == 1

    page1 = norm_doc.pages[0]
    assert page1.extraction_method == "ocr"
    assert page1.doc_id == "scanned_paper"
    assert page1.page_number == 1
    assert "Extracted text via Tesseract OCR" in page1.text
    assert page1.quality_metrics.is_usable is True

    # Verify mock process_pymupdf_page was called exactly once
    mock_ocr.process_pymupdf_page.assert_called_once()
