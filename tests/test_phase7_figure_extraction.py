"""
Unit tests for Phase 7: Scientific figure extraction, VLM visual analysis, figure retrieval, and citation validation.
"""

import os
import tempfile
import fitz  # PyMuPDF
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch

from src.ingestion.figure_parser import PyMuPDFFigureExtractor, ExtractedFigure
from src.retrieval.figure_repository import FigureRepository
from src.models.vlm_cache import VLMCache
from src.models.vlm_analyzer import VLMFigureAnalyzer, FigureAnalysisResult
from src.retrieval.figure_retriever import FigureRetriever, FigureSearchResult
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_validator import CitationValidator


def create_synthetic_pdf_with_image(pdf_path: str, include_caption: bool = True):
    """Helper to generate a synthetic PDF containing an image and optional caption."""
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)

    # 1. Create a dummy PNG image in memory and save to disk
    img = Image.new("RGB", (200, 150), color="blue")
    temp_img_path = pdf_path + ".tmp.png"
    img.save(temp_img_path)

    rect = fitz.Rect(100, 100, 300, 250)
    page.insert_image(rect, filename=temp_img_path)

    # 2. Optionally insert a figure caption text block
    if include_caption:
        page.insert_text((100, 270), "Figure 1: Convolutional LSTM architecture diagram.", fontsize=11)

    doc.save(pdf_path)
    doc.close()

    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)


def test_figure_parser_extraction_and_determinism():
    """Test PDF figure extraction, deterministic ID generation, and caption matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "test_paper.pdf")
        fig_out_dir = os.path.join(tmpdir, "figures")
        create_synthetic_pdf_with_image(pdf_path, include_caption=True)

        extractor = PyMuPDFFigureExtractor(output_dir=fig_out_dir, min_dimension=30, min_area=900)
        doc_id = "doc_hash_123"

        figures = extractor.extract_figures(pdf_path, doc_id)
        assert len(figures) == 1

        fig = figures[0]
        assert fig.figure_id == "doc_hash_123_p001_f00"
        assert fig.doc_id == doc_id
        assert fig.page_number == 1
        assert fig.figure_index == 0
        assert fig.caption == "Figure 1: Convolutional LSTM architecture diagram."
        assert os.path.exists(fig.image_path)

        # Determinism check
        figures_repeat = extractor.extract_figures(pdf_path, doc_id)
        assert figures_repeat[0].figure_id == fig.figure_id


def test_figure_parser_missing_caption_handling():
    """Test extraction when no caption text exists near the image."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "no_caption.pdf")
        fig_out_dir = os.path.join(tmpdir, "figures")
        create_synthetic_pdf_with_image(pdf_path, include_caption=False)

        extractor = PyMuPDFFigureExtractor(output_dir=fig_out_dir, min_dimension=30, min_area=900)
        figures = extractor.extract_figures(pdf_path, "doc_nocap")
        assert len(figures) == 1
        assert figures[0].caption is None


def test_figure_repository():
    """Test in-memory FigureRepository storage, retrieval by ID, doc_id, and page."""
    repo = FigureRepository()
    fig1 = ExtractedFigure(
        figure_id="doc1_p001_f00",
        doc_id="doc1",
        source_filename="paper1.pdf",
        page_number=1,
        figure_index=0,
        image_path="/path/fig1.png",
        caption="Fig 1: Plot",
        extraction_method="pymupdf_cropped",
    )
    fig2 = ExtractedFigure(
        figure_id="doc1_p002_f00",
        doc_id="doc1",
        source_filename="paper1.pdf",
        page_number=2,
        figure_index=0,
        image_path="/path/fig2.png",
        caption="Fig 2: Diagram",
        extraction_method="pymupdf_cropped",
    )

    repo.add_figures([fig1, fig2])

    assert repo.get_by_id("doc1_p001_f00") == fig1
    assert len(repo.get_by_doc_id("doc1")) == 2
    assert repo.get_by_page("doc1", 2) == [fig2]
    assert len(repo.list_figures()) == 2


def test_vlm_cache_behavior():
    """Test that VLM analysis results are stored and re-used from disk cache."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_file = os.path.join(tmpdir, "vlm_cache.json")
        cache = VLMCache(cache_file=cache_file)

        key = cache.make_key("doc1", "doc1_p001_f00")
        assert cache.get(key) is None

        data = {
            "figure_summary": "Architecture flowchart",
            "figure_type": "flowchart",
            "observations": ["3 boxes"],
            "visible_values": [],
            "uncertainties": [],
        }
        cache.set(key, data)

        # Reload cache from disk
        cache2 = VLMCache(cache_file=cache_file)
        cached = cache2.get(key)
        assert cached is not None
        assert cached["figure_type"] == "flowchart"


def test_vlm_structured_json_parsing_and_fallback():
    """Test structured VLM JSON response parsing and handling of malformed text."""
    analyzer = VLMFigureAnalyzer()
    fig = ExtractedFigure(
        figure_id="f1",
        doc_id="d1",
        source_filename="p.pdf",
        page_number=1,
        figure_index=0,
        image_path="/path/img.png",
        caption="Fig 1",
        extraction_method="pymupdf_cropped",
    )

    # 1. Valid markdown fenced JSON
    raw_json = """```json
    {
        "figure_summary": "Line plot showing loss curve.",
        "figure_type": "line plot",
        "observations": ["Loss decreases steadily"],
        "visible_values": ["0.12", "0.05"],
        "uncertainties": []
    }
    ```"""
    parsed = analyzer._parse_json_response(raw_json, fig)
    assert parsed.figure_summary == "Line plot showing loss curve."
    assert parsed.figure_type == "line plot"
    assert parsed.visible_values == ["0.12", "0.05"]

    # 2. Malformed non-JSON fallback
    malformed = "This is plain prose response without JSON."
    parsed_fallback = analyzer._parse_json_response(malformed, fig)
    assert parsed_fallback.figure_type == "unknown"
    assert "not valid JSON" in parsed_fallback.uncertainties[0]


def test_figure_retriever_vector_search():
    """Test FigureRetriever indexing and semantic vector retrieval over research_paper_figures."""
    retriever = FigureRetriever()
    fig1 = ExtractedFigure(
        figure_id="doc1_p005_f00",
        doc_id="doc1",
        source_filename="convlstm.pdf",
        page_number=5,
        figure_index=0,
        image_path="/path/convlstm_net.png",
        caption="Figure 5: ConvLSTM network architecture with recurrent connections.",
        extraction_method="pymupdf_cropped",
    )
    analysis1 = FigureAnalysisResult(
        figure_summary="Diagram illustrating ConvLSTM cell structure and hidden state transitions.",
        figure_type="architecture diagram",
        observations=["Input gate", "Forget gate", "Hidden state H_t"],
        visible_values=[],
        uncertainties=[],
    )

    retriever.add_figure(fig1, analysis1)

    results = retriever.search("ConvLSTM architecture diagram", top_k=1)
    assert len(results) == 1
    top = results[0]
    assert top.figure_id == "doc1_p005_f00"
    assert top.page_number == 5
    assert top.score > 0.5
    assert top.analysis.figure_type == "architecture diagram"


def test_figure_evidence_registration_and_citation_validation():
    """Test registering figure evidence in EvidenceRegistry and validating citation format."""
    registry = EvidenceRegistry()
    validator = CitationValidator()

    fig_search_res = FigureSearchResult(
        figure_id="convlstm_p006_f00",
        doc_id="convlstm_hash",
        source_filename="convlstm_precipitation_nowcasting.pdf",
        page_number=6,
        image_path="/path/fig.png",
        caption="Figure 3: Hidden state propagation.",
        searchable_text="Figure 3: Hidden state propagation.",
        score=0.88,
    )

    item = registry.register_figure(fig_search_res)
    assert item.evidence_id == "E1"
    assert item.evidence_type == "figure"
    assert item.figure_id == "convlstm_p006_f00"

    # Validate citation
    answer = "The ConvLSTM cell architecture is illustrated in detail [E1]."
    val_res = validator.validate_citations(answer, registry)

    assert val_res.validation_status == "valid"
    assert len(val_res.citations) == 1
    cit = val_res.citations[0]
    assert cit.formatted_citation == "[E1] convlstm_precipitation_nowcasting.pdf, Figure (p. 6)"
