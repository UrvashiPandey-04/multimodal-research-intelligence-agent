"""
Unit tests for Phase 5: Citation Validation and Evidence Grounding.
"""

import pytest
from unittest.mock import MagicMock
from src.retrieval.vector_store import SearchResult, VectorStore
from src.retrieval.text_retriever import TextRetriever
from src.citations.evidence_registry import EvidenceRegistry, EvidenceRegistryItem
from src.citations.citation_model import Citation, ValidationResult
from src.citations.citation_validator import CitationValidator
from src.generation.response_generator import ResponseGenerator, RAGAnswer
from src.models.llm import GeminiLLM


@pytest.fixture
def sample_registry() -> EvidenceRegistry:
    """Fixture returning an EvidenceRegistry populated with 2 canonical search results."""
    reg = EvidenceRegistry()

    c1 = SearchResult(
        chunk_id="chunk_convlstm_p004_c001",
        doc_id="convlstm_paper",
        source_filename="convlstm_precipitation_nowcasting.pdf",
        page_number=4,
        chunk_index=1,
        text="ConvLSTM replaces matrix multiplication with convolution in state transitions.",
        extraction_method="native",
        score=0.89,
        distance=0.22,
        metadata={},
    )
    c2 = SearchResult(
        chunk_id="chunk_dcnn_p008_c000",
        doc_id="dcnn_paper",
        source_filename="dcnn_fluvial_flood.pdf",
        page_number=8,
        chunk_index=0,
        text="Rapid flood inundation prediction model achieves an RMSE of 0.12m.",
        extraction_method="ocr",
        score=0.82,
        distance=0.36,
        metadata={},
    )

    reg.register_chunks([c1, c2])
    return reg


def test_valid_evidence_id_accepted(sample_registry: EvidenceRegistry):
    """1. Test that valid evidence IDs (E1, E2) resolve to authentic citations."""
    validator = CitationValidator()
    answer_text = "The ConvLSTM model handles spatio-temporal sequences [E1]."

    val_result = validator.validate_citations(answer_text, sample_registry)

    assert val_result.validation_status == "valid"
    assert len(val_result.citations) == 1
    assert len(val_result.invalid_citations) == 0

    cit = val_result.citations[0]
    assert cit.evidence_id == "E1"
    assert cit.document_id == "convlstm_paper"
    assert cit.source_filename == "convlstm_precipitation_nowcasting.pdf"
    assert cit.page_number == 4
    assert cit.chunk_id == "chunk_convlstm_p004_c001"
    assert cit.formatted_citation == "[E1] convlstm_precipitation_nowcasting.pdf, p. 4"


def test_unknown_evidence_id_rejected(sample_registry: EvidenceRegistry):
    """2 & 3. Test that unknown evidence IDs (e.g. E99) or IDs from another request are rejected."""
    validator = CitationValidator()
    answer_text = "Model achieves high accuracy [E99]."

    val_result = validator.validate_citations(answer_text, sample_registry)

    assert val_result.validation_status == "failed"
    assert len(val_result.citations) == 0
    assert len(val_result.invalid_citations) == 1

    inv_cit = val_result.invalid_citations[0]
    assert inv_cit.evidence_id == "E99"
    assert inv_cit.validation_status == "invalid"
    assert "Unknown evidence ID 'E99'" in inv_cit.validation_reason


def test_missing_page_number_rejected():
    """4. Test that an evidence item with an invalid page number (<= 0) is marked invalid."""
    reg = EvidenceRegistry()
    invalid_chunk = SearchResult(
        chunk_id="chunk_bad_page",
        doc_id="doc1",
        source_filename="paper.pdf",
        page_number=0,  # Invalid page number
        chunk_index=0,
        text="Sample text",
        extraction_method="native",
        score=0.85,
        distance=0.3,
        metadata={},
    )
    reg.register_chunk(invalid_chunk)

    validator = CitationValidator()
    val_result = validator.validate_citations("Claim statement [E1].", reg)

    assert val_result.validation_status == "failed"
    assert len(val_result.invalid_citations) == 1
    assert "invalid page_number (0)" in val_result.invalid_citations[0].validation_reason


def test_missing_chunk_id_rejected():
    """5. Test that an evidence item with an empty chunk_id is marked invalid."""
    reg = EvidenceRegistry()
    bad_chunk = SearchResult(
        chunk_id="",  # Missing chunk ID
        doc_id="doc1",
        source_filename="paper.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample text",
        extraction_method="native",
        score=0.85,
        distance=0.3,
        metadata={},
    )
    reg.register_chunk(bad_chunk)

    validator = CitationValidator()
    val_result = validator.validate_citations("Claim statement [E1].", reg)

    assert val_result.validation_status == "failed"
    assert "missing chunk_id" in val_result.invalid_citations[0].validation_reason


def test_missing_document_id_rejected():
    """6. Test that an evidence item with missing document_id is marked invalid."""
    reg = EvidenceRegistry()
    bad_chunk = SearchResult(
        chunk_id="chunk123",
        doc_id="",  # Missing doc ID
        source_filename="paper.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample text",
        extraction_method="native",
        score=0.85,
        distance=0.3,
        metadata={},
    )
    reg.register_chunk(bad_chunk)

    validator = CitationValidator()
    val_result = validator.validate_citations("Claim statement [E1].", reg)

    assert val_result.validation_status == "failed"
    assert "missing document_id" in val_result.invalid_citations[0].validation_reason


def test_multiple_valid_citations_preserved(sample_registry: EvidenceRegistry):
    """7 & 9. Test that multiple valid citations across different papers are all preserved."""
    validator = CitationValidator()
    answer_text = "ConvLSTM is evaluated on precipitation data [E1]. The 2D DCNN predicts flood inundation [E2]."

    val_result = validator.validate_citations(answer_text, sample_registry)

    assert val_result.validation_status == "valid"
    assert len(val_result.citations) == 2

    c1 = val_result.citations[0]
    c2 = val_result.citations[1]

    assert c1.source_filename == "convlstm_precipitation_nowcasting.pdf"
    assert c1.page_number == 4
    assert c2.source_filename == "dcnn_fluvial_flood.pdf"
    assert c2.page_number == 8


def test_duplicate_evidence_ids_deduplicated(sample_registry: EvidenceRegistry):
    """8. Test that duplicate evidence references [E1, E1, E2] are deterministically deduplicated."""
    validator = CitationValidator()
    answer_text = "First claim [E1]. Repeating E1 claim [E1]. Second claim [E2]."

    val_result = validator.validate_citations(answer_text, sample_registry)

    assert val_result.validation_status == "valid"
    assert len(val_result.citations) == 2  # E1 and E2 only
    assert [c.evidence_id for c in val_result.citations] == ["E1", "E2"]


def test_llm_invented_page_number_ignored(sample_registry: EvidenceRegistry):
    """10. Test that if the LLM invents a page number in text (e.g. 'Page 99'), authoritative metadata comes from EvidenceRegistry (Page 4)."""
    validator = CitationValidator()
    answer_text = "ConvLSTM model is discussed on Page 99 [E1]."

    val_result = validator.validate_citations(answer_text, sample_registry)

    assert val_result.validation_status == "valid"
    cit = val_result.citations[0]
    # Authoritative page number MUST be 4 from EvidenceRegistry, ignoring 'Page 99' in LLM text
    assert cit.page_number == 4
    assert cit.formatted_citation == "[E1] convlstm_precipitation_nowcasting.pdf, p. 4"


def test_empty_and_malformed_citation_tags(sample_registry: EvidenceRegistry):
    """11 & 12. Test empty answer text and malformed tag extraction."""
    validator = CitationValidator()

    # Empty text
    res_empty = validator.validate_citations("", sample_registry)
    assert res_empty.validation_status == "none"
    assert len(res_empty.citations) == 0

    # Malformed text without bracket tags
    res_no_tags = validator.validate_citations("Statement without any bracket tags.", sample_registry)
    assert res_no_tags.validation_status == "none"

    # Numeric tag [1] mapped to E1
    res_num = validator.validate_citations("Statement with numeric tag [1].", sample_registry)
    assert res_num.validation_status == "valid"
    assert res_num.citations[0].evidence_id == "E1"


def test_full_rag_pipeline_with_citation_validation():
    """13. Integration test for full RAG pipeline resolving citations through EvidenceRegistry."""
    mock_retriever = MagicMock(spec=TextRetriever)
    mock_retriever.search.return_value = [
        SearchResult(
            chunk_id="chunk_test_p003_c000",
            doc_id="test_paper",
            source_filename="test_paper.pdf",
            page_number=3,
            chunk_index=0,
            text="The ConvLSTM model achieves superior precipitation nowcasting performance.",
            extraction_method="native",
            score=0.87,
            distance=0.26,
            metadata={},
        )
    ]

    mock_llm = MagicMock(spec=GeminiLLM)
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.is_configured.return_value = True
    mock_llm.generate_text.return_value = (
        "The ConvLSTM model delivers state-of-the-art nowcasting performance [E1]."
    )

    generator = ResponseGenerator(retriever=mock_retriever, llm=mock_llm)
    rag_answer = generator.generate_answer("What performance does ConvLSTM achieve?")

    assert isinstance(rag_answer, RAGAnswer)
    assert rag_answer.validation_status == "valid"
    assert len(rag_answer.citations) == 1

    cit = rag_answer.citations[0]
    assert cit["evidence_id"] == "E1"
    assert cit["source_filename"] == "test_paper.pdf"
    assert cit["page_number"] == 3
    assert cit["chunk_id"] == "chunk_test_p003_c000"
    assert cit["formatted_citation"] == "[E1] test_paper.pdf, p. 3"
