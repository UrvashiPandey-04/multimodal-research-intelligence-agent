"""
Unit tests for Gemini API quota (HTTP 429 / RESOURCE_EXHAUSTED) error handling,
fail-fast retry logic, evidence preservation, and UI alert formatting.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.models.llm import (
    GeminiLLM,
    GeminiAPIError,
    GeminiQuotaExceededError,
    is_quota_error,
    is_daily_quota_error,
)
from src.agent.nodes import ResearchAgentNodes
from src.generation.response_generator import ResponseGenerator
from src.api import answer_query, ResearchResponse
from src.models.vlm_analyzer import VLMFigureAnalyzer, FigureAnalysisResult
from src.ingestion.figure_parser import ExtractedFigure
from src.retrieval.vector_store import SearchResult


def test_quota_error_detection_helpers():
    """Test helper functions for detecting 429 / RESOURCE_EXHAUSTED / quota error patterns."""
    err_429 = Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota, option: check plan.")
    err_daily = Exception("Quota exceeded for quota metric 'Generate Content API requests' and limit 'Requests per day'.")
    err_503 = Exception("503 Service Unavailable: Server overloaded.")
    err_generic = Exception("ValueError: Invalid argument")

    assert is_quota_error(err_429) is True
    assert is_quota_error(err_daily) is True
    assert is_quota_error(err_503) is False
    assert is_quota_error(err_generic) is False

    assert is_daily_quota_error(err_daily) is True
    assert is_daily_quota_error(err_429) is True
    assert is_daily_quota_error(err_503) is False


def test_gemini_llm_fails_fast_on_daily_quota_error():
    """Verify GeminiLLM.generate_text fails fast without retrying 3 times on daily quota error."""
    llm = GeminiLLM(api_key="mock_key")
    llm.client = MagicMock()
    
    quota_exception = Exception("429 RESOURCE_EXHAUSTED: You exceeded your current quota per day.")
    llm.client.models.generate_content.side_effect = quota_exception

    with pytest.raises(GeminiQuotaExceededError) as exc_info:
        llm.generate_text("Test prompt")

    # Should raise user-friendly message
    assert "Gemini API quota has been reached" in str(exc_info.value)
    # Crucial Requirement 4: Should fail fast (only 1 attempt, no 3x retries)
    assert llm.client.models.generate_content.call_count == 1


def test_answer_generation_node_handles_quota_error():
    """Verify answer_generation_node catches GeminiQuotaExceededError and returns user message without raw error dump."""
    nodes = ResearchAgentNodes()
    nodes.llm = MagicMock()
    nodes.llm.is_configured.return_value = True
    nodes.llm.generate_text.side_effect = GeminiQuotaExceededError(
        "Gemini API quota has been reached. Answer generation is temporarily unavailable."
    )

    fused_mock = MagicMock()
    fused_mock.evidence_count = 3
    fused_mock.fused_text = "[E1 TEXT EVIDENCE] Sample evidence"
    state = {
        "original_query": "What is ConvLSTM?",
        "fused_context": fused_mock,
        "errors": [],
    }

    res = nodes.answer_generation_node(state)
    answer = res.get("generated_answer", "")

    assert "Gemini API quota has been reached" in answer
    assert "RESOURCE_EXHAUSTED" not in answer
    assert "call failed" not in answer
    assert res.get("errors") == ["Gemini API quota has been reached. Answer generation is temporarily unavailable."]


def test_answer_query_preserves_evidence_on_quota_error():
    """Verify answer_query returns quota_exceeded validation status AND retains retrieved evidence."""
    mock_agent = MagicMock()

    # Synthetic evidence items
    chunk = SearchResult(
        chunk_id="chunk_1",
        doc_id="doc1",
        source_filename="convlstm.pdf",
        page_number=1,
        chunk_index=0,
        text="ConvLSTM handles spatial-temporal correlations.",
        extraction_method="native",
        score=0.92,
        distance=0.08,
        metadata={},
    )

    from src.citations.evidence_registry import EvidenceRegistry
    registry = EvidenceRegistry()
    registry.register_chunk(chunk)

    mock_state = {
        "original_query": "What is ConvLSTM?",
        "generated_answer": "Gemini API quota has been reached. Answer generation is temporarily unavailable.",
        "selected_tools": ["text_retriever"],
        "evidence_registry": registry,
        "citations": [],
        "errors": ["Gemini API quota has been reached. Answer generation is temporarily unavailable."],
        "warnings": [],
    }
    mock_agent.run.return_value = mock_state

    response: ResearchResponse = answer_query("What is ConvLSTM?", agent=mock_agent)

    assert response.validation_status == "quota_exceeded"
    assert "Gemini API quota has been reached" in response.answer
    # Crucial Requirement 3: Retrieved evidence must be preserved and returned!
    assert len(response.evidence) == 1
    assert response.evidence[0]["source_filename"] == "convlstm.pdf"
    assert response.evidence[0]["text_snippet"] == "ConvLSTM handles spatial-temporal correlations."


def test_vlm_analyzer_handles_quota_error():
    """Verify VLMFigureAnalyzer handles quota error without leaking raw API dumps into uncertainties."""
    llm_mock = MagicMock()
    llm_mock.is_configured.return_value = True
    llm_mock.client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")

    analyzer = VLMFigureAnalyzer(llm_client=llm_mock)
    fig = ExtractedFigure(
        figure_id="fig_1",
        doc_id="convlstm",
        source_filename="convlstm.pdf",
        page_number=3,
        figure_index=0,
        image_path="./tests/fixtures/sample.png",
        caption="Sample figure caption",
        extraction_method="pymupdf_cropped",
    )

    from unittest.mock import mock_open
    with patch("os.path.exists", return_value=True), patch("builtins.open", mock_open(read_data=b"fake_png_data")):
        result: FigureAnalysisResult = analyzer.analyze_figure(fig, force_refresh=True)

    assert result.figure_type == "unknown"
    assert len(result.uncertainties) == 1
    assert "Gemini API quota has been reached" in result.uncertainties[0]
    assert "RESOURCE_EXHAUSTED" not in result.uncertainties[0]


def test_ordinary_successful_response_unaffected():
    """Verify ordinary successful response generation works normally."""
    mock_agent = MagicMock()
    from src.citations.evidence_registry import EvidenceRegistry
    registry = EvidenceRegistry()

    mock_state = {
        "original_query": "What is FC-LSTM?",
        "generated_answer": "FC-LSTM is a fully connected LSTM model [E1].",
        "selected_tools": ["text_retriever"],
        "evidence_registry": registry,
        "citations": [{"evidence_id": "E1", "formatted_citation": "[convlstm.pdf, p. 1]"}],
        "errors": [],
        "warnings": [],
    }
    mock_agent.run.return_value = mock_state

    response: ResearchResponse = answer_query("What is FC-LSTM?", agent=mock_agent)

    assert response.validation_status == "valid"
    assert response.answer == "FC-LSTM is a fully connected LSTM model [E1]."
    assert len(response.citations) == 1


def test_end_to_end_agent_quota_handling_with_mocked_gemini_client():
    """Regression test tracing a real ResearchAgent instance with mocked Gemini client 429 response."""
    from src.agent.graph import ResearchAgent
    agent = ResearchAgent()

    # Mock the LLM client generate_content method to raise 429 RESOURCE_EXHAUSTED
    agent.nodes.llm.is_configured = MagicMock(return_value=True)
    agent.nodes.llm.client = MagicMock()
    agent.nodes.llm.client.models.generate_content.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: You exceeded your current quota per day. Available evidence: 6 items."
    )

    response: ResearchResponse = answer_query("What main research problem does the ConvLSTM paper address?", agent=agent)

    # 1. Answer is sanitized
    assert response.answer == "Gemini API quota has been reached. Answer generation is temporarily unavailable."
    assert "RESOURCE_EXHAUSTED" not in response.answer
    assert "429" not in response.answer
    assert "Available evidence: 6 items" not in response.answer

    # 2. validation_status is quota_exceeded
    assert response.validation_status == "quota_exceeded"

    # 3. Retrieved evidence is preserved
    assert isinstance(response.evidence, list)
    assert len(response.evidence) > 0
