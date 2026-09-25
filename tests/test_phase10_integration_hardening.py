"""
Unit and Integration Tests for Phase 10: Backend Integration, Reliability Hardening, Request Isolation, and Error Resilience.
"""

import os
import pytest

from src.config import get_config_summary, GEMINI_MODEL, EMBEDDING_MODEL
from src.api import answer_query, ResearchResponse
from src.agent.graph import ResearchAgent
from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.text_retriever import TextRetriever
from src.citations.evidence_registry import EvidenceRegistry
from src.agent.nodes import ResearchAgentNodes
from src.agent.calculator import SafeCalculator
from src.models.vlm_analyzer import VLMFigureAnalyzer


def test_config_centralization():
    """Test that centralized configuration exports valid, non-machine-specific settings."""
    cfg = get_config_summary()
    assert isinstance(cfg, dict)
    assert "gemini_model" in cfg
    assert "embedding_model" in cfg
    assert cfg["gemini_model"] == GEMINI_MODEL
    assert cfg["embedding_model"] == EMBEDDING_MODEL


def test_api_answer_query_empty_query():
    """Test answer_query handles empty and whitespace query string gracefully."""
    res1 = answer_query("")
    assert isinstance(res1, ResearchResponse)
    assert res1.validation_status == "insufficient_evidence"
    assert "empty" in res1.errors[0]

    res2 = answer_query("   ")
    assert isinstance(res2, ResearchResponse)
    assert res2.validation_status == "insufficient_evidence"


def test_request_isolation():
    """Test that sequential query executions have strictly isolated EvidenceRegistry state and citations."""
    vector_store = VectorStore(in_memory=True)
    text_retriever = TextRetriever(vector_store=vector_store)

    # Ingestion mock
    from src.ingestion.chunker import TextChunk
    chunk1 = TextChunk("c1", "doc1", "paper1.pdf", 1, 0, "ConvLSTM precipitation nowcasting", "native", 4, 30)
    chunk2 = TextChunk("c2", "doc2", "paper2.pdf", 1, 0, "Fluvial flood forecasting DCNN", "native", 4, 30)

    from src.models.embeddings import get_embedding_model
    emb_model = get_embedding_model()
    vecs = emb_model.embed_batch([chunk1.text, chunk2.text])
    vector_store.add_chunks([chunk1, chunk2], vecs)

    nodes = ResearchAgentNodes(text_retriever=text_retriever)
    agent = ResearchAgent(nodes=nodes)

    # Execution 1
    res1 = answer_query("ConvLSTM architecture", agent=agent)
    eids1 = [e["evidence_id"] for e in res1.evidence]

    # Execution 2
    res2 = answer_query("Fluvial flood forecasting", agent=agent)
    eids2 = [e["evidence_id"] for e in res2.evidence]

    # Evidence IDs in execution 2 must restart from E1, not accumulate from execution 1
    assert len(eids1) > 0
    assert len(eids2) > 0
    assert eids2[0] == "E1"


def test_error_resilience_calculator_failure():
    """Test that safe calculator returns error object without throwing uncaught exceptions on bad syntax."""
    calc = SafeCalculator()
    res = calc.evaluate("10 / 0")
    assert res.success is False
    assert "division by zero" in res.error_message.lower()

    res_syntax = calc.evaluate("2 + * 3")
    assert res_syntax.success is False
    assert "syntax" in res_syntax.error_message.lower()


def test_vlm_fallback_on_missing_api_key(monkeypatch):
    """Test that VLMFigureAnalyzer handles missing Gemini API key without crashing."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    from src.models.llm import GeminiLLM
    llm = GeminiLLM()
    vlm = VLMFigureAnalyzer(llm_client=llm)
    from src.ingestion.figure_parser import ExtractedFigure
    fig = ExtractedFigure("fig1", "doc1", "paper1.pdf", 1, 0, "non_existent.png", "Sample caption", "pymupdf_cropped")

    res = vlm.analyze_figure(fig)
    assert len(res.uncertainties) > 0
    assert "GEMINI_API_KEY is not configured" in res.uncertainties[0]
    assert "Sample caption" in res.figure_summary


def test_end_to_end_insufficient_evidence():
    """Test end-to-end flow returns explicit insufficient evidence fallback when no chunks match."""
    v_text = VectorStore(in_memory=True, collection_name="empty_text")

    text_retriever = TextRetriever(vector_store=v_text)
    from src.retrieval.table_retriever import TableRetriever
    from src.retrieval.figure_retriever import FigureRetriever

    table_retriever = TableRetriever(in_memory=True, collection_name="empty_tables")
    figure_retriever = FigureRetriever(in_memory=True)

    nodes = ResearchAgentNodes(
        text_retriever=text_retriever,
        table_retriever=table_retriever,
        figure_retriever=figure_retriever,
    )
    agent = ResearchAgent(nodes=nodes)

    res = answer_query("What is the exact weight of a blue whale in grams according to this paper?", agent=agent)
    assert isinstance(res, ResearchResponse)
    assert res.validation_status == "insufficient_evidence"
    assert "don't have enough evidence" in res.answer.lower()
