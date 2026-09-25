"""
Unit tests for Phase 9: Cross-paper reasoning, Document diversification, Evaluation dataset, Metrics engine, and Runner.
"""

import os
import tempfile
import pytest

from src.retrieval.text_retriever import TextRetriever
from src.retrieval.vector_store import VectorStore, SearchResult
from src.ingestion.chunker import TextChunk
from src.evaluation.schema import EvaluationItem, SingleEvaluationResult, AggregateMetrics
from src.evaluation.dataset import build_30_question_dataset
from src.evaluation.metrics import MetricsEngine
from src.citations.evidence_registry import EvidenceRegistry
from src.agent.context_fuser import MultimodalContextFuser


def test_document_diversified_retrieval():
    """Test TextRetriever.search_diversified balances top_k chunks across multiple document IDs."""
    vector_store = VectorStore(in_memory=True)
    text_retriever = TextRetriever(vector_store=vector_store)

    chunks = [
        TextChunk(chunk_id="doc1_c1", doc_id="doc1", source_filename="paper1.pdf", page_number=1, chunk_index=0, text="Rainfall forecasting ConvLSTM", extraction_method="native", word_count=5, char_count=30),
        TextChunk(chunk_id="doc1_c2", doc_id="doc1", source_filename="paper1.pdf", page_number=2, chunk_index=1, text="Precipitation model architecture", extraction_method="native", word_count=5, char_count=30),
        TextChunk(chunk_id="doc2_c1", doc_id="doc2", source_filename="paper2.pdf", page_number=1, chunk_index=0, text="Precipitation nowcasting benchmark", extraction_method="native", word_count=5, char_count=30),
    ]

    from src.models.embeddings import get_embedding_model
    emb_model = get_embedding_model()
    embeddings = emb_model.embed_batch([c.text for c in chunks])
    vector_store.add_chunks(chunks, embeddings)

    # Standard search might be dominated by doc1
    div_results = text_retriever.search_diversified("precipitation nowcasting", top_k=2, per_doc_limit=1)
    assert len(div_results) == 2
    doc_ids = {r.doc_id for r in div_results}
    assert len(doc_ids) == 2
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_evaluation_dataset_30_questions_composition():
    """Test that build_30_question_dataset returns exactly 30 valid EvaluationItems matching category targets."""
    dataset = build_30_question_dataset()
    assert len(dataset) == 30

    cat_counts = {}
    for item in dataset:
        assert isinstance(item.question_id, str)
        assert isinstance(item.question, str)
        assert len(item.expected_documents) > 0
        assert len(item.reference_citations) > 0
        cat_counts[item.category] = cat_counts.get(item.category, 0) + 1

    assert cat_counts.get("text", 0) == 10
    assert cat_counts.get("table", 0) == 8
    assert cat_counts.get("figure", 0) == 4
    assert cat_counts.get("cross_paper", 0) == 5
    assert cat_counts.get("calculation", 0) == 3


def test_metrics_engine_recall_and_routing():
    """Test MetricsEngine calculations for Recall@K and tool selection routing accuracy."""
    engine = MetricsEngine()

    item = EvaluationItem(
        question_id="Q01",
        question="What is the main contribution of paper 1?",
        category="text",
        expected_tools=["text_retriever"],
        expected_documents=["paper1.pdf"],
        expected_pages=[1],
        reference_answer="Reference contribution answer.",
        reference_citations=["paper1.pdf, p. 1"],
    )

    # 1. Routing evaluation
    assert engine.evaluate_routing(["text_retriever"], ["text_retriever"]) is True
    assert engine.evaluate_routing(["table_retriever"], ["text_retriever"]) is False

    # 2. Recall@K evaluation
    registry = EvidenceRegistry()
    chunk = SearchResult(
        chunk_id="c1",
        doc_id="doc1",
        source_filename="paper1.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample text content.",
        extraction_method="native",
        score=0.9,
        distance=0.1,
        metadata={},
    )
    registry.register_chunk(chunk)
    items = registry.get_all_items()

    r1 = engine.evaluate_retrieval_recall(items, ["paper1.pdf"], [1], k=1)
    assert r1 == 1.0

    r_miss = engine.evaluate_retrieval_recall(items, ["missing_paper.pdf"], [1], k=1)
    assert r_miss == 0.0


def test_metrics_aggregation():
    """Test aggregating single evaluation results into overall AggregateMetrics."""
    engine = MetricsEngine()

    res1 = SingleEvaluationResult(
        question_id="Q01",
        category="text",
        question="Q1 text",
        selected_tools=["text_retriever"],
        expected_tools=["text_retriever"],
        routing_exact_match=True,
        retrieved_documents=["paper1.pdf"],
        retrieved_pages=[1],
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        valid_citation_count=1,
        invalid_citation_count=0,
        citation_validity_rate=1.0,
        expected_paper_coverage=1.0,
        generated_answer="Answer 1",
        reference_answer="Ref 1",
    )

    res2 = SingleEvaluationResult(
        question_id="Q11",
        category="table",
        question="Q2 table",
        selected_tools=["table_retriever"],
        expected_tools=["table_retriever"],
        routing_exact_match=True,
        retrieved_documents=["paper1.pdf"],
        retrieved_pages=[8],
        recall_at_1=1.0,
        recall_at_3=1.0,
        recall_at_5=1.0,
        valid_citation_count=1,
        invalid_citation_count=0,
        citation_validity_rate=1.0,
        expected_paper_coverage=1.0,
        generated_answer="Answer 2",
        reference_answer="Ref 2",
    )

    aggregate = engine.aggregate_results([res1, res2])

    assert aggregate.total_questions == 2
    assert aggregate.successful_executions == 2
    assert aggregate.overall_routing_accuracy == 1.0
    assert aggregate.recall_at_1 == 1.0
    assert aggregate.provenance_valid_citation_rate == 1.0
    assert aggregate.category_routing_accuracy["text"] == 1.0
    assert aggregate.category_routing_accuracy["table"] == 1.0
