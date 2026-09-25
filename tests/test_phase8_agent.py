"""
Unit tests for Phase 8: LangGraph Research Agent orchestration, Query Analysis, Safe Calculator, Context Fusion, and End-to-End Graph Execution.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.agent.calculator import SafeCalculator, CalculationResult
from src.agent.query_analyzer import QueryAnalyzer, QueryAnalysis
from src.agent.context_fuser import MultimodalContextFuser, FusedContext
from src.agent.nodes import ResearchAgentNodes
from src.agent.graph import ResearchAgent, build_research_agent_graph
from src.citations.evidence_registry import EvidenceRegistry
from src.retrieval.vector_store import SearchResult
from src.retrieval.table_retriever import TableSearchResult
from src.retrieval.figure_retriever import FigureSearchResult


def test_safe_calculator_valid_expressions():
    """Test SafeCalculator AST parsing for valid mathematical expressions and percentage operations."""
    calc = SafeCalculator()

    # 1. Basic arithmetic
    res1 = calc.evaluate("1.42 - 0.98")
    assert res1.success is True
    assert round(res1.result, 2) == 0.44

    # 2. Complex formula with parentheses
    res2 = calc.evaluate("((1.42 - 0.98) / 1.42) * 100")
    assert res2.success is True
    assert round(res2.result, 1) == 31.0

    # 3. Percentage phrase
    res3 = calc.evaluate("25% of 200")
    assert res3.success is True
    assert res3.result == 50.0

    # 4. Division by zero safety
    res4 = calc.evaluate("10 / 0")
    assert res4.success is False
    assert "Division by zero" in res4.error_message


def test_query_analyzer_intent_classification():
    """Test QueryAnalyzer deterministic tool selection for various question modalities."""
    analyzer = QueryAnalyzer()

    # 1. Text query
    qa_text = analyzer.analyze("What is the main contribution of the paper?")
    assert qa_text.requires_text is True
    assert qa_text.requires_table is False
    assert qa_text.requires_figure is False
    assert "text_retriever" in qa_text.get_selected_tools()

    # 2. Table query
    qa_table = analyzer.analyze("Which model has the lowest Rainfall-MSE score?")
    assert qa_table.requires_table is True
    assert "table_retriever" in qa_table.get_selected_tools()

    # 3. Figure query
    qa_fig = analyzer.analyze("What does Figure 3 architecture diagram show?")
    assert qa_fig.requires_figure is True
    assert "figure_retriever" in qa_fig.get_selected_tools()
    assert "figure_vlm" in qa_fig.get_selected_tools()

    # 4. Calculation query
    qa_calc = analyzer.analyze("What is the percentage improvement from 1.42 to 0.98?")
    assert qa_calc.requires_calculation is True
    assert qa_calc.calc_expression is not None
    assert "calculator" in qa_calc.get_selected_tools()

    # 5. Multimodal query
    qa_multi = analyzer.analyze("How does the architecture in Figure 2 differ from the method in Paper B?")
    assert qa_multi.requires_figure is True
    assert qa_multi.requires_multiple_papers is True
    assert qa_multi.requires_multimodal_reasoning is True


def test_context_fuser_multimodal_formatting():
    """Test MultimodalContextFuser formatting text, table, figure, and calculation evidence."""
    fuser = MultimodalContextFuser()
    registry = EvidenceRegistry()

    # Register text chunk
    chunk = SearchResult(
        chunk_id="chunk_1",
        doc_id="doc1",
        source_filename="paper1.pdf",
        page_number=2,
        chunk_index=0,
        text="ConvLSTM uses convolutional structures in transitions.",
        extraction_method="native",
        score=0.9,
        distance=0.1,
        metadata={},
    )
    registry.register_chunk(chunk)

    # Register calculation
    registry.register_calculation("(1.42 - 0.98) / 1.42 * 100", "31%", 31.0)

    fused = fuser.fuse_context(registry)
    assert fused.evidence_count == 2
    assert fused.has_text is True
    assert fused.has_calculation is True
    assert "[E1 TEXT EVIDENCE]" in fused.fused_text
    assert "[E2 CALCULATION EVIDENCE]" in fused.fused_text


def test_langgraph_nodes_execution():
    """Test executing individual ResearchAgentNodes in sequence."""
    nodes = ResearchAgentNodes()

    state = {"original_query": "Which model has the lowest Rainfall-MSE?", "errors": [], "warnings": []}

    # 1. Query Analysis
    res1 = nodes.query_analysis_node(state)
    state.update(res1)
    assert state["query_analysis"].requires_table is True

    # 2. Tool Selection
    res2 = nodes.tool_selection_node(state)
    state.update(res2)
    assert "table_retriever" in state["selected_tools"]

    # 3. Context Fusion with empty registry
    res5 = nodes.context_fusion_node(state)
    state.update(res5)

    # 4. Answer Generation (Insufficient evidence fallback)
    res6 = nodes.answer_generation_node(state)
    state.update(res6)
    assert "don't have enough evidence" in state["generated_answer"]


def test_research_agent_end_to_end_execution():
    """Test running full LangGraph ResearchAgent workflow on synthetic state."""
    agent = ResearchAgent()
    final_state = agent.run("What is the percentage improvement from 100 to 50?")

    assert final_state["original_query"] == "What is the percentage improvement from 100 to 50?"
    assert final_state["query_analysis"].requires_calculation is True
    assert "calculator" in final_state["selected_tools"]
    assert final_state["fused_context"].evidence_count > 0
    assert final_state["validation_result"].validation_status in ["valid", "none"]
    assert len(final_state["final_response"]) > 0
