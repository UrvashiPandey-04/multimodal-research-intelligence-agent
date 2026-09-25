"""
LangGraph Workflow Graph Builder and Orchestrator for the Research Agent.
"""

from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from src.agent.state import ResearchAgentState
from src.agent.nodes import ResearchAgentNodes


def build_research_agent_graph(nodes: Optional[ResearchAgentNodes] = None):
    """
    Build and compile the deterministic LangGraph workflow graph.

    Args:
        nodes: ResearchAgentNodes instance encapsulating node handlers.

    Returns:
        Compiled LangGraph StateGraph runnable application.
    """
    node_handler = nodes or ResearchAgentNodes()
    workflow = StateGraph(ResearchAgentState)

    # 1. Add Nodes
    workflow.add_node("query_analysis", node_handler.query_analysis_node)
    workflow.add_node("tool_selection", node_handler.tool_selection_node)
    workflow.add_node("tool_execution", node_handler.tool_execution_node)
    workflow.add_node("evidence_collection", node_handler.evidence_collection_node)
    workflow.add_node("context_fusion", node_handler.context_fusion_node)
    workflow.add_node("answer_generation", node_handler.answer_generation_node)
    workflow.add_node("citation_validation", node_handler.citation_validation_node)
    workflow.add_node("final_response", node_handler.final_response_node)

    # 2. Add Linear Edges (START -> Node 1 -> ... -> Node 8 -> END)
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "tool_selection")
    workflow.add_edge("tool_selection", "tool_execution")
    workflow.add_edge("tool_execution", "evidence_collection")
    workflow.add_edge("evidence_collection", "context_fusion")
    workflow.add_edge("context_fusion", "answer_generation")
    workflow.add_edge("answer_generation", "citation_validation")
    workflow.add_edge("citation_validation", "final_response")
    workflow.add_edge("final_response", END)

    # 3. Compile Graph Application
    app = workflow.compile()
    return app


class ResearchAgent:
    """High-level Research Agent executing queries through LangGraph orchestration."""

    def __init__(self, nodes: Optional[ResearchAgentNodes] = None):
        """
        Initialize ResearchAgent.

        Args:
            nodes: ResearchAgentNodes instance.
        """
        self.nodes = nodes or ResearchAgentNodes()
        self.graph = build_research_agent_graph(self.nodes)

    def run(self, query: str) -> ResearchAgentState:
        """
        Execute user query through the LangGraph workflow end-to-end.

        Args:
            query: Natural language query string.

        Returns:
            Final ResearchAgentState dictionary containing answer, citations, and evidence provenance.
        """
        initial_state: ResearchAgentState = {
            "original_query": query,
            "selected_tools": [],
            "tool_results": {},
            "errors": [],
            "warnings": [],
        }

        final_state = self.graph.invoke(initial_state)
        return final_state

    def get_graph_ascii(self) -> str:
        """Return ASCII diagram representation of the LangGraph structure."""
        return (
            "START\n"
            "  │\n"
            "  ▼\n"
            "query_analysis ────────► Analyze query & classify intent\n"
            "  │\n"
            "  ▼\n"
            "tool_selection ────────► Select tools (text, table, figure, calc)\n"
            "  │\n"
            "  ▼\n"
            "tool_execution ────────► Execute tools & collect raw outputs\n"
            "  │\n"
            "  ▼\n"
            "evidence_collection ───► Register outputs into EvidenceRegistry\n"
            "  │\n"
            "  ▼\n"
            "context_fusion ────────► Structure multimodal evidence blocks\n"
            "  │\n"
            "  ▼\n"
            "answer_generation ─────► Synthesize grounded answer via Gemini\n"
            "  │\n"
            "  ▼\n"
            "citation_validation ───► Authoritatively validate citations\n"
            "  │\n"
            "  ▼\n"
            "final_response ────────► Format final response package\n"
            "  │\n"
            "  ▼\n"
            "END"
        )
