"""
Typed Graph State definition for LangGraph research agent workflow.
"""

from typing import List, Dict, Any, Optional, TypedDict
from src.agent.query_analyzer import QueryAnalysis
from src.agent.context_fuser import FusedContext
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_model import ValidationResult


class ResearchAgentState(TypedDict, total=False):
    """
    Typed graph state passed through LangGraph research agent nodes.
    """
    original_query: str
    query_analysis: Optional[QueryAnalysis]
    selected_tools: List[str]
    tool_results: Dict[str, Any]
    evidence_registry: EvidenceRegistry
    fused_context: Optional[FusedContext]
    generated_answer: str
    validation_result: Optional[ValidationResult]
    final_response: str
    errors: List[str]
    warnings: List[str]
