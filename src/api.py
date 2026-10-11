"""
Application-Level Single Entry Point for the Multimodal Research Intelligence Agent.
Provides the primary answer_query interface returning structured ResearchResponse objects.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.agent.graph import ResearchAgent
from src.agent.state import ResearchAgentState
from src.citations.evidence_registry import EvidenceRegistry
from src.utils.logger import log_event


@dataclass
class ResearchResponse:
    """Structured application-level response object encapsulating query execution outputs."""
    query: str
    answer: str
    selected_tools: List[str]
    evidence: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    validation_status: str  # "valid", "insufficient_evidence", "error"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert response object to dictionary format."""
        return {
            "query": self.query,
            "answer": self.answer,
            "selected_tools": self.selected_tools,
            "evidence": self.evidence,
            "citations": self.citations,
            "validation_status": self.validation_status,
            "warnings": self.warnings,
            "errors": self.errors,
        }


def answer_query(
    query: str,
    agent: Optional[ResearchAgent] = None,
    collection: Optional[Any] = None,
    paper_filter: Optional[Any] = None,
) -> ResearchResponse:
    """
    Single application-level entry point for answering research paper questions.

    Args:
        query: User natural language research query.
        agent: Optional pre-configured ResearchAgent instance.
        collection: Optional target collection name or Chroma collection object.
        paper_filter: Optional source_filename paper filter (str or list of str).

    Returns:
        Structured ResearchResponse containing grounded answer, selected tools,
        evidence items, verified citations, validation status, warnings, and errors.
    """
    clean_query = str(query).strip() if query else ""

    # 1. Handle empty or whitespace-only query gracefully
    if not clean_query:
        log_event("query_rejected", {"reason": "empty_query"}, level=30)
        return ResearchResponse(
            query=query or "",
            answer="I don't have enough evidence in the provided research papers to answer an empty query.",
            selected_tools=[],
            evidence=[],
            citations=[],
            validation_status="insufficient_evidence",
            warnings=["Input query was empty."],
            errors=["empty_query"],
        )

    log_event("query_received", {"query": clean_query, "paper_filter": paper_filter})

    # 2. Execute ResearchAgent LangGraph workflow
    try:
        agent_instance = agent or ResearchAgent()
        state: ResearchAgentState = agent_instance.run(clean_query, paper_filter=paper_filter)
    except Exception as e:
        log_event("agent_execution_failed", {"error": str(e)}, level=40)
        return ResearchResponse(
            query=clean_query,
            answer="An internal system error occurred while processing your research query.",
            selected_tools=[],
            evidence=[],
            citations=[],
            validation_status="error",
            warnings=[],
            errors=[f"Agent execution exception: {e}"],
        )


    # 3. Extract execution details from state
    selected_tools = state.get("selected_tools", [])
    answer = state.get("generated_answer", "")
    citations_raw = state.get("citations", [])
    errors = list(state.get("errors", []))
    warnings = list(state.get("warnings", []))

    # Extract evidence objects
    registry: Optional[EvidenceRegistry] = state.get("evidence_registry")
    evidence_items: List[Dict[str, Any]] = []
    if registry:
        evidence_items = [item.to_dict() for item in registry.get_all_items()]

    # Extract citation objects
    citations: List[Dict[str, Any]] = []
    if citations_raw:
        for cit in citations_raw:
            if hasattr(cit, "to_dict"):
                citations.append(cit.to_dict())
            elif isinstance(cit, dict):
                citations.append(cit)

    # 4. Determine validation status
    insufficient_phrases = [
        "don't have enough evidence",
        "insufficient evidence",
        "no relevant evidence",
    ]
    if "quota has been reached" in answer.lower() or "quota" in answer.lower():
        validation_status = "quota_exceeded"
    elif any(phrase in answer.lower() for phrase in insufficient_phrases):
        validation_status = "insufficient_evidence"
    elif errors and not answer:
        validation_status = "error"
    else:
        validation_status = "valid"

    log_event(
        "query_completed",
        {
            "query": clean_query,
            "validation_status": validation_status,
            "tool_count": len(selected_tools),
            "evidence_count": len(evidence_items),
            "citation_count": len(citations),
        },
    )

    return ResearchResponse(
        query=clean_query,
        answer=answer,
        selected_tools=selected_tools,
        evidence=evidence_items,
        citations=citations,
        validation_status=validation_status,
        warnings=warnings,
        errors=errors,
    )
