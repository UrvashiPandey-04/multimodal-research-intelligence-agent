"""
Data models and schemas for research agent evaluation framework.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class EvaluationItem:
    """Dataclass representing a single benchmark evaluation test case."""
    question_id: str
    question: str
    category: str  # "text", "table", "figure", "cross_paper", "calculation"
    expected_tools: List[str]
    expected_documents: List[str]  # e.g. ["convlstm_precipitation_nowcasting.pdf"]
    expected_pages: List[int]      # Expected 1-indexed page numbers
    reference_answer: str          # Verified reference answer string
    reference_citations: List[str] # Expected citation strings

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation item to dictionary format."""
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "expected_tools": self.expected_tools,
            "expected_documents": self.expected_documents,
            "expected_pages": self.expected_pages,
            "reference_answer": self.reference_answer,
            "reference_citations": self.reference_citations,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationItem":
        """Instantiate EvaluationItem from dictionary representation."""
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            category=data["category"],
            expected_tools=data.get("expected_tools", []),
            expected_documents=data.get("expected_documents", []),
            expected_pages=data.get("expected_pages", []),
            reference_answer=data.get("reference_answer", ""),
            reference_citations=data.get("reference_citations", []),
        )


@dataclass
class SingleEvaluationResult:
    """Dataclass storing evaluation results for a single test question execution."""
    question_id: str
    category: str
    question: str
    selected_tools: List[str]
    expected_tools: List[str]
    routing_exact_match: bool
    retrieved_documents: List[str]
    retrieved_pages: List[int]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    valid_citation_count: int
    invalid_citation_count: int
    citation_validity_rate: float
    expected_paper_coverage: float
    generated_answer: str
    reference_answer: str
    error_category: Optional[str] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert single result to dictionary format."""
        return {
            "question_id": self.question_id,
            "category": self.category,
            "question": self.question,
            "selected_tools": self.selected_tools,
            "expected_tools": self.expected_tools,
            "routing_exact_match": self.routing_exact_match,
            "retrieved_documents": self.retrieved_documents,
            "retrieved_pages": self.retrieved_pages,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "valid_citation_count": self.valid_citation_count,
            "invalid_citation_count": self.invalid_citation_count,
            "citation_validity_rate": round(self.citation_validity_rate, 4),
            "expected_paper_coverage": round(self.expected_paper_coverage, 4),
            "generated_answer": self.generated_answer,
            "reference_answer": self.reference_answer,
            "error_category": self.error_category,
            "notes": self.notes,
        }


@dataclass
class AggregateMetrics:
    """Dataclass storing aggregated benchmark metrics across all questions."""
    total_questions: int
    successful_executions: int
    failed_executions: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    overall_routing_accuracy: float
    category_routing_accuracy: Dict[str, float]
    provenance_valid_citation_rate: float
    invalid_citation_rate: float
    table_retrieval_success_rate: float
    figure_retrieval_success_rate: float
    calculation_workflow_accuracy: float
    cross_paper_coverage_rate: float
    error_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "total_questions": self.total_questions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "recall_at_1": round(self.recall_at_1, 4),
            "recall_at_3": round(self.recall_at_3, 4),
            "recall_at_5": round(self.recall_at_5, 4),
            "overall_routing_accuracy": round(self.overall_routing_accuracy, 4),
            "category_routing_accuracy": {k: round(v, 4) for k, v in self.category_routing_accuracy.items()},
            "provenance_valid_citation_rate": round(self.provenance_valid_citation_rate, 4),
            "invalid_citation_rate": round(self.invalid_citation_rate, 4),
            "table_retrieval_success_rate": round(self.table_retrieval_success_rate, 4),
            "figure_retrieval_success_rate": round(self.figure_retrieval_success_rate, 4),
            "calculation_workflow_accuracy": round(self.calculation_workflow_accuracy, 4),
            "cross_paper_coverage_rate": round(self.cross_paper_coverage_rate, 4),
            "error_counts": self.error_counts,
        }
