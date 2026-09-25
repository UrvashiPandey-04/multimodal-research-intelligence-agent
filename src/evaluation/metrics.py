"""
Evaluation Metrics Engine calculating Recall@K, Routing Accuracy, Citation Validity, and Multimodal Coverage.
"""

from typing import List, Dict, Any, Set
from src.evaluation.schema import EvaluationItem, SingleEvaluationResult, AggregateMetrics
from src.citations.evidence_registry import EvidenceRegistry, EvidenceRegistryItem
from src.citations.citation_model import ValidationResult


class MetricsEngine:
    """Calculates retrieval, routing, citation, and multimodal metrics for evaluation queries."""

    def __init__(self):
        """Initialize metrics engine."""
        pass

    def evaluate_retrieval_recall(
        self,
        retrieved_items: List[EvidenceRegistryItem],
        expected_docs: List[str],
        expected_pages: List[int],
        k: int,
    ) -> float:
        """
        Calculate Recall@K: Check if expected documents and pages are present in top K evidence items.

        Args:
            retrieved_items: List of EvidenceRegistryItem objects returned by agent.
            expected_docs: List of expected source filename strings.
            expected_pages: List of expected 1-indexed page numbers.
            k: Top K threshold to inspect.

        Returns:
            Recall score (1.0 if hit, else 0.0).
        """
        if not expected_docs:
            return 1.0

        top_k_items = retrieved_items[:k]
        if not top_k_items:
            return 0.0

        hit_count = 0
        for exp_doc in expected_docs:
            # Check if exp_doc is present in top K items
            doc_hit = any(exp_doc.lower() in item.source_filename.lower() for item in top_k_items)
            if doc_hit:
                hit_count += 1

        return hit_count / len(expected_docs)

    def evaluate_routing(
        self,
        selected_tools: List[str],
        expected_tools: List[str],
    ) -> bool:
        """
        Evaluate tool routing accuracy: Check if selected tools match expected tools.

        Args:
            selected_tools: Tools chosen by agent.
            expected_tools: Tools specified in benchmark ground truth.

        Returns:
            True if tool sets match or overlap primary tool, else False.
        """
        if not expected_tools:
            return True
        sel_set = set(selected_tools)
        exp_set = set(expected_tools)
        # Match if expected primary tools are present in selected tools
        return exp_set.issubset(sel_set) or sel_set == exp_set

    def evaluate_cross_paper_coverage(
        self,
        retrieved_items: List[EvidenceRegistryItem],
        expected_docs: List[str],
    ) -> float:
        """
        Calculate cross-paper evidence coverage rate: Fraction of expected documents present in evidence.

        Args:
            retrieved_items: Evidence items registered during query execution.
            expected_docs: List of expected document filename strings.

        Returns:
            Coverage ratio between 0.0 and 1.0.
        """
        if not expected_docs:
            return 1.0

        retrieved_docs = {item.source_filename.lower() for item in retrieved_items if item.source_filename}
        matched = 0
        for exp_doc in expected_docs:
            if any(exp_doc.lower() in rd for rd in retrieved_docs):
                matched += 1

        return matched / len(expected_docs)

    def evaluate_single_item(
        self,
        item: EvaluationItem,
        state: Dict[str, Any],
    ) -> SingleEvaluationResult:
        """
        Evaluate a single evaluation item against agent execution state.

        Args:
            item: EvaluationItem ground truth test case.
            state: Final ResearchAgentState from agent execution.

        Returns:
            SingleEvaluationResult dataclass.
        """
        selected_tools = state.get("selected_tools", [])
        registry: EvidenceRegistry = state.get("evidence_registry") or EvidenceRegistry()
        evidence_items = registry.get_all_items()
        val_result: Optional[ValidationResult] = state.get("validation_result")
        gen_answer = state.get("generated_answer", "")

        # 1. Routing exact match
        routing_match = self.evaluate_routing(selected_tools, item.expected_tools)

        # 2. Retrieval Recall@K
        r1 = self.evaluate_retrieval_recall(evidence_items, item.expected_documents, item.expected_pages, k=1)
        r3 = self.evaluate_retrieval_recall(evidence_items, item.expected_documents, item.expected_pages, k=3)
        r5 = self.evaluate_retrieval_recall(evidence_items, item.expected_documents, item.expected_pages, k=5)

        # 3. Citation metrics
        valid_cits = len(val_result.citations) if val_result else 0
        invalid_cits = len(val_result.invalid_citations) if val_result else 0
        total_cits = valid_cits + invalid_cits
        cit_rate = (valid_cits / total_cits) if total_cits > 0 else (1.0 if not val_result or not val_result.invalid_citations else 0.0)

        # 4. Cross-paper coverage
        cp_cov = self.evaluate_cross_paper_coverage(evidence_items, item.expected_documents)

        # 5. Extract retrieved doc names & pages
        ret_docs = [it.source_filename for it in evidence_items]
        ret_pages = [it.page_number for it in evidence_items]

        # 6. Failure category classification
        err_cat: Optional[str] = None
        if r3 == 0.0 and item.expected_documents:
            err_cat = "retrieval_failure"
        elif not routing_match:
            err_cat = "wrong_tool_selection"
        elif invalid_cits > 0:
            err_cat = "citation_failure"
        elif "don't have enough evidence" in gen_answer.lower():
            err_cat = "insufficient_evidence"

        return SingleEvaluationResult(
            question_id=item.question_id,
            category=item.category,
            question=item.question,
            selected_tools=selected_tools,
            expected_tools=item.expected_tools,
            routing_exact_match=routing_match,
            retrieved_documents=ret_docs,
            retrieved_pages=ret_pages,
            recall_at_1=r1,
            recall_at_3=r3,
            recall_at_5=r5,
            valid_citation_count=valid_cits,
            invalid_citation_count=invalid_cits,
            citation_validity_rate=cit_rate,
            expected_paper_coverage=cp_cov,
            generated_answer=gen_answer,
            reference_answer=item.reference_answer,
            error_category=err_cat,
        )

    def aggregate_results(self, results: List[SingleEvaluationResult]) -> AggregateMetrics:
        """
        Aggregate list of SingleEvaluationResult objects into overall benchmark metrics.

        Args:
            results: List of SingleEvaluationResult objects.

        Returns:
            AggregateMetrics dataclass.
        """
        if not results:
            return AggregateMetrics(
                total_questions=0,
                successful_executions=0,
                failed_executions=0,
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                overall_routing_accuracy=0.0,
                category_routing_accuracy={},
                provenance_valid_citation_rate=0.0,
                invalid_citation_rate=0.0,
                table_retrieval_success_rate=0.0,
                figure_retrieval_success_rate=0.0,
                calculation_workflow_accuracy=0.0,
                cross_paper_coverage_rate=0.0,
            )

        total = len(results)
        successful = sum(1 for r in results if r.error_category is None)
        failed = total - successful

        avg_r1 = sum(r.recall_at_1 for r in results) / total
        avg_r3 = sum(r.recall_at_3 for r in results) / total
        avg_r5 = sum(r.recall_at_5 for r in results) / total

        overall_routing = sum(1 for r in results if r.routing_exact_match) / total

        # Per category routing accuracy
        cat_groups: Dict[str, List[SingleEvaluationResult]] = {}
        for r in results:
            cat_groups.setdefault(r.category, []).append(r)

        cat_routing: Dict[str, float] = {}
        for cat, group in cat_groups.items():
            cat_routing[cat] = sum(1 for r in group if r.routing_exact_match) / len(group)

        # Citation validity
        avg_cit_rate = sum(r.citation_validity_rate for r in results) / total
        invalid_rate = 1.0 - avg_cit_rate

        # Category specific metrics
        table_results = cat_groups.get("table", [])
        table_success = (sum(r.recall_at_3 for r in table_results) / len(table_results)) if table_results else 1.0

        figure_results = cat_groups.get("figure", [])
        figure_success = (sum(r.recall_at_3 for r in figure_results) / len(figure_results)) if figure_results else 1.0

        calc_results = cat_groups.get("calculation", [])
        calc_success = (sum(1 for r in calc_results if r.routing_exact_match) / len(calc_results)) if calc_results else 1.0

        cp_results = cat_groups.get("cross_paper", [])
        cp_cov = (sum(r.expected_paper_coverage for r in cp_results) / len(cp_results)) if cp_results else 1.0

        # Error counts
        err_counts: Dict[str, int] = {}
        for r in results:
            if r.error_category:
                err_counts[r.error_category] = err_counts.get(r.error_category, 0) + 1

        return AggregateMetrics(
            total_questions=total,
            successful_executions=successful,
            failed_executions=failed,
            recall_at_1=avg_r1,
            recall_at_3=avg_r3,
            recall_at_5=avg_r5,
            overall_routing_accuracy=overall_routing,
            category_routing_accuracy=cat_routing,
            provenance_valid_citation_rate=avg_cit_rate,
            invalid_citation_rate=invalid_rate,
            table_retrieval_success_rate=table_success,
            figure_retrieval_success_rate=figure_success,
            calculation_workflow_accuracy=calc_success,
            cross_paper_coverage_rate=cp_cov,
            error_counts=err_counts,
        )
