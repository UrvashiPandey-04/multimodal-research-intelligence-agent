"""
Evaluation Runner executing benchmark test cases through the LangGraph agent and producing reports.
"""

import os
import json
from typing import List, Dict, Any, Optional
from src.evaluation.schema import EvaluationItem, SingleEvaluationResult, AggregateMetrics
from src.evaluation.dataset import load_dataset_from_file
from src.evaluation.metrics import MetricsEngine
from src.agent.graph import ResearchAgent
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.retrieval.vector_store import VectorStore
from src.models.embeddings import get_embedding_model
from src.retrieval.text_retriever import TextRetriever
from src.ingestion.table_parser import TableParser
from src.retrieval.table_repository import TableRepository
from src.retrieval.table_retriever import TableRetriever
from src.ingestion.figure_parser import PyMuPDFFigureExtractor
from src.retrieval.figure_repository import FigureRepository
from src.retrieval.figure_retriever import FigureRetriever


class EvaluationRunner:
    """Benchmark runner executing the evaluation dataset and generating machine and human readable reports."""

    def __init__(
        self,
        dataset_path: str = "data/evaluation/evaluation_dataset.json",
        results_path: str = "reports/evaluation_results.json",
        summary_path: str = "reports/evaluation_summary.md",
    ):
        """
        Initialize evaluation runner.

        Args:
            dataset_path: Path to evaluation dataset JSON file.
            results_path: Destination path for evaluation results JSON.
            summary_path: Destination path for human-readable summary Markdown.
        """
        self.dataset_path = dataset_path
        self.results_path = results_path
        self.summary_path = summary_path
        self.metrics_engine = MetricsEngine()

    def prepare_environment_and_index_papers(self, papers_dir: str = "data/papers") -> ResearchAgent:
        """
        Index canonical papers into TextRetriever, TableRetriever, and FigureRetriever vector collections.

        Args:
            papers_dir: Directory containing research PDFs.

        Returns:
            Initialized ResearchAgent instance ready for evaluation execution.
        """
        if not os.path.exists(papers_dir):
            raise FileNotFoundError(f"Papers directory not found at {papers_dir}")

        pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in {papers_dir}")

        parser = PDFParser()
        chunker = DocumentChunker()
        vector_store = VectorStore(in_memory=True)
        text_retriever = TextRetriever(vector_store=vector_store)

        table_repo = TableRepository()
        table_retriever = TableRetriever(in_memory=True)

        fig_extractor = PyMuPDFFigureExtractor(output_dir="data/figures")
        fig_repo = FigureRepository()
        figure_retriever = FigureRetriever(in_memory=True)

        for fname in pdf_files:
            fpath = os.path.join(papers_dir, fname)

            # 1. Text parsing & indexing
            norm_doc = parser.parse_pdf(fpath)
            chunks = chunker.chunk_document(norm_doc)

            embedding_model = get_embedding_model()
            chunk_texts = [c.text for c in chunks]
            embeddings = embedding_model.embed_batch(chunk_texts)
            vector_store.add_chunks(chunks, embeddings)

            # 2. Table parsing & indexing
            table_parser = TableParser()
            tables = table_parser.extract_tables_from_pdf(fpath)
            table_repo.add_tables(tables)
            table_retriever.add_tables(tables)

            # 3. Figure parsing & indexing
            figs = fig_extractor.extract_figures(fpath, norm_doc.doc_id)
            fig_repo.add_figures(figs)
            figure_retriever.add_figures(figs)

        from src.agent.nodes import ResearchAgentNodes
        agent_nodes = ResearchAgentNodes(
            text_retriever=text_retriever,
            table_retriever=table_retriever,
            figure_retriever=figure_retriever,
        )
        agent = ResearchAgent(nodes=agent_nodes)
        return agent

    def run_evaluation(self, agent: Optional[ResearchAgent] = None) -> Tuple[AggregateMetrics, List[SingleEvaluationResult]]:
        """
        Execute full benchmark evaluation workflow.

        Args:
            agent: Optional pre-configured ResearchAgent instance.

        Returns:
            Tuple of (AggregateMetrics, List[SingleEvaluationResult]).
        """
        items = load_dataset_from_file(self.dataset_path)
        if agent is None:
            agent = self.prepare_environment_and_index_papers()

        single_results: List[SingleEvaluationResult] = []

        print(f"Running evaluation benchmark over {len(items)} questions...")

        for idx, item in enumerate(items, start=1):
            print(f"[{idx}/{len(items)}] Evaluating {item.question_id} ({item.category}): '{item.question[:45]}...'")
            try:
                state = agent.run(item.question)
                res = self.metrics_engine.evaluate_single_item(item, state)
            except Exception as e:
                res = SingleEvaluationResult(
                    question_id=item.question_id,
                    category=item.category,
                    question=item.question,
                    selected_tools=[],
                    expected_tools=item.expected_tools,
                    routing_exact_match=False,
                    retrieved_documents=[],
                    retrieved_pages=[],
                    recall_at_1=0.0,
                    recall_at_3=0.0,
                    recall_at_5=0.0,
                    valid_citation_count=0,
                    invalid_citation_count=0,
                    citation_validity_rate=0.0,
                    expected_paper_coverage=0.0,
                    generated_answer="",
                    reference_answer=item.reference_answer,
                    error_category="pipeline_exception",
                    notes=str(e),
                )
            single_results.append(res)

        aggregate = self.metrics_engine.aggregate_results(single_results)
        self._write_results_json(aggregate, single_results)
        self._write_summary_markdown(aggregate, single_results)

        return aggregate, single_results

    def _write_results_json(self, aggregate: AggregateMetrics, single_results: List[SingleEvaluationResult]) -> None:
        """Write machine-readable evaluation results JSON file."""
        os.makedirs(os.path.dirname(self.results_path), exist_ok=True)
        payload = {
            "summary": aggregate.to_dict(),
            "results": [r.to_dict() for r in single_results],
        }
        with open(self.results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _write_summary_markdown(self, aggregate: AggregateMetrics, single_results: List[SingleEvaluationResult]) -> None:
        """Write human-readable evaluation summary Markdown report."""
        os.makedirs(os.path.dirname(self.summary_path), exist_ok=True)

        md_lines = [
            "# Research Agent Benchmark Evaluation Summary Report",
            "",
            "## 1. Executive Summary & Aggregate Benchmark Metrics",
            "",
            "| Metric Category | Metric Name | Value |",
            "|---|---|---:|",
            f"| **Benchmark Overview** | Total Benchmark Questions | {aggregate.total_questions} |",
            f"| | Successful Executions | {aggregate.successful_executions} |",
            f"| | Failed Executions | {aggregate.failed_executions} |",
            f"| **Retrieval Quality** | Recall@1 | {aggregate.recall_at_1:.4f} |",
            f"| | Recall@3 | {aggregate.recall_at_3:.4f} |",
            f"| | Recall@5 | {aggregate.recall_at_5:.4f} |",
            f"| **Tool Routing Accuracy** | Overall Tool Selection Match | {aggregate.overall_routing_accuracy * 100:.1f}% |",
            f"| **Citation Provenance** | Verified Citation Validity Rate | {aggregate.provenance_valid_citation_rate * 100:.1f}% |",
            f"| | Invalid Citation Rate | {aggregate.invalid_citation_rate * 100:.1f}% |",
            f"| **Multimodal Retrieval** | Table Retrieval Success Rate | {aggregate.table_retrieval_success_rate * 100:.1f}% |",
            f"| | Figure Retrieval Success Rate | {aggregate.figure_retrieval_success_rate * 100:.1f}% |",
            f"| | Calculation Workflow Accuracy | {aggregate.calculation_workflow_accuracy * 100:.1f}% |",
            f"| **Cross-Paper Retrieval** | Expected-Paper Evidence Coverage | {aggregate.cross_paper_coverage_rate * 100:.1f}% |",
            "",
            "## 2. Per-Category Routing Accuracy",
            "",
            "| Category | Total Questions | Exact Routing Accuracy |",
            "|---|---:|---:|",
        ]

        cat_counts: Dict[str, int] = {}
        for r in single_results:
            cat_counts[r.category] = cat_counts.get(r.category, 0) + 1

        for cat, acc in aggregate.category_routing_accuracy.items():
            cnt = cat_counts.get(cat, 0)
            md_lines.append(f"| `{cat}` | {cnt} | {acc * 100:.1f}% |")

        md_lines.extend([
            "",
            "## 3. Error Analysis & Failure Categorization",
            "",
            "| Error Category | Failure Count | Description & Mitigation |",
            "|---|---:|---|",
        ])

        if not aggregate.error_counts:
            md_lines.append("| `none` | 0 | All 30 evaluation test cases executed cleanly without failures. |")
        else:
            descriptions = {
                "retrieval_failure": "Expected paper/page evidence was not retrieved in top K candidates.",
                "wrong_tool_selection": "Query analyzer selected a different set of tools than benchmark expected.",
                "citation_failure": "Generated response referenced an unknown or unvalidated evidence tag.",
                "insufficient_evidence": "Agent triggered fallback due to low similarity retrieval scores.",
                "pipeline_exception": "System runtime exception occurred during workflow execution.",
            }
            for err_cat, cnt in aggregate.error_counts.items():
                desc = descriptions.get(err_cat, "Unclassified error.")
                md_lines.append(f"| `{err_cat}` | {cnt} | {desc} |")

        md_lines.extend([
            "",
            "## 4. Question-Level Execution Breakdown",
            "",
            "| Question ID | Category | Question | Selected Tools | Recall@3 | Routing Match | Citation Rate |",
            "|---|---|---|---|---:|:---:|---:|",
        ])

        for r in single_results:
            q_snip = (r.question[:40] + "...") if len(r.question) > 40 else r.question
            tools_str = ",".join(r.selected_tools)
            match_str = "YES" if r.routing_exact_match else "NO"
            md_lines.append(
                f"| `{r.question_id}` | `{r.category}` | {q_snip} | `{tools_str}` | {r.recall_at_3:.2f} | {match_str} | {r.citation_validity_rate * 100:.0f}% |"
            )

        md_lines.append("\n---\n*Report generated automatically by Research Agent Evaluation Runner.*")

        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
