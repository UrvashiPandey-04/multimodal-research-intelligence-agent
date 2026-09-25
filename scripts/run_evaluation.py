"""
CLI Evaluation Script running benchmark evaluation over canonical research papers and generating metric reports.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluation.dataset import save_dataset_to_file
from src.evaluation.runner import EvaluationRunner


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("==================================================")
    print("PHASE 9 RESEARCH AGENT EVALUATION BENCHMARK")
    print("==================================================")

    # 1. Ensure evaluation dataset exists
    dataset_file = os.path.join(project_root, "data", "evaluation", "evaluation_dataset.json")
    if not os.path.exists(dataset_file):
        print("Generating 30-question benchmark dataset...")
        save_dataset_to_file(dataset_file)

    results_file = os.path.join(project_root, "reports", "evaluation_results.json")
    summary_file = os.path.join(project_root, "reports", "evaluation_summary.md")

    runner = EvaluationRunner(
        dataset_path=dataset_file,
        results_path=results_file,
        summary_path=summary_file,
    )

    papers_dir = os.path.join(project_root, "data", "papers")
    print(f"Indexing canonical papers in '{papers_dir}'...")
    agent = runner.prepare_environment_and_index_papers(papers_dir=papers_dir)

    print("Starting evaluation execution over 30 benchmark questions...\n")
    aggregate, single_results = runner.run_evaluation(agent=agent)

    print("\n==================================================")
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("==================================================")
    print(f"Total Questions Evaluated:          {aggregate.total_questions}")
    print(f"Successful Executions:              {aggregate.successful_executions}")
    print(f"Failed Executions:                  {aggregate.failed_executions}")
    print(f"Recall@1:                           {aggregate.recall_at_1:.4f}")
    print(f"Recall@3:                           {aggregate.recall_at_3:.4f}")
    print(f"Recall@5:                           {aggregate.recall_at_5:.4f}")
    print(f"Overall Tool Routing Accuracy:      {aggregate.overall_routing_accuracy * 100:.1f}%")
    print(f"Provenace Citation Validity Rate:   {aggregate.provenance_valid_citation_rate * 100:.1f}%")
    print(f"Table Retrieval Success Rate:       {aggregate.table_retrieval_success_rate * 100:.1f}%")
    print(f"Figure Retrieval Success Rate:      {aggregate.figure_retrieval_success_rate * 100:.1f}%")
    print(f"Calculation Workflow Accuracy:      {aggregate.calculation_workflow_accuracy * 100:.1f}%")
    print(f"Cross-Paper Evidence Coverage:      {aggregate.cross_paper_coverage_rate * 100:.1f}%")
    print("--------------------------------------------------")
    print(f"Machine-readable results saved to: {results_file}")
    print(f"Human-readable summary saved to:  {summary_file}")
    print("==================================================\n")


if __name__ == "__main__":
    main()
