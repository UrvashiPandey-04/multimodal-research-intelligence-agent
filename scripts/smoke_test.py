"""
Lightweight End-to-End Smoke Test verifying full agent integration across all query modalities.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.api import answer_query, ResearchResponse
from src.evaluation.runner import EvaluationRunner


def run_smoke_test():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("==================================================")
    print("END-TO-END SYSTEM SMOKE TEST")
    print("==================================================")

    # 1. Environment & Paper Indexing
    papers_dir = os.path.join(project_root, "data", "papers")
    print(f"Step 1: Loading & indexing canonical papers from '{papers_dir}'...")
    
    runner = EvaluationRunner()
    agent = runner.prepare_environment_and_index_papers(papers_dir=papers_dir)
    print("   ✓ Papers parsed and vector collections populated successfully.\n")

    # 2. Test Cases across 4 core modalities
    test_queries = [
        ("Text Query", "What main research problem does the ConvLSTM paper address?"),
        ("Table Query", "Which model achieves the lowest Rainfall-MSE score in Table 1?"),
        ("Figure Query", "What does Figure 1 illustrate regarding 2D image transformations over time?"),
        ("Calculation Query", "What is the percentage improvement from Rainfall-MSE 0.120 to 0.089?"),
        ("Insufficient Evidence Query", "What color is the author's cat in the ConvLSTM paper?"),
    ]

    success_count = 0

    for idx, (label, query) in enumerate(test_queries, start=1):
        print(f"Step {idx+1}: Executing {label} -> '{query}'")
        res: ResearchResponse = answer_query(query, agent=agent)

        print(f"   Selected Tools:    {res.selected_tools}")
        print(f"   Validation Status: {res.validation_status}")
        print(f"   Evidence Count:    {len(res.evidence)}")
        print(f"   Citation Count:    {len(res.citations)}")
        answer_snippet = (res.answer[:80] + "...") if len(res.answer) > 80 else res.answer
        print(f"   Answer:            {answer_snippet}")

        # Basic assertion validation
        if res.validation_status in ("valid", "insufficient_evidence"):
            print("   ✓ Status verified.\n")
            success_count += 1
        else:
            print(f"   ✗ Failure status: {res.errors}\n")

    print("==================================================")
    print(f"SMOKE TEST COMPLETED: {success_count}/{len(test_queries)} TESTS PASSED")
    print("==================================================")

    if success_count == len(test_queries):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_smoke_test()
