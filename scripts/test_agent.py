"""
CLI Demonstration Script for Phase 8 LangGraph Research Agent Orchestration.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.agent.graph import ResearchAgent


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("==================================================")
    print("PHASE 8 LANGGRAPH RESEARCH AGENT DEMO")
    print("==================================================")

    agent = ResearchAgent()
    print("\nLangGraph Architecture Layout:")
    print(agent.get_graph_ascii())

    test_queries = [
        ("1. TEXT-ONLY QUERY", "What is the main contribution of the ConvLSTM paper?"),
        ("2. TABLE QUERY", "Which model has the lowest Rainfall-MSE score?"),
        ("3. FIGURE QUERY", "What does Figure 3 encoding-forecasting network show?"),
        ("4. CALCULATION QUERY", "What is the percentage improvement from 1.42 to 0.98?"),
        ("5. MULTIMODAL QUERY", "How does the ConvLSTM architecture in Figure 3 relate to the Rainfall-MSE performance reported in Table 5?"),
        ("6. MULTIPLE-PAPER QUERY", "Compare the main methodology across canonical papers in the dataset."),
    ]

    for label, query in test_queries:
        print("\n" + "=" * 55)
        print(f"DEMO SCENARIO: {label}")
        print(f"QUERY: \"{query}\"")

        state = agent.run(query)

        analysis = state.get("query_analysis")
        tools = state.get("selected_tools", [])
        reg = state.get("evidence_registry")
        fused = state.get("fused_context")
        val = state.get("validation_result")
        response = state.get("final_response", "")

        print("\n--- QUERY ANALYSIS ---")
        if analysis:
            print(f"  Requires Text:        {analysis.requires_text}")
            print(f"  Requires Table:       {analysis.requires_table}")
            print(f"  Requires Figure:      {analysis.requires_figure}")
            print(f"  Requires Calculation: {analysis.requires_calculation}")
            print(f"  Requires Multi-Paper: {analysis.requires_multiple_papers}")
            print(f"  Reasoning Summary:    {analysis.reasoning_summary}")

        print("\n--- SELECTED TOOLS ---")
        print(f"  Tools: {tools}")

        print("\n--- EVIDENCE REGISTRY ---")
        if reg:
            items = reg.get_all_items()
            print(f"  Total Evidence Items Registered: {len(items)}")
            for it in items:
                print(f"    |-- [{it.evidence_id}] Type: {it.evidence_type.upper():<11} | Source: {it.source_filename:<38} | Page: {it.page_number:02d}")
        else:
            print("  No evidence items registered.")

        print("\n--- FUSED CONTEXT SUMMARY ---")
        if fused:
            print(f"  Total Evidence Blocks: {fused.evidence_count}")
            print(f"  Modalities Present:    Text={fused.has_text}, Table={fused.has_table}, Figure={fused.has_figure}, Calc={fused.has_calculation}")

        print("\n--- GENERATED ANSWER & VERIFIED CITATIONS ---")
        print(f"{response}")

        print("\n--- CITATION VALIDATION STATUS ---")
        if val:
            print(f"  Status: {val.validation_status.upper()}")
            for cit in val.citations:
                print(f"    Validated: {cit.formatted_citation}")

    print("\n==================================================")
    print("PHASE 8 LANGGRAPH RESEARCH AGENT DEMO COMPLETE")
    print("==================================================\n")


if __name__ == "__main__":
    main()
