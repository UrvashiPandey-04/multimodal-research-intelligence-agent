"""
Query Analysis module for classifying research queries and determining required tools.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class QueryAnalysis:
    """Dataclass representing the analyzed intent and tool requirements of a query."""
    query: str
    requires_text: bool = True
    requires_table: bool = False
    requires_figure: bool = False
    requires_calculation: bool = False
    requires_multiple_papers: bool = False
    requires_multimodal_reasoning: bool = False
    calc_expression: Optional[str] = None
    reasoning_summary: str = ""

    def get_selected_tools(self) -> List[str]:
        """Return list of explicit tool names required based on query analysis."""
        tools: List[str] = []
        if self.requires_text:
            tools.append("text_retriever")
        if self.requires_table:
            tools.append("table_retriever")
        if self.requires_figure:
            tools.append("figure_retriever")
            tools.append("figure_vlm")
        if self.requires_calculation:
            tools.append("calculator")
        return tools

    def to_dict(self) -> Dict[str, Any]:
        """Convert query analysis to dictionary representation."""
        return {
            "query": self.query,
            "requires_text": self.requires_text,
            "requires_table": self.requires_table,
            "requires_figure": self.requires_figure,
            "requires_calculation": self.requires_calculation,
            "requires_multiple_papers": self.requires_multiple_papers,
            "requires_multimodal_reasoning": self.requires_multimodal_reasoning,
            "calc_expression": self.calc_expression,
            "selected_tools": self.get_selected_tools(),
            "reasoning_summary": self.reasoning_summary,
        }


class QueryAnalyzer:
    """Deterministic Query Analyzer identifying tool requirements and multi-paper intents."""

    def __init__(self):
        """Initialize query analyzer."""
        self.table_keywords = re.compile(
            r"\b(table|mse|csi|far|pod|rmse|r2|metric|score|benchmark|row|column|parameters|optical flow|rover)\b",
            re.IGNORECASE,
        )
        self.figure_keywords = re.compile(
            r"\b(figure|fig|fig\.|diagram|architecture|flowchart|plot|heatmap|loss curve|structure|look like|schematic)\b",
            re.IGNORECASE,
        )
        self.calc_keywords = re.compile(
            r"\b(percentage|percent|improvement|difference|calculate|change from|increase|decrease|ratio)\b",
            re.IGNORECASE,
        )
        self.multi_paper_keywords = re.compile(
            r"\b(papers|both papers|all papers|compare|across papers|paper a|paper b|differ from the method|between)\b",
            re.IGNORECASE,
        )
        # Regex to extract numbers e.g. "from 1.42 to 0.98" -> expression "(1.42 - 0.98) / 1.42 * 100"
        self.two_number_pattern = re.compile(
            r"(?:from|between)?\s*(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )

    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze user query string and return typed QueryAnalysis object.

        Args:
            query: Natural language query string.

        Returns:
            QueryAnalysis object specifying tool flags and selected tools.
        """
        q_clean = query.strip()
        q_lower = q_clean.lower()

        is_table = bool(self.table_keywords.search(q_lower))
        is_figure = bool(self.figure_keywords.search(q_lower))
        is_calc = bool(self.calc_keywords.search(q_lower))
        is_multi_paper = bool(self.multi_paper_keywords.search(q_lower))

        calc_expr: Optional[str] = None

        # Extract explicit math or percentage expression if calculation requested
        if is_calc or "%" in q_clean or any(op in q_clean for op in ["+", "-", "*", "/"]):
            num_match = self.two_number_pattern.search(q_clean)
            if num_match and ("percentage" in q_lower or "improvement" in q_lower or "change" in q_lower):
                val1 = float(num_match.group(1))
                val2 = float(num_match.group(2))
                if val1 > 0:
                    calc_expr = f"(({val1} - {val2}) / {val1}) * 100"
                    is_calc = True
            elif num_match and ("difference" in q_lower or "subtract" in q_lower):
                val1 = float(num_match.group(1))
                val2 = float(num_match.group(2))
                calc_expr = f"{val1} - {val2}"
                is_calc = True

        # Check for direct mathematical formula string in query e.g. "1.42 - 0.98"
        direct_math = re.search(r"(\d+(?:\.\d+)?\s*[\+\-\*/]\s*\d+(?:\.\d+)?)", q_clean)
        if direct_math and not calc_expr:
            calc_expr = direct_math.group(1)
            is_calc = True

        # Multimodal reasoning flag
        is_multimodal = (is_figure and (is_table or is_multi_paper)) or (is_table and is_calc)

        # Text retrieval is enabled unless query is purely a direct math evaluation
        is_text = True
        if is_calc and not is_table and not is_figure and not is_multi_paper and calc_expr:
            # Pure math expression like "1.42 - 0.98"
            if len(q_clean.split()) <= 6 and not any(k in q_lower for k in ["paper", "model", "convlstm", "what", "how", "why"]):
                is_text = False

        reasoning_parts = []
        if is_table:
            reasoning_parts.append("table metrics/parameters detected")
        if is_figure:
            reasoning_parts.append("figure/diagram keywords detected")
        if is_calc:
            reasoning_parts.append(f"numerical calculation requested (expr: {calc_expr})")
        if is_multi_paper:
            reasoning_parts.append("cross-paper comparison intent detected")
        if is_multimodal:
            reasoning_parts.append("multimodal reasoning across modalities required")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "standard text RAG retrieval"

        return QueryAnalysis(
            query=q_clean,
            requires_text=is_text,
            requires_table=is_table,
            requires_figure=is_figure,
            requires_calculation=is_calc,
            requires_multiple_papers=is_multi_paper,
            requires_multimodal_reasoning=is_multimodal,
            calc_expression=calc_expr,
            reasoning_summary=reasoning,
        )
