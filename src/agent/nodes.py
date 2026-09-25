"""
LangGraph Nodes implementing the research agent orchestration steps.
"""

from typing import Dict, Any, Optional
from src.agent.state import ResearchAgentState
from src.agent.query_analyzer import QueryAnalyzer, QueryAnalysis
from src.agent.calculator import SafeCalculator, CalculationResult
from src.agent.context_fuser import MultimodalContextFuser, FusedContext
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_validator import CitationValidator
from src.models.llm import GeminiLLM
from src.retrieval.text_retriever import TextRetriever
from src.retrieval.table_retriever import TableRetriever
from src.retrieval.figure_retriever import FigureRetriever
from src.models.vlm_analyzer import VLMFigureAnalyzer


class ResearchAgentNodes:
    """Class encapsulating stateful LangGraph node execution logic."""

    def __init__(
        self,
        text_retriever: Optional[TextRetriever] = None,
        table_retriever: Optional[TableRetriever] = None,
        figure_retriever: Optional[FigureRetriever] = None,
        vlm_analyzer: Optional[VLMFigureAnalyzer] = None,
        calculator: Optional[SafeCalculator] = None,
        llm_client: Optional[GeminiLLM] = None,
    ):
        """
        Initialize agent nodes with existing component instances.

        Args:
            text_retriever: TextRetriever instance.
            table_retriever: TableRetriever instance.
            figure_retriever: FigureRetriever instance.
            vlm_analyzer: VLMFigureAnalyzer instance.
            calculator: SafeCalculator instance.
            llm_client: GeminiLLM instance.
        """
        self.text_retriever = text_retriever or TextRetriever()
        self.table_retriever = table_retriever or TableRetriever()
        self.figure_retriever = figure_retriever or FigureRetriever()
        self.vlm_analyzer = vlm_analyzer or VLMFigureAnalyzer()
        self.calculator = calculator or SafeCalculator()
        self.llm = llm_client or GeminiLLM()
        self.query_analyzer = QueryAnalyzer()
        self.context_fuser = MultimodalContextFuser()
        self.citation_validator = CitationValidator()

    def query_analysis_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 1: Analyze user query intent and requirements."""
        query = state.get("original_query", "")
        analysis = self.query_analyzer.analyze(query)
        return {
            "query_analysis": analysis,
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        }

    def tool_selection_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 2: Select explicit required tools based on query analysis."""
        analysis: Optional[QueryAnalysis] = state.get("query_analysis")
        if analysis:
            tools = analysis.get_selected_tools()
        else:
            tools = ["text_retriever"]
        return {"selected_tools": tools}

    def tool_execution_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 3: Safely execute selected tools and collect outputs."""
        query = state.get("original_query", "")
        analysis: Optional[QueryAnalysis] = state.get("query_analysis")
        tools = state.get("selected_tools", [])

        tool_results: Dict[str, Any] = {}
        errors = list(state.get("errors", []))

        # 1. Text Retrieval
        if "text_retriever" in tools:
            try:
                if analysis and analysis.requires_multiple_papers:
                    chunks = self.text_retriever.search_diversified(query, top_k=6, per_doc_limit=3)
                else:
                    chunks = self.text_retriever.search(query, top_k=4)
                tool_results["text_chunks"] = chunks
            except Exception as e:
                errors.append(f"Text retriever error: {e}")
                tool_results["text_chunks"] = []

        # 2. Table Retrieval
        if "table_retriever" in tools:
            try:
                tables = self.table_retriever.search(query, top_k=3)
                tool_results["tables"] = tables
            except Exception as e:
                errors.append(f"Table retriever error: {e}")
                tool_results["tables"] = []

        # 3. Figure Retrieval & VLM Analysis
        if "figure_retriever" in tools or "figure_vlm" in tools:
            try:
                figures = self.figure_retriever.search(query, top_k=2)
                tool_results["figures"] = figures

                # VLM visual analysis for top retrieved figures
                if "figure_vlm" in tools and figures:
                    for fig_res in figures:
                        # Reconstruct ExtractedFigure context for VLM
                        from src.ingestion.figure_parser import ExtractedFigure
                        fig_obj = ExtractedFigure(
                            figure_id=fig_res.figure_id,
                            doc_id=fig_res.doc_id,
                            source_filename=fig_res.source_filename,
                            page_number=fig_res.page_number,
                            figure_index=0,
                            image_path=fig_res.image_path,
                            caption=fig_res.caption,
                            extraction_method="pymupdf_cropped",
                        )
                        fig_res.analysis = self.vlm_analyzer.analyze_figure(fig_obj)
            except Exception as e:
                errors.append(f"Figure retriever/VLM error: {e}")
                tool_results["figures"] = []

        # 4. Calculator Tool
        if "calculator" in tools:
            try:
                expr = analysis.calc_expression if analysis and analysis.calc_expression else query
                calc_res = self.calculator.evaluate(expr)
                tool_results["calculation"] = calc_res
            except Exception as e:
                errors.append(f"Calculator tool error: {e}")

        return {"tool_results": tool_results, "errors": errors}

    def evidence_collection_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 4: Normalize all tool outputs into authoritative request EvidenceRegistry."""
        registry = EvidenceRegistry()
        tool_results = state.get("tool_results", {})

        # Text chunks
        chunks = tool_results.get("text_chunks", [])
        if chunks:
            registry.register_chunks(chunks)

        # Tables
        tables = tool_results.get("tables", [])
        if tables:
            registry.register_tables(tables)

        # Figures
        figures = tool_results.get("figures", [])
        if figures:
            registry.register_figures(figures)

        # Calculation
        calc: Optional[CalculationResult] = tool_results.get("calculation")
        if calc and calc.success:
            registry.register_calculation(calc.expression, calc.formatted_result, calc.result)

        return {"evidence_registry": registry}

    def context_fusion_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 5: Fuse multi-modal evidence items into structured prompt context."""
        registry: EvidenceRegistry = state.get("evidence_registry") or EvidenceRegistry()
        fused_context = self.context_fuser.fuse_context(registry)
        return {"fused_context": fused_context}

    def answer_generation_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 6: Generate grounded answer using Gemini 2.5 Flash and fused evidence."""
        query = state.get("original_query", "")
        fused: Optional[FusedContext] = state.get("fused_context")

        if not fused or fused.evidence_count == 0:
            fallback = "I don't have enough evidence in the provided research papers to answer this question."
            return {"generated_answer": fallback}

        if not self.llm.is_configured():
            # Fallback answer when LLM API key is missing
            reg: EvidenceRegistry = state.get("evidence_registry")
            items = reg.get_all_items() if reg else []
            eids = " ".join([f"[{it.evidence_id}]" for it in items])
            fallback = f"Based on retrieved research paper evidence {eids}, here is the grounded summary for '{query}'."
            return {"generated_answer": fallback}

        system_instruction = (
            "You are a strict, evidence-grounded scientific research assistant.\n"
            "Rules:\n"
            "1. Answer the user query using ONLY the provided evidence blocks [E1], [E2], etc.\n"
            "2. Cite evidence IDs inline (e.g., [E1], [E2]) immediately after stating facts derived from them.\n"
            "3. Do NOT invent page numbers, paper names, metric scores, or figures not present in the evidence.\n"
            "4. If the provided evidence is insufficient to answer the query fully, explicitly state:\n"
            "   'I don't have enough evidence in the provided papers to answer this.'\n"
        )

        user_prompt = f"User Query: {query}\n\nRetrieved Evidence Blocks:\n{fused.fused_text}\n\nPlease provide a clear, grounded response with inline citation tags [E1], [E2]."

        try:
            raw_answer = self.llm.generate_text(
                prompt=user_prompt,
                system_instruction=system_instruction,
                temperature=0.1,
            )
            return {"generated_answer": raw_answer}
        except Exception as e:
            fallback = f"Answer generation failed: {e}. Available evidence: {fused.evidence_count} items."
            return {"generated_answer": fallback}

    def citation_validation_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 7: Deterministically validate citation tags against EvidenceRegistry provenance."""
        answer = state.get("generated_answer", "")
        registry: EvidenceRegistry = state.get("evidence_registry") or EvidenceRegistry()
        val_result = self.citation_validator.validate_citations(answer, registry)
        return {"validation_result": val_result}

    def final_response_node(self, state: ResearchAgentState) -> Dict[str, Any]:
        """Node 8: Assemble final inspectable response package."""
        answer = state.get("generated_answer", "")
        val_result = state.get("validation_result")

        final_str_parts = [answer]

        if val_result and val_result.citations:
            final_str_parts.append("\n\nVerified Citations:")
            for cit in val_result.citations:
                final_str_parts.append(f"  {cit.formatted_citation}")

        final_response = "\n".join(final_str_parts)
        return {"final_response": final_response}
