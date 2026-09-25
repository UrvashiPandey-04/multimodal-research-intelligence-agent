"""
CLI Inspection Script for Phase 7 Scientific Figure Extraction, VLM Analysis, Vector Search, and Citation Validation.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.figure_parser import PyMuPDFFigureExtractor, ExtractedFigure
from src.retrieval.figure_repository import FigureRepository
from src.models.vlm_cache import VLMCache
from src.models.vlm_analyzer import VLMFigureAnalyzer, FigureAnalysisResult
from src.retrieval.figure_retriever import FigureRetriever
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_validator import CitationValidator


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    papers_dir = os.path.join(project_root, "data", "papers")
    if not os.path.exists(papers_dir):
        print(f"Error: Papers directory not found at {papers_dir}")
        sys.exit(1)

    pdf_files = [f for f in os.listdir(papers_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}")
        sys.exit(1)

    print("==================================================")
    print("PHASE 7 SCIENTIFIC FIGURE EXTRACTION & VLM DEMO")
    print("==================================================")

    pdf_parser = PDFParser()
    figure_extractor = PyMuPDFFigureExtractor(output_dir=os.path.join(project_root, "data", "figures"))
    figure_repo = FigureRepository()
    vlm_cache = VLMCache(cache_file=os.path.join(project_root, "data", "cache", "vlm_cache.json"))
    vlm_analyzer = VLMFigureAnalyzer(cache=vlm_cache)
    figure_retriever = FigureRetriever()

    all_figures = []

    # 1. FIGURE EXTRACTION REPORT
    print("\n--- 1. FIGURE EXTRACTION REPORT ---")
    for pdf_filename in pdf_files:
        pdf_path = os.path.join(papers_dir, pdf_filename)
        doc = pdf_parser.parse_pdf(pdf_path)
        figs = figure_extractor.extract_figures(pdf_path, doc.doc_id)
        figure_repo.add_figures(figs)
        all_figures.extend(figs)

        print(f"Paper: {pdf_filename:<42} | Figures Extracted: {len(figs)}")
        for fig in figs:
            cap_snippet = (fig.caption[:45] + "...") if fig.caption else "None"
            print(f"   |-- Figure ID: {fig.figure_id} | Page: {fig.page_number:02d} | Caption: {cap_snippet}")

    print(f"Total Extracted Figures Across Collection: {len(all_figures)}")

    if not all_figures:
        print("No figures detected across papers.")
        return

    # 2. VLM ANALYSIS & CHROMADB VECTOR INDEXING
    print("\n--- 2. VLM ANALYSIS & VECTOR INDEXING ---")
    is_vlm_live = vlm_analyzer.llm.is_configured()
    if is_vlm_live:
        print("GEMINI_API_KEY detected. Running live Gemini 2.5 Flash VLM analysis (with caching)...")
    else:
        print("GEMINI_API_KEY not set. Running structured fallback VLM analysis...")

    for fig in all_figures:
        analysis = vlm_analyzer.analyze_figure(fig)
        figure_retriever.add_figure(fig, analysis)

    print(f"Successfully indexed {len(all_figures)} figures into ChromaDB collection '{FigureRetriever.COLLECTION_NAME}'.")

    # 3. SEMANTIC FIGURE RETRIEVAL DEMO
    print("\n--- 3. SEMANTIC FIGURE RETRIEVAL DEMO ---")
    sample_queries = [
        "What does the ConvLSTM architecture or network structure look like?",
        "Show radar reflectivity or precipitation prediction comparison plots",
    ]

    for q_idx, query in enumerate(sample_queries, start=1):
        print(f"\nQUERY #{q_idx}: \"{query}\"")
        results = figure_retriever.search(query, top_k=2)

        if not results:
            print("  No matching figure results returned.")
            continue

        for rank, res in enumerate(results, start=1):
            print(f"  Rank #{rank} | Score: {res.score:.4f}")
            print(f"  Paper:       {res.source_filename}")
            print(f"  Page Number: {res.page_number}")
            print(f"  Figure ID:   {res.figure_id}")
            print(f"  Caption:     {res.caption or 'None'}")
            print(f"  Image Path:  {res.image_path}")
            if res.analysis:
                print(f"  Summary:     {res.analysis.figure_summary}")
                print(f"  Type:        {res.analysis.figure_type}")
                if res.analysis.observations:
                    print(f"  Observations: {', '.join(res.analysis.observations[:3])}")
                if res.analysis.uncertainties:
                    print(f"  Uncertainties: {', '.join(res.analysis.uncertainties[:2])}")
            print("  " + "-" * 45)

    # 4. EVIDENCE REGISTRATION & CITATION VALIDATION
    print("\n--- 4. FIGURE EVIDENCE & CITATION VALIDATION ---")
    registry = EvidenceRegistry()
    validator = CitationValidator()

    # Search top figure for query
    top_fig_results = figure_retriever.search(sample_queries[0], top_k=1)
    if top_fig_results:
        top_fig = top_fig_results[0]
        ev_item = registry.register_figure(top_fig)

        sample_answer = f"The ConvLSTM network structure and hidden state transitions are illustrated in the architecture diagram [{ev_item.evidence_id}]."
        val_result = validator.validate_citations(sample_answer, registry)

        print(f"Answer:            \"{sample_answer}\"")
        print(f"Validation Status: {val_result.validation_status.upper()}")
        for cit in val_result.citations:
            print(f"  Validated Citation: {cit.formatted_citation} (Figure ID: {cit.chunk_id})")

    print("==================================================\n")


if __name__ == "__main__":
    main()
