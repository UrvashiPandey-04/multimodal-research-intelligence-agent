"""
CLI Inspection & Retrieval Script for Phase 6 Scientific Table Extraction and Search.
Run this script to extract tables from canonical papers, index their Markdown embeddings, and run semantic table queries.
"""

import sys
from pathlib import Path

# Configure UTF-8 stdout encoding for Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.ingestion.table_parser import TableParser, ExtractedTable
from src.models.embeddings import get_embedding_model
from src.retrieval.table_repository import TableRepository
from src.retrieval.table_retriever import TableRetriever, TableSearchResult
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_validator import CitationValidator


def run_table_demo():
    """Demonstrate Phase 6 table extraction, indexing, semantic search, and citation provenance."""
    print("\n==================================================")
    print("PHASE 6 SCIENTIFIC TABLE EXTRACTION & RETRIEVAL DEMO")
    print("==================================================")

    papers_dir = root_dir / "data" / "papers"
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/papers/. Exiting.")
        sys.exit(1)

    table_parser = TableParser()
    table_retriever = TableRetriever(collection_name="research_paper_tables")
    all_tables: list[ExtractedTable] = []

    # 1. Extract tables page-by-page from canonical papers
    print("--- 1. TABLE EXTRACTION REPORT ---")
    for pdf in pdf_files:
        try:
            tables = table_parser.extract_tables_from_pdf(pdf)
            all_tables.extend(tables)
            print(f"Paper: {pdf.name:42s} | Tables Extracted: {len(tables)}")
            for t in tables:
                print(f"   └── Table ID: {t.table_id:30s} | Page: {t.page_number:02d} | Cols: {len(t.headers):02d} | Rows: {len(t.rows):02d}")
        except Exception as e:
            print(f"Paper: {pdf.name:42s} | Extraction Error: {e}")

    print(f"Total Extracted Tables Across Collection: {len(all_tables)}\n")

    if not all_tables:
        print("No extractable tables found in target PDFs. Exiting.")
        return

    # 2. Index Table Markdown Embeddings in ChromaDB
    print("--- 2. CHROMADB TABLE VECTOR INDEXING ---")
    indexed_count = table_retriever.add_tables(all_tables)
    print(f"Successfully indexed {indexed_count} tables into ChromaDB collection '{table_retriever.collection_name}'.\n")

    # 3. Execute Semantic Table Retrieval Queries
    sample_queries = [
        "Which model has the lowest MSE or best Rainfall-MSE score?",
        "What are the parameters for optical flow estimator in ROVER?",
    ]

    print("--- 3. SEMANTIC TABLE RETRIEVAL RESULTS ---")
    registry = EvidenceRegistry()

    for q_idx, query in enumerate(sample_queries, start=1):
        print(f"QUERY #{q_idx}: \"{query}\"")
        results = table_retriever.retrieve_tables(query, top_k=2)

        if not results:
            print("  [No matching tables retrieved]\n")
            continue

        for rank, res in enumerate(results, start=1):
            # Register in Evidence Registry
            ev_item = registry.register_table(res)

            print(f"  Rank #{rank} | Evidence ID: {ev_item.evidence_id} | Score: {res.score:.4f} (Dist: {res.distance:.4f})")
            print(f"  Paper:       {res.source_filename}")
            print(f"  Page Number: {res.page_number}")
            print(f"  Table ID:    {res.table_id}")
            print("  MARKDOWN TABLE:")
            for line in res.markdown_text.split("\n"):
                print(f"    {line}")
            print()

        print("--------------------------------------------------")

    # 4. Demonstrate Citation Validation for Table Evidence
    print("\n--- 4. TABLE EVIDENCE & CITATION VALIDATION ---")
    validator = CitationValidator()
    answer_text = "ConvLSTM achieves the lowest Rainfall-MSE score among all compared models [E1]."
    val_result = validator.validate_citations(answer_text, registry)

    print(f"Answer:            \"{answer_text}\"")
    print(f"Validation Status: {val_result.validation_status.upper()}")
    for cit in val_result.citations:
        print(f"  Validated Citation: {cit.formatted_citation} (Chunk/Table ID: {cit.chunk_id})")

    print("==================================================\n")


if __name__ == "__main__":
    run_table_demo()
