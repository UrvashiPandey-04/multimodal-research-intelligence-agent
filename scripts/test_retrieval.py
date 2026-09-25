"""
CLI Retrieval Inspection Script for Phase 3.
Run this script to index research papers into ChromaDB and execute semantic similarity search.
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

from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.models.embeddings import get_embedding_model
from src.retrieval.vector_store import VectorStore
from src.retrieval.text_retriever import TextRetriever


def index_and_search(query_strings: list[str], pdf_path: Path):
    """Index a target PDF into ChromaDB and run semantic query evaluation."""
    print("\n==================================================")
    print("PHASE 3 INGESTION & RETRIEVAL VALIDATION")
    print("==================================================")

    # 1. Parse & Normalize PDF (Phase 1 & 2)
    parser = PDFParser()
    norm_doc = parser.parse_pdf(pdf_path)
    print(f"Parsed Document: {norm_doc.source_filename} (Doc ID: {norm_doc.doc_id})")
    print(f"Total Pages: {norm_doc.total_pages} | Native: {norm_doc.native_page_count} | OCR: {norm_doc.ocr_page_count}")

    # 2. Chunk Document
    chunker = DocumentChunker(target_chunk_chars=2400, overlap_chars=400)
    chunks = chunker.chunk_document(norm_doc)
    print(f"Generated Chunks: {len(chunks)} chunks")

    # 3. Generate Embeddings (BGE Small v1.5)
    embedding_model = get_embedding_model()
    chunk_texts = [c.text for c in chunks]
    print(f"Generating BGE embeddings (384-dim) for {len(chunk_texts)} chunks...")
    embeddings = embedding_model.embed_batch(chunk_texts)

    # 4. Upsert into ChromaDB
    vector_store = VectorStore(collection_name="research_paper_text_chunks")
    added_count = vector_store.add_chunks(chunks, embeddings)
    print(f"Vector Store Collection Count: {vector_store.count()} chunks in ChromaDB")
    print("--------------------------------------------------\n")

    # 5. Execute Semantic Queries
    retriever = TextRetriever(vector_store=vector_store, embedding_model=embedding_model)

    for q in query_strings:
        print(f"QUERY: \"{q}\"")
        results = retriever.search(q, top_k=3)

        for rank, res in enumerate(results, start=1):
            snippet = res.text[:140].replace("\n", " ")
            print(
                f"  Rank #{rank} | Score: {res.score:.4f} (Dist: {res.distance:.4f}) | "
                f"Paper: {res.source_filename} | Page: {res.page_number:02d} | Method: {res.extraction_method}"
            )
            print(f"  Snippet: {snippet}...\n")

        print("--------------------------------------------------")

    print("==================================================\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_pdfs = [Path(sys.argv[1])]
    else:
        papers_dir = root_dir / "data" / "papers"
        target_pdfs = list(papers_dir.glob("*.pdf"))

    if not target_pdfs:
        print("No PDF files found in data/papers/. Exiting.")
        sys.exit(1)

    sample_queries = [
        "What dataset or rainfall data did the authors use?",
        "What architecture or model does the paper propose?",
        "What is the reported RMSE or prediction performance score?",
    ]

    for pdf in target_pdfs:
        index_and_search(sample_queries, pdf)
