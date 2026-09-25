"""
CLI RAG Inspection Script for Phase 5.
Run this script to test semantic retrieval, Evidence Registry mapping, Gemini LLM generation, and deterministic citation validation.
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
from src.generation.response_generator import ResponseGenerator
from src.models.llm import GeminiLLM


def run_rag_demo():
    """Run Phase 5 RAG demonstration with citation validation on canonical research papers."""
    print("\n==================================================")
    print("PHASE 5 CITATION VALIDATION & GROUNDING DEMO")
    print("==================================================")

    # 1. Check/Index target paper into ChromaDB
    papers_dir = root_dir / "data" / "papers"
    pdf_files = list(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/papers/. Exiting.")
        sys.exit(1)

    target_pdf = pdf_files[0]
    print(f"Indexing Target PDF: {target_pdf.name}")

    parser = PDFParser()
    norm_doc = parser.parse_pdf(target_pdf)

    chunker = DocumentChunker()
    chunks = chunker.chunk_document(norm_doc)

    embedding_model = get_embedding_model()
    chunk_texts = [c.text for c in chunks]
    embeddings = embedding_model.embed_batch(chunk_texts)

    vector_store = VectorStore(collection_name="research_paper_text_chunks")
    vector_store.add_chunks(chunks, embeddings)
    print(f"Vector Store Ready: {vector_store.count()} chunks in ChromaDB")
    print("--------------------------------------------------\n")

    retriever = TextRetriever(vector_store=vector_store, embedding_model=embedding_model)
    llm = GeminiLLM()

    if not llm.is_configured():
        print("[NOTICE] GEMINI_API_KEY is not configured in .env.")
        print("[NOTICE] Retrieval, Evidence Registry, and Citation Validation will execute without API calls.\n")

    generator = ResponseGenerator(retriever=retriever, llm=llm)

    test_questions = [
        # 1. Straightforward factual question
        "What dataset or rainfall data did the authors evaluate?",
        # 2. Question requiring multiple retrieved chunks
        "What architecture did the authors propose and what performance did it achieve?",
        # 3. Question where evidence is insufficient
        "What quantum computing hardware processor was used to train the ConvLSTM model?",
    ]

    for idx, q in enumerate(test_questions, start=1):
        print(f"=== QUESTION #{idx}: \"{q}\" ===")
        rag_answer = generator.generate_answer(q, top_k=3)

        print(f"Sufficient Evidence: {rag_answer.is_sufficient}")
        print(f"Validation Status:   {rag_answer.validation_status.upper()}")
        print(f"Model:               {rag_answer.model}")
        print("--------------------------------------------------")
        print("GENERATED GROUNDED ANSWER:")
        print(rag_answer.answer)
        print("--------------------------------------------------")
        print("VALIDATED CITATIONS (From Evidence Registry):")
        if rag_answer.citations:
            for c in rag_answer.citations:
                print(f"  {c['formatted_citation']} (Chunk ID: {c['chunk_id']})")
        else:
            print("  [No valid citations]")

        if rag_answer.invalid_citations:
            print("INVALID / REJECTED CITATIONS:")
            for ic in rag_answer.invalid_citations:
                print(f"  [REJECTED] Evidence ID '{ic['evidence_id']}': {ic['validation_reason']}")

        print("--------------------------------------------------")
        print("AUTHORITATIVE EVIDENCE REGISTRY:")
        for ev in rag_answer.retrieved_evidence:
            print(f"  {ev['evidence_id']} ──► {ev['source_filename']} | Page {ev['page_number']} | Method: {ev['extraction_method']} | Score: {ev['score']:.4f}")
        print("==================================================\n")


if __name__ == "__main__":
    run_rag_demo()
