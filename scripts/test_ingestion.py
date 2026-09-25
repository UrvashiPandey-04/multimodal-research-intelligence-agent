"""
Minimal Command-Line Ingestion Test Script for Phase 2.
Run this script to verify PDF ingestion, quality assessment, normalization, and OCR fallback metrics on target documents.
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


def run_ingestion_test(pdf_path: str | Path):
    """Run normalized page-level text extraction test on a given PDF file."""
    path = Path(pdf_path)
    print(f"\n==========================================")
    print(f"Testing PDF Ingestion (Phase 2): {path.name}")
    print(f"==========================================")

    parser = PDFParser()
    doc_data = parser.parse_pdf(path)

    print(f"Filename:      {doc_data.source_filename}")
    print(f"Doc ID:        {doc_data.doc_id}")
    print(f"Total Pages:   {doc_data.total_pages}")
    print(f"Native Pages:  {doc_data.native_page_count}")
    print(f"OCR Pages:     {doc_data.ocr_page_count}")
    print(f"File Path:     {doc_data.filepath}")
    print(f"------------------------------------------")

    for p in doc_data.pages:
        snippet = p.text[:100].replace("\n", " ") if p.text else "[EMPTY PAGE]"
        q = p.quality_metrics
        print(
            f"Page {p.page_number:02d} | Method: {p.extraction_method:6s} | Usable: {str(q.is_usable):5s} | "
            f"Words: {q.word_count:04d} | Chars: {q.char_count:05d} | Printable: {q.printable_ratio:.2f} | Snippet: {snippet}..."
        )

    print(f"==========================================\n")
    return doc_data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Check papers directory for actual paper
        papers_dir = root_dir / "data" / "papers"
        pdf_files = list(papers_dir.glob("*.pdf"))

        if pdf_files:
            # Prefer real arXiv paper if present
            real_papers = [f for f in pdf_files if "convlstm" in f.name.lower()]
            target_path = real_papers[0] if real_papers else pdf_files[0]
        else:
            print("No PDF files found in data/papers/. Creating a sample synthetic PDF for test...")
            import pymupdf as fitz
            sample_file = papers_dir / "canonical_sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Phase 2 Ingestion Test Document.")
            page.insert_text((50, 80), "Paper: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.")
            doc.save(str(sample_file))
            doc.close()
            target_path = sample_file

    run_ingestion_test(target_path)
