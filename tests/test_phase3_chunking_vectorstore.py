"""
Unit tests for Phase 3: Text Chunking, BGE Embeddings, and ChromaDB Vector Storage.
"""

import pytest
from unittest.mock import MagicMock
from src.ingestion.schema import NormalizedPage, NormalizedDocument, QualityMetrics
from src.ingestion.chunker import DocumentChunker, TextChunk
from src.models.embeddings import BGEEmbeddings
from src.retrieval.vector_store import VectorStore, SearchResult
from src.retrieval.text_retriever import TextRetriever


def test_deterministic_chunking_and_metadata_preservation():
    """Test that document chunking generates deterministic chunk IDs and preserves provenance metadata."""
    page = NormalizedPage(
        doc_id="convlstm_nowcasting",
        source_filename="convlstm_nowcasting.pdf",
        page_number=4,
        text=(
            "Paragraph 1: Convolutional LSTM Network handles spatio-temporal sequence forecasting.\n\n"
            "Paragraph 2: The model extends FC-LSTM by having convolutional structures in both input-to-state and state-to-state transitions.\n\n"
            "Paragraph 3: Experimental evaluations show superior performance on Moving-MNIST and radar datasets."
        ),
        raw_text="...",
        extraction_method="native",
        quality_metrics=QualityMetrics(150, 25, 1.0, 6.0, True),
        has_images=False,
    )

    chunker = DocumentChunker(target_chunk_chars=120, overlap_chars=30)
    chunks = chunker.chunk_page(page)

    assert len(chunks) > 1
    for idx, c in enumerate(chunks):
        assert isinstance(c, TextChunk)
        assert c.doc_id == "convlstm_nowcasting"
        assert c.source_filename == "convlstm_nowcasting.pdf"
        assert c.page_number == 4
        assert c.chunk_index == idx
        assert c.extraction_method == "native"
        assert c.chunk_id.startswith("convlstm_nowcasting_p004_c")

    # Verify determinism: re-chunking identical input yields identical chunk IDs
    chunks_retry = chunker.chunk_page(page)
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in chunks_retry]


def test_chromadb_upsert_prevents_duplicates():
    """Test that ChromaDB upserts with deterministic chunk IDs prevent duplicate records upon re-ingestion."""
    vstore = VectorStore(in_memory=True, collection_name="test_collection")
    assert vstore.count() == 0

    chunk = TextChunk(
        chunk_id="doc1_p001_c000_abc12345",
        doc_id="doc1",
        source_filename="doc1.pdf",
        page_number=1,
        chunk_index=0,
        text="Sample text content for vector indexing.",
        word_count=6,
        char_count=40,
        extraction_method="native",
    )

    dummy_embedding = [0.1] * 384

    # First ingestion
    vstore.add_chunks([chunk], [dummy_embedding])
    assert vstore.count() == 1

    # Second ingestion with identical chunk_id
    vstore.add_chunks([chunk], [dummy_embedding])
    assert vstore.count() == 1  # Record count must remain 1 (no duplicate)


def test_semantic_retrieval_and_metadata_roundtrip():
    """Test semantic retrieval and metadata round-trip using mock embeddings and ephemeral ChromaDB."""
    vstore = VectorStore(in_memory=True, collection_name="test_retrieval_collection")

    chunk1 = TextChunk(
        chunk_id="paperA_p001_c000",
        doc_id="paperA",
        source_filename="paperA.pdf",
        page_number=1,
        chunk_index=0,
        text="Convolutional LSTM achieves high precipitation nowcasting accuracy.",
        word_count=8,
        char_count=65,
        extraction_method="native",
    )
    chunk2 = TextChunk(
        chunk_id="paperB_p008_c000",
        doc_id="paperB",
        source_filename="paperB.pdf",
        page_number=8,
        chunk_index=0,
        text="Rapid fluvial flood inundation prediction using deep 2D CNN models.",
        word_count=10,
        char_count=67,
        extraction_method="ocr",
    )

    vec1 = [1.0] + [0.0] * 383
    vec2 = [0.0] * 383 + [1.0]

    vstore.add_chunks([chunk1, chunk2], [vec1, vec2])

    mock_embeddings = MagicMock(spec=BGEEmbeddings)
    # Query close to vec1
    mock_embeddings.embed_text.return_value = [0.9] + [0.0] * 383

    retriever = TextRetriever(vector_store=vstore, embedding_model=mock_embeddings)
    results = retriever.search("precipitation nowcasting", top_k=2)

    assert len(results) == 2
    top_res = results[0]
    assert top_res.chunk_id == "paperA_p001_c000"
    assert top_res.doc_id == "paperA"
    assert top_res.source_filename == "paperA.pdf"
    assert top_res.page_number == 1
    assert top_res.extraction_method == "native"
    assert "precipitation nowcasting" in top_res.text
