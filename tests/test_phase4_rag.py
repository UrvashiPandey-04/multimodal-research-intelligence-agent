"""
Unit tests for Phase 4: Basic Retrieval-Augmented Generation (RAG).
"""

import os
import pytest
from unittest.mock import MagicMock
from src.retrieval.vector_store import SearchResult, VectorStore
from src.retrieval.text_retriever import TextRetriever
from src.generation.context_builder import ContextBuilder, StructuredContext
from src.generation.prompts import build_rag_prompt, RAG_SYSTEM_INSTRUCTION
from src.generation.response_generator import ResponseGenerator, RAGAnswer
from src.models.llm import GeminiLLM


def test_context_builder_metadata_preservation():
    """Test that ContextBuilder formats evidence blocks and preserves paper/page metadata."""
    chunks = [
        SearchResult(
            chunk_id="chunk1",
            doc_id="doc1",
            source_filename="convlstm.pdf",
            page_number=4,
            chunk_index=0,
            text="ConvLSTM replaces matrix multiplication with convolution in state transitions.",
            extraction_method="native",
            score=0.88,
            distance=0.24,
            metadata={},
        ),
        SearchResult(
            chunk_id="chunk2",
            doc_id="doc2",
            source_filename="dcnn_flood.pdf",
            page_number=8,
            chunk_index=0,
            text="The 2D DCNN model achieves an RMSE of 0.12m for flood inundation mapping.",
            extraction_method="ocr",
            score=0.82,
            distance=0.36,
            metadata={},
        ),
    ]

    builder = ContextBuilder(min_score_threshold=0.78)
    context = builder.build_context(chunks)

    assert isinstance(context, StructuredContext)
    assert context.is_sufficient is True
    assert context.max_score == 0.88
    assert len(context.evidence_blocks) == 2

    # Check Evidence 1 metadata
    e1 = context.evidence_blocks[0]
    assert e1.paper == "convlstm.pdf"
    assert e1.page == 4
    assert e1.extraction_method == "native"
    assert "ConvLSTM" in e1.text

    # Check Evidence 2 metadata
    e2 = context.evidence_blocks[1]
    assert e2.paper == "dcnn_flood.pdf"
    assert e2.page == 8
    assert e2.extraction_method == "ocr"

    # Check formatted string contains evidence tags and paper/page provenance
    assert "[EVIDENCE E1]" in context.formatted_context
    assert "Paper: convlstm.pdf" in context.formatted_context
    assert "Page: 4" in context.formatted_context
    assert "[EVIDENCE E2]" in context.formatted_context
    assert "Paper: dcnn_flood.pdf" in context.formatted_context
    assert "Page: 8" in context.formatted_context


def test_context_builder_insufficient_evidence():
    """Test that low similarity scores mark evidence as insufficient."""
    low_chunks = [
        SearchResult(
            chunk_id="c1",
            doc_id="d1",
            source_filename="paper.pdf",
            page_number=1,
            chunk_index=0,
            text="Irrelevant text snippet.",
            extraction_method="native",
            score=0.15,  # Below 0.35 threshold
            distance=1.70,
            metadata={},
        )
    ]

    builder = ContextBuilder(min_score_threshold=0.78)
    context = builder.build_context(low_chunks)

    assert context.is_sufficient is False
    assert context.max_score == 0.15


def test_response_generator_insufficient_evidence_fallback():
    """Test that ResponseGenerator produces the required fallback response when evidence is insufficient."""
    mock_retriever = MagicMock(spec=TextRetriever)
    mock_retriever.search.return_value = []  # No relevant chunks found

    mock_llm = MagicMock(spec=GeminiLLM)
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.is_configured.return_value = True

    generator = ResponseGenerator(retriever=mock_retriever, llm=mock_llm)
    answer = generator.generate_answer("What quantum processor was used?")

    assert isinstance(answer, RAGAnswer)
    assert answer.is_sufficient is False
    assert answer.answer == "I don't have enough evidence in the provided papers to answer this."
    assert len(answer.citations) == 0


def test_response_generator_grounded_answer_and_citations():
    """Test full RAG generation pipeline with mocked Gemini LLM."""
    mock_retriever = MagicMock(spec=TextRetriever)
    mock_retriever.search.return_value = [
        SearchResult(
            chunk_id="conv_p6_c0",
            doc_id="convlstm",
            source_filename="convlstm_precipitation_nowcasting.pdf",
            page_number=6,
            chunk_index=0,
            text="ConvLSTM model was evaluated on Moving-MNIST dataset.",
            extraction_method="native",
            score=0.86,
            distance=0.28,
            metadata={},
        )
    ]

    mock_llm = MagicMock(spec=GeminiLLM)
    mock_llm.model_name = "gemini-2.5-flash"
    mock_llm.is_configured.return_value = True
    mock_llm.generate_text.return_value = (
        "The authors evaluated the ConvLSTM model using the Moving-MNIST dataset [E1]."
    )

    generator = ResponseGenerator(retriever=mock_retriever, llm=mock_llm)
    answer = generator.generate_answer("What dataset was used?")

    assert isinstance(answer, RAGAnswer)
    assert answer.is_grounded is True
    assert answer.is_sufficient is True
    assert answer.validation_status == "valid"
    assert "Moving-MNIST dataset [E1]" in answer.answer
    assert len(answer.citations) == 1

    citation = answer.citations[0]
    assert citation["evidence_id"] == "E1"
    assert citation["source_filename"] == "convlstm_precipitation_nowcasting.pdf"
    assert citation["page_number"] == 6
    assert citation["chunk_id"] == "conv_p6_c0"
    assert citation["formatted_citation"] == "[E1] convlstm_precipitation_nowcasting.pdf, p. 6"

    # Verify Gemini LLM received prompt containing retrieved context
    mock_llm.generate_text.assert_called_once()
    call_args = mock_llm.generate_text.call_args
    prompt_sent = call_args.kwargs.get("prompt") or call_args.args[0]
    assert "What dataset was used?" in prompt_sent
    assert "convlstm_precipitation_nowcasting.pdf" in prompt_sent


@pytest.mark.skipif(
    not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_gemini_api_key_here",
    reason="Real Gemini API Key not present in environment",
)
def test_real_gemini_rag_integration():
    """Optional real integration test executed only when GEMINI_API_KEY is available."""
    vstore = VectorStore(in_memory=True, collection_name="test_real_rag")

    from src.ingestion.chunker import TextChunk
    chunk = TextChunk(
        chunk_id="real_test_chunk",
        doc_id="convlstm_paper",
        source_filename="convlstm_paper.pdf",
        page_number=3,
        chunk_index=0,
        text="The proposed Convolutional LSTM (ConvLSTM) model extends FC-LSTM by introducing 3D tensor state transitions.",
        word_count=16,
        char_count=115,
        extraction_method="native",
    )

    from src.models.embeddings import get_embedding_model
    emb_model = get_embedding_model()
    vec = emb_model.embed_text(chunk.text)
    vstore.add_chunks([chunk], [vec])

    retriever = TextRetriever(vector_store=vstore, embedding_model=emb_model)
    generator = ResponseGenerator(retriever=retriever)

    answer = generator.generate_answer("What model is proposed in the paper?")
    assert answer.is_grounded is True
    assert answer.is_sufficient is True
    assert len(answer.answer) > 10
    assert len(answer.citations) > 0
