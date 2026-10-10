"""
Response Generator module integrating Evidence Registry and Citation Validation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

from src.retrieval.text_retriever import TextRetriever
from src.generation.context_builder import ContextBuilder, StructuredContext, EvidenceBlock
from src.generation.prompts import RAG_SYSTEM_INSTRUCTION, build_rag_prompt
from src.models.llm import GeminiLLM, MissingAPIKeyError
from src.citations.evidence_registry import EvidenceRegistry
from src.citations.citation_validator import CitationValidator
from src.citations.citation_model import ValidationResult, Citation


@dataclass
class RAGAnswer:
    """Structured response object containing query, answer, evidence, validated citations, and validation metadata."""
    query: str
    answer: str
    retrieved_evidence: List[Dict[str, Any]]
    citations: List[Dict[str, Any]]
    invalid_citations: List[Dict[str, Any]] = field(default_factory=list)
    validation_status: str = "none"
    validation_messages: List[str] = field(default_factory=list)
    model: str = "gemini-2.5-flash"
    is_grounded: bool = True
    is_sufficient: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert RAGAnswer to dictionary format."""
        return {
            "query": self.query,
            "answer": self.answer,
            "is_grounded": self.is_grounded,
            "is_sufficient": self.is_sufficient,
            "validation_status": self.validation_status,
            "validation_messages": self.validation_messages,
            "model": self.model,
            "retrieved_evidence": self.retrieved_evidence,
            "citations": self.citations,
            "invalid_citations": self.invalid_citations,
        }


class ResponseGenerator:
    """End-to-end evidence-grounded RAG generator with deterministic citation validation."""

    def __init__(
        self,
        retriever: Optional[TextRetriever] = None,
        context_builder: Optional[ContextBuilder] = None,
        validator: Optional[CitationValidator] = None,
        llm: Optional[GeminiLLM] = None,
    ):
        """
        Initialize ResponseGenerator.

        Args:
            retriever: TextRetriever instance.
            context_builder: ContextBuilder instance.
            validator: CitationValidator instance.
            llm: GeminiLLM instance.
        """
        self.retriever = retriever or TextRetriever()
        self.context_builder = context_builder or ContextBuilder(min_score_threshold=0.78)
        self.validator = validator or CitationValidator()
        self.llm = llm or GeminiLLM()

    def generate_answer(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        source_filename: Optional[str] = None,
    ) -> RAGAnswer:
        """
        Execute full Phase 5 RAG pipeline:
        Query -> Retrieve -> Register Evidence -> Build Context -> Gemini LLM -> Validate Citations -> Grounded RAGAnswer.

        Args:
            query: User research question.
            top_k: Number of chunks to retrieve.
            doc_id: Optional metadata filter for specific document ID.
            source_filename: Optional metadata filter for specific PDF.

        Returns:
            RAGAnswer object containing answer, evidence, validated citations, and validation status.
        """
        clean_query = query.strip()
        if not clean_query:
            return RAGAnswer(
                query=query,
                answer="Please provide a valid non-empty research question.",
                retrieved_evidence=[],
                citations=[],
                invalid_citations=[],
                validation_status="none",
                is_grounded=False,
                is_sufficient=False,
            )

        # 1. Semantic Retrieval
        search_results = self.retriever.search(
            query=clean_query,
            top_k=top_k,
            doc_id=doc_id,
            source_filename=source_filename,
        )

        # 2. Build Request Evidence Registry
        registry = EvidenceRegistry()
        registry.register_chunks(search_results)

        # 3. Context Building & Sufficiency Assessment
        structured_context = self.context_builder.build_context(search_results)
        evidence_dicts = [item.to_dict() for item in registry.get_all_items()]

        # 4. Insufficient Evidence Fallback
        if not structured_context.is_sufficient:
            return RAGAnswer(
                query=clean_query,
                answer="I don't have enough evidence in the provided papers to answer this.",
                retrieved_evidence=evidence_dicts,
                citations=[],
                invalid_citations=[],
                validation_status="none",
                model=self.llm.model_name if self.llm else "gemini-2.5-flash",
                is_grounded=True,
                is_sufficient=False,
            )

        # 5. Prompt Construction
        prompt = build_rag_prompt(clean_query, structured_context.formatted_context)

        # 6. Gemini LLM Text Generation
        if not self.llm.is_configured():
            answer_text = "[API KEY MISSING] Cannot generate Gemini response without GEMINI_API_KEY."
            is_grounded = False
        else:
            try:
                answer_text = self.llm.generate_text(
                    prompt=prompt,
                    system_instruction=RAG_SYSTEM_INSTRUCTION,
                )
                is_grounded = True
            except Exception as e:
                from src.models.llm import GeminiQuotaExceededError
                if isinstance(e, GeminiQuotaExceededError) or "quota" in str(e).lower():
                    answer_text = "Gemini API quota has been reached. Answer generation is temporarily unavailable."
                else:
                    answer_text = "Answer generation is currently unavailable due to an API error."
                is_grounded = False

        # 7. Deterministic Citation Validation against Request EvidenceRegistry
        val_result: ValidationResult = self.validator.validate_citations(
            answer_text=answer_text,
            registry=registry,
        )

        valid_citations_dict = [c.to_dict() for c in val_result.citations]
        invalid_citations_dict = [c.to_dict() for c in val_result.invalid_citations]

        return RAGAnswer(
            query=clean_query,
            answer=answer_text,
            retrieved_evidence=evidence_dicts,
            citations=valid_citations_dict,
            invalid_citations=invalid_citations_dict,
            validation_status=val_result.validation_status,
            validation_messages=val_result.validation_messages,
            model=self.llm.model_name,
            is_grounded=is_grounded,
            is_sufficient=True,
        )
