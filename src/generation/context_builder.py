"""
Context Builder module for formatting retrieved chunks into structured, provenance-preserving LLM context.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from src.retrieval.vector_store import SearchResult


@dataclass
class EvidenceBlock:
    """Represents a structured evidence item formatted for LLM ingestion."""
    evidence_id: int  # 1-indexed (e.g., 1, 2, 3)
    chunk_id: str
    doc_id: str
    paper: str
    page: int
    extraction_method: str
    score: float
    text: str

    def to_citation_dict(self) -> Dict[str, Any]:
        """Convert evidence block to citation metadata dictionary."""
        return {
            "citation_label": f"[{self.evidence_id}]",
            "paper": self.paper,
            "page": self.page,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "extraction_method": self.extraction_method,
            "score": self.score,
        }


@dataclass
class StructuredContext:
    """Represents the complete structured context passed to the LLM."""
    formatted_context: str
    evidence_blocks: List[EvidenceBlock]
    is_sufficient: bool
    max_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert structured context to dictionary."""
        return {
            "is_sufficient": self.is_sufficient,
            "max_score": round(self.max_score, 4),
            "evidence_count": len(self.evidence_blocks),
            "evidence_blocks": [e.to_citation_dict() for e in self.evidence_blocks],
        }


class ContextBuilder:
    """Formats retrieved vector search results into structured, evidence-grounded prompt context."""

    def __init__(self, min_score_threshold: float = 0.78):
        """
        Initialize ContextBuilder.

        Args:
            min_score_threshold: Minimum similarity score threshold for considering evidence sufficient.
                                 For BGE-small embeddings, relevant hits score >= 0.78.
        """
        self.min_score_threshold = min_score_threshold

    def build_context(self, chunks: List[SearchResult]) -> StructuredContext:
        """
        Build structured context string from retrieved search result chunks.

        Args:
            chunks: List of SearchResult objects from TextRetriever.

        Returns:
            StructuredContext object containing formatted context, metadata, and sufficiency flag.
        """
        if not chunks:
            return StructuredContext(
                formatted_context="",
                evidence_blocks=[],
                is_sufficient=False,
                max_score=0.0,
            )

        evidence_blocks: List[EvidenceBlock] = []
        context_parts: List[str] = []
        max_score = max(c.score for c in chunks)

        for idx, chunk in enumerate(chunks, start=1):
            block = EvidenceBlock(
                evidence_id=idx,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                paper=chunk.source_filename,
                page=chunk.page_number,
                extraction_method=chunk.extraction_method,
                score=chunk.score,
                text=chunk.text,
            )
            evidence_blocks.append(block)

            formatted_block = (
                f"[EVIDENCE E{idx}]\n"
                f"Paper: {block.paper}\n"
                f"Page: {block.page}\n"
                f"Method: {block.extraction_method}\n"
                f"Text: {block.text}\n"
            )
            context_parts.append(formatted_block)

        formatted_context = "\n".join(context_parts)
        is_sufficient = len(chunks) > 0 and max_score >= self.min_score_threshold

        return StructuredContext(
            formatted_context=formatted_context,
            evidence_blocks=evidence_blocks,
            is_sufficient=is_sufficient,
            max_score=max_score,
        )
