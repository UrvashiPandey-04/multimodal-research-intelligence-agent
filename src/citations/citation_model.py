"""
Citation Data Models for verified evidence provenance and validation results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Citation:
    """Represents a single verified or rejected citation derived from authoritative evidence."""
    citation_id: str  # e.g., "CIT_1"
    evidence_id: str  # e.g., "E1"
    document_id: str
    source_filename: str
    page_number: int
    chunk_id: str
    extraction_method: str
    validation_status: str  # "valid" or "invalid"
    validation_reason: str
    formatted_citation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary representation."""
        return {
            "citation_id": self.citation_id,
            "evidence_id": self.evidence_id,
            "document_id": self.document_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "extraction_method": self.extraction_method,
            "validation_status": self.validation_status,
            "validation_reason": self.validation_reason,
            "formatted_citation": self.formatted_citation,
        }


@dataclass
class ValidationResult:
    """Container holding validated citations, rejected citations, and overall validation status."""
    citations: List[Citation] = field(default_factory=list)
    invalid_citations: List[Citation] = field(default_factory=list)
    validation_status: str = "none"  # "valid", "partial", "failed", "none"
    validation_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation result to dictionary format."""
        return {
            "validation_status": self.validation_status,
            "valid_count": len(self.citations),
            "invalid_count": len(self.invalid_citations),
            "citations": [c.to_dict() for c in self.citations],
            "invalid_citations": [c.to_dict() for c in self.invalid_citations],
            "validation_messages": self.validation_messages,
        }
