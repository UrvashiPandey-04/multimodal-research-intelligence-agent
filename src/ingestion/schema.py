"""
Data models and schema definitions for normalized document representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class QualityMetrics:
    """Detailed quality assessment metrics for extracted page text."""
    char_count: int
    word_count: int
    printable_ratio: float
    avg_word_length: float
    is_usable: bool
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "char_count": self.char_count,
            "word_count": self.word_count,
            "printable_ratio": round(self.printable_ratio, 4),
            "avg_word_length": round(self.avg_word_length, 2),
            "is_usable": self.is_usable,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class NormalizedPage:
    """Represents a normalized, page-aware document block ready for chunking."""
    doc_id: str
    source_filename: str
    page_number: int  # 1-indexed
    text: str
    raw_text: str
    extraction_method: str  # "native" or "ocr"
    quality_metrics: QualityMetrics
    has_images: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalized page to dictionary representation."""
        return {
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "text": self.text,
            "raw_text": self.raw_text,
            "extraction_method": self.extraction_method,
            "quality_metrics": self.quality_metrics.to_dict(),
            "has_images": self.has_images,
            "metadata": self.metadata,
        }


@dataclass
class NormalizedDocument:
    """Represents a fully processed document containing normalized pages."""
    doc_id: str
    source_filename: str
    filepath: str
    total_pages: int
    pages: List[NormalizedPage]
    native_page_count: int
    ocr_page_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document object to dictionary format."""
        return {
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "filepath": self.filepath,
            "total_pages": self.total_pages,
            "native_page_count": self.native_page_count,
            "ocr_page_count": self.ocr_page_count,
            "pages": [p.to_dict() for p in self.pages],
            "metadata": self.metadata,
        }
