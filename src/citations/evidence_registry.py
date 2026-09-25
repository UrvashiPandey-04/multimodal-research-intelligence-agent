"""
Authoritative Evidence Registry supporting Text, Table, and Figure evidence modalities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from src.retrieval.vector_store import SearchResult
from src.retrieval.table_retriever import TableSearchResult
from src.retrieval.figure_retriever import FigureSearchResult


@dataclass
class EvidenceRegistryItem:
    """Represents a single registered evidence item (text chunk, structured table, or figure) with authoritative metadata."""
    evidence_id: str  # e.g., "E1", "E2"
    evidence_type: str  # "text", "table", or "figure"
    chunk_id: str  # chunk_id, table_id, or figure_id
    doc_id: str
    source_filename: str
    page_number: int  # 1-indexed
    extraction_method: str
    score: float
    text_snippet: str
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    image_path: Optional[str] = None
    caption: Optional[str] = None

    def is_metadata_valid(self) -> bool:
        """Verify that critical provenance metadata fields are present and valid."""
        if self.evidence_type == "calculation":
            return bool(self.chunk_id and self.text_snippet)
        id_valid = bool(self.chunk_id or self.table_id or self.figure_id)
        return bool(
            id_valid
            and self.doc_id
            and self.source_filename
            and isinstance(self.page_number, int)
            and self.page_number > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert registry item to dictionary representation."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type,
            "chunk_id": self.chunk_id,
            "table_id": self.table_id if self.evidence_type == "table" else None,
            "figure_id": self.figure_id if self.evidence_type == "figure" else None,
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "extraction_method": self.extraction_method,
            "score": round(self.score, 4),
            "text_snippet": self.text_snippet,
            "image_path": self.image_path,
            "caption": self.caption,
        }


class EvidenceRegistry:
    """Authoritative, request-scoped registry mapping evidence IDs (E1, E2, ...) to text chunks, tables, or figures."""

    def __init__(self):
        self._registry: Dict[str, EvidenceRegistryItem] = {}
        self._key_to_id: Dict[str, str] = {}

    def register_chunk(self, chunk: SearchResult) -> EvidenceRegistryItem:
        """Register a retrieved text SearchResult chunk."""
        if chunk.chunk_id in self._key_to_id:
            existing_id = self._key_to_id[chunk.chunk_id]
            return self._registry[existing_id]

        evidence_id = f"E{len(self._registry) + 1}"
        item = EvidenceRegistryItem(
            evidence_id=evidence_id,
            evidence_type="text",
            chunk_id=chunk.chunk_id,
            table_id=None,
            figure_id=None,
            doc_id=chunk.doc_id,
            source_filename=chunk.source_filename,
            page_number=chunk.page_number,
            extraction_method=chunk.extraction_method,
            score=chunk.score,
            text_snippet=chunk.text,
        )

        self._registry[evidence_id] = item
        self._key_to_id[chunk.chunk_id] = evidence_id
        return item

    def register_table(self, table: TableSearchResult) -> EvidenceRegistryItem:
        """Register a retrieved TableSearchResult object."""
        if table.table_id in self._key_to_id:
            existing_id = self._key_to_id[table.table_id]
            return self._registry[existing_id]

        evidence_id = f"E{len(self._registry) + 1}"
        item = EvidenceRegistryItem(
            evidence_id=evidence_id,
            evidence_type="table",
            chunk_id=table.table_id,
            table_id=table.table_id,
            figure_id=None,
            doc_id=table.doc_id,
            source_filename=table.source_filename,
            page_number=table.page_number,
            extraction_method="pdfplumber",
            score=table.score,
            text_snippet=table.markdown_text,
        )

        self._registry[evidence_id] = item
        self._key_to_id[table.table_id] = evidence_id
        return item

    def register_figure(self, figure: FigureSearchResult) -> EvidenceRegistryItem:
        """Register a retrieved FigureSearchResult object."""
        if figure.figure_id in self._key_to_id:
            existing_id = self._key_to_id[figure.figure_id]
            return self._registry[existing_id]

        evidence_id = f"E{len(self._registry) + 1}"
        item = EvidenceRegistryItem(
            evidence_id=evidence_id,
            evidence_type="figure",
            chunk_id=figure.figure_id,
            table_id=None,
            figure_id=figure.figure_id,
            doc_id=figure.doc_id,
            source_filename=figure.source_filename,
            page_number=figure.page_number,
            extraction_method="pymupdf_cropped",
            score=figure.score,
            text_snippet=figure.caption or figure.searchable_text,
            image_path=figure.image_path,
            caption=figure.caption,
        )

        self._registry[evidence_id] = item
        self._key_to_id[figure.figure_id] = evidence_id
        return item

    def register_calculation(self, expression: str, formatted_result: str, result_val: float) -> EvidenceRegistryItem:
        """
        Register a calculated mathematical result.

        Args:
            expression: Evaluated mathematical expression string.
            formatted_result: Formatted string result.
            result_val: Numeric float result.

        Returns:
            EvidenceRegistryItem object.
        """
        calc_key = f"calc_{expression}"
        if calc_key in self._key_to_id:
            existing_id = self._key_to_id[calc_key]
            return self._registry[existing_id]

        evidence_id = f"E{len(self._registry) + 1}"
        item = EvidenceRegistryItem(
            evidence_id=evidence_id,
            evidence_type="calculation",
            chunk_id=calc_key,
            table_id=None,
            figure_id=None,
            doc_id="calculator",
            source_filename="calculator",
            page_number=1,
            extraction_method="calculator_tool",
            score=1.0,
            text_snippet=f"Calculation: {expression} = {formatted_result}",
        )

        self._registry[evidence_id] = item
        self._key_to_id[calc_key] = evidence_id
        return item

    def register_chunks(self, chunks: List[SearchResult]) -> List[EvidenceRegistryItem]:
        """Register multiple text SearchResult chunks."""
        return [self.register_chunk(c) for c in chunks]

    def register_tables(self, tables: List[TableSearchResult]) -> List[EvidenceRegistryItem]:
        """Register multiple TableSearchResult objects."""
        return [self.register_table(t) for t in tables]

    def register_figures(self, figures: List[FigureSearchResult]) -> List[EvidenceRegistryItem]:
        """Register multiple FigureSearchResult objects."""
        return [self.register_figure(f) for f in figures]

    def get_item(self, evidence_id: str) -> Optional[EvidenceRegistryItem]:
        """Retrieve authoritative metadata for a given evidence ID."""
        clean_id = evidence_id.strip().upper().replace("[", "").replace("]", "")
        if clean_id.isdigit():
            clean_id = f"E{clean_id}"
        return self._registry.get(clean_id)

    def has_item(self, evidence_id: str) -> bool:
        """Check if evidence ID exists in this registry."""
        return self.get_item(evidence_id) is not None

    def get_all_items(self) -> List[EvidenceRegistryItem]:
        """Return all registered evidence items in order of registration."""
        return list(self._registry.values())

