"""
Text Chunker module for page-aware deterministic text segmentation with metadata preservation.
"""

import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Any
from src.ingestion.schema import NormalizedPage, NormalizedDocument


@dataclass
class TextChunk:
    """Represents a single semantically split text chunk with full provenance metadata."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int  # 1-indexed
    chunk_index: int  # 0-indexed within page
    text: str
    word_count: int
    char_count: int
    extraction_method: str  # "native" or "ocr"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk object to dictionary format for ChromaDB metadata storage."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "extraction_method": self.extraction_method,
            **self.metadata,
        }


class DocumentChunker:
    """Page-aware, paragraph & sentence structure preserving deterministic text chunker."""

    def __init__(
        self,
        target_chunk_chars: int = 2400,  # ~600-800 tokens
        overlap_chars: int = 400,        # ~100-120 tokens
        separators: List[str] = None,
    ):
        """
        Initialize chunker parameters.

        Args:
            target_chunk_chars: Target character length per chunk (~800 tokens).
            overlap_chars: Overlap character count between consecutive chunks (~120 tokens).
            separators: List of recursive splitting separators (default: ["\n\n", "\n", ". ", " "]).
        """
        self.target_chunk_chars = target_chunk_chars
        self.overlap_chars = max(0, overlap_chars)
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def _generate_chunk_id(self, doc_id: str, page_num: int, chunk_idx: int, text: str) -> str:
        """
        Generate a deterministic chunk ID based on document ID, page number, chunk index, and content hash.
        """
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_p{page_num:03d}_c{chunk_idx:03d}_{text_hash}"

    def chunk_page(self, page: NormalizedPage) -> List[TextChunk]:
        """
        Chunk a single NormalizedPage into deterministic TextChunk objects.

        Args:
            page: NormalizedPage instance from Phase 2.

        Returns:
            List of TextChunk instances preserving page provenance.
        """
        text = page.text.strip()
        if not text:
            return []

        # If text fits within target chunk size, return single chunk
        if len(text) <= self.target_chunk_chars:
            chunk_id = self._generate_chunk_id(page.doc_id, page.page_number, 0, text)
            words = text.split()
            return [
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    chunk_index=0,
                    text=text,
                    word_count=len(words),
                    char_count=len(text),
                    extraction_method=page.extraction_method,
                    metadata={"has_images": page.has_images},
                )
            ]

        # Recursive text splitting logic preserving structure
        split_segments = self._split_text(text, self.target_chunk_chars, self.overlap_chars)
        chunks: List[TextChunk] = []

        for idx, segment in enumerate(split_segments):
            seg_text = segment.strip()
            if not seg_text:
                continue
            chunk_id = self._generate_chunk_id(page.doc_id, page.page_number, idx, seg_text)
            words = seg_text.split()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    source_filename=page.source_filename,
                    page_number=page.page_number,
                    chunk_index=idx,
                    text=seg_text,
                    word_count=len(words),
                    char_count=len(seg_text),
                    extraction_method=page.extraction_method,
                    metadata={"has_images": page.has_images},
                )
            )

        return chunks

    def _split_text(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Helper to recursively split text by natural separators with overlap."""
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para_str = para.strip()
            if not para_str:
                continue

            if len(current_chunk) + len(para_str) + 2 <= max_size:
                current_chunk = f"{current_chunk}\n\n{para_str}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    # Create overlap string from end of current_chunk
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = f"{overlap_text}\n\n{para_str}".strip()
                else:
                    current_chunk = para_str

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def chunk_document(self, doc: NormalizedDocument) -> List[TextChunk]:
        """
        Chunk an entire NormalizedDocument across all pages.

        Args:
            doc: NormalizedDocument instance.

        Returns:
            Flattened list of TextChunk objects.
        """
        all_chunks: List[TextChunk] = []
        for page in doc.pages:
            all_chunks.extend(self.chunk_page(page))
        return all_chunks
