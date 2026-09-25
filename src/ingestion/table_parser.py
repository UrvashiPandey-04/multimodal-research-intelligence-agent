"""
Table Parser module using pdfplumber for page-by-page scientific table extraction, cleaning, and Markdown representation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
import pdfplumber


@dataclass
class ExtractedTable:
    """Represents a structured, page-aware scientific table extracted from a PDF."""
    table_id: str  # Deterministic ID e.g. "convlstm_p006_t00"
    doc_id: str
    source_filename: str
    page_number: int  # 1-indexed
    table_index: int  # 0-indexed table order on page
    headers: List[str]
    rows: List[List[str]]
    markdown_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert extracted table to dictionary representation."""
        return {
            "table_id": self.table_id,
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "table_index": self.table_index,
            "headers": self.headers,
            "rows": self.rows,
            "markdown_text": self.markdown_text,
            "metadata": self.metadata,
        }


class TableParser:
    """Scientific table extractor backed by pdfplumber."""

    def __init__(self):
        pass

    def _clean_cell(self, cell: Optional[str]) -> str:
        """Conservatively clean a single table cell without altering numbers or scientific notation."""
        if cell is None:
            return ""
        # Normalize internal line breaks and redundant whitespace
        cleaned = str(cell).replace("\r\n", " ").replace("\n", " ").strip()
        return cleaned

    def _generate_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """
        Generate a clean, searchable Markdown table representation.

        Example:
        | Model | MSE | CSI |
        |---|---|---|
        | ConvLSTM | 0.09 | 0.76 |
        """
        if not headers and not rows:
            return ""

        effective_headers = headers if headers else [f"Col_{i+1}" for i in range(len(rows[0]) if rows else 0)]
        num_cols = len(effective_headers)

        lines: List[str] = []
        # Header row
        header_str = "| " + " | ".join(effective_headers) + " |"
        lines.append(header_str)
        # Separator row
        lines.append("| " + " | ".join(["---"] * num_cols) + " |")

        # Data rows
        for row in rows:
            # Pad or truncate row to match header column count
            padded_row = [self._clean_cell(c) for c in row[:num_cols]]
            if len(padded_row) < num_cols:
                padded_row.extend([""] * (num_cols - len(padded_row)))
            row_str = "| " + " | ".join(padded_row) + " |"
            lines.append(row_str)

        return "\n".join(lines)

    def extract_tables_from_pdf(self, pdf_path: str | Path) -> List[ExtractedTable]:
        """
        Extract all tables page-by-page from a PDF file using pdfplumber.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            List of ExtractedTable objects with deterministic IDs and page provenance.

        Raises:
            FileNotFoundError: If PDF file does not exist.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at path: {pdf_path}")

        filename = path.name
        doc_id = path.stem
        extracted_tables: List[ExtractedTable] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    page_num = page_idx + 1  # 1-indexed

                    # Detect tables on page using pdfplumber grid analysis
                    raw_tables = page.extract_tables() or []

                    for table_idx, raw_table in enumerate(raw_tables):
                        if not raw_table or len(raw_table) == 0:
                            continue

                        # Clean cell grid
                        cleaned_grid: List[List[str]] = []
                        for row in raw_table:
                            cleaned_row = [self._clean_cell(c) for c in row]
                            # Include row if it contains at least one non-empty cell
                            if any(c != "" for c in cleaned_row):
                                cleaned_grid.append(cleaned_row)

                        if not cleaned_grid:
                            continue

                        # Determine headers and data rows
                        headers = cleaned_grid[0]
                        rows = cleaned_grid[1:] if len(cleaned_grid) > 1 else []

                        # Generate Markdown representation
                        markdown_text = self._generate_markdown(headers, rows)

                        # Generate deterministic Table ID
                        table_id = f"{doc_id}_p{page_num:03d}_t{table_idx:02d}"

                        extracted_table = ExtractedTable(
                            table_id=table_id,
                            doc_id=doc_id,
                            source_filename=filename,
                            page_number=page_num,
                            table_index=table_idx,
                            headers=headers,
                            rows=rows,
                            markdown_text=markdown_text,
                            metadata={
                                "col_count": len(headers),
                                "row_count": len(rows),
                            },
                        )
                        extracted_tables.append(extracted_table)

        except Exception as e:
            # Record exception gracefully or raise ValueError
            raise ValueError(f"Failed to extract tables from PDF {pdf_path}: {e}") from e

        return extracted_tables
