"""
Table Repository module for storing and retrieving extracted ExtractedTable objects by ID, document, or page.
"""

from typing import List, Dict, Optional
from src.ingestion.table_parser import ExtractedTable


class TableRepository:
    """In-memory table repository for managing extracted structured table objects."""

    _instance: Optional["TableRepository"] = None

    def __init__(self):
        self._tables: Dict[str, ExtractedTable] = {}

    def add_table(self, table: ExtractedTable):
        """Store an ExtractedTable object in the repository."""
        self._tables[table.table_id] = table

    def add_tables(self, tables: List[ExtractedTable]):
        """Store multiple ExtractedTable objects."""
        for t in tables:
            self.add_table(t)

    def get_table(self, table_id: str) -> Optional[ExtractedTable]:
        """Retrieve an ExtractedTable object by its unique table_id."""
        return self._tables.get(table_id)

    def get_tables_by_doc(self, doc_id: str) -> List[ExtractedTable]:
        """Retrieve all tables extracted from a specific document ID."""
        return [t for t in self._tables.values() if t.doc_id == doc_id]

    def get_tables_by_page(self, doc_id: str, page_number: int) -> List[ExtractedTable]:
        """Retrieve all tables extracted from a specific page number of a document."""
        return [t for t in self._tables.values() if t.doc_id == doc_id and t.page_number == page_number]

    def count(self) -> int:
        """Return total count of stored tables."""
        return len(self._tables)

    def clear(self):
        """Clear all stored tables."""
        self._tables.clear()
