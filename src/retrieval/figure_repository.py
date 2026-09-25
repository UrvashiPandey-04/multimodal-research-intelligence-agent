"""
In-memory Figure Repository storage component for ExtractedFigure objects.
"""

from typing import List, Dict, Optional
from src.ingestion.figure_parser import ExtractedFigure


class FigureRepository:
    """Independent storage and retrieval layer for extracted scientific figures."""

    def __init__(self):
        """Initialize empty figure repository."""
        self._figures: Dict[str, ExtractedFigure] = {}
        self._doc_index: Dict[str, List[str]] = {}

    def add_figure(self, figure: ExtractedFigure) -> None:
        """
        Store an extracted figure in repository.

        Args:
            figure: ExtractedFigure dataclass instance.
        """
        self._figures[figure.figure_id] = figure
        if figure.doc_id not in self._doc_index:
            self._doc_index[figure.doc_id] = []
        if figure.figure_id not in self._doc_index[figure.doc_id]:
            self._doc_index[figure.doc_id].append(figure.figure_id)

    def add_figures(self, figures: List[ExtractedFigure]) -> None:
        """Store multiple extracted figures."""
        for fig in figures:
            self.add_figure(fig)

    def get_by_id(self, figure_id: str) -> Optional[ExtractedFigure]:
        """
        Retrieve an extracted figure by figure_id.

        Args:
            figure_id: Deterministic figure identifier string.

        Returns:
            ExtractedFigure instance if found, else None.
        """
        return self._figures.get(figure_id)

    def get_by_doc_id(self, doc_id: str) -> List[ExtractedFigure]:
        """
        Retrieve all extracted figures for a specific document.

        Args:
            doc_id: Document hash identifier.

        Returns:
            List of ExtractedFigure instances.
        """
        fig_ids = self._doc_index.get(doc_id, [])
        return [self._figures[fid] for fid in fig_ids if fid in self._figures]

    def get_by_page(self, doc_id: str, page_number: int) -> List[ExtractedFigure]:
        """
        Retrieve all figures located on a specific page of a document.

        Args:
            doc_id: Document hash identifier.
            page_number: 1-indexed page number.

        Returns:
            List of ExtractedFigure instances on that page.
        """
        doc_figs = self.get_by_doc_id(doc_id)
        return [f for f in doc_figs if f.page_number == page_number]

    def list_figures(self) -> List[ExtractedFigure]:
        """Return all stored figures across all documents."""
        return list(self._figures.values())

    def clear(self) -> None:
        """Clear all stored figures."""
        self._figures.clear()
        self._doc_index.clear()
