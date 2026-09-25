"""
Semantic vector retrieval layer for scientific figures using BAAI/bge-small-en-v1.5 embeddings and ChromaDB.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import chromadb
from src.models.embeddings import BGEEmbeddings
from src.ingestion.figure_parser import ExtractedFigure
from src.models.vlm_analyzer import FigureAnalysisResult, VLMFigureAnalyzer


@dataclass
class FigureSearchResult:
    """Dataclass representing a semantically retrieved figure search result."""
    figure_id: str
    doc_id: str
    source_filename: str
    page_number: int
    image_path: str
    caption: Optional[str]
    searchable_text: str
    score: float
    analysis: Optional[FigureAnalysisResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert figure search result to dictionary format."""
        return {
            "figure_id": self.figure_id,
            "doc_id": self.doc_id,
            "source_filename": self.source_filename,
            "page_number": self.page_number,
            "image_path": self.image_path,
            "caption": self.caption,
            "searchable_text": self.searchable_text,
            "score": round(self.score, 4),
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }


from chromadb.config import Settings


class FigureRetriever:
    """Isolated ChromaDB retriever for scientific figures in collection 'research_paper_figures'."""

    COLLECTION_NAME = "research_paper_figures"

    def __init__(
        self,
        embedding_model: Optional[BGEEmbeddings] = None,
        chroma_client: Optional[chromadb.ClientAPI] = None,
        in_memory: bool = True,
        persist_dir: Optional[str] = None,
    ):
        """
        Initialize FigureRetriever.

        Args:
            embedding_model: BGEEmbeddings instance.
            chroma_client: Ephemeral or persistent ChromaDB client.
            in_memory: If True, uses EphemeralClient.
            persist_dir: Storage path for persistent database.
        """
        self.embedding_model = embedding_model or BGEEmbeddings()
        if chroma_client:
            self.client = chroma_client
        elif in_memory:
            self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        else:
            p_dir = persist_dir or "./data/chroma"
            self.client = chromadb.PersistentClient(
                path=p_dir,
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._analysis_store: Dict[str, FigureAnalysisResult] = {}

    def build_searchable_representation(
        self,
        figure: ExtractedFigure,
        analysis: Optional[FigureAnalysisResult] = None,
    ) -> str:
        """
        Construct a searchable textual representation from figure metadata and VLM analysis.

        Args:
            figure: ExtractedFigure object.
            analysis: FigureAnalysisResult object.

        Returns:
            Searchable string representation suitable for BGE embedding.
        """
        parts = [
            f"Figure ID: {figure.figure_id}",
            f"Paper: {figure.source_filename}",
            f"Page: {figure.page_number}",
        ]
        if figure.caption:
            parts.append(f"Caption: {figure.caption}")

        if analysis:
            parts.append(f"Figure Summary: {analysis.figure_summary}")
            parts.append(f"Figure Type: {analysis.figure_type}")
            if analysis.observations:
                parts.append(f"Observations: {' '.join(analysis.observations)}")
            if analysis.visible_values:
                parts.append(f"Visible Values: {' '.join(analysis.visible_values)}")

        if figure.surrounding_text:
            parts.append(f"Context: {figure.surrounding_text}")

        return "\n".join(parts)

    def add_figure(
        self,
        figure: ExtractedFigure,
        analysis: Optional[FigureAnalysisResult] = None,
    ) -> None:
        """
        Index an extracted figure into the dedicated ChromaDB collection.

        Args:
            figure: ExtractedFigure dataclass.
            analysis: Optional FigureAnalysisResult object.
        """
        searchable_text = self.build_searchable_representation(figure, analysis)
        vector = self.embedding_model.embed_text(searchable_text)

        metadata = {
            "figure_id": figure.figure_id,
            "doc_id": figure.doc_id,
            "source_filename": figure.source_filename,
            "page_number": figure.page_number,
            "image_path": figure.image_path,
            "caption": figure.caption or "",
            "searchable_text": searchable_text,
        }

        if analysis:
            self._analysis_store[figure.figure_id] = analysis

        self.collection.upsert(
            ids=[figure.figure_id],
            embeddings=[vector],
            documents=[searchable_text],
            metadatas=[metadata],
        )

    def add_figures(
        self,
        figures: List[ExtractedFigure],
        analyzer: Optional[VLMFigureAnalyzer] = None,
    ) -> None:
        """
        Index multiple extracted figures, running VLM analysis if analyzer provided.

        Args:
            figures: List of ExtractedFigure dataclass objects.
            analyzer: Optional VLMFigureAnalyzer instance.
        """
        for fig in figures:
            analysis = analyzer.analyze_figure(fig) if analyzer else None
            self.add_figure(fig, analysis)

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_id_filter: Optional[str] = None,
    ) -> List[FigureSearchResult]:
        """
        Perform semantic vector search for scientific figures.

        Args:
            query: Natural language query string.
            top_k: Number of top results to return.
            doc_id_filter: Optional doc_id to restrict search.

        Returns:
            List of ranked FigureSearchResult objects.
        """
        if self.collection.count() == 0:
            return []

        query_vector = self.embedding_model.embed_text(query)
        where_clause = {"doc_id": doc_id_filter} if doc_id_filter else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            where=where_clause,
        )

        search_results: List[FigureSearchResult] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
        metadatas = results["metadatas"][0] if "metadatas" in results and results["metadatas"] else [{}] * len(ids)
        documents = results["documents"][0] if "documents" in results and results["documents"] else [""] * len(ids)

        for fid, dist, meta, doc_text in zip(ids, distances, metadatas, documents):
            score = max(0.0, 1.0 - float(dist))
            analysis = self._analysis_store.get(fid)

            res = FigureSearchResult(
                figure_id=meta.get("figure_id", fid),
                doc_id=meta.get("doc_id", ""),
                source_filename=meta.get("source_filename", ""),
                page_number=int(meta.get("page_number", 1)),
                image_path=meta.get("image_path", ""),
                caption=meta.get("caption") if meta.get("caption") else None,
                searchable_text=doc_text,
                score=score,
                analysis=analysis,
            )
            search_results.append(res)

        # Sort by similarity score descending
        search_results.sort(key=lambda r: r.score, reverse=True)
        return search_results

    def clear(self) -> None:
        """Clear the ChromaDB figures collection."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._analysis_store.clear()
