"""
ChromaDB Vector Store layer for indexing and querying semantic chunk embeddings.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from src.ingestion.chunker import TextChunk

load_dotenv()


@dataclass
class SearchResult:
    """Represents a retrieved vector search result with similarity metadata."""
    chunk_id: str
    doc_id: str
    source_filename: str
    page_number: int
    chunk_index: int
    text: str
    extraction_method: str
    score: float  # Cosine similarity score (0.0 to 1.0)
    distance: float
    metadata: Dict[str, Any]


class VectorStore:
    """Isolated ChromaDB vector database manager for document chunks."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "research_paper_text_chunks",
        in_memory: bool = False,
    ):
        """
        Initialize ChromaDB VectorStore.

        Args:
            persist_dir: Storage path for persistent database.
            collection_name: Default ChromaDB collection name.
            in_memory: If True, uses EphemeralClient (ideal for unit testing).
        """
        self.collection_name = collection_name
        self.in_memory = in_memory

        if in_memory:
            self.client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        else:
            p_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
            Path(p_dir).mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(Path(p_dir).resolve()),
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return total number of chunks stored in the active collection."""
        return self.collection.count()

    def add_chunks(self, chunks: List[TextChunk], embeddings: List[List[float]]) -> int:
        """
        Upsert text chunks and their pre-computed vector embeddings into ChromaDB.
        Uses deterministic chunk_ids to prevent duplicate records upon re-ingestion.

        Args:
            chunks: List of TextChunk objects.
            embeddings: Corresponding list of 384-dim float vectors.

        Returns:
            Number of chunks successfully added/upserted.
        """
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch between chunk count ({len(chunks)}) and embeddings count ({len(embeddings)})")

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "source_filename": c.source_filename,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "extraction_method": c.extraction_method,
                "word_count": c.word_count,
                "char_count": c.char_count,
            }
            for c in chunks
        ]

        # Use upsert to guarantee idempotency and avoid duplicates
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(chunks)

    def search_by_vector(
        self,
        query_vector: List[float],
        top_k: int = 5,
        doc_id: Optional[str] = None,
        source_filename: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform nearest-neighbor similarity search using a query vector.

        Args:
            query_vector: 384-dim float query embedding.
            top_k: Number of top results to retrieve.
            doc_id: Optional metadata filter for specific document ID.
            source_filename: Optional metadata filter for specific source PDF.

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        if self.count() == 0:
            return []

        where_filter: Optional[Dict[str, Any]] = None
        if doc_id:
            where_filter = {"doc_id": doc_id}
        elif source_filename:
            where_filter = {"source_filename": source_filename}

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results: List[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for chunk_id, doc_text, meta, distance in zip(ids, docs, metas, dists):
            # ChromaDB cosine distance range: 0.0 (identical) to 2.0 (opposite)
            # Convert cosine distance to similarity score: similarity = 1 - (distance / 2)
            similarity = max(0.0, 1.0 - (float(distance) / 2.0))

            res = SearchResult(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                source_filename=meta.get("source_filename", ""),
                page_number=int(meta.get("page_number", 1)),
                chunk_index=int(meta.get("chunk_index", 0)),
                text=doc_text,
                extraction_method=meta.get("extraction_method", "native"),
                score=round(similarity, 4),
                distance=round(float(distance), 4),
                metadata=meta,
            )
            search_results.append(res)

        return search_results

    def clear(self):
        """Reset and delete all documents in the active collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
