"""
Text Retriever module providing high-level semantic search across document chunks.
"""

from typing import List, Optional
from src.models.embeddings import BGEEmbeddings, get_embedding_model
from src.retrieval.vector_store import VectorStore, SearchResult


class TextRetriever:
    """Semantic text retriever coupling BGEEmbeddings and VectorStore."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[BGEEmbeddings] = None,
    ):
        """
        Initialize TextRetriever.

        Args:
            vector_store: VectorStore instance. If None, initializes default persistent store.
            embedding_model: BGEEmbeddings instance. If None, uses cached singleton model.
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_model = embedding_model or get_embedding_model()

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
        source_filename: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Execute semantic similarity search for a natural language query string.

        Args:
            query: User search query string.
            top_k: Number of top relevant chunks to retrieve.
            doc_id: Optional document ID metadata filter.
            source_filename: Optional source filename metadata filter.

        Returns:
            List of SearchResult objects containing ranked chunks and page provenance.
        """
        clean_query = query.strip()
        if not clean_query:
            return []

        # 1. Embed query locally using BGEEmbeddings
        query_vector = self.embedding_model.embed_text(clean_query)

        # 2. Retrieve nearest neighbors from ChromaDB
        results = self.vector_store.search_by_vector(
            query_vector=query_vector,
            top_k=top_k,
            doc_id=doc_id,
            source_filename=source_filename,
        )

        return results

    def search_diversified(
        self,
        query: str,
        top_k: int = 6,
        per_doc_limit: int = 2,
    ) -> List[SearchResult]:
        """
        Perform document-aware diversified text search across multiple papers.
        Prevents a single paper from dominating all top_k results.

        Args:
            query: User search query string.
            top_k: Total number of chunks to return.
            per_doc_limit: Maximum chunks to include per document.

        Returns:
            List of SearchResult objects balanced across documents.
        """
        candidates = self.search(query, top_k=top_k * 3)
        if not candidates:
            return []

        doc_groups: dict = {}
        for c in candidates:
            doc_groups.setdefault(c.doc_id, []).append(c)

        diversified: List[SearchResult] = []
        doc_ids = list(doc_groups.keys())
        idx = 0

        # Round-robin selection across document groups up to per_doc_limit
        counts = {did: 0 for did in doc_ids}
        while len(diversified) < top_k and any(counts[did] < len(doc_groups[did]) and counts[did] < per_doc_limit for did in doc_ids):
            did = doc_ids[idx % len(doc_ids)]
            if counts[did] < len(doc_groups[did]) and counts[did] < per_doc_limit:
                diversified.append(doc_groups[did][counts[did]])
                counts[did] += 1
            idx += 1

        # If still under top_k, fill with remaining highest-scoring candidates
        if len(diversified) < top_k:
            for c in candidates:
                if c not in diversified:
                    diversified.append(c)
                    if len(diversified) >= top_k:
                        break

        return diversified
