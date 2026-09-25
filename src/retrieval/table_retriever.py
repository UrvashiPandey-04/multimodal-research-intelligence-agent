"""
Table Retriever module providing semantic retrieval across structured PDF tables via ChromaDB.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from src.ingestion.table_parser import ExtractedTable
from src.models.embeddings import BGEEmbeddings, get_embedding_model
from src.retrieval.vector_store import VectorStore
from src.retrieval.table_repository import TableRepository


@dataclass
class TableSearchResult:
    """Represents a retrieved table search result with similarity score and cell structure metadata."""
    table_id: str
    doc_id: str
    source_filename: str
    page_number: int
    table_index: int
    headers: List[str]
    rows: List[List[str]]
    markdown_text: str
    score: float
    distance: float
    metadata: Dict[str, Any]


class TableRetriever:
    """Semantic retriever for searching scientific tables stored in an isolated ChromaDB collection."""

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        embedding_model: Optional[BGEEmbeddings] = None,
        repository: Optional[TableRepository] = None,
        collection_name: str = "research_paper_tables",
        in_memory: bool = False,
    ):
        """
        Initialize TableRetriever.

        Args:
            vector_store: Dedicated VectorStore instance for tables.
            embedding_model: BGEEmbeddings instance.
            repository: TableRepository instance.
            collection_name: ChromaDB collection name for tables (default: research_paper_tables).
            in_memory: If True, uses ephemeral ChromaDB client for unit tests.
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model or get_embedding_model()
        self.repository = repository or TableRepository()

        if vector_store:
            self.vector_store = vector_store
        else:
            self.vector_store = VectorStore(
                collection_name=self.collection_name,
                in_memory=in_memory,
            )

    def add_tables(self, tables: List[ExtractedTable]) -> int:
        """
        Index extracted tables into TableRepository and ChromaDB vector store.

        Args:
            tables: List of ExtractedTable objects.

        Returns:
            Number of tables successfully indexed.
        """
        if not tables:
            return 0

        # Store in table repository
        self.repository.add_tables(tables)

        # Generate embeddings for table Markdown text
        markdown_texts = [t.markdown_text for t in tables]
        embeddings = self.embedding_model.embed_batch(markdown_texts)

        # Prepare metadata payload for ChromaDB
        ids = [t.table_id for t in tables]
        documents = [t.markdown_text for t in tables]
        metadatas = [
            {
                "table_id": t.table_id,
                "doc_id": t.doc_id,
                "source_filename": t.source_filename,
                "page_number": t.page_number,
                "table_index": t.table_index,
                "headers_json": json.dumps(t.headers),
                "col_count": len(t.headers),
                "row_count": len(t.rows),
            }
            for t in tables
        ]

        self.vector_store.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        return len(tables)

    def retrieve_tables(
        self,
        query: str,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        source_filename: Optional[str] = None,
    ) -> List[TableSearchResult]:
        """
        Retrieve nearest matching tables for a query using semantic similarity search.

        Args:
            query: Natural language query string (e.g. "Which model has the lowest MSE?").
            top_k: Top k tables to return.
            doc_id: Optional document ID filter.
            source_filename: Optional source filename filter.

        Returns:
            List of TableSearchResult objects.
        """
        clean_query = query.strip()
        if not clean_query or self.vector_store.count() == 0:
            return []

        # 1. Embed query locally
        query_vector = self.embedding_model.embed_text(clean_query)

        # 2. Vector search in ChromaDB tables collection
        where_filter: Optional[Dict[str, Any]] = None
        if doc_id:
            where_filter = {"doc_id": doc_id}
        elif source_filename:
            where_filter = {"source_filename": source_filename}

        results = self.vector_store.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.vector_store.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        table_results: List[TableSearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return table_results

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for table_id, markdown_doc, meta, distance in zip(ids, docs, metas, dists):
            similarity = max(0.0, 1.0 - (float(distance) / 2.0))

            # Fetch structured ExtractedTable from repository if available
            stored_table = self.repository.get_table(table_id)

            headers = stored_table.headers if stored_table else json.loads(meta.get("headers_json", "[]"))
            rows = stored_table.rows if stored_table else []

            res = TableSearchResult(
                table_id=table_id,
                doc_id=meta.get("doc_id", ""),
                source_filename=meta.get("source_filename", ""),
                page_number=int(meta.get("page_number", 1)),
                table_index=int(meta.get("table_index", 0)),
                headers=headers,
                rows=rows,
                markdown_text=markdown_doc,
                score=round(similarity, 4),
                distance=round(float(distance), 4),
                metadata=meta,
            )
            table_results.append(res)

        return table_results

    def search(
        self,
        query: str,
        top_k: int = 3,
        doc_id: Optional[str] = None,
        source_filename: Optional[str] = None,
    ) -> List[TableSearchResult]:
        """Alias method for retrieve_tables providing uniform retriever interface."""
        return self.retrieve_tables(query=query, top_k=top_k, doc_id=doc_id, source_filename=source_filename)
