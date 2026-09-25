"""
BGE Embeddings module using sentence-transformers (BAAI/bge-small-en-v1.5).
"""

import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


class BGEEmbeddings:
    """Local embedding generator using BAAI/bge-small-en-v1.5 with cached model loading."""

    _instance: Optional["BGEEmbeddings"] = None
    _model: Optional[SentenceTransformer] = None

    def __new__(cls, model_name: Optional[str] = None):
        """Singleton pattern ensuring model is loaded into memory only once."""
        if cls._instance is None:
            cls._instance = super(BGEEmbeddings, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize BGE Embeddings generator.

        Args:
            model_name: Name of sentence-transformers model (default: BAAI/bge-small-en-v1.5).
        """
        if self._model is None:
            name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
            self.model_name = name
            # Load local model
            self._model = SentenceTransformer(name)
            self.embedding_dim = self._model.get_embedding_dimension() or 384

    def embed_text(self, text: str) -> List[float]:
        """
        Generate a normalized 384-dimensional dense embedding for a single string.

        Args:
            text: Input text string.

        Returns:
            List of floats representing vector embedding.
        """
        if not text:
            return [0.0] * self.embedding_dim

        # BGE query instruction prefix is recommended for queries, but for chunk indexing standard embedding is used.
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of strings in batches.

        Args:
            texts: List of input text strings.
            batch_size: Processing batch size.

        Returns:
            List of vector embeddings.
        """
        if not texts:
            return []

        vecs = self._model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return vecs.tolist()


def get_embedding_model(model_name: Optional[str] = None) -> BGEEmbeddings:
    """Helper function to obtain cached BGEEmbeddings singleton instance."""
    return BGEEmbeddings(model_name=model_name)
