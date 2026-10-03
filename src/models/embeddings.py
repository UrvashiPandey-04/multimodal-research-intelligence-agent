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


class GeminiEmbeddings:
    """API-based embedding generator using Google Gemini API (text-embedding-004)."""

    def __init__(self, model_name: Optional[str] = None):
        from google import genai
        from src.config import get_gemini_api_key

        key = get_gemini_api_key()
        if not key:
            raise ValueError("GEMINI_API_KEY is required for Gemini embeddings provider.")
        self.client = genai.Client(api_key=key)
        self.model_name = model_name or os.getenv("GEMINI_EMBEDDING_MODEL", "text-embedding-004")
        self.embedding_dim = 768

    def embed_text(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self.embedding_dim
        res = self.client.models.embed_content(
            model=self.model_name,
            contents=text,
        )
        return res.embedding.values or [0.0] * self.embedding_dim

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        if not texts:
            return []
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            res = self.client.models.embed_content(
                model=self.model_name,
                contents=batch,
            )
            if hasattr(res, "embeddings") and res.embeddings:
                for emb in res.embeddings:
                    embeddings.append(emb.values)
            elif hasattr(res, "embedding") and res.embedding:
                embeddings.append(res.embedding.values)
        return embeddings


def get_embedding_model(model_name: Optional[str] = None, provider: Optional[str] = None):
    """Factory function to obtain requested embedding model (local BGE or API Gemini)."""
    prov = (provider or os.getenv("EMBEDDING_PROVIDER", "local")).lower()
    if prov == "gemini":
        return GeminiEmbeddings(model_name=model_name)
    return BGEEmbeddings(model_name=model_name)

