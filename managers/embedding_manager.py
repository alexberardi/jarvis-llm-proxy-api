"""Embedding Manager — singleton for generating text embeddings.

Uses sentence-transformers to produce dense vector embeddings. Runs on CPU
independently of the GGUF chat model. Thread-safe via the sentence-transformers
encode() implementation.

Configurable via JARVIS_EMBEDDING_MODEL env var (default: all-MiniLM-L6-v2).
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger("uvicorn")

# Default model: 384 dimensions, ~80MB, ~2ms/sentence on M2 Max
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingManager:
    """Singleton manager for text embedding generation."""

    _instance: Optional["EmbeddingManager"] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._model_name: str = os.getenv("JARVIS_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self._dimension: int = 0

    @classmethod
    def get_instance(cls) -> "EmbeddingManager":
        """Get or create the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the embedding dimension. Loads model if needed."""
        if self._dimension == 0:
            self._ensure_loaded()
        return self._dimension

    def _ensure_loaded(self) -> None:
        """Load the model if not already loaded."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {self._model_name}")
        t0 = time.time()
        self._model = SentenceTransformer(self._model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model loaded in {time.time() - t0:.2f}s "
            f"(dim={self._dimension})"
        )

    def encode(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        """Encode texts into embedding vectors.

        Args:
            texts: List of text strings to embed.
            normalize: If True, L2-normalize embeddings (default for cosine similarity).

        Returns:
            List of embedding vectors (list of floats).
        """
        self._ensure_loaded()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
        )
        return embeddings.tolist()
