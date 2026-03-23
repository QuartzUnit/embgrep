"""fastembed wrapper for embgrep — lazy-loading embedding model."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


class Embedder:
    """Wrapper around fastembed TextEmbedding with lazy initialization.

    The ONNX model is downloaded and loaded only on first use.
    """

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model_name = model
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> TextEmbedding:
        """Lazily initialize the fastembed model."""
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of numpy arrays, each of shape (EMBEDDING_DIM,).
        """
        if not texts:
            return []
        model = self._ensure_model()
        embeddings = list(model.embed(texts))
        return [e.astype(np.float32) for e in embeddings]

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string.

        Args:
            query: The query text to embed.

        Returns:
            Numpy array of shape (EMBEDDING_DIM,).
        """
        model = self._ensure_model()
        embeddings = list(model.embed([query]))
        return embeddings[0].astype(np.float32)
