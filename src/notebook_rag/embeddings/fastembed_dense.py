"""fastembed-based dense embedder (BAAI/bge-small-en-v1.5).

The TextEmbedding instance is created ONCE at app startup and reused.
"""
from __future__ import annotations

import logging

from fastembed import TextEmbedding

log = logging.getLogger(__name__)


class DenseEmbedder:
    """Wraps fastembed TextEmbedding for dense vector generation."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        log.info("Loading dense embedding model: %s", model_name)
        self._model = TextEmbedding(model_name=model_name)
        log.info("Dense embedding model ready.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of float vectors."""
        if not texts:
            return []
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query (uses the model's query-side path when it has one)."""
        fn = getattr(self._model, "query_embed", None)
        if fn is None:
            return self.embed([text])[0]
        return next(iter(fn(text))).tolist()

    @property
    def dimension(self) -> int:
        """Return embedding dimension (needed to configure Qdrant collection)."""
        probe = self.embed(["probe"])
        return len(probe[0])
