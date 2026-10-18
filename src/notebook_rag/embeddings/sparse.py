"""fastembed BM25 sparse encoder.

Produces Qdrant SparseVector objects for hybrid retrieval.
IDF statistics are applied server-side by Qdrant: the collection's sparse
vector is configured with `modifier=Modifier.IDF`. The encoder only produces
TF weights (documents) or presence weights (queries).
"""
from __future__ import annotations

import logging

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

log = logging.getLogger(__name__)


class SparseEmbedder:
    """Wraps fastembed SparseTextEmbedding (BM25) for sparse vector generation."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        log.info("Loading sparse embedding model: %s", model_name)
        self._model = SparseTextEmbedding(model_name=model_name)
        log.info("Sparse embedding model ready.")

    def encode(self, texts: list[str]) -> list[SparseVector]:
        """Encode a batch of texts into SparseVector objects."""
        if not texts:
            return []
        results: list[SparseVector] = []
        for emb in self._model.embed(texts):
            # fastembed SparseEmbedding has .indices and .values
            results.append(
                SparseVector(
                    indices=emb.indices.tolist(),
                    values=emb.values.tolist(),
                )
            )
        return results

    def encode_query(self, text: str) -> SparseVector:
        """Encode a search query using the model's query-side path."""
        fn = getattr(self._model, "query_embed", None)
        if fn is None:
            return self.encode([text])[0]
        emb = next(iter(fn(text)))
        return SparseVector(indices=emb.indices.tolist(), values=emb.values.tolist())
