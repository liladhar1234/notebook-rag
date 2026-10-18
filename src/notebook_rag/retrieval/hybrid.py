"""HybridRetriever: dense + BM25 search for one or many queries -> one RRF pass -> top-k SearchHits."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import replace

from notebook_rag.config import get_settings
from notebook_rag.embeddings.fastembed_dense import DenseEmbedder
from notebook_rag.embeddings.sparse import SparseEmbedder
from notebook_rag.models import SearchHit
from notebook_rag.retrieval.rrf import reciprocal_rank_fusion
from notebook_rag.retrieval.store import VectorStore

log = logging.getLogger(__name__)


class HybridRetriever:
    """Combines dense + sparse Qdrant search via Python-side RRF."""

    def __init__(
        self,
        store: VectorStore,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
    ) -> None:
        self._store = store
        self._dense = dense_embedder
        self._sparse = sparse_embedder

    @property
    def store(self) -> VectorStore:
        return self._store

    def retrieve(
        self,
        queries: str | Sequence[str],
        chat_id: str,
        top_n: int | None = None,
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Run hybrid retrieval for one query (or several variants) and return the top-k fused hits.

        Every query contributes a dense list and a BM25 list; all lists are fused in ONE RRF pass.
        """
        cfg = get_settings()
        top_n = top_n or cfg.retrieval_top_n
        top_k = top_k or cfg.retrieval_top_k
        if isinstance(queries, str):
            queries = [queries]

        ranked_lists: list[list[str]] = []
        for query in queries:
            dense_vec = self._dense.embed_query(query)
            sparse_vec = self._sparse.encode_query(query)
            for results in (
                self._store.search_dense(dense_vec, chat_id, top_n),
                self._store.search_sparse(sparse_vec, chat_id, top_n),
            ):
                ids = [r[0] for r in results]
                if ids:
                    ranked_lists.append(ids)

        if not ranked_lists:
            return []
        log.debug("Hybrid: %d queries -> %d ranked lists", len(queries), len(ranked_lists))

        fused = reciprocal_rank_fusion(ranked_lists, k=cfg.rrf_k)
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]
        score_map = {doc_id: score for doc_id, score in fused}

        # Fetch full payloads
        hits = self._store.fetch_by_ids(top_ids)

        # Attach fused scores and preserve ranking order
        scored = [replace(h, score=score_map.get(str(h.chunk_id), 0.0)) for h in hits]
        scored.sort(key=lambda h: h.score, reverse=True)

        return scored
