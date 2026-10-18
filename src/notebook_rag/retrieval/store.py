"""Qdrant vector store wrapper.

Runs in local disk-backed mode (no separate daemon).
Creates the collection on first use with named dense + sparse vectors.

Uses the modern qdrant-client API (>=1.7):
  - Plain dicts for vectors_config / sparse_vectors_config
  - query_points(using=...) instead of search() with NamedVector wrappers
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from uuid import UUID

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    Modifier,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from notebook_rag.models import Chunk, SearchHit

log = logging.getLogger(__name__)

COLLECTION = "notebook_chunks"
DENSE_VEC = "dense"
SPARSE_VEC = "sparse"


class VectorStore:
    """Thin wrapper around QdrantClient for notebook RAG operations.

    Local-mode Qdrant is not safe for concurrent use from several threads, and
    ingestion now runs in worker threads, so every client call takes `_lock`.
    """

    def __init__(self, path: Path, dense_dim: int) -> None:
        self._client = QdrantClient(path=str(path))
        self._dense_dim = dense_dim
        self._lock = threading.RLock()
        # True when an older collection (no IDF modifier) was dropped and rebuilt:
        # the caller must re-index every registered document.
        self.needs_reindex = False
        self._ensure_collection()

    def _sparse_has_idf(self) -> bool:
        info = self._client.get_collection(COLLECTION)
        sparse = info.config.params.sparse_vectors or {}
        cfg = sparse.get(SPARSE_VEC)
        return bool(cfg and cfg.modifier == Modifier.IDF)

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION in existing:
            if self._sparse_has_idf():
                return
            log.warning(
                "Collection '%s' was created without the IDF modifier; "
                "rebuilding it (documents will be re-indexed).", COLLECTION,
            )
            self._client.delete_collection(COLLECTION)
            self.needs_reindex = True
        log.info("Creating Qdrant collection '%s'", COLLECTION)
        self._client.create_collection(
            collection_name=COLLECTION,
            # Plain dict — no VectorsConfig wrapper needed in modern qdrant-client
            vectors_config={
                DENSE_VEC: VectorParams(
                    size=self._dense_dim,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                SPARSE_VEC: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                    # BM25 needs IDF; Qdrant computes it from collection statistics.
                    modifier=Modifier.IDF,
                )
            },
        )
        for key in ("chat_id", "doc_id"):
            try:  # no-op (warning) in local mode, real index on Qdrant server/cloud
                self._client.create_payload_index(
                    COLLECTION, field_name=key, field_schema="keyword"
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("Payload index on %s skipped: %s", key, exc)

    async def upsert(
        self,
        chunks: list[Chunk],
        dense_vecs: list[list[float]],
        sparse_vecs: list[SparseVector],
    ) -> None:
        """Async wrapper kept for API compatibility."""
        self.upsert_sync(chunks, dense_vecs, sparse_vecs)

    def upsert_sync(
        self,
        chunks: list[Chunk],
        dense_vecs: list[list[float]],
        sparse_vecs: list[SparseVector],
    ) -> None:
        """Upsert chunk vectors + payloads into Qdrant (blocking; run in a thread)."""
        points: list[PointStruct] = []
        for chunk, dvec, svec in zip(chunks, dense_vecs, sparse_vecs):
            points.append(
                PointStruct(
                    id=str(chunk.chunk_id),
                    vector={
                        DENSE_VEC: dvec,
                        SPARSE_VEC: svec,
                    },
                    payload={
                        "chunk_id": str(chunk.chunk_id),
                        "doc_id": chunk.doc_id,
                        "chat_id": chunk.chat_id,
                        "page": chunk.page,
                        "section": chunk.section,
                        "text": chunk.text,
                        "position": chunk.position,
                    },
                )
            )
        with self._lock:
            self._client.upsert(collection_name=COLLECTION, points=points)
        log.debug("Upserted %d points", len(points))

    def search_dense(
        self, query_vec: list[float], chat_id: str, top_n: int
    ) -> list[tuple[str, float]]:
        """Return (point_id, score) pairs from dense ANN search."""
        chat_filter = Filter(
            must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]
        )
        with self._lock:
            result = self._client.query_points(
                collection_name=COLLECTION,
                query=query_vec,          # plain list[float] for dense
                using=DENSE_VEC,
                query_filter=chat_filter,
                limit=top_n,
                with_payload=False,
            )
        return [(str(p.id), p.score) for p in result.points]

    def search_sparse(
        self, query_sparse: SparseVector, chat_id: str, top_n: int
    ) -> list[tuple[str, float]]:
        """Return (point_id, score) pairs from sparse BM25 search."""
        chat_filter = Filter(
            must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))]
        )
        with self._lock:
            result = self._client.query_points(
                collection_name=COLLECTION,
                query=query_sparse,       # SparseVector for sparse
                using=SPARSE_VEC,
                query_filter=chat_filter,
                limit=top_n,
                with_payload=False,
            )
        return [(str(p.id), p.score) for p in result.points]

    def fetch_by_ids(self, ids: list[str]) -> list[SearchHit]:
        """Retrieve full payloads for a list of point IDs."""
        if not ids:
            return []
        with self._lock:
            records = self._client.retrieve(
                collection_name=COLLECTION,
                ids=ids,
                with_payload=True,
            )
        hits: list[SearchHit] = []
        for r in records:
            p = r.payload
            hits.append(
                SearchHit(
                    chunk_id=UUID(p["chunk_id"]),
                    doc_id=p["doc_id"],
                    page=p.get("page"),
                    section=p.get("section"),
                    text=p["text"],
                    score=0.0,  # fused score set by HybridRetriever
                    position=p.get("position", 0),
                )
            )
        return hits

    def list_chat_chunks(self, chat_id: str, batch: int = 256) -> list[SearchHit]:
        """Every chunk of every document in a chat (a chat holds only a few documents)."""
        flt = Filter(must=[FieldCondition(key="chat_id", match=MatchValue(value=chat_id))])
        hits: list[SearchHit] = []
        offset = None
        while True:
            with self._lock:
                records, offset = self._client.scroll(
                    collection_name=COLLECTION,
                    scroll_filter=flt,
                    limit=batch,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            for r in records:
                p = r.payload
                hits.append(
                    SearchHit(
                        chunk_id=UUID(p["chunk_id"]),
                        doc_id=p["doc_id"],
                        page=p.get("page"),
                        section=p.get("section"),
                        text=p["text"],
                        score=0.0,
                        position=p.get("position", 0),
                    )
                )
            if offset is None:
                return hits

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all points belonging to a document."""
        with self._lock:
            self._client.delete(
                collection_name=COLLECTION,
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                    )
                ),
            )
