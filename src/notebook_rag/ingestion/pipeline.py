"""Ingestion pipeline: parse -> chunk -> embed -> index -> register.

Uploads are non-blocking: the route saves the file, registers it as
`processing` and returns at once. `process_document` then runs in the
background; the CPU-heavy work happens in a worker thread so the event loop
(queries, polling, other uploads) stays responsive. The UI polls
GET /documents/{doc_id} until the status is `ready` or `failed`.
"""
from __future__ import annotations

import asyncio
import logging
import weakref
from pathlib import Path

from qdrant_client.models import SparseVector

from notebook_rag.embeddings.fastembed_dense import DenseEmbedder
from notebook_rag.embeddings.sparse import SparseEmbedder
from notebook_rag.ingestion import progress
from notebook_rag.ingestion.chunker import chunk_pages
from notebook_rag.ingestion.parsers import get_parser
from notebook_rag.models import Chunk
from notebook_rag.registry.repository import (
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_READY,
    DocumentRepository,
)
from notebook_rag.retrieval.store import VectorStore

log = logging.getLogger(__name__)

# One ingestion at a time per event loop: embedding is CPU-bound and several
# parallel jobs would just slow each other (and queries) down.
_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _SEMAPHORES.get(loop)
    if sem is None:
        sem = _SEMAPHORES[loop] = asyncio.Semaphore(1)
    return sem


class NoExtractableTextError(ValueError):
    """The file parsed but produced no text (e.g. a scanned PDF)."""


def _empty_message(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "No text could be extracted from this PDF, even with OCR. The pages may be blank or unreadable."
    if suffix in _IMAGE_SUFFIXES:
        return (
            "No readable text found in this image (OCR). Photos and diagrams without "
            "text can't be indexed yet."
        )
    return "No extractable text found in this file."


_EMBED_BATCH = 32  # chunks embedded per step; progress is reported after each

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"}


def _prepare(
    path: Path,
    doc_id: str,
    chat_id: str,
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
) -> tuple[list[Chunk], list[list[float]], list[SparseVector], str | None]:
    """Blocking: parse, chunk and embed. Run in a worker thread.

    Returns the chunks, their vectors and an optional warning for partially
    indexed files (e.g. OCR page limit reached).
    """
    progress.update(doc_id, "Reading file…")
    parser = get_parser(path)
    pages = parser.parse(path, doc_id)
    progress.update(doc_id, "Splitting into chunks…")
    log.info("Parsed %d pages/sections from '%s'", len(pages), path.name)

    chunks = chunk_pages(pages, doc_id=doc_id, chat_id=chat_id)
    if not chunks:
        raise NoExtractableTextError(_empty_message(path))
    log.info("Produced %d chunks", len(chunks))

    warnings = list(dict.fromkeys(
        p.metadata["warning"] for p in pages if p.metadata.get("warning")
    ))
    texts = [c.text for c in chunks]
    dense: list = []
    sparse: list = []
    for i in range(0, len(texts), _EMBED_BATCH):
        progress.update(doc_id, f"Embedding {min(i + _EMBED_BATCH, len(texts))}/{len(texts)} chunks…")
        batch = texts[i : i + _EMBED_BATCH]
        dense.extend(dense_embedder.embed(batch))
        sparse.extend(sparse_embedder.encode(batch))
    return chunks, dense, sparse, " ".join(warnings) or None


async def process_document(
    source_path: Path,
    doc_id: str,
    chat_id: str,
    store: VectorStore,
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
    repo: DocumentRepository,
) -> None:
    """Index one saved file and update its registry status. Never raises."""
    progress.update(doc_id, "Waiting in queue…")
    try:
        await _process(source_path, doc_id, chat_id, store, dense_embedder, sparse_embedder, repo)
    finally:
        progress.clear(doc_id)


async def _process(
    source_path: Path,
    doc_id: str,
    chat_id: str,
    store: VectorStore,
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
    repo: DocumentRepository,
) -> None:
    async with _semaphore():
        try:
            chunks, dvecs, svecs, warning = await asyncio.to_thread(
                _prepare, source_path, doc_id, chat_id, dense_embedder, sparse_embedder
            )
            # The user may have deleted the document while it was processing.
            if await repo.get(doc_id) is None:
                log.info("Document %s deleted during ingestion; discarding.", doc_id)
                return
            progress.update(doc_id, "Saving to the search index…")
            await asyncio.to_thread(store.delete_by_doc_id, doc_id)  # idempotent re-index
            await asyncio.to_thread(store.upsert_sync, chunks, dvecs, svecs)
            await repo.set_status(
                doc_id, STATUS_READY, chunk_count=len(chunks), error=warning
            )
            log.info("Document %s ready (%d chunks)", doc_id, len(chunks))
        except NoExtractableTextError as exc:
            log.warning("Document %s: %s", doc_id, exc)
            await repo.set_status(doc_id, STATUS_FAILED, chunk_count=0, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("Ingestion failed for %s", doc_id)
            await repo.set_status(
                doc_id, STATUS_FAILED, chunk_count=0, error=f"Ingestion failed: {exc}"
            )


async def reindex_all(
    upload_dir: Path,
    store: VectorStore,
    dense_embedder: DenseEmbedder,
    sparse_embedder: SparseEmbedder,
    repo: DocumentRepository,
) -> None:
    """Rebuild vectors for every registered document from its saved original.

    Needed after the vector collection is recreated (schema change).
    """
    records = await repo.list_all()
    log.info("Re-indexing %d documents", len(records))
    for rec in records:
        path = upload_dir / rec.doc_id / rec.filename
        if not path.exists():
            await repo.set_status(
                rec.doc_id, STATUS_FAILED, chunk_count=0,
                error="Original file missing; please upload again.",
            )
            continue
        await repo.set_status(rec.doc_id, STATUS_PROCESSING)
        await process_document(
            path, rec.doc_id, rec.chat_id, store, dense_embedder, sparse_embedder, repo
        )
