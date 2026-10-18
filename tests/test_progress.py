"""Live indexing progress shown next to a document that is still processing."""
from __future__ import annotations

import asyncio
import io

import pytest

from notebook_rag.ingestion import progress

pytestmark = pytest.mark.asyncio


def test_progress_store_roundtrip():
    progress.update("d1", "Reading page 1/3…")
    assert progress.get("d1") == "Reading page 1/3…"
    progress.clear("d1")
    assert progress.get("d1") is None
    progress.clear("d1")  # clearing twice is harmless


async def test_processing_document_reports_its_progress(client, wired):
    chat = (await client.post("/chats")).json()["chat_id"]
    await wired.state.document_repository.upsert(
        doc_id="doc-live", chat_id=chat, filename="big.pdf", file_type="pdf", chunk_count=0, status="processing"
    )
    progress.update("doc-live", "OCR on page 3/10 (slow)…")
    try:
        one = (await client.get("/documents/doc-live")).json()
        listed = (await client.get("/documents", params={"chat_id": chat})).json()["documents"]
    finally:
        progress.clear("doc-live")
    assert one["status"] == "processing" and one["progress"] == "OCR on page 3/10 (slow)…"
    assert listed[0]["progress"] == "OCR on page 3/10 (slow)…"


async def test_pipeline_reports_steps_and_clears_when_done(client, monkeypatch):
    seen: list[str] = []
    real_update = progress.update
    monkeypatch.setattr(progress, "update", lambda doc_id, msg: (seen.append(msg), real_update(doc_id, msg))[1])

    chat = (await client.post("/chats")).json()["chat_id"]
    text = " ".join(f"Sentence number {i} about pipelines and streaming." for i in range(400))
    resp = await client.post(
        "/upload", files={"file": ("long.txt", io.BytesIO(text.encode()), "text/plain")}, data={"chat_id": chat}
    )
    doc_id = resp.json()["doc_id"]
    for _ in range(200):
        body = (await client.get(f"/documents/{doc_id}")).json()
        if body["status"] != "processing":
            break
        await asyncio.sleep(0.05)

    assert body["status"] == "ready" and body["progress"] is None
    assert progress.get(doc_id) is None                      # cleaned up
    assert seen[0] == "Waiting in queue…"
    assert "Reading file…" in seen and "Splitting into chunks…" in seen
    assert any(m.startswith("Embedding ") and m.endswith("chunks…") for m in seen)
    assert seen.index("Reading file…") < seen.index("Splitting into chunks…") < seen.index("Saving to the search index…")


async def test_failed_document_also_clears_progress(client):
    chat = (await client.post("/chats")).json()["chat_id"]
    resp = await client.post(
        "/upload", files={"file": ("blank.txt", io.BytesIO(b"   \n  "), "text/plain")}, data={"chat_id": chat}
    )
    if resp.status_code != 202:  # whitespace-only files may be rejected up front
        return
    doc_id = resp.json()["doc_id"]
    for _ in range(200):
        body = (await client.get(f"/documents/{doc_id}")).json()
        if body["status"] != "processing":
            break
        await asyncio.sleep(0.05)
    assert body["status"] == "failed" and body["progress"] is None and progress.get(doc_id) is None
