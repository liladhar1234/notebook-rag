"""Integration tests for the FastAPI endpoints.

Uses httpx + ASGITransport to test routes without a live server.
Embedding models and Qdrant are real but isolated to temp directories.
"""
from __future__ import annotations

import asyncio
import io

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def _wait_ready(client, doc_id, timeout=60.0):
    """Poll GET /documents/{id} until the background indexing finishes."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        body = (await client.get(f"/documents/{doc_id}")).json()
        if body["status"] != "processing":
            return body
        assert asyncio.get_event_loop().time() < deadline, "indexing timed out"
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_upload_txt_is_accepted_then_becomes_ready(client):
    content = "This is a test document for notebook RAG testing purposes."
    files = {"file": ("test.txt", io.BytesIO(content.encode()), "text/plain")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-test-1"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["status"] == "processing"
    assert body["filename"] == "test.txt"

    final = await _wait_ready(client, body["doc_id"])
    assert final["status"] == "ready"
    assert final["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_empty_file_rejected(client):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-empty"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_oversized_file_rejected_and_not_kept(client, monkeypatch):
    from notebook_rag.config import get_settings
    monkeypatch.setattr(get_settings(), "max_upload_bytes", 1024)
    files = {"file": ("big.txt", io.BytesIO(b"x" * 5000), "text/plain")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-big"})
    assert resp.status_code == 413
    docs = (await client.get("/documents", params={"chat_id": "chat-big"})).json()
    assert docs["documents"] == []


@pytest.mark.asyncio
async def test_per_chat_document_limit(client, monkeypatch):
    from notebook_rag.config import get_settings
    monkeypatch.setattr(get_settings(), "max_docs_per_chat", 2)
    for i in range(2):
        files = {"file": (f"d{i}.txt", io.BytesIO(b"hello world content"), "text/plain")}
        r = await client.post("/upload", files=files, data={"chat_id": "chat-limit"})
        assert r.status_code == 202
    files = {"file": ("d3.txt", io.BytesIO(b"hello again"), "text/plain")}
    r = await client.post("/upload", files=files, data={"chat_id": "chat-limit"})
    assert r.status_code == 409
    # another chat is unaffected
    files = {"file": ("d4.txt", io.BytesIO(b"hello again"), "text/plain")}
    r = await client.post("/upload", files=files, data={"chat_id": "other-chat"})
    assert r.status_code == 202


@pytest.mark.asyncio
async def test_pdf_without_text_fails_with_clear_error(client):
    import fitz
    pdf = fitz.open()
    pdf.new_page()  # blank page: no extractable text, like a scan
    data = pdf.tobytes()
    files = {"file": ("scan.pdf", io.BytesIO(data), "application/pdf")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-scan"})
    assert resp.status_code == 202
    final = await _wait_ready(client, resp.json()["doc_id"])
    assert final["status"] == "failed"
    assert "OCR" in final["error"]
    assert final["chunk_count"] == 0


@pytest.mark.asyncio
async def test_filename_path_traversal_is_neutralised(client):
    files = {"file": ("../../evil.txt", io.BytesIO(b"harmless text"), "text/plain")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-trav"})
    assert resp.status_code == 202
    assert resp.json()["filename"] == "evil.txt"


@pytest.mark.asyncio
async def test_unknown_binary_file_rejected(client):
    files = {"file": ("blob.bin", io.BytesIO(bytes(range(256)) * 8), "application/octet-stream")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-test-1"})
    assert resp.status_code == 415


@pytest.mark.asyncio
async def test_legacy_office_format_gives_helpful_error(client):
    files = {"file": ("old.doc", io.BytesIO(b"\xd0\xcf\x11\xe0 fake"), "application/msword")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-test-1"})
    assert resp.status_code == 415
    assert ".docx" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_csv_is_indexed(client):
    csv_bytes = b"name,cost\nAlpha,10\nBeta,20\n"
    files = {"file": ("prices.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-csv"})
    assert resp.status_code == 202
    final = await _wait_ready(client, resp.json()["doc_id"])
    assert final["status"] == "ready" and final["chunk_count"] >= 1


@pytest.mark.asyncio
async def test_unknown_extension_with_text_content_is_accepted(client):
    files = {"file": ("notes.weird", io.BytesIO(b"plain readable text here"), "text/plain")}
    resp = await client.post("/upload", files=files, data={"chat_id": "chat-weird"})
    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    resp = await client.get("/documents", params={"chat_id": "nonexistent-chat"})
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client):
    resp = await client.delete("/documents/nonexistent-doc-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_upload_and_list(client):
    """Upload a document then verify it appears in the list with its status."""
    files = {"file": ("sample.txt", io.BytesIO(b"Sample content for list test."), "text/plain")}
    upload_resp = await client.post("/upload", files=files, data={"chat_id": "chat-list-test"})
    assert upload_resp.status_code == 202
    doc_id = upload_resp.json()["doc_id"]
    await _wait_ready(client, doc_id)

    docs = (await client.get("/documents", params={"chat_id": "chat-list-test"})).json()["documents"]
    match = [d for d in docs if d["doc_id"] == doc_id]
    assert match and match[0]["status"] == "ready"


@pytest.mark.asyncio
async def test_delete_removes_document(client):
    files = {"file": ("gone.txt", io.BytesIO(b"to be deleted"), "text/plain")}
    doc_id = (await client.post("/upload", files=files, data={"chat_id": "chat-del"})).json()["doc_id"]
    await _wait_ready(client, doc_id)
    assert (await client.delete(f"/documents/{doc_id}")).status_code == 200
    assert (await client.get(f"/documents/{doc_id}")).status_code == 404
