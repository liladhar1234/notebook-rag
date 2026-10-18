"""POST /upload endpoint (non-blocking).

The file is validated and saved, registered as `processing`, and indexed in a
background task. Poll GET /documents/{doc_id} for the final status.
"""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

from notebook_rag.api.deps import (
    get_chat_repository,
    get_current_user,
    get_dense_embedder,
    get_document_repository,
    get_sparse_embedder,
    get_vector_store,
)
from notebook_rag.api.schemas import UploadResponse
from notebook_rag.config import get_settings
from notebook_rag.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    UNSUPPORTED_HINTS,
    looks_like_text,
)
from notebook_rag.ingestion.pipeline import process_document
from notebook_rag.registry.repository import STATUS_PROCESSING

router = APIRouter()

_READ_CHUNK = 1024 * 1024  # 1 MiB


@router.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    chat_id: str = Form(...),
    user=Depends(get_current_user),
    store=Depends(get_vector_store),
    dense=Depends(get_dense_embedder),
    sparse=Depends(get_sparse_embedder),
    repo=Depends(get_document_repository),
    chats=Depends(get_chat_repository),
):
    """Accept any readable file (PDF, Office, sheets, images, HTML, text/code...) and index it in the background."""
    settings = get_settings()

    # The chat must be yours (a new id is created for you; someone else's id -> 404).
    if await chats.ensure(user.user_id, chat_id) is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")

    # Keep only the base name: never trust a client-supplied path.
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix in UNSUPPORTED_HINTS:
        raise HTTPException(status_code=415, detail=UNSUPPORTED_HINTS[suffix])

    if await repo.count_by_chat(chat_id) >= settings.max_docs_per_chat:
        raise HTTPException(
            status_code=409,
            detail=f"Limit reached: at most {settings.max_docs_per_chat} documents per chat. "
            "Delete one to upload another.",
        )

    doc_id = str(uuid.uuid4())
    dest = settings.upload_dir / doc_id / filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Stream to disk, aborting as soon as the size limit is exceeded.
    size = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(_READ_CHUNK):
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is "
                        f"{settings.max_upload_bytes // (1024 * 1024)} MB.",
                    )
                out.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")
        # Unknown extension: accept only if the content is readable text.
        if suffix not in SUPPORTED_EXTENSIONS and not looks_like_text(dest):
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{suffix or 'no extension'}'. "
                "Binary files of this kind can't be read.",
            )
    except BaseException:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise

    await repo.upsert(
        doc_id=doc_id,
        chat_id=chat_id,
        filename=filename,
        file_type=suffix.lstrip(".") or "txt",
        chunk_count=0,
        status=STATUS_PROCESSING,
    )
    background_tasks.add_task(
        process_document, dest, doc_id, chat_id, store, dense, sparse, repo
    )

    return UploadResponse(doc_id=doc_id, filename=filename, status=STATUS_PROCESSING)
