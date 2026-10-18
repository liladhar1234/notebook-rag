"""GET /documents and DELETE /documents/{doc_id} endpoints."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, Depends, HTTPException, Query

from notebook_rag.api.deps import (
    get_chat_repository,
    get_current_user,
    get_document_repository,
    get_vector_store,
)
from notebook_rag.api.schemas import DeleteResponse, DocumentInfo, DocumentsResponse
from notebook_rag.config import get_settings
from notebook_rag.ingestion import progress
from notebook_rag.registry.chats import ChatRepository
from notebook_rag.registry.repository import DocumentRecord, DocumentRepository
from notebook_rag.registry.users import UserRecord
from notebook_rag.retrieval.store import VectorStore

router = APIRouter()


def _info(r: DocumentRecord) -> DocumentInfo:
    return DocumentInfo(
        doc_id=r.doc_id,
        filename=r.filename,
        file_type=r.file_type,
        chunk_count=r.chunk_count,
        created_at=r.created_at,
        status=r.status or "ready",
        error=r.error,
        progress=progress.get(r.doc_id) if r.status == "processing" else None,
    )


async def _owned_document(doc_id, user, repo, chats) -> DocumentRecord | None:
    """The document only if its chat belongs to the caller (else None -> 404)."""
    record = await repo.get(doc_id)
    if record is None or await chats.get_owned(user.user_id, record.chat_id) is None:
        return None
    return record


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    chat_id: str = Query(..., description="Filter documents by chat session ID"),
    user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_document_repository),
    chats: ChatRepository = Depends(get_chat_repository),
):
    """List all documents for a given chat session."""
    chat = await chats.get(chat_id)
    if chat is not None and chat.user_id != user.user_id:  # someone else's chat
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    records = await repo.list_by_chat(chat_id)
    return DocumentsResponse(documents=[_info(r) for r in records])


@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_document_repository),
    chats: ChatRepository = Depends(get_chat_repository),
):
    """Status of one document (poll this after an upload until ready/failed)."""
    record = await _owned_document(doc_id, user, repo, chats)
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return _info(record)


@router.delete("/documents/{doc_id}", response_model=DeleteResponse)
async def delete_document(
    doc_id: str,
    user: UserRecord = Depends(get_current_user),
    repo: DocumentRepository = Depends(get_document_repository),
    chats: ChatRepository = Depends(get_chat_repository),
    store: VectorStore = Depends(get_vector_store),
):
    """Delete a document and all its associated vectors."""
    record = await _owned_document(doc_id, user, repo, chats)
    if not record:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")

    # Remove from vector store first, then registry
    store.delete_by_doc_id(doc_id)
    deleted = await repo.delete(doc_id)
    shutil.rmtree(get_settings().upload_dir / doc_id, ignore_errors=True)

    return DeleteResponse(
        doc_id=doc_id,
        deleted=deleted,
        message="Document and all its vectors deleted successfully.",
    )
