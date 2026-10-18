"""Chat management: POST/GET /chats, GET /chats/{id}/messages, DELETE /chats/{id}.

A chat is an isolated workspace: its own documents, vectors and message history.
Deleting a chat removes all of them. Every route is login-only and sees only the
caller's own chats (someone else's chat id answers 404, same as a missing one).
"""
from __future__ import annotations

import asyncio
import shutil

from fastapi import APIRouter, Depends, HTTPException

from notebook_rag.api.deps import (
    get_chat_repository,
    get_current_user,
    get_document_repository,
    get_vector_store,
)
from notebook_rag.api.schemas import (
    ChatInfo,
    ChatsResponse,
    DeleteChatResponse,
    MessageInfo,
    MessagesResponse,
    SourceChunk,
)
from notebook_rag.config import get_settings
from notebook_rag.registry.chats import ChatRecord, ChatRepository
from notebook_rag.registry.repository import DocumentRepository
from notebook_rag.registry.users import UserRecord
from notebook_rag.retrieval.store import VectorStore

router = APIRouter()


def _info(c: ChatRecord) -> ChatInfo:
    return ChatInfo(chat_id=c.chat_id, title=c.title, created_at=c.created_at, updated_at=c.updated_at)


@router.post("/chats", response_model=ChatInfo, status_code=201)
async def create_chat(
    user: UserRecord = Depends(get_current_user),
    chats: ChatRepository = Depends(get_chat_repository),
):
    """Start a new, empty chat (no documents, no history)."""
    return _info(await chats.create(user.user_id))


@router.get("/chats", response_model=ChatsResponse)
async def list_chats(
    user: UserRecord = Depends(get_current_user),
    chats: ChatRepository = Depends(get_chat_repository),
):
    """Your chats, most recently active first."""
    return ChatsResponse(chats=[_info(c) for c in await chats.list(user.user_id)])


@router.get("/chats/{chat_id}/messages", response_model=MessagesResponse)
async def chat_messages(
    chat_id: str,
    user: UserRecord = Depends(get_current_user),
    chats: ChatRepository = Depends(get_chat_repository),
):
    """The full message history of a chat, with the sources of each answer."""
    if await chats.get_owned(user.user_id, chat_id) is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")
    return MessagesResponse(
        chat_id=chat_id,
        messages=[
            MessageInfo(
                role=m.role,
                content=m.content,
                sources=[SourceChunk(**s) for s in m.sources],
                gate_passed=m.gate_passed,
                retrieval_mode=m.retrieval_mode,
                created_at=m.created_at,
            )
            for m in await chats.messages(chat_id)
        ],
    )


@router.delete("/chats/{chat_id}", response_model=DeleteChatResponse)
async def delete_chat(
    chat_id: str,
    user: UserRecord = Depends(get_current_user),
    chats: ChatRepository = Depends(get_chat_repository),
    docs: DocumentRepository = Depends(get_document_repository),
    store: VectorStore = Depends(get_vector_store),
):
    """Delete a chat together with its documents (vectors + saved files) and messages."""
    if await chats.get_owned(user.user_id, chat_id) is None:
        raise HTTPException(status_code=404, detail=f"Chat '{chat_id}' not found.")

    upload_dir = get_settings().upload_dir
    records = await docs.list_by_chat(chat_id)
    for record in records:
        await asyncio.to_thread(store.delete_by_doc_id, record.doc_id)
        await docs.delete(record.doc_id)
        shutil.rmtree(upload_dir / record.doc_id, ignore_errors=True)
    deleted = await chats.delete(chat_id)
    return DeleteChatResponse(chat_id=chat_id, deleted=deleted, documents_deleted=len(records))
