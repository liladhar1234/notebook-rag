"""POST /query endpoint."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from notebook_rag.api.deps import (
    get_chat_repository,
    get_current_user,
    get_document_repository,
    get_generation_service,
    get_usage_repository,
)
from notebook_rag.api.schemas import QueryRequest, QueryResponse, SourceChunk
from notebook_rag.config import get_settings
from notebook_rag.generation.service import GenerationService
from notebook_rag.registry.chats import ChatRepository
from notebook_rag.registry.repository import DocumentRepository
from notebook_rag.registry.usage import UsageRepository, limit_message
from notebook_rag.registry.users import UserRecord

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    user: UserRecord = Depends(get_current_user),
    svc: GenerationService = Depends(get_generation_service),
    chats: ChatRepository = Depends(get_chat_repository),
    docs: DocumentRepository = Depends(get_document_repository),
    usage: UsageRepository = Depends(get_usage_repository),
):
    """Answer a question from the documents of one chat, using the chat's recent history."""
    if await chats.ensure(user.user_id, body.chat_id) is None:
        raise HTTPException(status_code=404, detail=f"Chat '{body.chat_id}' not found.")
    cfg = get_settings()
    quota = await usage.consume(user.user_id, 1, cfg.daily_questions_per_user, cfg.daily_questions_global)
    if not quota.allowed:
        raise HTTPException(status_code=429, detail=limit_message(quota))

    recent = await chats.recent_messages(body.chat_id, get_settings().history_messages)
    history = [(m.role, m.content) for m in recent]

    names = {d.doc_id: d.filename for d in await docs.list_by_chat(body.chat_id)}
    try:
        # Planning, retrieval and the LLM calls block: keep them off the event loop.
        result = await asyncio.to_thread(
            svc.answer, question=body.question, chat_id=body.chat_id, history=history, doc_names=names
        )
    except Exception as exc:
        await usage.refund(user.user_id, 1, quota.day)  # our failure: don't charge the question
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sources = [
        SourceChunk(
            doc_id=h.doc_id,
            doc_name=names.get(h.doc_id, h.doc_id),
            page=h.page,
            section=h.section,
            text=h.text,
            score=h.score,
        )
        for h in result.get("hits", [])
    ]

    retrieval_mode = result.get("retrieval_mode", "hybrid+rrf")
    await chats.add_exchange(
        body.chat_id,
        question=body.question,
        answer=result["answer"],
        sources=[s.model_dump() for s in sources],
        gate_passed=result["gate_passed"],
        retrieval_mode=retrieval_mode,
    )

    return QueryResponse(
        answer=result["answer"],
        gate_passed=result["gate_passed"],
        sources=sources,
        hallucinated_citations=result.get("hallucinated_citations", []),
        retrieval_mode=retrieval_mode,
        standalone_question=result.get("standalone_question"),
    )
