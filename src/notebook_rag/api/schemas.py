"""Pydantic v2 request and response schemas for the API layer."""
from __future__ import annotations

from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str = "processing"   # processing | ready | failed
    chunks: int = 0              # known only once status == "ready"
    message: str = "Upload received; indexing in the background."


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_id: str = Field(..., min_length=1)


class SourceChunk(BaseModel):
    doc_id: str
    doc_name: str = ""           # original filename, for display
    page: int | None
    section: str | None
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    gate_passed: bool
    sources: list[SourceChunk]
    hallucinated_citations: list[str] = []
    retrieval_mode: str = "hybrid+rrf"
    standalone_question: str | None = None   # the follow-up resolved against the chat history


# ── Chats ─────────────────────────────────────────────────────────────────────

class ChatInfo(BaseModel):
    chat_id: str
    title: str
    created_at: str
    updated_at: str


class ChatsResponse(BaseModel):
    chats: list[ChatInfo]


class MessageInfo(BaseModel):
    role: str                    # user | assistant
    content: str
    sources: list[SourceChunk] = []
    gate_passed: bool | None = None
    retrieval_mode: str | None = None
    created_at: str


class MessagesResponse(BaseModel):
    chat_id: str
    messages: list[MessageInfo]


class DeleteChatResponse(BaseModel):
    chat_id: str
    deleted: bool
    documents_deleted: int


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    chunk_count: int
    created_at: str
    status: str = "ready"        # processing | ready | failed
    error: str | None = None
    progress: str | None = None  # live step while status == "processing"


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str


# ── Evaluate (RAGAS) ──────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_id: str | None = None          # required for auto mode
    answer: str | None = None           # supply for manual mode
    contexts: list[str] | None = None   # supply for manual mode
    ground_truth: str | None = None     # optional; enables context_recall


class EvaluateResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    note: str | None = None
