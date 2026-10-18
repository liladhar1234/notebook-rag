"""FastAPI dependency injection.

All heavy singletons are stored on `app.state` during lifespan startup.
Route handlers pull them via these typed dependencies.
"""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request

from notebook_rag.embeddings.fastembed_dense import DenseEmbedder
from notebook_rag.embeddings.sparse import SparseEmbedder
from notebook_rag.generation.service import GenerationService
from notebook_rag.registry.chats import ChatRepository
from notebook_rag.registry.repository import DocumentRepository
from notebook_rag.registry.usage import UsageRepository
from notebook_rag.registry.users import UserRecord, UserRepository
from notebook_rag.retrieval.hybrid import HybridRetriever
from notebook_rag.retrieval.store import VectorStore


def get_dense_embedder(request: Request) -> DenseEmbedder:
    return request.app.state.dense_embedder


def get_sparse_embedder(request: Request) -> SparseEmbedder:
    return request.app.state.sparse_embedder


def get_vector_store(request: Request) -> VectorStore:
    return request.app.state.vector_store


def get_retriever(request: Request) -> HybridRetriever:
    return request.app.state.retriever


def get_generation_service(request: Request) -> GenerationService:
    return request.app.state.generation_service


def get_document_repository(request: Request) -> DocumentRepository:
    return request.app.state.document_repository


def get_chat_repository(request: Request) -> ChatRepository:
    return request.app.state.chat_repository


def get_user_repository(request: Request) -> UserRepository:
    return request.app.state.user_repository


def get_usage_repository(request: Request) -> UsageRepository:
    return request.app.state.usage_repository


COOKIE_NAME = "nb_session"


async def get_current_user(
    request: Request,
    users: UserRepository = Depends(get_user_repository),
) -> UserRecord:
    """The logged-in user, or 401. Every data route depends on this."""
    token = request.cookies.get(COOKIE_NAME)
    user = await users.user_for_token(token) if token else None
    if user is None:
        raise HTTPException(status_code=401, detail="Not logged in.")
    return user


def get_reranker(request: Request):
    """Returns CrossEncoderReranker or None if disabled."""
    return getattr(request.app.state, "reranker", None)


def get_ragas_evaluator(request: Request):
    """Returns RagasEvaluator instance."""
    return request.app.state.ragas_evaluator
