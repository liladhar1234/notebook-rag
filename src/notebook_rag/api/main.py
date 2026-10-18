"""FastAPI application factory with lifespan model loading.

All heavy resources (embedding models, vector store, LLM client,
reranker, multi-query retriever, RAGAS evaluator) are loaded ONCE
during the lifespan startup event and stored on app.state.
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from notebook_rag.config import get_settings
from notebook_rag.embeddings.fastembed_dense import DenseEmbedder
from notebook_rag.embeddings.sparse import SparseEmbedder
from notebook_rag.evaluation.ragas_eval import RagasEvaluator
from notebook_rag.generation.agent import DocumentAgent
from notebook_rag.generation.llm_client import build_llm_client
from notebook_rag.generation.service import GenerationService
from notebook_rag.ingestion.pipeline import reindex_all
from notebook_rag.registry.chats import ChatRepository
from notebook_rag.registry.db import Base, make_engine, make_session_factory, migrate
from notebook_rag.registry.repository import DocumentRepository
from notebook_rag.registry.usage import UsageRepository
from notebook_rag.registry.users import UserRepository
from notebook_rag.retrieval.hybrid import HybridRetriever
from notebook_rag.retrieval.planner import QueryPlanner
from notebook_rag.retrieval.reranker import CrossEncoderReranker
from notebook_rag.retrieval.store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
log = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup; clean up on shutdown."""
    cfg = get_settings()
    cfg.ensure_dirs()

    log.info("Starting up Notebook RAG...")

    # ── Embeddings ────────────────────────────────────────────────────────────
    app.state.dense_embedder = DenseEmbedder(cfg.dense_model)
    app.state.sparse_embedder = SparseEmbedder(cfg.sparse_model)

    # ── Vector store ──────────────────────────────────────────────────────────
    app.state.vector_store = VectorStore(
        path=cfg.qdrant_path,
        dense_dim=app.state.dense_embedder.dimension,
    )

    # ── Base hybrid retriever ─────────────────────────────────────────────────
    app.state.retriever = HybridRetriever(
        store=app.state.vector_store,
        dense_embedder=app.state.dense_embedder,
        sparse_embedder=app.state.sparse_embedder,
    )

    # ── LLM client ────────────────────────────────────────────────────────────
    llm = build_llm_client(
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        anthropic_api_key=cfg.anthropic_api_key,
    )

    # ── Query planner: follow-up rewriting + (optional) multi-query, one LLM call ──
    variants = cfg.multi_query_count if cfg.multi_query_enabled else 0
    log.info("Query planner ready (history=%d messages, multi-query variants=%d)",
             cfg.history_messages, variants)
    planner = QueryPlanner(llm=llm, num_variants=variants)

    # ── Cross-Encoder Re-ranker (optional) ────────────────────────────────────
    reranker = None
    if cfg.reranker_enabled:
        log.info("Cross-encoder reranker ENABLED (model=%s)", cfg.reranker_model)
        reranker = CrossEncoderReranker(model_name=cfg.reranker_model)
        app.state.reranker = reranker

    # ── Generation service ────────────────────────────────────────────────────
    app.state.generation_service = GenerationService(
        retriever=app.state.retriever,
        llm=llm,
        planner=planner,
        reranker=reranker,
        agent=DocumentAgent(llm=llm, retriever=app.state.retriever, reranker=reranker)
        if cfg.agent_enabled else None,
    )

    # ── RAGAS Evaluator ───────────────────────────────────────────────────────
    app.state.ragas_evaluator = RagasEvaluator(
        llm_provider=cfg.llm_provider,
        llm_model=cfg.llm_model,
        llm_api_key=cfg.llm_api_key,
        llm_base_url=cfg.llm_base_url,
        anthropic_api_key=cfg.anthropic_api_key,
        dense_embedder=app.state.dense_embedder,
    )

    # ── SQLite registry ───────────────────────────────────────────────────────
    engine = make_engine(cfg.registry_db_path)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await migrate(conn)
    session_factory = make_session_factory(engine)
    app.state.document_repository = DocumentRepository(session_factory)
    app.state.chat_repository = ChatRepository(session_factory)
    app.state.user_repository = UserRepository(session_factory)
    app.state.usage_repository = UsageRepository(session_factory)

    # Chats that only exist as documents (created before chat history) get a chat row.
    # Chats from before accounts have no owner: the first account created adopts them all.
    for chat_id in sorted({r.chat_id for r in await app.state.document_repository.list_all()}):
        await app.state.chat_repository.ensure_legacy(chat_id, title=chat_id)

    # Anything still 'processing' belongs to a previous run that never finished.
    stale = await app.state.document_repository.fail_stale()
    if stale:
        log.warning("Marked %d interrupted documents as failed", stale)

    # The vector collection was rebuilt (schema change): re-index from saved files.
    reindex_task = None
    if app.state.vector_store.needs_reindex:
        reindex_task = asyncio.create_task(
            reindex_all(
                cfg.upload_dir,
                app.state.vector_store,
                app.state.dense_embedder,
                app.state.sparse_embedder,
                app.state.document_repository,
            )
        )
        app.state.reindex_task = reindex_task  # keep a reference

    log.info("Notebook RAG ready — retrieval: multi_query=%s, reranker=%s",
             cfg.multi_query_enabled, cfg.reranker_enabled)
    yield

    # Shutdown
    log.info("Shutting down...")
    if reindex_task is not None and not reindex_task.done():
        reindex_task.cancel()
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Notebook RAG",
        version="0.2.0",
        description="Hybrid RAG with Multi-Query Retrieval, Cross-Encoder Reranking, and RAGAS Evaluation",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────────────────
    from notebook_rag.api.routes import auth, chats, documents, evaluate, query, upload

    app.include_router(auth.router, tags=["auth"])
    app.include_router(chats.router, tags=["chats"])
    app.include_router(upload.router, tags=["ingestion"])
    app.include_router(query.router, tags=["retrieval"])
    app.include_router(documents.router, tags=["documents"])
    app.include_router(evaluate.router, tags=["evaluation"])

    @app.get("/health", tags=["system"])
    async def health():
        cfg = get_settings()
        return {
            "status": "ok",
            "multi_query": cfg.multi_query_enabled,
            "reranker": cfg.reranker_enabled,
        }

    # ── Serve the UI ──────────────────────────────────────────────────────────
    app.mount("/ui", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(_STATIC_DIR / "index.html")

    return app


app = create_app()
