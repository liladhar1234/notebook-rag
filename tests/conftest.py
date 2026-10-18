"""Shared pytest fixtures.

The suite runs OFFLINE: embeddings, reranker and LLM are replaced by small fakes,
so no model downloads and no API key are needed. Qdrant (local mode) and SQLite
are real but live in temp directories.
"""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
import zlib
from pathlib import Path

import pytest
import pytest_asyncio
import tiktoken
from httpx import ASGITransport, AsyncClient
from qdrant_client.models import SparseVector

# Set env vars BEFORE importing the app so Settings loads them
os.environ.setdefault("QDRANT_PATH", tempfile.mkdtemp())
os.environ.setdefault("REGISTRY_DB_PATH", str(Path(tempfile.mkdtemp()) / "reg.db"))
os.environ.setdefault("UPLOAD_DIR", tempfile.mkdtemp())
os.environ.setdefault("LLM_API_KEY", "test")
os.environ.setdefault("LLM_PROVIDER", "openai_compat")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_BASE_URL", "http://localhost:9999")

# The chunker loads tiktoken's vocabulary at import time (needs network on first use).
# If that is unavailable, fall back to a whitespace stand-in so the suite still runs.
try:
    tiktoken.get_encoding("cl100k_base")
except Exception:  # noqa: BLE001

    class _WordEncoding:
        def encode(self, text, **_):
            return list(range(len(text.split())))

        def decode(self, tokens):
            return " ".join(f"w{t}" for t in tokens)

    tiktoken.get_encoding = lambda name: _WordEncoding()

from notebook_rag.api.main import create_app  # noqa: E402
from notebook_rag.generation.service import GenerationService  # noqa: E402
from notebook_rag.registry.chats import ChatRepository  # noqa: E402
from notebook_rag.registry.db import Base, make_engine, make_session_factory, migrate  # noqa: E402
from notebook_rag.registry.repository import DocumentRepository  # noqa: E402
from notebook_rag.registry.usage import UsageRepository  # noqa: E402
from notebook_rag.registry.users import UserRepository  # noqa: E402
from notebook_rag.retrieval.hybrid import HybridRetriever  # noqa: E402
from notebook_rag.retrieval.planner import QueryPlanner  # noqa: E402
from notebook_rag.retrieval.store import VectorStore  # noqa: E402

DIM = 64


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeDense:
    """Hashed bag-of-words vectors: similar wording -> similar vector."""

    dimension = DIM

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * DIM
            for w in words(t):
                v[zlib.crc32(w.encode()) % DIM] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out

    def embed_query(self, text):
        return self.embed([text])[0]


class FakeSparse:
    def encode(self, texts):
        out = []
        for t in texts:
            counts: dict[int, float] = {}
            for w in words(t):
                i = zlib.crc32(w.encode()) % 100_000
                counts[i] = counts.get(i, 0.0) + 1.0
            out.append(SparseVector(indices=list(counts), values=list(counts.values())))
        return out

    def encode_query(self, text):
        return self.encode([text])[0]


class FakeReranker:
    """score = 4 * (query words found in the chunk) - 6: unrelated text scores -6."""

    def __init__(self):
        self.seen: list[int] = []  # candidate-pool size of each rerank call
        self.queries: list[str] = []

    def rerank(self, query, hits, top_k=None):
        from dataclasses import replace

        self.seen.append(len(hits))
        self.queries.append(query)
        q = set(words(query))
        scored = [replace(h, score=4.0 * len(q & set(words(h.text))) - 6.0) for h in hits]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k] if top_k else scored


class FakeLLM:
    """Plays both LLM roles. Planner calls (recognised by their system prompt) get
    `plan_reply`; everything else is an answer, recorded in `answers`."""

    def __init__(self):
        self.plan_reply = None  # None -> echo the latest message; str -> that text; callable(user) -> text; Exception -> raise
        self.plan_prompts: list[str] = []
        self.answers: list[str] = []  # user prompts of generation calls

    def complete(self, system: str, user: str) -> str:
        if "search queries for a document search engine" in system:
            self.plan_prompts.append(user)
            reply = self.plan_reply
            if isinstance(reply, Exception):
                raise reply
            if callable(reply):
                return reply(user)
            if isinstance(reply, str):
                return reply
            latest = re.search(r"Latest user message: (.*)", user).group(1)
            return json.dumps({"standalone_question": latest, "search_queries": [f"{latest} details"]})
        self.answers.append(user)
        return "The documents say so [f043841f-3815-4039-a442-d22fdcde71e1, page 1]."


@pytest.fixture(scope="session")
def app():
    return create_app()


PASSWORD = "correct horse battery"


async def register(ac: AsyncClient, email: str = "tester@example.com", password: str = PASSWORD):
    """Sign up through the API; the client's cookie jar keeps the session."""
    resp = await ac.post("/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    return resp.json()


@pytest_asyncio.fixture
async def wired(app, tmp_path, monkeypatch):
    """app.state wired with fakes (ASGITransport skips the real lifespan). Yields the app.

    Every data location points into a temp folder, so running the tests inside the
    Docker container never touches your real data/ (uploads, registry, vectors)."""
    from notebook_rag.config import get_settings

    cfg = get_settings()
    monkeypatch.setattr(cfg, "upload_dir", tmp_path / "uploads")
    monkeypatch.setattr(cfg, "qdrant_path", tmp_path / "qdrant")
    monkeypatch.setattr(cfg, "registry_db_path", tmp_path / "reg.db")
    # No daily limits by default, so tests that ask many questions are not throttled.
    monkeypatch.setattr(cfg, "daily_questions_per_user", 0)
    monkeypatch.setattr(cfg, "daily_questions_global", 0)
    engine = make_engine(tmp_path / "reg.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await migrate(conn)
    sf = make_session_factory(engine)

    dense, sparse = FakeDense(), FakeSparse()
    store = VectorStore(path=tmp_path / "qdrant", dense_dim=DIM)
    retriever = HybridRetriever(store, dense, sparse)
    llm, reranker = FakeLLM(), FakeReranker()

    app.state.dense_embedder, app.state.sparse_embedder = dense, sparse
    app.state.vector_store, app.state.retriever = store, retriever
    app.state.llm, app.state.reranker = llm, reranker
    app.state.generation_service = GenerationService(
        retriever, llm, planner=QueryPlanner(llm, num_variants=2), reranker=reranker
    )
    app.state.document_repository = DocumentRepository(sf)
    app.state.chat_repository = ChatRepository(sf)
    app.state.user_repository = UserRepository(sf)
    app.state.usage_repository = UsageRepository(sf)
    app.state.ragas_evaluator = object()  # /evaluate needs one to resolve its dependencies
    if hasattr(app.state, "login_limiter"):
        del app.state.login_limiter
    yield app
    await engine.dispose()


@pytest_asyncio.fixture
async def anon_client(wired):
    """A client that is NOT logged in."""
    async with AsyncClient(transport=ASGITransport(app=wired), base_url="http://test") as ac:
        yield ac


@pytest_asyncio.fixture
async def client(wired):
    """A client already logged in as tester@example.com."""
    async with AsyncClient(transport=ASGITransport(app=wired), base_url="http://test") as ac:
        await register(ac)
        yield ac


@pytest_asyncio.fixture
async def other_client(wired, client):
    """A second, separate user (own cookie jar) on the same app. Depends on `client`
    so that tester@example.com is the first account."""
    async with AsyncClient(transport=ASGITransport(app=wired), base_url="http://test") as ac:
        await register(ac, "other@example.com")
        yield ac
