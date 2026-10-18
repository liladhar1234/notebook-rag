"""Application configuration loaded from environment / .env file.

All tuneable parameters live here. Import `get_settings()` everywhere else.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Embeddings ────────────────────────────────────────────────────────────
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"

    # ── Storage ───────────────────────────────────────────────────────────────
    qdrant_path: Path = Path("./data/qdrant")
    registry_db_path: Path = Path("./data/registry.db")
    upload_dir: Path = Path("./data/uploads")

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_n: int = Field(30, ge=1, le=200)
    retrieval_top_k: int = Field(6, ge=1, le=50)
    rrf_k: int = Field(60, ge=1)
    # Candidates handed to the cross-encoder; the final context is cut to top_k afterwards.
    rerank_candidates: int = Field(30, ge=1, le=200)

    # ── Multi-Query Retrieval ─────────────────────────────────────────────────
    multi_query_enabled: bool = True
    multi_query_count: int = Field(3, ge=1, le=5)

    # ── Re-ranking ────────────────────────────────────────────────────────────
    reranker_enabled: bool = True
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    # Evidence gate: reject if the best reranker logit is below this value.
    # ms-marco logits: clearly relevant passages score > 0, unrelated ones about -8 to -11.
    # Tune with POST /evaluate on your own documents. Only used when the reranker is on.
    min_rerank_score: float = -5.0

    # Whole-document questions ("summarize this"): opening chunks per document / in total.
    overview_chunks_per_doc: int = Field(6, ge=1, le=30)
    overview_max_chunks: int = Field(12, ge=1, le=60)

    # ── Agent loop (comparisons / multi-part questions) ──────────────────────
    agent_enabled: bool = True
    agent_max_steps: int = Field(4, ge=2, le=8)          # max LLM calls per question; the last must answer
    agent_timeout_s: float = Field(60.0, ge=5.0)         # after this the next call is forced to answer
    agent_max_context_chars: int = Field(18000, ge=2000) # tool-result text fed back to the model

    # ── Conversation ────────────────────────────────────────────────────────────
    # How many recent messages (questions + answers) are used to resolve follow-ups.
    history_messages: int = Field(6, ge=0, le=20)

    # ── Accounts ──────────────────────────────────────────────────────────────
    allow_signup: bool = True                      # open sign-up; set ALLOW_SIGNUP=false to close it
    min_password_length: int = Field(8, ge=6, le=64)
    session_days: int = Field(30, ge=1, le=365)    # how long a login lasts
    cookie_secure: bool = False                    # set COOKIE_SECURE=true when served over HTTPS

    # ── Daily limits (protect the shared Gemini quota); 0 = unlimited ─────────
    daily_questions_per_user: int = Field(10, ge=0)    # DAILY_QUESTIONS_PER_USER
    daily_questions_global: int = Field(100, ge=0)     # DAILY_QUESTIONS_GLOBAL: all users together
    evaluate_cost: int = Field(3, ge=1)                # a RAGAS evaluation counts as this many questions

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_tokens: int = Field(400, ge=50, le=2048)
    chunk_overlap_tokens: int = Field(60, ge=0, le=512)

    # ── OCR / tabular limits ──────────────────────────────────────────────────
    ocr_enabled: bool = True
    ocr_min_chars: int = Field(25, ge=0)         # PDF page with fewer chars is OCR'd
    ocr_dpi: int = Field(200, ge=72, le=400)
    max_pdf_ocr_pages: int = Field(40, ge=0)     # OCR is slow on CPU; cap per file
    max_sheet_rows: int = Field(20000, ge=10)

    # ── Upload limits ─────────────────────────────────────────────────────────
    max_upload_bytes: int = Field(20 * 1024 * 1024, ge=1)
    max_docs_per_chat: int = Field(5, ge=1)

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = "openai_compat"  # "openai_compat" | "anthropic"
    llm_model: str = "gemini-3.8-flash"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: str = ""
    anthropic_api_key: str = ""

    def ensure_dirs(self) -> None:
        """Create data directories if they don't exist."""
        for p in (self.qdrant_path, self.upload_dir):
            p.mkdir(parents=True, exist_ok=True)
        self.registry_db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
