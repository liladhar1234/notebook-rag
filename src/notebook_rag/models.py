"""Domain models for Notebook RAG.

All layers import from here; nothing here imports from other app modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class ParsedPage:
    """A single logical page (or section for Markdown/TXT) from a parsed document."""

    doc_id: str
    page: int | None  # None for Markdown/TXT
    section: str | None  # e.g. "Setup > Installation" for Markdown
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A token-bounded, page-confined chunk ready for embedding."""

    chunk_id: UUID
    doc_id: str
    chat_id: str
    page: int | None
    section: str | None
    text: str
    token_count: int
    position: int = 0  # order of the chunk within its document


@dataclass
class SearchHit:
    """A single retrieval result from Qdrant with its fused score."""

    chunk_id: UUID
    doc_id: str
    page: int | None
    section: str | None
    text: str
    score: float
    position: int = 0  # order of the chunk within its document (0 for older indexes)


@dataclass
class Citation:
    """A parsed citation extracted from a generated answer."""

    doc_id: str
    page: int | None  # None for Markdown/TXT docs
    section: str | None
    raw: str  # original citation string as it appeared in the answer
