"""Live progress text for documents that are being indexed.

Kept in memory (one server process): the background worker writes short messages
such as "Reading page 12/40 (OCR)", and GET /documents reads them while the
document's status is `processing`. Nothing is persisted: a restart marks
interrupted documents as failed anyway.
"""
from __future__ import annotations

_PROGRESS: dict[str, str] = {}


def update(doc_id: str, message: str) -> None:
    """Safe to call from any thread (a single dict assignment)."""
    _PROGRESS[doc_id] = message


def get(doc_id: str) -> str | None:
    return _PROGRESS.get(doc_id)


def clear(doc_id: str) -> None:
    _PROGRESS.pop(doc_id, None)
