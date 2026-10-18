"""DocumentParser protocol — all parsers must satisfy this interface."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from notebook_rag.models import ParsedPage


@runtime_checkable
class DocumentParser(Protocol):
    """Converts a file on disk into a flat list of ParsedPage objects."""

    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]: ...
