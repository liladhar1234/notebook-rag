"""Generic text parser: logs, JSON, XML, YAML, source code and unknown text files."""
from __future__ import annotations

import json
from pathlib import Path

from notebook_rag.models import ParsedPage


def decode_bytes(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


class TextParser:
    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]:
        text = decode_bytes(path.read_bytes()).strip()
        if path.suffix.lower() == ".json":
            try:  # pretty-print so the chunker has line structure to work with
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)
            except ValueError:
                pass
        if not text:
            return []
        return [ParsedPage(doc_id=doc_id, page=None, section=path.name, text=text)]
