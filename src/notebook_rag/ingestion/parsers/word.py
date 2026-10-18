"""DOCX parser (python-docx).

Walks the document body in order so headings, paragraphs, lists and tables stay
where they were. One ParsedPage per heading section (H1-H3); tables become
Markdown table blocks that the chunker keeps whole.
"""
from __future__ import annotations

import re
from pathlib import Path

from notebook_rag.ingestion.parsers._tables import rows_to_markdown
from notebook_rag.models import ParsedPage

_HEADING = re.compile(r"^heading\s*(\d)$", re.IGNORECASE)


class WordParser:
    def __init__(self, max_heading_level: int = 3) -> None:
        self._max_level = max_heading_level

    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(str(path))
        stack: list[tuple[int, str]] = []
        blocks: list[str] = []
        pages: list[ParsedPage] = []

        def flush() -> None:
            text = "\n\n".join(b for b in blocks if b.strip()).strip()
            blocks.clear()
            if text:
                section = " > ".join(t for _, t in stack) or path.stem
                pages.append(ParsedPage(doc_id=doc_id, page=None, section=section, text=text))

        for child in doc.element.body.iterchildren():
            tag = child.tag.split("}")[-1]
            if tag == "p":
                para = Paragraph(child, doc)
                text = para.text.strip()
                if not text:
                    continue
                style = (para.style.name if para.style is not None else "") or ""
                m = _HEADING.match(style)
                if style.lower() == "title":
                    level = 1
                elif m:
                    level = int(m.group(1))
                else:
                    level = 0
                if level and level <= self._max_level:
                    flush()
                    stack[:] = [(l, t) for l, t in stack if l < level]
                    stack.append((level, text))
                elif style.lower().startswith("list"):
                    blocks.append(f"- {text}")
                else:
                    blocks.append(text)
            elif tag == "tbl":
                table = Table(child, doc)
                rows = [[cell.text for cell in row.cells] for row in table.rows]
                md = rows_to_markdown(rows)
                if md:
                    blocks.append(md)
        flush()
        return pages
