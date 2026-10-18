"""Heading-aware Markdown / TXT parser.

Produces one ParsedPage per logical section (H1/H2/H3 boundary).
Plain TXT files are treated as a single section with no headings.
"""
from __future__ import annotations

from pathlib import Path

from markdown_it import MarkdownIt

from notebook_rag.models import ParsedPage


class MarkdownParser:
    """Parse Markdown or plain-text files into section-based ParsedPages."""

    def __init__(self, max_heading_level: int = 3) -> None:
        self._md = MarkdownIt()
        self._max_heading_level = max_heading_level

    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]:
        raw = path.read_text(encoding="utf-8", errors="replace")

        # Plain TXT → single section
        if path.suffix.lower() == ".txt":
            return [
                ParsedPage(
                    doc_id=doc_id,
                    page=None,
                    section=path.stem,
                    text=raw.strip(),
                )
            ]

        return self._parse_markdown(raw, doc_id)

    def _parse_markdown(self, raw: str, doc_id: str) -> list[ParsedPage]:
        tokens = self._md.parse(raw)
        pages: list[ParsedPage] = []

        # heading stack: list of (level, title) in current nesting
        heading_stack: list[tuple[int, str]] = []
        current_lines: list[str] = []
        lines = raw.splitlines(keepends=True)
        line_idx = 0  # current source line pointer (1-indexed for display)

        def flush(section_path: str) -> None:
            text = "".join(current_lines).strip()
            if text:
                pages.append(
                    ParsedPage(
                        doc_id=doc_id,
                        page=None,
                        section=section_path or doc_id,
                        text=text,
                    )
                )
            current_lines.clear()

        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.type == "heading_open" and int(tok.tag[1]) <= self._max_heading_level:
                level = int(tok.tag[1])
                # Inline token is next
                inline_tok = tokens[i + 1] if i + 1 < len(tokens) else None
                heading_text = inline_tok.content if inline_tok else ""

                # Flush current section before starting new one
                section_path = " > ".join(t for _, t in heading_stack)
                flush(section_path)

                # Pop stack back to parent level
                heading_stack[:] = [(l, t) for l, t in heading_stack if l < level]
                heading_stack.append((level, heading_text))
                i += 3  # heading_open, inline, heading_close
                continue

            # Accumulate raw source lines for each TOP-LEVEL block only (nested
            # tokens such as list items are already inside their parent's range),
            # separated by a blank line so the chunker can see block boundaries.
            if tok.map and tok.level == 0:
                start, end = tok.map
                current_lines.extend(lines[start:end])
                if current_lines and not current_lines[-1].endswith("\n"):
                    current_lines[-1] += "\n"
                current_lines.append("\n")
            i += 1

        # Flush final section
        section_path = " > ".join(t for _, t in heading_stack)
        flush(section_path)

        return pages
