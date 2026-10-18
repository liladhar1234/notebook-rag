"""HTML parser (stdlib only): visible text, H1-H3 sections, tables as Markdown."""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from notebook_rag.ingestion.parsers._tables import rows_to_markdown
from notebook_rag.ingestion.parsers.text import decode_bytes
from notebook_rag.models import ParsedPage

_SKIP = {"script", "style", "noscript", "head", "template", "svg"}
_BLOCK = {"p", "div", "section", "article", "li", "ul", "ol", "br", "tr", "pre",
          "blockquote", "header", "footer", "main", "nav", "aside", "figure", "dd", "dt"}
_HEADINGS = {"h1": 1, "h2": 2, "h3": 3}


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.stack: list[tuple[int, str]] = []
        self.sections: list[tuple[str, str]] = []   # (section path, text)
        self.blocks: list[str] = []
        self.buf: list[str] = []
        self.heading_level = 0
        self.heading_buf: list[str] = []
        # table state (outermost table only)
        self.table_depth = 0
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    # -- helpers
    def _end_block(self) -> None:
        text = " ".join("".join(self.buf).split())
        self.buf = []
        if text:
            self.blocks.append(text)

    def _flush_section(self) -> None:
        self._end_block()
        text = "\n\n".join(self.blocks).strip()
        self.blocks = []
        if text:
            self.sections.append((" > ".join(t for _, t in self.stack), text))

    # -- parser callbacks
    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
            return
        if self.skip:
            return
        if tag == "table":
            self.table_depth += 1
            if self.table_depth == 1:
                self._end_block()
                self.rows = []
            return
        if self.table_depth == 1:
            if tag == "tr":
                self.row = []
            elif tag in ("td", "th"):
                self.cell = []
            return
        if tag in _HEADINGS:
            self._end_block()
            self.heading_level = _HEADINGS[tag]
            self.heading_buf = []
        elif tag in _BLOCK:
            self._end_block()

    def handle_endtag(self, tag):
        if tag in _SKIP:
            self.skip = max(0, self.skip - 1)
            return
        if self.skip:
            return
        if tag == "table" and self.table_depth:
            self.table_depth -= 1
            if self.table_depth == 0:
                if self.row:
                    self.rows.append(self.row)
                    self.row = None
                md = rows_to_markdown(self.rows)
                if md:
                    self.blocks.append(md)
                self.rows = []
            return
        if self.table_depth == 1:
            if tag in ("td", "th") and self.cell is not None and self.row is not None:
                self.row.append(" ".join("".join(self.cell).split()))
                self.cell = None
            elif tag == "tr" and self.row is not None:
                self.rows.append(self.row)
                self.row = None
            return
        if tag in _HEADINGS and self.heading_level:
            title = " ".join("".join(self.heading_buf).split())
            level, self.heading_level = self.heading_level, 0
            if title:
                self._flush_section()
                self.stack[:] = [(l, t) for l, t in self.stack if l < level]
                self.stack.append((level, title))
        elif tag in _BLOCK:
            self._end_block()

    def handle_data(self, data):
        if self.skip:
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.heading_level:
            self.heading_buf.append(data)
        elif self.table_depth == 0:
            self.buf.append(data)


class WebParser:
    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]:
        ex = _Extractor()
        ex.feed(decode_bytes(path.read_bytes()))
        ex.close()
        ex._flush_section()
        return [
            ParsedPage(doc_id=doc_id, page=None, section=sec or path.stem, text=text)
            for sec, text in ex.sections
        ]
