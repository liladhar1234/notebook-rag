"""Structure-aware, token-bounded chunker.

Rules:
- Chunks NEVER cross page/section boundaries (each ParsedPage is chunked alone).
- Text is split into *blocks* on blank lines. Blocks are packed whole into chunks
  up to ~chunk_tokens, so paragraphs and list items are not cut mid-sentence.
- A Markdown table is an atomic block: it is never sliced mid-row. If it fits in
  `MAX_TABLE_FACTOR x chunk_tokens` it is kept in one chunk; a bigger table is
  split between rows and the header row is repeated in every piece.
- A single plain paragraph larger than the window falls back to overlapping token
  windows (chunk_overlap_tokens).
- A tiny trailing chunk is merged into the previous text chunk when it fits.
- Chunk IDs are deterministic SHA-1-derived UUIDs.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from uuid import UUID

import tiktoken

from notebook_rag.config import get_settings
from notebook_rag.models import Chunk, ParsedPage

log = logging.getLogger(__name__)

# Use cl100k_base (same tokeniser as many modern LLMs) for counting
_ENCODING = tiktoken.get_encoding("cl100k_base")

# A table up to this many windows long is kept as ONE chunk (tables lose meaning when cut).
MAX_TABLE_FACTOR = 3
_BLANK_LINES = re.compile(r"\n\s*\n")


def _derive_chunk_id(doc_id: str, locator: str, index: int) -> UUID:
    """Deterministic UUID from SHA-1 of doc_id + locator + index."""
    raw = f"{doc_id}:{locator}:{index}".encode()
    digest = hashlib.sha1(raw).digest()[:16]  # noqa: S324 – not cryptographic
    # Force version 5 UUID structure
    digest = bytearray(digest)
    digest[6] = (digest[6] & 0x0F) | 0x50  # version 5
    digest[8] = (digest[8] & 0x3F) | 0x80  # variant 1
    return UUID(bytes=bytes(digest))


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


@dataclass
class _Piece:
    text: str
    tokens: int
    is_table: bool = False


def _is_table_block(block: str) -> bool:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    return len(lines) >= 2 and all(ln.lstrip().startswith("|") for ln in lines)


def _split_table(block: str, limit: int) -> list[str]:
    """Split a Markdown table between rows, repeating the header in each part."""
    lines = [ln for ln in block.splitlines() if ln.strip()]
    # header = first row + separator row (|---|---|) when present
    n_header = 2 if len(lines) > 1 and re.fullmatch(r"[\s|:\-]+", lines[1]) else 1
    header, rows = lines[:n_header], lines[n_header:]
    header_text = "\n".join(header)
    header_tokens = _count_tokens(header_text)

    parts: list[str] = []
    current: list[str] = []
    used = header_tokens
    for row in rows:
        row_tokens = _count_tokens(row) + 1
        if current and used + row_tokens > limit:
            parts.append(header_text + "\n" + "\n".join(current))
            current, used = [], header_tokens
        current.append(row)
        used += row_tokens
    if current:
        parts.append(header_text + "\n" + "\n".join(current))
    return parts or [block]


def _split_long_text(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Overlapping token windows for one paragraph that exceeds the window."""
    tokens = _ENCODING.encode(text, disallowed_special=())
    overlap = min(overlap_tokens, chunk_tokens // 2)
    out: list[str] = []
    pos = 0
    while pos < len(tokens):
        end = min(pos + chunk_tokens, len(tokens))
        out.append(_ENCODING.decode(tokens[pos:end]))
        if end == len(tokens):
            break
        pos = end - overlap
    return out


def _to_pieces(text: str, chunk_tokens: int, overlap_tokens: int) -> list[_Piece]:
    """Break page text into atomic pieces no larger than the packing rules allow."""
    pieces: list[_Piece] = []
    for block in _BLANK_LINES.split(text):
        block = block.strip()
        if not block:
            continue
        tokens = _count_tokens(block)
        if _is_table_block(block):
            if tokens <= chunk_tokens * MAX_TABLE_FACTOR:
                pieces.append(_Piece(block, tokens, True))
            else:
                for part in _split_table(block, chunk_tokens * MAX_TABLE_FACTOR // 2):
                    pieces.append(_Piece(part, _count_tokens(part), True))
        elif tokens <= chunk_tokens:
            pieces.append(_Piece(block, tokens))
        else:
            for part in _split_long_text(block, chunk_tokens, overlap_tokens):
                pieces.append(_Piece(part, _count_tokens(part)))
    return pieces


def _pack(pieces: list[_Piece], chunk_tokens: int) -> list[_Piece]:
    """Greedily pack pieces into chunks; tables always start their own chunk."""
    packed: list[_Piece] = []
    cur: list[_Piece] = []
    used = 0

    def flush() -> None:
        nonlocal cur, used
        if cur:
            text = "\n\n".join(p.text for p in cur)
            packed.append(_Piece(text, _count_tokens(text), len(cur) == 1 and cur[0].is_table))
        cur, used = [], 0

    for piece in pieces:
        if piece.is_table:
            flush()
            packed.append(piece)
            continue
        if cur and used + piece.tokens > chunk_tokens:
            flush()
        cur.append(piece)
        used += piece.tokens
    flush()

    # Merge a tiny trailing text chunk into the previous text chunk if it fits.
    if len(packed) >= 2:
        last, prev = packed[-1], packed[-2]
        if (
            not last.is_table
            and not prev.is_table
            and last.tokens < chunk_tokens // 4
            and prev.tokens + last.tokens <= int(chunk_tokens * 1.5)
        ):
            text = prev.text + "\n\n" + last.text
            packed[-2:] = [_Piece(text, _count_tokens(text))]
    return packed


def chunk_pages(pages: list[ParsedPage], doc_id: str, chat_id: str) -> list[Chunk]:
    """Chunk a list of ParsedPages into token-bounded Chunks.

    Pages are processed independently (no cross-page merging).
    """
    settings = get_settings()
    chunk_tokens = settings.chunk_tokens
    overlap_tokens = settings.chunk_overlap_tokens

    all_chunks: list[Chunk] = []
    global_idx = 0

    for parsed_page in pages:
        pieces = _to_pieces(parsed_page.text, chunk_tokens, overlap_tokens)
        packed = _pack(pieces, chunk_tokens)
        locator = (
            str(parsed_page.page)
            if parsed_page.page is not None
            else (parsed_page.section or "0")
        )
        for piece in packed:
            all_chunks.append(
                Chunk(
                    chunk_id=_derive_chunk_id(doc_id, locator, global_idx),
                    doc_id=doc_id,
                    chat_id=chat_id,
                    page=parsed_page.page,
                    section=parsed_page.section,
                    text=piece.text,
                    token_count=piece.tokens,
                    position=global_idx,
                )
            )
            global_idx += 1

    log.debug("Chunked %d pages into %d chunks", len(pages), len(all_chunks))
    return all_chunks
