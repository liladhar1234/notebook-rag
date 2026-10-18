"""Unit tests for the token-aware page-bounded chunker."""
from __future__ import annotations

import os

import pytest

from notebook_rag.models import ParsedPage

# Make sure tiktoken can load without network on CI
os.environ.setdefault("TIKTOKEN_CACHE_DIR", "/tmp/tiktoken")


def _make_page(text: str, doc_id: str = "doc1", page: int = 1) -> ParsedPage:
    return ParsedPage(doc_id=doc_id, page=page, section=None, text=text)


def test_single_short_page_produces_one_chunk():
    from notebook_rag.ingestion.chunker import chunk_pages
    page = _make_page("Hello world. This is a short document.")
    chunks = chunk_pages([page], doc_id="doc1", chat_id="chat1")
    assert len(chunks) == 1
    assert chunks[0].page == 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].chat_id == "chat1"


def test_long_text_splits_into_multiple_chunks():
    from notebook_rag.ingestion.chunker import chunk_pages
    # ~600 words → should produce at least 2 chunks with 400-token window
    long_text = " ".join([f"word{i}" for i in range(600)])
    page = _make_page(long_text)
    chunks = chunk_pages([page], doc_id="doc1", chat_id="chat1")
    assert len(chunks) >= 2


def test_chunks_do_not_cross_pages():
    from notebook_rag.ingestion.chunker import chunk_pages
    pages = [
        _make_page("Page one content. " * 30, page=1),
        _make_page("Page two content. " * 30, page=2),
    ]
    chunks = chunk_pages(pages, doc_id="doc1", chat_id="chat1")
    page_nums = {c.page for c in chunks}
    assert page_nums == {1, 2}
    # No chunk should span both pages
    for c in chunks:
        assert c.page in {1, 2}


def test_chunk_ids_are_deterministic():
    from notebook_rag.ingestion.chunker import chunk_pages
    page = _make_page("Determinism test content.")
    chunks1 = chunk_pages([page], doc_id="doc1", chat_id="chat1")
    chunks2 = chunk_pages([page], doc_id="doc1", chat_id="chat1")
    assert [c.chunk_id for c in chunks1] == [c.chunk_id for c in chunks2]


def test_chunk_ids_differ_for_different_docs():
    from notebook_rag.ingestion.chunker import chunk_pages
    page = _make_page("Same text.")
    chunks1 = chunk_pages([page], doc_id="doc1", chat_id="chat1")
    chunks2 = chunk_pages([page], doc_id="doc2", chat_id="chat1")
    assert chunks1[0].chunk_id != chunks2[0].chunk_id


def test_all_chunks_have_correct_metadata():
    from notebook_rag.ingestion.chunker import chunk_pages
    page = _make_page("Test content.")
    chunks = chunk_pages([page], doc_id="testdoc", chat_id="testchat")
    for c in chunks:
        assert c.doc_id == "testdoc"
        assert c.chat_id == "testchat"
        assert c.token_count > 0
        assert c.text.strip()


# ── Structure-aware behaviour ─────────────────────────────────────────────────

def _table(rows: int, cols: int = 3) -> str:
    head = "| " + " | ".join(f"col{c}" for c in range(cols)) + " |"
    sep = "|" + "|".join(["---"] * cols) + "|"
    body = [
        "| " + " | ".join(f"r{r}c{c}" for c in range(cols)) + " |" for r in range(rows)
    ]
    return "\n".join([head, sep, *body])


def test_small_table_is_never_cut_mid_row():
    from notebook_rag.ingestion.chunker import chunk_pages
    text = "Intro paragraph. " * 40 + "\n\n" + _table(10) + "\n\n" + "Outro text. " * 40
    chunks = chunk_pages([_make_page(text)], doc_id="d", chat_id="c")
    table_chunks = [c for c in chunks if "col0" in c.text]
    assert len(table_chunks) == 1
    for r in range(10):
        assert f"r{r}c0" in table_chunks[0].text
    # every line of the table chunk is a complete row
    for line in table_chunks[0].text.splitlines():
        assert line.startswith("|") and line.rstrip().endswith("|")


def test_table_gets_its_own_chunk_apart_from_prose():
    from notebook_rag.ingestion.chunker import chunk_pages
    text = "Some prose before.\n\n" + _table(3) + "\n\nSome prose after."
    chunks = chunk_pages([_make_page(text)], doc_id="d", chat_id="c")
    assert any(c.text.startswith("|") for c in chunks)
    assert all(not ("Some prose" in c.text and "col0" in c.text) for c in chunks)


def test_huge_table_splits_on_rows_and_repeats_header():
    from notebook_rag.ingestion.chunker import chunk_pages
    chunks = chunk_pages([_make_page(_table(400, cols=4))], doc_id="d", chat_id="c")
    assert len(chunks) >= 2
    for c in chunks:
        lines = c.text.splitlines()
        assert lines[0].startswith("| col0")        # header repeated
        assert lines[1].startswith("|---")
        assert all(l.startswith("|") and l.endswith("|") for l in lines)
    # no row lost or duplicated
    body = [l for c in chunks for l in c.text.splitlines()[2:]]
    assert len(body) == 400


def test_paragraphs_are_packed_whole_not_cut_mid_sentence():
    from notebook_rag.ingestion.chunker import chunk_pages
    paras = [(f"Paragraph {i}. " + "word " * 80).strip() for i in range(12)]
    chunks = chunk_pages([_make_page("\n\n".join(paras))], doc_id="d", chat_id="c")
    assert len(chunks) >= 2
    for c in chunks:
        for part in c.text.split("\n\n"):
            assert part in paras  # every piece is an intact original paragraph


def test_empty_page_text_produces_no_chunks():
    from notebook_rag.ingestion.chunker import chunk_pages
    assert chunk_pages([_make_page("   \n\n  ")], doc_id="d", chat_id="c") == []
