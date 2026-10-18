"""Whole-document question detection and chunk selection (pure logic, no models)."""
from __future__ import annotations

from uuid import uuid4

import pytest

from notebook_rag.models import SearchHit
from notebook_rag.retrieval.overview import is_overview_question, pick_overview_hits


@pytest.mark.parametrize("q", [
    "what is this image", "explain the document", "give me summerized resume of the given resume",
    "Summarize this PDF", "summarise the report", "tl;dr", "what is this?", "what's in this file",
    "describe the uploaded image", "tell me about the document", "give me an overview",
    "what is this document about", "what does this contain",
])
def test_overview_questions_detected(q):
    assert is_overview_question(q)


@pytest.mark.parametrize("q", [
    "what is the roll number?", "which course is this person enrolled in",
    "what is the capital of France", "how much does the pro plan cost",
    "list the skills mentioned", "when does the card expire",
])
def test_specific_questions_not_overview(q):
    assert not is_overview_question(q)


def _hit(doc, pos, page=None):
    return SearchHit(chunk_id=uuid4(), doc_id=doc, page=page, section=None,
                     text=f"{doc}-{pos}", score=0.0, position=pos)


def test_picks_opening_chunks_in_reading_order():
    chunks = [_hit("a", p) for p in (5, 2, 9, 0, 1, 3, 4, 6, 7, 8)]
    hits, truncated = pick_overview_hits(chunks, per_doc=3, max_total=12)
    assert [h.position for h in hits] == [0, 1, 2]
    assert truncated


def test_budget_is_shared_between_documents():
    chunks = [_hit("a", i) for i in range(20)] + [_hit("b", i) for i in range(20)]
    hits, _ = pick_overview_hits(chunks, per_doc=10, max_total=8)
    assert len(hits) == 8
    assert {h.doc_id for h in hits} == {"a", "b"} and sum(h.doc_id == "a" for h in hits) == 4


def test_small_document_is_not_truncated():
    hits, truncated = pick_overview_hits([_hit("a", 0)], per_doc=6, max_total=12)
    assert len(hits) == 1 and not truncated


def test_empty_chat():
    assert pick_overview_hits([], 6, 12) == ([], False)


def test_old_chunks_without_position_fall_back_to_page_order():
    chunks = [_hit("a", 0, page=3), _hit("a", 0, page=1), _hit("a", 0, page=2)]
    hits, _ = pick_overview_hits(chunks, 3, 12)
    assert [h.page for h in hits] == [1, 2, 3]


# ── Service-level behaviour (stubs, no models / API key) ─────────────────────

class _Store:
    def __init__(self, chunks): self._c = chunks
    def list_chat_chunks(self, chat_id): return self._c


class _Retriever:
    """Search finds nothing relevant, like a generic question against an ID card."""
    def __init__(self, chunks): self.store = _Store(chunks)
    def retrieve(self, queries, chat_id, top_n=None, top_k=None): return []


class _LLM:
    def __init__(self): self.prompts = []
    def complete(self, system, user):
        self.prompts.append(user)
        return "It is an ID card."


def _service(chunks):
    from notebook_rag.generation.service import GenerationService
    llm = _LLM()
    return GenerationService(retriever=_Retriever(chunks), llm=llm), llm


def test_overview_question_is_answered_from_opening_chunks():
    svc, llm = _service([_hit("a", 0), _hit("a", 1)])
    out = svc.answer("what is this image", "chat")
    assert out["gate_passed"] and out["retrieval_mode"] == "overview"
    assert len(out["hits"]) == 2 and "OPENING sections" in llm.prompts[0]


def test_specific_question_is_still_refused():
    svc, llm = _service([_hit("a", 0)])
    out = svc.answer("what is the capital of France", "chat")
    assert out["gate_passed"] is False and not llm.prompts


def test_overview_on_empty_chat_is_refused():
    svc, llm = _service([])
    out = svc.answer("summarize this", "chat")
    assert out["gate_passed"] is False and not llm.prompts
