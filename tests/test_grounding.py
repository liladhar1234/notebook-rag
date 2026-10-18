"""Unit tests for evidence gate and citation validator."""
from __future__ import annotations

from uuid import UUID

import pytest

from notebook_rag.generation.grounding import (
    EvidenceGateError,
    check_evidence,
    extract_citations,
    verify_citations,
)
from notebook_rag.models import SearchHit


def _hit(doc_id="doc1", page=1, section=None, score=0.5) -> SearchHit:
    return SearchHit(
        chunk_id=UUID("12345678-1234-5678-1234-567812345678"),
        doc_id=doc_id,
        page=page,
        section=section,
        text="Some relevant text.",
        score=score,
    )


# ── Evidence gate ─────────────────────────────────────────────────────────────

def test_gate_passes_with_valid_hits():
    check_evidence([_hit(score=0.5)])


def test_gate_raises_on_empty_hits():
    with pytest.raises(EvidenceGateError):
        check_evidence([])


def test_gate_raises_when_best_score_below_threshold():
    with pytest.raises(EvidenceGateError):
        check_evidence([_hit(score=-9.0), _hit(score=-7.5)], min_score=-5.0)


def test_gate_passes_when_best_score_meets_threshold():
    check_evidence([_hit(score=-9.0), _hit(score=2.3)], min_score=-5.0)


def test_gate_without_threshold_only_rejects_empty():
    # RRF scores (no reranker) are tiny and must not be gated by a logit threshold.
    check_evidence([_hit(score=0.0)])
    check_evidence([_hit(score=0.0)], min_score=None)


# ── Citation extraction ───────────────────────────────────────────────────────

def test_extract_pdf_citation():
    answer = "The answer is here [Report, Page 5]."
    cits = extract_citations(answer)
    assert len(cits) == 1
    assert cits[0].doc_id == "Report"
    assert cits[0].page == 5
    assert cits[0].section is None


def test_extract_markdown_citation():
    answer = "See [MyDoc, Installation] for details."
    cits = extract_citations(answer)
    assert len(cits) == 1
    assert cits[0].doc_id == "MyDoc"
    assert cits[0].section == "Installation"
    assert cits[0].page is None


def test_extract_multiple_citations():
    answer = "Ref1 [DocA, Page 1] and Ref2 [DocB, Setup]."
    cits = extract_citations(answer)
    assert len(cits) == 2


def test_extract_no_citations():
    cits = extract_citations("No citations here.")
    assert cits == []


# ── Citation verification ─────────────────────────────────────────────────────

def test_valid_citation_not_flagged():
    hit = _hit(doc_id="doc1", page=5)
    answer = "See [doc1, Page 5]."
    hallucinated = verify_citations(answer, [hit])
    assert hallucinated == []


def test_hallucinated_citation_flagged():
    hit = _hit(doc_id="doc1", page=5)
    answer = "See [doc1, Page 99]."
    hallucinated = verify_citations(answer, [hit])
    assert len(hallucinated) == 1


def test_no_citations_returns_empty():
    hit = _hit(doc_id="doc1", page=5)
    hallucinated = verify_citations("No citations here.", [hit])
    assert hallucinated == []
