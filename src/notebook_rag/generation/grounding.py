"""Evidence gate and post-generation citation validator.

Evidence Gate:
    Rejects early (before LLM call) if retrieval is empty or scores are too low.

Citation Validator:
    Extracts citations from the generated answer and verifies they reference
    only chunks that were actually injected into the prompt.
"""
from __future__ import annotations

import re

from notebook_rag.models import Citation, SearchHit

NOT_IN_DOCS = "The requested information is not in your documents."

_CITATION_RE = re.compile(
    r"\["
    r"(?P<doc>[^,\]]+),\s*"
    r"(?:Page\s*(?P<page>\d+)|(?P<section>[^\]]+))"
    r"\]",
    re.IGNORECASE,
)


# Matches bracketed tags that look like "[<uuid>, page 1]" or "[report.pdf, Page 3]".
# Brackets without a page number or UUID (e.g. "[0, 1]") are left alone.
_STRIP_RE = re.compile(
    r"\s*\[[^\]]*?(?:page\s*\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})[^\]]*\]",
    re.IGNORECASE,
)


def strip_citations(text: str) -> str:
    """Remove inline [doc, page N] style tags from a generated answer."""
    return _STRIP_RE.sub("", text)


class EvidenceGateError(Exception):
    """Raised when retrieval evidence is insufficient."""


def check_evidence(hits: list[SearchHit], min_score: float | None = None) -> None:
    """Raise EvidenceGateError if evidence is insufficient to answer.

    Args:
        hits:      Final hits, best first.
        min_score: Minimum acceptable score for the best hit. Pass it only when the
                   scores are comparable to it (cross-encoder logits). None means
                   "only reject empty retrieval" (e.g. raw RRF scores).
    """
    if not hits:
        raise EvidenceGateError(NOT_IN_DOCS)
    best = max(h.score for h in hits)
    if min_score is not None and best < min_score:
        raise EvidenceGateError(NOT_IN_DOCS)


def extract_citations(text: str) -> list[Citation]:
    """Parse all [DocName, Page X] or [DocName, SectionName] citations from text."""
    citations: list[Citation] = []
    for m in _CITATION_RE.finditer(text):
        doc = m.group("doc").strip()
        page_str = m.group("page")
        section = m.group("section")
        citations.append(
            Citation(
                doc_id=doc,
                page=int(page_str) if page_str else None,
                section=section.strip() if section else None,
                raw=m.group(0),
            )
        )
    return citations


def verify_citations(answer: str, hits: list[SearchHit]) -> list[str]:
    """Return a list of hallucinated citation strings not found in hits.

    A citation is valid if its doc_id AND page/section appear in at least
    one of the injected chunks.
    """
    parsed = extract_citations(answer)
    if not parsed:
        return []

    # Build a set of valid (doc_id, page) and (doc_id, section) pairs
    valid_pairs: set[tuple[str, str]] = set()
    for h in hits:
        if h.page is not None:
            valid_pairs.add((h.doc_id, str(h.page)))
        if h.section:
            valid_pairs.add((h.doc_id, h.section))

    hallucinated: list[str] = []
    for cit in parsed:
        key = (cit.doc_id, str(cit.page) if cit.page else (cit.section or ""))
        if key not in valid_pairs:
            hallucinated.append(cit.raw)

    return hallucinated