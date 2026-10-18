"""Whole-document ("overview") questions.

"Summarize this", "explain the document", "what is this image?" don't share words
with any one passage, so search + the evidence gate (correctly) find nothing
relevant. For those questions we skip search and give the model the OPENING
chunks of every document in the chat instead.

Both helpers are pure functions so they are easy to test.
"""
from __future__ import annotations

import re
from collections import OrderedDict

from notebook_rag.models import SearchHit

_NOUN = (
    r"(?:doc\w*|file|pdf|image|picture|photo|screenshot|scan|resume|cv|paper|report|"
    r"page|card|sheet|spreadsheet|table|slides?|deck|presentation|contract|invoice|"
    r"article|notes?|letter|book|chapter)"
)

_PATTERNS = [
    # summarize / summary / summerized (typo-tolerant) / summarise
    re.compile(r"\bsumm?[ae]r\w*", re.I),
    re.compile(r"\b(overview|gist|tl;?dr|key (?:points|takeaways))\b", re.I),
    # explain / describe / "what is" + this|the|my|given + <document noun>
    re.compile(
        rf"\b(?:explain|describe|analy[sz]e|review|tell me about|what(?:'s| is| are| does)|walk me through)\b"
        rf".{{0,30}}\b(?:this|these|the|my|our|given|attached|uploaded)\b.{{0,30}}\b{_NOUN}\b",
        re.I,
    ),
    re.compile(r"^\s*what(?:'s| is)\s+(?:this|it|that)\s*\??\s*$", re.I),
    re.compile(r"\bwhat(?:'s| is| are)\s+(?:in|inside)\s+(?:this|the|it|them)\b", re.I),
    re.compile(r"\bwhat\s+(?:is|does)\s+(?:this|it)\s+(?:about|contain|say|show)\b", re.I),
]


def is_overview_question(text: str) -> bool:
    """True if the user is asking about a document as a whole, not a specific fact."""
    return any(p.search(text or "") for p in _PATTERNS)


def pick_overview_hits(
    chunks: list[SearchHit], per_doc: int, max_total: int
) -> tuple[list[SearchHit], bool]:
    """Choose the opening chunks of each document.

    Returns (hits in reading order grouped by document, truncated) where `truncated`
    is True when some document has more content than was included.
    """
    by_doc: "OrderedDict[str, list[SearchHit]]" = OrderedDict()
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    if not by_doc:
        return [], False

    # Share the budget between documents so one long file can't crowd out the rest.
    share = max(1, min(per_doc, max_total // len(by_doc)))
    picked: list[SearchHit] = []
    truncated = False
    for doc_chunks in by_doc.values():
        doc_chunks.sort(key=lambda c: (c.position, c.page if c.page is not None else 0, str(c.section or "")))
        picked.extend(doc_chunks[:share])
        truncated = truncated or len(doc_chunks) > share
    return picked[:max_total], truncated or len(picked) > max_total
