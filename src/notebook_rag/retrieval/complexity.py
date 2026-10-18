"""Router: does this question need the agent loop, or is the plain pipeline enough?

Pure heuristics, so routing costs no LLM call (and no free-tier quota).
Simple questions ("what is the roll number?") stay on the fast retrieve->answer path;
comparisons and multi-part questions go to the agent.
"""
from __future__ import annotations

import re

_DOC = r"(?:documents?|docs?|files?|pdfs?|resumes?|cvs?|papers?|reports?|sheets?|slides?|decks?|them)"

_PATTERNS = [
    re.compile(r"\b(compare|comparison|versus|vs\.?|differences?|differ(?:s|ent)?|contrast|similarit\w+)\b", re.I),
    re.compile(r"\b(?:better|worse|stronger|weaker|cheaper|more expensive)\s+than\b", re.I),
    re.compile(rf"\b(?:across|each|every|all|both)\s+(?:of\s+)?(?:the\s+)?(?:my\s+)?{_DOC}\b", re.I),
    re.compile(rf"\bbetween\s+(?:the\s+)?(?:two\s+)?{_DOC}\b", re.I),
    re.compile(r"\bsumm?[ae]ri[sz]e\s+(?:everything|all)\b", re.I),
    re.compile(r"\b(?:list|find|extract)\s+all\b", re.I),
]
_NUMBERED = re.compile(r"(?:^|\n)\s*(?:\d+[.)]|[-*•])\s+\S", re.M)
_CONNECTIVES = re.compile(r"\b(?:and also|as well as|in addition|additionally|and then|also)\b", re.I)


def is_complex_question(text: str) -> bool:
    """True for comparisons, cross-document and multi-part questions."""
    text = text or ""
    if any(p.search(text) for p in _PATTERNS):
        return True
    if text.count("?") >= 2:
        return True
    if len(_NUMBERED.findall(text)) >= 2:
        return True
    words = len(text.split())
    if words >= 45:
        return True
    return words >= 14 and bool(_CONNECTIVES.search(text))
