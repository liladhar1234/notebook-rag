"""Query planner: ONE LLM call that prepares everything retrieval needs.

  1. Follow-up rewriting - "and what about its cost?" becomes a self-contained
     question using the recent conversation.
  2. Multi-query expansion - a few differently worded search queries.

Doing both in a single call keeps a follow-up at the same LLM cost as a first
question, which matters on free-tier rate limits. If the call fails, or the
reply can't be parsed, we fall back to the original question: planning is an
enhancement, never a dependency. With no history and expansion disabled the
LLM is not called at all.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Sequence

from notebook_rag.generation.llm_client import LLMClient

log = logging.getLogger(__name__)

History = Sequence[tuple[str, str]]  # (role, content), oldest first

_MAX_TURN_CHARS = 600  # keep the planner prompt small
_MAX_STANDALONE_CHARS = 500

_SYSTEM = (
    "You prepare search queries for a document search engine. "
    "You never answer the question and never add facts of your own."
)

_TEMPLATE = """{conversation}Latest user message: {question}

Reply with ONLY a JSON object with these keys:
- "standalone_question": the latest message rewritten so it makes sense without the \
conversation (replace pronouns and references such as "it", "that", "the second one" using \
the conversation; keep formatting requests such as "as a table"). If it is already \
self-contained, return it unchanged.
{variants_key}
Example: {example}"""

_VARIANTS_KEY = (
    '- "search_queries": {n} alternative search queries for the standalone question, worded '
    "differently from it; include one short keyword-style query."
)


@dataclass(frozen=True)
class QueryPlan:
    standalone: str  # question with references resolved (== question if nothing to resolve)
    queries: list[str]  # what to search with: standalone first, then the rest
    variants: int = 0  # how many LLM-generated variants are in `queries`
    rewritten: bool = False  # True if the standalone differs from the user's message


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _clip(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= _MAX_TURN_CHARS else text[:_MAX_TURN_CHARS] + "…"


def _parse_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data


class QueryPlanner:
    def __init__(self, llm: LLMClient, num_variants: int = 3) -> None:
        self._llm = llm
        self._n = max(0, num_variants)

    def plan(self, question: str, history: History = ()) -> QueryPlan:
        fallback = QueryPlan(standalone=question, queries=[question])
        if not history and self._n == 0:
            return fallback  # nothing to rewrite or expand: skip the LLM call

        try:
            raw = self._llm.complete(_SYSTEM, self._prompt(question, history))
            data = _parse_json(raw)
        except Exception as exc:  # noqa: BLE001 - degrade to plain retrieval
            log.warning("Query planning failed; using the original question only: %s", exc)
            return fallback

        standalone = data.get("standalone_question")
        if not isinstance(standalone, str) or not standalone.strip() or len(standalone) > _MAX_STANDALONE_CHARS:
            standalone = question
        standalone = " ".join(standalone.split())

        raw_variants = data.get("search_queries") if self._n else []
        variants = [v for v in raw_variants if isinstance(v, str)] if isinstance(raw_variants, list) else []

        queries: list[str] = []
        seen: set[str] = set()
        for q in [standalone, question]:  # keep the user's own words too: a wrong rewrite can't hurt recall
            if _norm(q) not in seen:
                seen.add(_norm(q))
                queries.append(q)
        added = 0
        for v in variants:
            v = " ".join(v.split())
            if len(v) > 3 and _norm(v) not in seen and added < self._n:
                seen.add(_norm(v))
                queries.append(v)
                added += 1

        plan = QueryPlan(
            standalone=standalone,
            queries=queries,
            variants=added,
            rewritten=_norm(standalone) != _norm(question),
        )
        log.info("Query plan: standalone=%r, %d queries (%d variants)", standalone, len(queries), added)
        return plan

    def _prompt(self, question: str, history: History) -> str:
        conversation = ""
        if history:
            lines = "\n".join(
                f"{'User' if role == 'user' else 'Assistant'}: {_clip(text)}" for role, text in history
            )
            conversation = f"Conversation so far:\n{lines}\n\n"
        if self._n:
            variants_key = _VARIANTS_KEY.format(n=self._n)
            example = '{"standalone_question": "...", "search_queries": ["...", "..."]}'
        else:
            variants_key = ""
            example = '{"standalone_question": "..."}'
        return _TEMPLATE.format(
            conversation=conversation, question=question, variants_key=variants_key, example=example
        )
