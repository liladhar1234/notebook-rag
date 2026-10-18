"""Generation service: plan -> retrieve -> rerank -> gate -> generate.

Pipeline:
  1. QueryPlanner: resolve follow-ups against the chat history and (optionally)
     write alternative search queries, in ONE LLM call
  2. HybridRetriever: dense + BM25 for every query, fused in one RRF pass
  3. CrossEncoderReranker (if enabled) scores the whole candidate pool
  4. Evidence gate on the best reranker score
  5. XML context assembly and LLM generation
  6. Strip any inline citations the model adds anyway (answers stay clean;
     provenance is returned separately in `hits`)
"""
from __future__ import annotations

import logging
import time

from notebook_rag.config import get_settings
from notebook_rag.generation.agent import AgentError, DocumentAgent
from notebook_rag.generation.grounding import (
    EvidenceGateError,
    check_evidence,
    strip_citations,
)
from notebook_rag.generation.llm_client import LLMClient
from notebook_rag.generation.prompts import (
    OVERVIEW_NOTE,
    OVERVIEW_TRUNCATED_NOTE,
    SYSTEM_PROMPT,
    build_context_xml,
    build_user_prompt,
)
from notebook_rag.models import SearchHit
from notebook_rag.retrieval.hybrid import HybridRetriever
from notebook_rag.retrieval.complexity import is_complex_question
from notebook_rag.retrieval.overview import is_overview_question, pick_overview_hits
from notebook_rag.retrieval.planner import QueryPlan, QueryPlanner

log = logging.getLogger(__name__)


class GenerationService:
    """Orchestrates the full enhanced RAG pipeline for a single query."""

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLMClient,
        planner: QueryPlanner | None = None,
        reranker=None,      # CrossEncoderReranker | None
        agent: DocumentAgent | None = None,
    ) -> None:
        self._agent = agent
        self._retriever = retriever
        self._llm = llm
        self._planner = planner
        self._reranker = reranker

    def answer(
        self,
        question: str,
        chat_id: str,
        history: list[tuple[str, str]] | None = None,
        doc_names: dict[str, str] | None = None,
    ) -> dict:
        """Run the full RAG pipeline and return a structured response.

        Returns a dict with keys:
          - answer: str
          - hits: list[SearchHit]
          - hallucinated_citations: list[str]
          - gate_passed: bool
          - retrieval_mode: str  (which stages ran)
          - standalone_question: str  (follow-up resolved against the history)
          - top_score: float | None  (best hit's score; log it to tune the gate)
        """
        history = history or []
        cfg = get_settings()
        use_reranker = self._reranker is not None

        # 0. Hard questions (comparisons, multi-part, cross-document) go to the bounded agent loop.
        if self._agent is not None and cfg.agent_enabled and is_complex_question(question):
            t_agent = time.perf_counter()
            try:
                result = self._agent.run(question, chat_id, history, doc_names)
                log.info("Timing [agent]: total=%.1fs mode=%s", time.perf_counter() - t_agent, result.get("retrieval_mode"))
            except AgentError as exc:  # internal problem only; LLM/API errors propagate as before
                log.warning("Agent failed, using the standard pipeline: %s", exc)
                result = None
            if result is not None:
                result["standalone_question"] = question
                return result

        t0 = time.perf_counter()

        # 1. Plan: resolve follow-ups and (optionally) expand into several queries.
        plan = (
            self._planner.plan(question, history)
            if self._planner is not None
            else QueryPlan(standalone=question, queries=[question])
        )

        t_plan = time.perf_counter()

        # 2. Retrieve. With a reranker, fetch a wide candidate pool (the cross-encoder
        #    picks the best few); without one the fused RRF order is final.
        pool = cfg.rerank_candidates if use_reranker else cfg.retrieval_top_k
        hits: list[SearchHit] = self._retriever.retrieve(plan.queries, chat_id, top_k=pool)

        t_retrieve = time.perf_counter()

        # 3. Rerank the pool against the self-contained question, keep the best top_k.
        pool_size = len(hits)
        if use_reranker and hits:
            hits = self._reranker.rerank(plan.standalone, hits, top_k=cfg.retrieval_top_k)

        t_rerank = time.perf_counter()

        modes = (["rewrite"] if plan.rewritten else []) + (["multi_query"] if plan.variants else [])
        modes += ["hybrid", "rrf"] + (["reranker"] if use_reranker else [])
        retrieval_mode = "+".join(modes)
        top_score = hits[0].score if hits else None
        log.info(
            "Retrieval [%s]: %d queries -> %d hits, top score %s | question '%s...'",
            retrieval_mode, len(plan.queries), len(hits),
            f"{top_score:.2f}" if top_score is not None else "n/a", question[:60],
        )

        base = {
            "hallucinated_citations": [],
            "retrieval_mode": retrieval_mode,
            "standalone_question": plan.standalone,
            "top_score": top_score,
        }

        # 4. Evidence gate (reranker scores are logits, so the threshold only applies with them).
        try:
            check_evidence(hits, min_score=cfg.min_rerank_score if use_reranker else None)
        except EvidenceGateError as exc:
            # "Summarize this" / "what is this image?" can't match any single passage, so the
            # gate rejects them. Answer from the opening chunks of each document instead.
            if is_overview_question(question) or is_overview_question(plan.standalone):
                overview = self._answer_overview(question, chat_id, history, plan, base)
                if overview is not None:
                    return overview
            log.info("Evidence gate rejected query: %s", exc)
            return {**base, "answer": str(exc), "hits": [], "gate_passed": False}

        # 5. Assemble context and generate; 6. strip any inline citations.
        user_prompt = build_user_prompt(
            question, build_context_xml(hits), history=history, standalone=plan.standalone
        )
        log.info("Calling LLM with %d context chunks", len(hits))
        t_gen = time.perf_counter()
        answer = strip_citations(self._llm.complete(SYSTEM_PROMPT, user_prompt))
        t_end = time.perf_counter()
        log.info(
            "Timing: plan=%.1fs retrieve=%.1fs rerank=%.1fs (%d candidates) generate=%.1fs total=%.1fs",
            t_plan - t0, t_retrieve - t_plan, t_rerank - t_retrieve, pool_size, t_end - t_gen, t_end - t0,
        )
        return {**base, "answer": answer, "hits": hits, "gate_passed": True}

    def _answer_overview(self, question, chat_id, history, plan, base) -> dict | None:
        """Answer a whole-document question from the opening chunks. None if the chat is empty."""
        cfg = get_settings()
        chunks = self._retriever.store.list_chat_chunks(chat_id)
        hits, truncated = pick_overview_hits(
            chunks, cfg.overview_chunks_per_doc, cfg.overview_max_chunks
        )
        if not hits:
            return None
        log.info("Overview mode: %d chunks from %d chunks in chat", len(hits), len(chunks))
        note = OVERVIEW_NOTE + (OVERVIEW_TRUNCATED_NOTE if truncated else "")
        user_prompt = build_user_prompt(
            question, build_context_xml(hits), history=history,
            standalone=plan.standalone, note=note,
        )
        answer = strip_citations(self._llm.complete(SYSTEM_PROMPT, user_prompt))
        return {
            **base,
            "retrieval_mode": "overview",
            "answer": answer,
            "hits": hits,
            "gate_passed": True,
        }