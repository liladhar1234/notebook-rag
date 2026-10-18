"""POST /evaluate endpoint — RAGAS metrics for a RAG result.

Two modes:
  Auto   → {question, chat_id}  → runs full RAG pipeline, evaluates result
  Manual → {question, answer, contexts, [ground_truth]} → evaluates as-is
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request

from notebook_rag.api.deps import get_chat_repository, get_current_user, get_usage_repository
from notebook_rag.config import get_settings
from notebook_rag.registry.usage import limit_message
from notebook_rag.api.schemas import EvaluateRequest, EvaluateResponse

router = APIRouter()


def _get_generation_service(request: Request):
    return request.app.state.generation_service


def _get_evaluator(request: Request):
    return request.app.state.ragas_evaluator


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate_rag(
    body: EvaluateRequest,
    svc=Depends(_get_generation_service),
    evaluator=Depends(_get_evaluator),
    user=Depends(get_current_user),
    chats=Depends(get_chat_repository),
    usage=Depends(get_usage_repository),
):
    """Run RAGAS evaluation on a RAG response."""
    cfg = get_settings()
    cost = cfg.evaluate_cost  # RAGAS makes many LLM calls, so it counts as several questions
    quota = await usage.consume(user.user_id, cost, cfg.daily_questions_per_user, cfg.daily_questions_global)
    if not quota.allowed:
        raise HTTPException(status_code=429, detail=limit_message(quota))
    try:
        return await _evaluate(body, svc, evaluator, user, chats)
    except HTTPException as exc:
        if exc.status_code >= 500 or exc.status_code in (404, 422):
            await usage.refund(user.user_id, cost, quota.day)
        raise
    except Exception:
        await usage.refund(user.user_id, cost, quota.day)
        raise


async def _evaluate(body, svc, evaluator, user, chats):
    answer = body.answer
    contexts = body.contexts or []

    # Auto mode: run the RAG pipeline to get an answer + contexts
    if not answer or not contexts:
        if not body.chat_id:
            raise HTTPException(
                status_code=422,
                detail="Provide either (answer + contexts) or chat_id to auto-run the pipeline.",
            )
        if await chats.get_owned(user.user_id, body.chat_id) is None:
            raise HTTPException(status_code=404, detail=f"Chat '{body.chat_id}' not found.")
        try:
            result = await asyncio.to_thread(
                svc.answer, question=body.question, chat_id=body.chat_id
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        answer = result["answer"]
        contexts = [h.text for h in result.get("hits", [])]

        if not result.get("gate_passed"):
            return EvaluateResponse(
                question=body.question,
                answer=answer,
                contexts=contexts,
                faithfulness=None,
                answer_relevancy=None,
                context_precision=None,
                context_recall=None,
                note="Evidence gate rejected the query — no documents to evaluate against.",
            )

    try:
        scores = await evaluator.evaluate(
            question=body.question,
            answer=answer,
            contexts=contexts,
            ground_truth=body.ground_truth,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    return EvaluateResponse(
        question=body.question,
        answer=answer,
        contexts=contexts,
        faithfulness=scores.get("faithfulness"),
        answer_relevancy=scores.get("answer_relevancy"),
        context_precision=scores.get("context_precision"),
        context_recall=scores.get("context_recall"),
    )
