"""RAGAS evaluation wrapper.

Metrics computed (all reference-free — no ground truth needed):
  - Faithfulness:       Is every claim in the answer supported by the retrieved context?
  - Answer Relevancy:   Is the answer responsive to the question? (embedding-based)
  - Context Precision:  Did the retrieved chunks actually help answer the question?

Optional (when ground_truth is supplied):
  - Context Recall:     Did the chunks cover everything in the ground truth?

All metrics use the same LLM + embeddings already configured in the app.
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


class RagasEvaluator:
    """Wraps RAGAS metrics, lazily importing ragas to keep startup fast."""

    def __init__(
        self,
        llm_provider: str,
        llm_model: str,
        llm_api_key: str,
        llm_base_url: str,
        anthropic_api_key: str,
        dense_embedder,  # DenseEmbedder instance for embedding-based metrics
    ) -> None:
        self._provider = llm_provider
        self._model = llm_model
        self._api_key = llm_api_key
        self._base_url = llm_base_url
        self._anthropic_key = anthropic_api_key
        self._dense_embedder = dense_embedder
        self._ragas_llm = None
        self._ragas_emb = None
        self._initialized = False

    def _init_ragas(self) -> None:
        """Lazy init so heavy ragas/langchain imports don't slow startup."""
        if self._initialized:
            return
        try:
            from langchain_openai import ChatOpenAI
            from ragas.embeddings import BaseRagasEmbeddings
            from ragas.llms import LangchainLLMWrapper

            if self._provider == "anthropic":
                from langchain_anthropic import ChatAnthropic
                lc_llm = ChatAnthropic(
                    model=self._model,
                    api_key=self._anthropic_key,
                )
            else:
                lc_llm = ChatOpenAI(
                    model=self._model,
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=60,
                )

            self._ragas_llm = LangchainLLMWrapper(lc_llm)

            # Use our local fastembed model for embedding-based metrics
            dense = self._dense_embedder

            class _FastembedRagasEmbeddings(BaseRagasEmbeddings):
                async def aembed_documents(self, texts):
                    return dense.embed(texts)

                async def aembed_query(self, text):
                    return dense.embed([text])[0]

                def embed_documents(self, texts):
                    return dense.embed(texts)

                def embed_query(self, text):
                    return dense.embed([text])[0]

            self._ragas_emb = _FastembedRagasEmbeddings()
            self._initialized = True
            log.info("RAGAS evaluator initialized with model: %s", self._model)
        except ImportError as exc:
            raise RuntimeError(
                "RAGAS evaluation requires extra packages. "
                "Run: pip install ragas langchain-openai"
            ) from exc

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> dict[str, float | None]:
        """Run RAGAS metrics and return a score dict.

        Returns:
            Dict with keys: faithfulness, answer_relevancy, context_precision,
            context_recall (None if no ground_truth supplied).
        """
        self._init_ragas()

        from ragas.dataset_schema import SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
        )
        try:
            from ragas.metrics import LLMContextPrecisionWithoutReference as ContextPrecision
        except ImportError:
            from ragas.metrics import ContextPrecision

        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )

        results: dict[str, float | None] = {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
            "context_recall": None,
        }

        # Run each metric independently so one failure doesn't block others
        metrics_to_run = [
            ("faithfulness",      Faithfulness(llm=self._ragas_llm)),
            ("answer_relevancy",  ResponseRelevancy(llm=self._ragas_llm, embeddings=self._ragas_emb)),
            ("context_precision", ContextPrecision(llm=self._ragas_llm)),
        ]

        if ground_truth:
            try:
                from ragas.metrics import ContextRecall
                metrics_to_run.append(("context_recall", ContextRecall(llm=self._ragas_llm)))
            except ImportError:
                pass

        for name, metric in metrics_to_run:
            try:
                score = await metric.single_turn_ascore(sample)
                results[name] = round(float(score), 4)
                log.debug("RAGAS %s = %.4f", name, score)
            except Exception as exc:  # noqa: BLE001
                log.warning("RAGAS metric '%s' failed: %s", name, exc)
                results[name] = None

        return results
