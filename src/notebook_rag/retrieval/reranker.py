"""Cross-encoder re-ranking via fastembed (ONNX, no PyTorch needed).

After RRF produces the initial top-K, a cross-encoder re-scores every
(query, chunk) pair with full token interaction, yielding a much more
precise final ranking.

Model: Xenova/ms-marco-MiniLM-L-6-v2 (ONNX, ~23 MB)
  - CPU inference, ~10 ms per batch
  - Trained on MS MARCO passage ranking
  - Loaded through fastembed's TextCrossEncoder (avoids PyTorch entirely)
"""
from __future__ import annotations

import logging
from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

from notebook_rag.models import SearchHit

log = logging.getLogger(__name__)

DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-ranks SearchHit results using a cross-encoder model via fastembed."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        log.info("Loading cross-encoder model: %s", model_name)
        self._model = TextCrossEncoder(model_name=model_name)
        log.info("Cross-encoder model ready.")

    def rerank(
        self,
        query: str,
        hits: list[SearchHit],
        top_k: int | None = None,
    ) -> list[SearchHit]:
        """Re-score hits with the cross-encoder and return sorted results.

        Args:
            query:  The user's original question.
            hits:   Hits from upstream retrieval (RRF fused).
            top_k:  If set, truncate to this many results after reranking.

        Returns:
            Hits sorted descending by cross-encoder score.
        """
        if not hits:
            return hits

        documents = [h.text for h in hits]
        # fastembed rerank() returns an iterable of plain floats
        # (one score per document, in the same order as input)
        scores: list[float] = list(self._model.rerank(query, documents))

        reranked = [
            replace(h, score=float(s)) for h, s in zip(hits, scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        log.debug(
            "Reranker: top score=%.4f, bottom score=%.4f",
            reranked[0].score if reranked else 0,
            reranked[-1].score if reranked else 0,
        )
        return reranked
