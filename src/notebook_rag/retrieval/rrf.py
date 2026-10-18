"""Reciprocal Rank Fusion — pure Python, fully unit-testable.

Formula:
    RRF_Score(d) = sum(w_m / (k + r_m(d)) for m in methods)

Default: k=60, w=1.0 (balanced).
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of document IDs into a single ranked list.

    Args:
        ranked_lists: Each inner list is an ordered list of document/chunk IDs
                      from best to worst for one retriever.
        k:            RRF constant (default 60).
        weights:      Per-list weights; defaults to 1.0 for all lists.

    Returns:
        List of (id, fused_score) sorted descending by score.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[str, float] = {}
    for ranked, w in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
