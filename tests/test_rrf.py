"""Unit tests for the RRF fusion function."""
from __future__ import annotations

import pytest

from notebook_rag.retrieval.rrf import reciprocal_rank_fusion


def test_single_list_preserves_order():
    ids = ["a", "b", "c", "d"]
    result = reciprocal_rank_fusion([ids])
    result_ids = [r[0] for r in result]
    assert result_ids == ids


def test_two_lists_boosts_common_items():
    list1 = ["a", "b", "c"]
    list2 = ["b", "a", "d"]
    result = reciprocal_rank_fusion([list1, list2])
    top_ids = [r[0] for r in result[:2]]
    # 'a' and 'b' appear in both lists and should rank highest
    assert set(top_ids) == {"a", "b"}


def test_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_scores_are_positive():
    result = reciprocal_rank_fusion([["x", "y", "z"]])
    assert all(score > 0 for _, score in result)


def test_scores_descending():
    result = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_custom_k_affects_scores():
    result_k10 = reciprocal_rank_fusion([["a"]], k=10)
    result_k100 = reciprocal_rank_fusion([["a"]], k=100)
    # smaller k → higher score for rank 1: 1/(10+1) > 1/(100+1)
    assert result_k10[0][1] > result_k100[0][1]


def test_weighted_fusion():
    list1 = ["a", "b"]
    list2 = ["b", "a"]
    # Weight list2 heavily: 'b' should win
    result = reciprocal_rank_fusion([list1, list2], weights=[0.1, 10.0])
    assert result[0][0] == "b"


def test_weight_length_mismatch_raises():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])
