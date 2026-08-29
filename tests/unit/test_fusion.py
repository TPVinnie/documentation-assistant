from __future__ import annotations

from app.retrieval.candidates import Candidate
from app.retrieval.fusion import reciprocal_rank_fusion


def _candidate(chunk_id: str, doc_status: str = "current") -> Candidate:
    return Candidate(chunk_id=chunk_id, text="text", metadata={"doc_status": doc_status})


def test_rrf_favors_items_ranked_highly_in_both_lists():
    dense = [_candidate("a"), _candidate("b"), _candidate("c")]
    dense[0].dense_rank, dense[1].dense_rank, dense[2].dense_rank = 0, 1, 2

    lexical = [_candidate("b"), _candidate("a"), _candidate("c")]
    lexical[0].lexical_rank, lexical[1].lexical_rank, lexical[2].lexical_rank = 0, 1, 2

    fused = reciprocal_rank_fusion(dense, lexical, rrf_k=60, use_hybrid=True)
    fused_ids = [c.chunk_id for c in fused]

    # "a" and "b" both appear near the top of both lists and should outrank "c"
    assert set(fused_ids[:2]) == {"a", "b"}
    assert fused_ids[-1] == "c"


def test_use_hybrid_false_ignores_lexical_candidates():
    dense = [_candidate("a")]
    dense[0].dense_rank = 0
    lexical = [_candidate("z")]
    lexical[0].lexical_rank = 0

    fused = reciprocal_rank_fusion(dense, lexical, use_hybrid=False)

    assert [c.chunk_id for c in fused] == ["a"]


def test_superseded_candidates_are_penalized():
    current = _candidate("current", doc_status="current")
    current.dense_rank = 0
    superseded = _candidate("superseded", doc_status="superseded")
    superseded.dense_rank = 0  # identical rank, differ only by doc_status

    fused = reciprocal_rank_fusion([current, superseded], [], use_hybrid=False)
    by_id = {c.chunk_id: c.fused_score for c in fused}

    assert by_id["superseded"] < by_id["current"]


def test_merging_keeps_both_dense_and_lexical_scores_for_shared_chunk():
    dense = [_candidate("shared")]
    dense[0].dense_rank = 0
    dense[0].dense_score = 0.9
    lexical = [_candidate("shared")]
    lexical[0].lexical_rank = 0
    lexical[0].lexical_score = 5.0

    fused = reciprocal_rank_fusion(dense, lexical, use_hybrid=True)

    assert len(fused) == 1
    assert fused[0].dense_score == 0.9
    assert fused[0].lexical_score == 5.0
