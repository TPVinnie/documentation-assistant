from __future__ import annotations

import pytest

from app.reranking import fallback
from app.retrieval.candidates import Candidate


def _candidate(chunk_id: str, fused_score: float) -> Candidate:
    c = Candidate(chunk_id=chunk_id, text=f"text for {chunk_id}", metadata={})
    c.fused_score = fused_score
    return c


def test_disabled_reranker_falls_back_to_fused_order_immediately():
    candidates = [_candidate("a", 0.5), _candidate("b", 0.9)]

    reranked, used, reason = fallback.apply_reranking("query", candidates, "some-model", enabled=False)

    assert used is False
    assert "disabled" in reason
    # fused order is left untouched (caller already sorted by fused_score upstream)
    assert [c.rerank_score for c in reranked] == [0.5, 0.9]


def test_reranker_exception_falls_back_to_fused_order_and_reports_reason(monkeypatch: pytest.MonkeyPatch):
    def _boom(model_name, query, texts):
        raise RuntimeError("model failed to load")

    monkeypatch.setattr(fallback, "score_pairs", _boom)
    candidates = [_candidate("a", 0.2), _candidate("b", 0.9)]

    reranked, used, reason = fallback.apply_reranking("query", candidates, "some-model", enabled=True)

    assert used is False
    assert "model failed to load" in reason
    assert [c.rerank_score for c in reranked] == [0.2, 0.9]


def test_successful_reranking_resorts_candidates_by_new_scores(monkeypatch: pytest.MonkeyPatch):
    def _fake_scores(model_name, query, texts):
        # deliberately inverts the fused order to prove reranking actually re-sorts
        return [0.1, 0.9]

    monkeypatch.setattr(fallback, "score_pairs", _fake_scores)
    candidates = [_candidate("a", 0.9), _candidate("b", 0.1)]

    reranked, used, reason = fallback.apply_reranking("query", candidates, "some-model", enabled=True)

    assert used is True
    assert reason == ""
    assert [c.chunk_id for c in reranked] == ["b", "a"]
