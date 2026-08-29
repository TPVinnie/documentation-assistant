from __future__ import annotations

from app.generation.abstention import decide_abstention
from app.retrieval.candidates import Candidate


def _hit(text: str, score: float) -> Candidate:
    c = Candidate(chunk_id="c1", text=text, metadata={})
    c.rerank_score = score
    return c


def test_abstains_when_no_hits():
    decision = decide_abstention("any question", [], min_score=0.15, min_coverage=0.2)
    assert decision.should_abstain is True
    assert decision.evidence_quality == "low"


def test_abstains_when_score_and_coverage_low():
    hits = [_hit("completely unrelated passage about weather", score=0.01)]
    decision = decide_abstention("what is the retention period?", hits, min_score=0.15, min_coverage=0.2)
    assert decision.should_abstain is True


def test_answers_when_score_and_coverage_sufficient():
    hits = [_hit("the retention period for logs is 60 days", score=0.9)]
    decision = decide_abstention("what is the retention period for logs?", hits, min_score=0.15, min_coverage=0.2)
    assert decision.should_abstain is False
    assert decision.evidence_quality in ("medium", "high")


def test_evidence_quality_is_qualitative_bucket_not_raw_score():
    hits = [_hit("the retention period for logs is 60 days", score=0.9)]
    decision = decide_abstention("what is the retention period for logs?", hits, min_score=0.15, min_coverage=0.2)
    assert decision.evidence_quality in {"low", "medium", "high"}
