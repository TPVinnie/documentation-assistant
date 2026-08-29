from __future__ import annotations

from app.evaluation.retrieval_metrics import (
    PerQuestionRetrieval,
    citation_source_coverage,
    compute_retrieval_metrics,
    hit_rate_at_k,
    latency_stats,
    mean_reciprocal_rank,
)


def _record(qid: str, category: str, expected: list[str], retrieved: list[str], latency: float = 10.0):
    return PerQuestionRetrieval(
        question_id=qid, category=category, expected_sources=expected,
        retrieved_file_names=retrieved, retrieval_latency_ms=latency,
    )


def test_hit_rate_counts_any_expected_source_present():
    records = [
        _record("q1", "direct_factual", ["a.md"], ["x.md", "a.md"]),
        _record("q2", "direct_factual", ["b.md"], ["x.md", "y.md"]),
    ]
    rate, n = hit_rate_at_k(records)
    assert n == 2
    assert rate == 0.5


def test_hit_rate_ignores_questions_with_no_expected_source():
    records = [
        _record("q1", "unanswerable", [], ["x.md"]),
    ]
    rate, n = hit_rate_at_k(records)
    assert n == 0
    assert rate == 0.0


def test_mrr_rewards_earlier_rank():
    records = [
        _record("q1", "direct_factual", ["a.md"], ["a.md", "b.md", "c.md"]),  # rank 1
        _record("q2", "direct_factual", ["c.md"], ["a.md", "b.md", "c.md"]),  # rank 3
    ]
    mrr, n = mean_reciprocal_rank(records)
    assert n == 2
    assert mrr == (1.0 + 1 / 3) / 2


def test_citation_source_coverage_requires_all_expected_sources():
    records = [
        _record("q1", "multi_document_comparison", ["a.md", "b.md"], ["a.md"]),
    ]
    coverage, n = citation_source_coverage(records)
    assert n == 1
    assert coverage == 0.5


def test_latency_stats_on_empty_list():
    assert latency_stats([]) == {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}


def test_compute_retrieval_metrics_breaks_down_by_category():
    records = [
        _record("q1", "direct_factual", ["a.md"], ["a.md"]),
        _record("q2", "unanswerable", [], []),
    ]
    report = compute_retrieval_metrics(records)
    assert "direct_factual" in report.by_category
    assert "unanswerable" in report.by_category
    assert report.overall["total_question_count"] == 2
