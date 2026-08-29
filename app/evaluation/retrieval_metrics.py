"""Retrieval-quality metrics (6.1).

Metrics that need a ground-truth source (Hit Rate, MRR, source coverage) are
computed only over questions whose `expected_sources` list is non-empty —
by design, `unanswerable` and some `ambiguous_query` questions have no
expected source, since the correct retrieval behavior there is "nothing
relevant enough to answer confidently", not a specific passage. The reported
`n` alongside each metric makes this scope explicit rather than silently
diluting the denominator.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PerQuestionRetrieval:
    question_id: str
    category: str
    expected_sources: list[str]
    retrieved_file_names: list[str]  # in rank order, top-K
    retrieval_latency_ms: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def hit_rate_at_k(records: list[PerQuestionRetrieval]) -> tuple[float, int]:
    scoped = [r for r in records if r.expected_sources]
    if not scoped:
        return 0.0, 0
    hits = sum(1 for r in scoped if set(r.retrieved_file_names) & set(r.expected_sources))
    return hits / len(scoped), len(scoped)


def mean_reciprocal_rank(records: list[PerQuestionRetrieval]) -> tuple[float, int]:
    scoped = [r for r in records if r.expected_sources]
    if not scoped:
        return 0.0, 0
    reciprocal_ranks = []
    for r in scoped:
        rank = next(
            (i + 1 for i, name in enumerate(r.retrieved_file_names) if name in r.expected_sources), None
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks), len(scoped)


def citation_source_coverage(records: list[PerQuestionRetrieval]) -> tuple[float, int]:
    scoped = [r for r in records if r.expected_sources]
    if not scoped:
        return 0.0, 0
    coverages = []
    for r in scoped:
        expected = set(r.expected_sources)
        found = expected & set(r.retrieved_file_names)
        coverages.append(len(found) / len(expected))
    return sum(coverages) / len(coverages), len(scoped)


def latency_stats(records: list[PerQuestionRetrieval]) -> dict[str, float]:
    values = [r.retrieval_latency_ms for r in records]
    if not values:
        return {"avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    return {
        "avg_ms": round(sum(values) / len(values), 2),
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
    }


@dataclass
class RetrievalMetricsReport:
    overall: dict = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)


def compute_retrieval_metrics(records: list[PerQuestionRetrieval]) -> RetrievalMetricsReport:
    def _bucket(bucket_records: list[PerQuestionRetrieval]) -> dict:
        hit_rate, hit_n = hit_rate_at_k(bucket_records)
        mrr, mrr_n = mean_reciprocal_rank(bucket_records)
        coverage, cov_n = citation_source_coverage(bucket_records)
        return {
            "hit_rate_at_k": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "citation_source_coverage": round(coverage, 4),
            "scored_question_count": hit_n,
            "total_question_count": len(bucket_records),
            "latency": latency_stats(bucket_records),
        }

    overall = _bucket(records)

    by_category: dict[str, dict] = {}
    categories = sorted({r.category for r in records})
    for category in categories:
        by_category[category] = _bucket([r for r in records if r.category == category])

    return RetrievalMetricsReport(overall=overall, by_category=by_category)
