"""Answer-quality metrics (6.2).

None of these are LLM-judge or NLI-based — they are cheap, deterministic,
lexical-overlap heuristics chosen to keep evaluation fully local and fast.
They are proxies, not proof: a high "groundedness" score means the answer's
words overlap heavily with its cited sources, not that every claim is
semantically entailed by them, and a low score can also mean the answer
paraphrased correctly. This is a documented limitation (see
TECHNICAL_REPORT.md), not a claim of hallucination-free generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.indexing.lexical_index import tokenize

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "by", "for", "with", "and", "or", "not",
    "must", "may", "should", "will", "this", "that", "these", "those",
    "it", "its", "as", "from", "per", "if", "than", "then", "so", "do",
    "does", "did", "has", "have", "had", "can", "could", "would",
}

_CONFLICT_KEYWORDS = ("disagree", "conflict", "contradict", "differ", "however", "on the other hand", "does not match")
_SUPERSEDED_KEYWORDS = ("supersed", "superceded", "older version", "previous version", "no longer")
_SOFT_REFUSAL_KEYWORDS = (
    "do not contain", "does not contain", "no mention of", "not mention", "not specify",
    "not provide", "does not provide", "not covered", "no information", "not stated",
    "no direct information", "no direct statement", "no direct mention", "not directly",
    "not enough evidence", "insufficient evidence", "cannot answer", "can't answer",
    "unable to answer", "will not reveal", "won't reveal", "will not comply", "do not have enough",
    "does not specify", "not explicitly", "no evidence", "not clear from the",
)


def _content_terms(text: str) -> set[str]:
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 2}


@dataclass
class PerQuestionAnswer:
    question_id: str
    category: str
    answerability: str  # gold label: "answerable" | "unanswerable"
    expected_answer: str
    abstained: bool
    llm_error: str
    answer_text: str
    citations_valid_count: int
    citations_invalid_count: int
    cited_chunk_texts: list[str]
    conflict_signal: bool
    end_to_end_latency_ms: float


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def _declined(record: PerQuestionAnswer) -> bool:
    """True if the system declined to give a confident substantive answer —
    either via the code-level abstention gate, or because the LLM's own
    generated text reads as a refusal/decline (e.g. "the sources do not
    mention..."). The two are tracked separately below because they mean
    different things: `correct_abstention_rate` measures the deterministic
    pre-generation gate in `app.generation.abstention` in isolation, while
    `correct_non_fabrication_rate` measures whether the user-visible
    behavior was correct regardless of which mechanism produced it. A gap
    between them means the gate is under-firing and correctness is
    currently resting on the model's own (non-deterministic) cooperation —
    see the baseline analysis in TECHNICAL_REPORT.md.
    """
    if record.abstained:
        return True
    lowered = record.answer_text.lower()
    return any(kw in lowered for kw in _SOFT_REFUSAL_KEYWORDS)


def abstention_accuracy(records: list[PerQuestionAnswer]) -> dict:
    unanswerable = [r for r in records if r.answerability == "unanswerable"]
    answerable = [r for r in records if r.answerability == "answerable"]
    correct_abstention = sum(1 for r in unanswerable if r.abstained) / len(unanswerable) if unanswerable else 0.0
    correct_non_fabrication = (
        sum(1 for r in unanswerable if _declined(r)) / len(unanswerable) if unanswerable else 0.0
    )
    over_abstention = sum(1 for r in answerable if r.abstained) / len(answerable) if answerable else 0.0
    return {
        "correct_abstention_rate": round(correct_abstention, 4),
        "correct_non_fabrication_rate": round(correct_non_fabrication, 4),
        "over_abstention_rate_on_answerable": round(over_abstention, 4),
        "unanswerable_n": len(unanswerable),
        "answerable_n": len(answerable),
    }


def citation_metrics(records: list[PerQuestionAnswer]) -> dict:
    answered = [r for r in records if not r.abstained]
    if not answered:
        return {"citation_correctness": 0.0, "citation_coverage": 0.0, "n": 0}
    total_valid = sum(r.citations_valid_count for r in answered)
    total_attempted = total_valid + sum(r.citations_invalid_count for r in answered)
    correctness = total_valid / total_attempted if total_attempted else 0.0
    coverage = sum(1 for r in answered if r.citations_valid_count > 0) / len(answered)
    return {
        "citation_correctness": round(correctness, 4),
        "citation_coverage": round(coverage, 4),
        "n": len(answered),
    }


def groundedness_proxy(records: list[PerQuestionAnswer]) -> dict:
    scoped = [r for r in records if not r.abstained and r.cited_chunk_texts]
    if not scoped:
        return {"lexical_overlap_score": 0.0, "n": 0}
    scores = []
    for r in scoped:
        answer_terms = _content_terms(r.answer_text)
        if not answer_terms:
            scores.append(0.0)
            continue
        source_terms: set[str] = set()
        for text in r.cited_chunk_texts:
            source_terms |= _content_terms(text)
        overlap = len(answer_terms & source_terms) / len(answer_terms)
        scores.append(overlap)
    return {"lexical_overlap_score": round(sum(scores) / len(scores), 4), "n": len(scoped)}


def answer_completeness_proxy(records: list[PerQuestionAnswer]) -> dict:
    scoped = [r for r in records if not r.abstained and r.answerability == "answerable"]
    if not scoped:
        return {"key_fact_coverage": 0.0, "n": 0}
    scores = []
    for r in scoped:
        expected_terms = _content_terms(r.expected_answer)
        if not expected_terms:
            continue
        answer_terms = _content_terms(r.answer_text)
        scores.append(len(expected_terms & answer_terms) / len(expected_terms))
    if not scores:
        return {"key_fact_coverage": 0.0, "n": 0}
    return {"key_fact_coverage": round(sum(scores) / len(scores), 4), "n": len(scores)}


def conflict_handling_rate(records: list[PerQuestionAnswer]) -> dict:
    scoped = [r for r in records if r.category == "contradictory_evidence"]
    if not scoped:
        return {"conflict_handling_rate": 0.0, "n": 0}
    handled = sum(
        1 for r in scoped if r.conflict_signal or any(kw in r.answer_text.lower() for kw in _CONFLICT_KEYWORDS)
    )
    return {"conflict_handling_rate": round(handled / len(scoped), 4), "n": len(scoped)}


def version_recency_accuracy(records: list[PerQuestionAnswer]) -> dict:
    scoped = [r for r in records if r.category == "version_recency"]
    if not scoped:
        return {"version_recency_accuracy": 0.0, "n": 0}
    handled = sum(1 for r in scoped if any(kw in r.answer_text.lower() for kw in _SUPERSEDED_KEYWORDS))
    return {"version_recency_accuracy": round(handled / len(scoped), 4), "n": len(scoped)}


def latency_and_failure(records: list[PerQuestionAnswer]) -> dict:
    values = [r.end_to_end_latency_ms for r in records]
    failures = sum(1 for r in records if r.llm_error)
    stats = {
        "avg_ms": round(sum(values) / len(values), 2) if values else 0.0,
        "p50_ms": round(_percentile(values, 50), 2),
        "p95_ms": round(_percentile(values, 95), 2),
    }
    return {"latency": stats, "failure_rate": round(failures / len(records), 4) if records else 0.0}


@dataclass
class AnswerMetricsReport:
    overall: dict = field(default_factory=dict)
    by_category: dict[str, dict] = field(default_factory=dict)


def compute_answer_metrics(records: list[PerQuestionAnswer]) -> AnswerMetricsReport:
    def _bucket(bucket_records: list[PerQuestionAnswer]) -> dict:
        return {
            "abstention": abstention_accuracy(bucket_records),
            "citations": citation_metrics(bucket_records),
            "groundedness_proxy": groundedness_proxy(bucket_records),
            "answer_completeness_proxy": answer_completeness_proxy(bucket_records),
            "conflict_handling": conflict_handling_rate(bucket_records),
            "version_recency": version_recency_accuracy(bucket_records),
            **latency_and_failure(bucket_records),
        }

    overall = _bucket(records)
    categories = sorted({r.category for r in records})
    by_category = {category: _bucket([r for r in records if r.category == category]) for category in categories}
    return AnswerMetricsReport(overall=overall, by_category=by_category)
