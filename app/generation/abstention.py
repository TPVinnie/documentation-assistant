"""Abstention policy (4.3): decide whether there is enough usable evidence
to answer at all, *before* calling the LLM.

The decision is based on two signals: the top retrieval/rerank score, and
lexical term-coverage between the question and the top evidence (does the
evidence even mention what's being asked about). Both are heuristics.
`evidence_quality` is an explicit qualitative bucket (low/medium/high) —
never presented as a calibrated probability of correctness, since raw
vector similarity and cross-encoder scores are not calibrated for that.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.indexing.lexical_index import tokenize
from app.retrieval.candidates import Candidate


@dataclass
class AbstentionDecision:
    should_abstain: bool
    reason: str
    evidence_quality: str  # "low" | "medium" | "high" — heuristic bucket, not a probability
    top_score: float
    term_coverage: float


def _term_coverage(query: str, hits: list[Candidate], top_n: int = 3) -> float:
    query_terms = set(tokenize(query))
    if not query_terms:
        return 0.0
    covered: set[str] = set()
    for hit in hits[:top_n]:
        covered |= query_terms & set(tokenize(hit.text))
    return len(covered) / len(query_terms)


def decide_abstention(
    query: str,
    hits: list[Candidate],
    min_score: float,
    min_coverage: float,
) -> AbstentionDecision:
    if not hits:
        return AbstentionDecision(
            should_abstain=True,
            reason="no evidence was retrieved for this question",
            evidence_quality="low",
            top_score=0.0,
            term_coverage=0.0,
        )

    top_score = hits[0].quality_score
    coverage = _term_coverage(query, hits)
    has_score_signal = top_score is not None

    if has_score_signal:
        should_abstain = top_score < min_score or coverage < min_coverage
        reason = (
            "the top retrieved evidence scored below the configured relevance threshold"
            if should_abstain and top_score < min_score
            else "the retrieved evidence does not share enough key terms with the question"
            if should_abstain
            else ""
        )
    else:
        # No dense score means this candidate came from lexical retrieval only
        # (BM25's unbounded scale isn't comparable to the cosine-similarity
        # threshold) — fall back to term coverage alone rather than guessing.
        should_abstain = coverage < min_coverage
        reason = "the retrieved evidence does not share enough key terms with the question" if should_abstain else ""

    if has_score_signal and top_score >= min_score * 2 and coverage >= min(1.0, min_coverage * 1.5):
        quality = "high"
    elif (has_score_signal and top_score >= min_score) or (not has_score_signal and coverage >= min_coverage):
        quality = "medium"
    else:
        quality = "low"

    top_score = top_score if has_score_signal else 0.0

    return AbstentionDecision(
        should_abstain=should_abstain,
        reason=reason,
        evidence_quality=quality,
        top_score=top_score,
        term_coverage=coverage,
    )
