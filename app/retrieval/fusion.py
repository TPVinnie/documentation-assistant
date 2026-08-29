"""Rank fusion (FR-06, pipeline stage 3).

Reciprocal Rank Fusion (RRF) combines the dense and lexical rankings using
only rank position, not raw scores — dense cosine similarity and BM25 scores
live on incomparable scales, so averaging them directly would silently let
whichever score happens to have larger magnitude dominate. RRF score for a
document is sum(1 / (k + rank)) over every list it appears in; `k` (default
60, the value used in the original RRF paper) damps the influence of any
single very-high rank.

A small, documented penalty is applied to superseded-version chunks here
(after fusion, before the top-k cut) so that current-version evidence for
the same fact naturally outranks it — without removing superseded chunks
from the candidate pool entirely, since FR-12 asks us to *flag* superseded
content, not hide it, and version/conflict questions need it retrievable.
"""

from __future__ import annotations

from app.retrieval.candidates import Candidate

SUPERSEDED_PENALTY = 0.85


def reciprocal_rank_fusion(
    dense_candidates: list[Candidate],
    lexical_candidates: list[Candidate],
    rrf_k: int = 60,
    use_hybrid: bool = True,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}

    for c in dense_candidates:
        merged[c.chunk_id] = c

    if use_hybrid:
        for c in lexical_candidates:
            if c.chunk_id in merged:
                existing = merged[c.chunk_id]
                existing.lexical_score = c.lexical_score
                existing.lexical_rank = c.lexical_rank
            else:
                merged[c.chunk_id] = c

    for candidate in merged.values():
        score = 0.0
        if candidate.dense_rank is not None:
            score += 1.0 / (rrf_k + candidate.dense_rank + 1)
        if use_hybrid and candidate.lexical_rank is not None:
            score += 1.0 / (rrf_k + candidate.lexical_rank + 1)
        if candidate.metadata.get("doc_status") == "superseded":
            score *= SUPERSEDED_PENALTY
        candidate.fused_score = score

    return sorted(merged.values(), key=lambda c: c.fused_score or 0.0, reverse=True)
