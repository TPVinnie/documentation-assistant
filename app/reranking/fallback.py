"""Fallback reranking strategy (4.3: explain behavior when the reranker is
unavailable). Used when reranking is disabled by configuration, or when the
cross-encoder raises (model failed to load, OOM, unexpected input) — in
either case we keep answering using the fused ranking rather than failing
the whole request.
"""

from __future__ import annotations

import logging

from app.reranking.cross_encoder import score_pairs
from app.retrieval.candidates import Candidate

logger = logging.getLogger(__name__)


def apply_reranking(
    query: str,
    candidates: list[Candidate],
    model_name: str,
    enabled: bool,
) -> tuple[list[Candidate], bool, str]:
    """Returns (reranked_candidates, reranker_used, fallback_reason)."""
    if not enabled:
        for c in candidates:
            c.rerank_score = c.fused_score
        return candidates, False, "reranker disabled by configuration"

    try:
        texts = [c.text for c in candidates]
        scores = score_pairs(model_name, query, texts)
        for c, s in zip(candidates, scores, strict=True):
            c.rerank_score = s
        reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return reranked, True, ""
    except Exception as exc:  # model unavailable, OOM, unexpected input, etc.
        logger.warning("reranker_unavailable_falling_back", extra={"error": str(exc)})
        for c in candidates:
            c.rerank_score = c.fused_score
        return candidates, False, f"reranker unavailable, used fused ranking instead: {exc}"
