"""Cross-encoder reranker (FR-07, pipeline stage 4).

A cross-encoder scores (query, chunk) pairs jointly, which is more accurate
than the bi-encoder similarity used for dense retrieval but too expensive to
run over the whole corpus — hence retrieve-broad-then-rerank-narrow.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder


@lru_cache(maxsize=2)
def _get_cross_encoder(model_name: str) -> CrossEncoder:
    return CrossEncoder(model_name)


def score_pairs(model_name: str, query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    model = _get_cross_encoder(model_name)
    scores = model.predict([(query, text) for text in texts])
    return [float(s) for s in scores]
