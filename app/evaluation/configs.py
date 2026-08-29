"""Baseline vs. improved retrieval configurations (4.2)."""

from __future__ import annotations

from app.retrieval.pipeline import RetrievalConfig

BASELINE = RetrievalConfig(
    name="baseline",
    chunk_strategy="fixed_window",
    top_k_dense=10,
    top_k_lexical=10,
    top_k_fused=5,
    top_k_final=5,
    use_hybrid=False,
    use_reranker=False,
)
"""Single chunking configuration (fixed-size window), dense retrieval only, no reranker."""

IMPROVED_A = RetrievalConfig(
    name="improved_a",
    chunk_strategy="fixed_window",
    top_k_dense=20,
    top_k_lexical=20,
    top_k_fused=10,
    top_k_final=5,
    use_hybrid=True,
    use_reranker=False,
    fusion_rrf_k=60,
)
"""Adds hybrid (dense + BM25) retrieval with explicit RRF fusion and a wider,
tuned top-k candidate pool, on the same chunking as the baseline."""

IMPROVED_B = RetrievalConfig(
    name="improved_b",
    chunk_strategy="structure_aware",
    top_k_dense=20,
    top_k_lexical=20,
    top_k_fused=10,
    top_k_final=5,
    use_hybrid=True,
    use_reranker=True,
    fusion_rrf_k=60,
)
"""Adds cross-encoder reranking on top of Improved A, and switches to the
structure-aware chunker so context selection respects section boundaries."""

ALL_CONFIGS: dict[str, RetrievalConfig] = {c.name: c for c in (BASELINE, IMPROVED_A, IMPROVED_B)}
