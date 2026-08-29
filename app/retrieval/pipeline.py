"""Orchestrates retrieval pipeline stages 1-4 (4.1): query preprocessing,
dense+lexical candidate retrieval, rank fusion + metadata filtering, and
reranking. Stages 5-7 (context selection, generation, citation assembly)
live in `app.generation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.indexing.lexical_index import LexicalIndex
from app.indexing.vector_store import VectorStore
from app.logging_config import StageTimer
from app.reranking.fallback import apply_reranking
from app.retrieval.candidates import Candidate
from app.retrieval.dense import dense_search
from app.retrieval.filters import RetrievalFilters
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.lexical import lexical_search
from app.retrieval.query_processing import ConversationTurn, ProcessedQuery, process_query

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """A named, fully-specified retrieval configuration (4.2)."""

    name: str
    chunk_strategy: str = "structure_aware"
    top_k_dense: int = 20
    top_k_lexical: int = 20
    top_k_fused: int = 10
    top_k_final: int = 5
    use_hybrid: bool = True
    use_reranker: bool = True
    fusion_rrf_k: int = 60


@dataclass
class RetrievalOutcome:
    processed_query: ProcessedQuery
    config: RetrievalConfig
    filters: RetrievalFilters
    fused_candidates: list[Candidate]
    hits: list[Candidate]
    reranker_used: bool
    reranker_fallback_reason: str
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


def retrieve(
    query: str,
    settings: Settings,
    config: RetrievalConfig,
    vector_store: VectorStore,
    lexical_index: LexicalIndex,
    filters: RetrievalFilters | None = None,
    history: list[ConversationTurn] | None = None,
) -> RetrievalOutcome:
    filters = filters or RetrievalFilters()
    timer = StageTimer()

    with timer.measure("query_processing"):
        processed = process_query(query, history)

    with timer.measure("dense_retrieval"):
        dense_candidates = dense_search(
            processed.retrieval_query,
            settings.embedding_model,
            vector_store,
            top_k=config.top_k_dense,
            chunk_strategy=config.chunk_strategy,
            filters=filters,
        )

    lexical_candidates: list[Candidate] = []
    if config.use_hybrid:
        with timer.measure("lexical_retrieval"):
            lexical_candidates = lexical_search(
                processed.retrieval_query,
                lexical_index,
                top_k=config.top_k_lexical,
                chunk_strategy=config.chunk_strategy,
                filters=filters,
            )

    with timer.measure("fusion"):
        fused = reciprocal_rank_fusion(
            dense_candidates,
            lexical_candidates,
            rrf_k=config.fusion_rrf_k,
            use_hybrid=config.use_hybrid,
        )
        fused_top = fused[: config.top_k_fused]

    with timer.measure("reranking"):
        reranked, reranker_used, fallback_reason = apply_reranking(
            processed.retrieval_query,
            fused_top,
            settings.reranker_model,
            enabled=config.use_reranker,
        )

    hits = reranked[: config.top_k_final]

    logger.info(
        "retrieval_complete",
        extra={
            "config": config.name,
            "candidate_count": len(fused),
            "hit_count": len(hits),
            "reranker_used": reranker_used,
            "stage_timings_ms": timer.stages_ms,
        },
    )

    return RetrievalOutcome(
        processed_query=processed,
        config=config,
        filters=filters,
        fused_candidates=fused,
        hits=hits,
        reranker_used=reranker_used,
        reranker_fallback_reason=fallback_reason,
        stage_timings_ms=timer.stages_ms,
    )
