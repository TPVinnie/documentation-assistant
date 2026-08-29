"""Lexical (BM25) candidate retrieval (FR-06, pipeline stage 2)."""

from __future__ import annotations

from app.indexing.lexical_index import LexicalIndex
from app.retrieval.candidates import Candidate
from app.retrieval.filters import RetrievalFilters, as_plain_dict


def lexical_search(
    query: str,
    lexical_index: LexicalIndex,
    top_k: int,
    chunk_strategy: str,
    filters: RetrievalFilters | None = None,
) -> list[Candidate]:
    where = as_plain_dict(filters) if filters else {}
    hits = lexical_index.search(chunk_strategy, query, top_k=top_k, where=where or None)

    candidates: list[Candidate] = []
    for rank, hit in enumerate(hits):
        candidates.append(
            Candidate(
                chunk_id=hit["chunk_id"],
                text=hit["text"],
                metadata=hit["metadata"],
                lexical_score=hit["score"],
                lexical_rank=rank,
            )
        )
    return candidates
