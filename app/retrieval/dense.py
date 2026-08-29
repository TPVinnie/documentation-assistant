"""Dense semantic candidate retrieval (FR-05/FR-06, pipeline stage 2)."""

from __future__ import annotations

from app.indexing.embeddings import embed_query
from app.indexing.vector_store import VectorStore
from app.retrieval.candidates import Candidate
from app.retrieval.filters import RetrievalFilters, as_chroma_where


def dense_search(
    query: str,
    embedding_model: str,
    vector_store: VectorStore,
    top_k: int,
    chunk_strategy: str,
    filters: RetrievalFilters | None = None,
) -> list[Candidate]:
    where = as_chroma_where(filters) if filters else None
    strategy_clause = {"chunk_strategy": chunk_strategy}
    if where is None:
        where = strategy_clause
    elif "$and" in where:
        where = {"$and": where["$and"] + [strategy_clause]}
    else:
        where = {"$and": [where, strategy_clause]}

    query_embedding = embed_query(embedding_model, query)
    hits = vector_store.query(query_embedding, n_results=top_k, where=where)

    candidates: list[Candidate] = []
    for rank, hit in enumerate(hits):
        candidates.append(
            Candidate(
                chunk_id=hit["chunk_id"],
                text=hit["text"],
                metadata=hit["metadata"],
                dense_score=hit["score"],
                dense_rank=rank,
            )
        )
    return candidates
