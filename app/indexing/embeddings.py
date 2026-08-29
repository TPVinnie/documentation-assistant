"""Embedding model loading, cached process-wide so ingestion and retrieval
share one loaded model instead of re-loading per call (FR-05, Performance)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=4)
def get_embedder(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_texts(model_name: str, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    model = get_embedder(model_name)
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(model_name: str, query: str) -> list[float]:
    return embed_texts(model_name, [query])[0]
