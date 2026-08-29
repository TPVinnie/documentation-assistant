"""Persistent local vector index (FR-05).

Wraps a Chroma persistent collection. Embeddings are always computed by us
(via `app.indexing.embeddings`) and passed in explicitly — we never let
Chroma pick or download its own embedding function — so the embedding model
choice stays a single, documented decision point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb

_COLLECTION_NAME = "chunks"


class VectorStore:
    def __init__(self, persist_dir: Path) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def delete_by_file_path(self, file_path: str) -> None:
        self._collection.delete(where={"file_path": file_path})

    def update_metadata(self, ids: list[str], metadatas: list[dict[str, Any]]) -> None:
        if not ids:
            return
        self._collection.update(ids=ids, metadatas=metadatas)

    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_embedding: list[float],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        n_results = min(n_results, self._collection.count())
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["metadatas", "documents", "distances"],
        )
        hits: list[dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for chunk_id, metadata, document, distance in zip(ids, metadatas, documents, distances, strict=True):
            similarity = 1.0 - distance  # cosine distance -> similarity, NOT a calibrated probability
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "metadata": metadata,
                    "text": document,
                    "score": similarity,
                }
            )
        return hits
