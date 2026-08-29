"""BM25 lexical index (FR-06), one per chunking strategy.

BM25 IDF statistics depend on the whole corpus, so rather than patching the
index in place on every incremental change, we rebuild it from the metadata
store's `chunks` table (the source of truth) after each ingestion run. That
rebuild is pure-Python tokenization over chunk text — cheap relative to
parsing/embedding, which is what FR-03's incremental-indexing requirement is
actually protecting against.
"""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _StrategyIndex:
    bm25: BM25Okapi
    chunk_ids: list[str]
    metadatas: list[dict[str, Any]]
    texts: list[str]


class LexicalIndex:
    def __init__(self, persist_path: Path) -> None:
        self._persist_path = persist_path
        self._by_strategy: dict[str, _StrategyIndex] = {}
        self._load()

    def _load(self) -> None:
        if self._persist_path.exists():
            with self._persist_path.open("rb") as f:
                self._by_strategy = pickle.load(f)

    def _save(self) -> None:
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persist_path.open("wb") as f:
            pickle.dump(self._by_strategy, f)

    def rebuild_strategy(
        self,
        chunk_strategy: str,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not chunk_ids:
            self._by_strategy.pop(chunk_strategy, None)
            self._save()
            return
        tokenized = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        self._by_strategy[chunk_strategy] = _StrategyIndex(
            bm25=bm25, chunk_ids=chunk_ids, metadatas=metadatas, texts=texts
        )
        self._save()

    def search(
        self,
        chunk_strategy: str,
        query: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        index = self._by_strategy.get(chunk_strategy)
        if index is None or not index.chunk_ids:
            return []
        scores = index.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[dict[str, Any]] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            metadata = index.metadatas[i]
            if where and any(metadata.get(k) != v for k, v in where.items()):
                continue
            results.append(
                {
                    "chunk_id": index.chunk_ids[i],
                    "metadata": metadata,
                    "text": index.texts[i],
                    "score": float(scores[i]),
                }
            )
            if len(results) >= top_k:
                break
        return results
