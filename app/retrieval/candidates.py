"""Shared candidate representation threaded through fusion, reranking, and
citation assembly."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Candidate:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        """Used to *rank* candidates — whichever scoring stage ran last wins."""
        if self.rerank_score is not None:
            return self.rerank_score
        if self.fused_score is not None:
            return self.fused_score
        return self.dense_score or self.lexical_score or 0.0

    @property
    def quality_score(self) -> float | None:
        """A scale-stable relevance signal for evidence-quality gating
        (abstention), deliberately independent of `final_score`.

        `final_score` mixes three incompatible scales depending on
        configuration: RRF fusion scores are bounded to roughly [0, 2/k]
        (tiny, e.g. ~0.016-0.03), cross-encoder rerank scores are unbounded
        logits (e.g. ~10+), and cosine similarity is bounded [0, 1]. A single
        absolute threshold compared against `final_score` would therefore
        need to change per configuration — thresholding baseline's fused
        score of 0.016 against a threshold calibrated for cosine similarity
        caused every answerable baseline question to abstain incorrectly.
        Dense cosine similarity is the one signal present and comparably
        scaled across every configuration, so abstention always gates on it.
        """
        return self.dense_score
