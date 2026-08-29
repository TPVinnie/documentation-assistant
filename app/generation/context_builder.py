"""Context selection and token-budget management (pipeline stage 5).

Packs reranked hits into a character budget (a simple, dependency-free proxy
for a token budget — documented as an approximation, not an exact tokenizer
count), tags each with a stable [S<n>] citation marker, and runs every
chunk through the injection guard before it is allowed into the prompt.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.generation.injection_guard import scan_and_sanitize
from app.retrieval.candidates import Candidate


@dataclass
class ContextBlock:
    tag: str
    chunk_id: str
    file_name: str
    unit_label: str
    category: str
    version: str
    doc_status: str
    text: str
    injection_flagged: bool


@dataclass
class ContextBundle:
    blocks: list[ContextBlock]
    dropped_for_budget: int

    @property
    def included_chunk_ids(self) -> set[str]:
        return {b.chunk_id for b in self.blocks}


def build_context(hits: list[Candidate], max_chars: int) -> ContextBundle:
    blocks: list[ContextBlock] = []
    used_chars = 0
    dropped = 0

    for i, hit in enumerate(hits):
        scan = scan_and_sanitize(hit.text)
        block_len = len(scan.sanitized_text)
        if blocks and used_chars + block_len > max_chars:
            dropped += 1
            continue
        blocks.append(
            ContextBlock(
                tag=f"S{i + 1}",
                chunk_id=hit.chunk_id,
                file_name=hit.metadata.get("file_name", ""),
                unit_label=hit.metadata.get("unit_label", ""),
                category=hit.metadata.get("category", ""),
                version=hit.metadata.get("version", ""),
                doc_status=hit.metadata.get("doc_status", "unknown"),
                text=scan.sanitized_text,
                injection_flagged=scan.flagged,
            )
        )
        used_chars += block_len

    return ContextBundle(blocks=blocks, dropped_for_budget=dropped)
