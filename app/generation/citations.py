"""Citation assembly and validation (FR-11, pipeline stage 7).

The model is asked to cite using [S<n>] tags that map 1:1 to the evidence
blocks it was given. Any tag that doesn't correspond to a block we actually
supplied (a hallucinated citation) is stripped from the answer rather than
trusted — citations must stay aligned with what was really retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.generation.context_builder import ContextBundle

_BRACKET_RE = re.compile(r"\[([^\]]+)\]")
_TAG_RE = re.compile(r"S(\d+)")


@dataclass
class Citation:
    tag: str
    chunk_id: str
    file_name: str
    unit_label: str
    category: str
    version: str
    doc_status: str


@dataclass
class CitationValidationResult:
    citations: list[Citation]
    invalid_tags_found: list[str]
    cleaned_answer: str


def assemble_and_validate_citations(raw_answer: str, bundle: ContextBundle) -> CitationValidationResult:
    tag_to_block = {b.tag: b for b in bundle.blocks}
    invalid_tags: list[str] = []
    valid_tags_used: set[str] = set()

    def _clean_bracket(match: re.Match) -> str:
        inner = match.group(1)
        found = _TAG_RE.findall(inner)
        if not found:
            return match.group(0)
        kept = []
        for num in found:
            tag = f"S{num}"
            if tag in tag_to_block:
                valid_tags_used.add(tag)
                kept.append(tag)
            else:
                invalid_tags.append(tag)
        return f"[{', '.join(kept)}]" if kept else ""

    cleaned_answer = _BRACKET_RE.sub(_clean_bracket, raw_answer).strip()

    citations = [
        Citation(
            tag=tag,
            chunk_id=block.chunk_id,
            file_name=block.file_name,
            unit_label=block.unit_label,
            category=block.category,
            version=block.version,
            doc_status=block.doc_status,
        )
        for tag, block in tag_to_block.items()
        if tag in valid_tags_used
    ]
    citations.sort(key=lambda c: int(c.tag[1:]))

    return CitationValidationResult(citations=citations, invalid_tags_found=invalid_tags, cleaned_answer=cleaned_answer)
