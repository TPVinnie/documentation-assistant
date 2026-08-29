"""Conflict *signal* detection (FR-13) — transparency metadata only.

Real contradiction detection (do two passages actually assert incompatible
facts) needs semantic judgment, which we delegate to the LLM via explicit
prompt rules (see `app.generation.prompt`, rule 3). This module only flags
that the *precondition* for a conflict exists — evidence for the answer
came from more than one distinct document — so the API response can surface
"multiple sources contributed, check for disagreement" as a hint even when
generation is unavailable (e.g. abstention or LLM failure) and the prompt
rules never ran. This is a known limitation: it signals "possible conflict",
not confirmed conflict, and is documented as such in the technical report.
"""

from __future__ import annotations

from app.retrieval.candidates import Candidate


def detect_conflict_signal(hits: list[Candidate], top_n: int = 4) -> bool:
    file_names = {hit.metadata.get("file_name") for hit in hits[:top_n]}
    return len(file_names) > 1
