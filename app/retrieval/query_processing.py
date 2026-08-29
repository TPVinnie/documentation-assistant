"""Query normalization and conversation-aware follow-up handling (FR-09).

This is deliberately rule-based rather than LLM-based: it needs to run
before we know whether the LLM backend is even reachable, and it must be
deterministic so unit tests don't depend on model output. The heuristic only
*biases retrieval* toward the right topic; the actual reference resolution
("what does 'it' mean") happens at generation time, where the full
conversation history is given to the LLM directly (see
`app.generation.prompt`). The original query is always preserved for
traceability, alongside whatever retrieval query was derived from it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_WHITESPACE_RE = re.compile(r"\s+")
_PRONOUN_LEAD_RE = re.compile(r"^(it|that|this|they|those|its|these|he|she|him|her)\b", re.IGNORECASE)
_FOLLOW_UP_WORD_COUNT = 6


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ProcessedQuery:
    original: str
    normalized: str
    retrieval_query: str
    used_conversation_context: bool
    history: list[ConversationTurn] = field(default_factory=list)


def normalize(query: str) -> str:
    return _WHITESPACE_RE.sub(" ", query).strip()


def process_query(query: str, history: list[ConversationTurn] | None = None) -> ProcessedQuery:
    history = history or []
    normalized = normalize(query)
    retrieval_query = normalized
    used_context = False

    if history:
        last_user = next((t.content for t in reversed(history) if t.role == "user"), None)
        looks_like_follow_up = bool(_PRONOUN_LEAD_RE.match(normalized)) or (
            len(normalized.split()) <= _FOLLOW_UP_WORD_COUNT
        )
        if last_user and looks_like_follow_up:
            retrieval_query = f"{last_user} {normalized}"
            used_context = True

    return ProcessedQuery(
        original=query,
        normalized=normalized,
        retrieval_query=retrieval_query,
        used_conversation_context=used_context,
        history=history,
    )
