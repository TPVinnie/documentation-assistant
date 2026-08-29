"""Pydantic request/response models for the API (FR-15)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConversationTurnIn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ConversationTurnIn] = Field(default_factory=list)
    file_name: str | None = None
    category: str | None = None
    version: str | None = None
    config_name: str = "improved_b"


class CitationOut(BaseModel):
    tag: str
    chunk_id: str
    file_name: str
    unit_label: str
    category: str
    version: str
    doc_status: str


class EvidenceOut(BaseModel):
    chunk_id: str
    file_name: str
    unit_label: str
    category: str
    version: str
    doc_status: str
    chunk_strategy: str
    dense_score: float | None
    lexical_score: float | None
    fused_score: float | None
    rerank_score: float | None
    text: str


class AskResponse(BaseModel):
    answer_id: str
    original_question: str
    retrieval_query: str
    used_conversation_context: bool
    answer: str
    citations: list[CitationOut]
    abstained: bool
    abstention_reason: str
    evidence_quality: str
    conflict_signal: bool
    reranker_used: bool
    reranker_fallback_reason: str
    config_used: str
    retrieved_evidence: list[EvidenceOut]
    stage_timings_ms: dict[str, float]


class FeedbackRequest(BaseModel):
    answer_id: str
    question: str
    useful: bool
    reason: str = ""


class IngestRequest(BaseModel):
    """Deliberately has no path field: the API always ingests the server's
    configured `documents_dir` (see `app.config.Settings`). Accepting an
    arbitrary filesystem path from an HTTP request body would be a
    path-traversal risk; ad-hoc paths are only supported via the trusted,
    local `scripts/ingest.py --path` CLI, never over the network API."""


class ComponentHealth(BaseModel):
    ok: bool
    message: str


class HealthResponse(BaseModel):
    status: str
    vector_store: ComponentHealth
    embedding_model: ComponentHealth
    reranker: ComponentHealth
    llm: ComponentHealth
