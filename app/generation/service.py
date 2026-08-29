"""Orchestrates pipeline stages 5-7 (context selection, grounded generation,
citation assembly) plus the abstention gate that runs before generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings
from app.generation.abstention import decide_abstention
from app.generation.citations import Citation, assemble_and_validate_citations
from app.generation.conflict import detect_conflict_signal
from app.generation.context_builder import build_context
from app.generation.llm_client import LLMClient, LLMUnavailableError
from app.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from app.logging_config import StageTimer
from app.retrieval.pipeline import RetrievalOutcome

ABSTENTION_MESSAGE_TEMPLATE = (
    "I don't have sufficient reliable evidence in the indexed documents to answer this "
    "confidently: {reason}. Try rephrasing the question, or check that the relevant document "
    "has been ingested."
)
LLM_UNAVAILABLE_MESSAGE = (
    "The generation model is currently unavailable, so I can't produce a grounded answer right "
    "now ({error}). Retrieved evidence is still available below for manual review."
)


@dataclass
class AnswerResult:
    answer: str
    citations: list[Citation]
    abstained: bool
    abstention_reason: str
    evidence_quality: str
    conflict_signal: bool
    reranker_used: bool
    reranker_fallback_reason: str
    llm_error: str
    invalid_citation_count: int = 0
    stage_timings_ms: dict[str, float] = field(default_factory=dict)


def answer_question(
    retrieval_outcome: RetrievalOutcome,
    settings: Settings,
    llm_client: LLMClient,
) -> AnswerResult:
    timer = StageTimer()
    hits = retrieval_outcome.hits
    # Use retrieval_query (not the bare original question) so the coverage check is judged
    # against what was actually searched for. For a follow-up like "What about backups?" the
    # original question alone has almost no content terms to match against — retrieval_query
    # already carries the prior-turn context (e.g. "...current log retention... What about
    # backups?"), which is the fair basis for "did the evidence cover the question."
    query = retrieval_outcome.processed_query.retrieval_query

    decision = decide_abstention(query, hits, settings.min_evidence_score, settings.min_citation_coverage)
    conflict_signal = detect_conflict_signal(hits)
    base_timings = dict(retrieval_outcome.stage_timings_ms)

    if decision.should_abstain:
        return AnswerResult(
            answer=ABSTENTION_MESSAGE_TEMPLATE.format(reason=decision.reason),
            citations=[],
            abstained=True,
            abstention_reason=decision.reason,
            evidence_quality=decision.evidence_quality,
            conflict_signal=conflict_signal,
            reranker_used=retrieval_outcome.reranker_used,
            reranker_fallback_reason=retrieval_outcome.reranker_fallback_reason,
            llm_error="",
            stage_timings_ms=base_timings,
        )

    with timer.measure("context_selection"):
        bundle = build_context(hits, settings.max_context_chars)

    user_prompt = build_user_prompt(retrieval_outcome.processed_query, bundle)

    llm_error = ""
    raw_answer = ""
    with timer.measure("generation"):
        try:
            raw_answer = llm_client.generate(SYSTEM_PROMPT, user_prompt, bundle.blocks)
        except LLMUnavailableError as exc:
            llm_error = str(exc)

    if llm_error:
        return AnswerResult(
            answer=LLM_UNAVAILABLE_MESSAGE.format(error=llm_error),
            citations=[],
            abstained=True,
            abstention_reason=f"generation backend unavailable: {llm_error}",
            evidence_quality=decision.evidence_quality,
            conflict_signal=conflict_signal,
            reranker_used=retrieval_outcome.reranker_used,
            reranker_fallback_reason=retrieval_outcome.reranker_fallback_reason,
            llm_error=llm_error,
            stage_timings_ms={**base_timings, **timer.stages_ms},
        )

    with timer.measure("citation_assembly"):
        citation_result = assemble_and_validate_citations(raw_answer, bundle)

    return AnswerResult(
        answer=citation_result.cleaned_answer,
        citations=citation_result.citations,
        abstained=False,
        abstention_reason="",
        evidence_quality=decision.evidence_quality,
        conflict_signal=conflict_signal,
        reranker_used=retrieval_outcome.reranker_used,
        reranker_fallback_reason=retrieval_outcome.reranker_fallback_reason,
        llm_error="",
        invalid_citation_count=len(citation_result.invalid_tags_found),
        stage_timings_ms={**base_timings, **timer.stages_ms},
    )
