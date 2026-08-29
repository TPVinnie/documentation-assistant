"""Runs one or more retrieval configurations over the labeled evaluation
dataset and produces machine-readable results (D5, D6, 6.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import Settings
from app.evaluation.answer_metrics import PerQuestionAnswer, compute_answer_metrics
from app.evaluation.dataset import EvalQuestion
from app.evaluation.retrieval_metrics import PerQuestionRetrieval, compute_retrieval_metrics
from app.generation.llm_client import LLMClient
from app.generation.service import answer_question
from app.indexing.lexical_index import LexicalIndex
from app.indexing.vector_store import VectorStore
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import RetrievalConfig, retrieve

logger = logging.getLogger(__name__)


@dataclass
class QuestionRunLog:
    """Full per-question trace, used for error analysis (6.3)."""

    question_id: str
    category: str
    question: str
    answerability: str
    expected_answer: str
    expected_sources: list[str]
    retrieved_file_names: list[str]
    answer_text: str
    abstained: bool
    abstention_reason: str
    citations: list[str]
    conflict_signal: bool
    reranker_used: bool
    llm_error: str
    retrieval_latency_ms: float
    end_to_end_latency_ms: float


def run_config(
    config: RetrievalConfig,
    questions: list[EvalQuestion],
    settings: Settings,
    vector_store: VectorStore,
    lexical_index: LexicalIndex,
    llm_client: LLMClient,
) -> dict:
    retrieval_records: list[PerQuestionRetrieval] = []
    answer_records: list[PerQuestionAnswer] = []
    run_logs: list[QuestionRunLog] = []

    for question in questions:
        outcome = retrieve(
            question.question,
            settings,
            config,
            vector_store,
            lexical_index,
            filters=RetrievalFilters(),
            history=question.history,
        )
        retrieval_latency_ms = sum(outcome.stage_timings_ms.values())
        retrieved_file_names = [hit.metadata.get("file_name", "") for hit in outcome.hits]
        retrieval_records.append(
            PerQuestionRetrieval(
                question_id=question.id,
                category=question.category,
                expected_sources=question.expected_sources,
                retrieved_file_names=retrieved_file_names,
                retrieval_latency_ms=retrieval_latency_ms,
            )
        )

        result = answer_question(outcome, settings, llm_client)
        end_to_end_latency_ms = sum(result.stage_timings_ms.values())
        cited_chunk_ids = {c.chunk_id for c in result.citations}
        cited_chunk_texts = [hit.text for hit in outcome.hits if hit.chunk_id in cited_chunk_ids]

        answer_records.append(
            PerQuestionAnswer(
                question_id=question.id,
                category=question.category,
                answerability=question.answerability,
                expected_answer=question.expected_answer,
                abstained=result.abstained,
                llm_error=result.llm_error,
                answer_text=result.answer,
                citations_valid_count=len(result.citations),
                citations_invalid_count=result.invalid_citation_count,
                cited_chunk_texts=cited_chunk_texts,
                conflict_signal=result.conflict_signal,
                end_to_end_latency_ms=end_to_end_latency_ms,
            )
        )

        run_logs.append(
            QuestionRunLog(
                question_id=question.id,
                category=question.category,
                question=question.question,
                answerability=question.answerability,
                expected_answer=question.expected_answer,
                expected_sources=question.expected_sources,
                retrieved_file_names=retrieved_file_names,
                answer_text=result.answer,
                abstained=result.abstained,
                abstention_reason=result.abstention_reason,
                citations=[f"[{c.tag}] {c.file_name} — {c.unit_label}" for c in result.citations],
                conflict_signal=result.conflict_signal,
                reranker_used=result.reranker_used,
                llm_error=result.llm_error,
                retrieval_latency_ms=round(retrieval_latency_ms, 2),
                end_to_end_latency_ms=round(end_to_end_latency_ms, 2),
            )
        )

    retrieval_report = compute_retrieval_metrics(retrieval_records)
    answer_report = compute_answer_metrics(answer_records)

    logger.info("evaluation_config_complete", extra={"config": config.name, "question_count": len(questions)})

    return {
        "config_name": config.name,
        "config": config.__dict__,
        "question_count": len(questions),
        "retrieval_metrics": {
            "overall": retrieval_report.overall,
            "by_category": retrieval_report.by_category,
        },
        "answer_metrics": {
            "overall": answer_report.overall,
            "by_category": answer_report.by_category,
        },
        "run_log": [log.__dict__ for log in run_logs],
    }
