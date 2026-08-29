"""Exercises app.evaluation.runner end to end (ingest -> retrieve -> generate
-> metrics) using the Mock LLM client, against a tiny fixture dataset."""

from __future__ import annotations

from app.config import Settings
from app.evaluation.configs import BASELINE
from app.evaluation.dataset import EvalQuestion
from app.evaluation.runner import run_config
from app.generation.llm_client import build_llm_client
from app.indexing.lexical_index import LexicalIndex
from app.indexing.vector_store import VectorStore
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.query_processing import ConversationTurn
from tests.conftest import write_doc


def test_run_config_produces_well_formed_metrics_and_run_log(test_settings: Settings):
    write_doc(
        test_settings.documents_dir,
        "retention.md",
        """---
title: Retention Policy
category: policy
version: 1.0
effective_date: 2024-01-01
---

Log data must be retained for 60 days.
""",
    )
    IngestionPipeline(test_settings).run()

    questions = [
        EvalQuestion(
            id="Q1", category="direct_factual", question="How long is log data retained?",
            answerability="answerable", expected_answer="60 days", expected_sources=["retention.md"],
        ),
        EvalQuestion(
            id="Q2", category="unanswerable", question="What is the capital of France?",
            answerability="unanswerable", expected_answer="not covered", expected_sources=[],
        ),
        EvalQuestion(
            id="Q3", category="conversation_follow_up", question="What about that?",
            answerability="answerable", expected_answer="60 days", expected_sources=["retention.md"],
            history=[ConversationTurn(role="user", content="How long is log data retained?")],
        ),
    ]

    vector_store = VectorStore(test_settings.chroma_dir)
    lexical_index = LexicalIndex(test_settings.bm25_path)
    llm_client = build_llm_client("mock", test_settings.ollama_host, test_settings.ollama_model, 5.0)

    result = run_config(BASELINE, questions, test_settings, vector_store, lexical_index, llm_client)

    assert result["config_name"] == "baseline"
    assert result["question_count"] == 3
    assert len(result["run_log"]) == 3

    retrieval_overall = result["retrieval_metrics"]["overall"]
    assert retrieval_overall["total_question_count"] == 3
    assert 0.0 <= retrieval_overall["hit_rate_at_k"] <= 1.0

    answer_overall = result["answer_metrics"]["overall"]
    assert answer_overall["abstention"]["unanswerable_n"] == 1
    assert answer_overall["abstention"]["answerable_n"] == 2

    q2_log = next(entry for entry in result["run_log"] if entry["question_id"] == "Q2")
    assert q2_log["abstained"] is True
