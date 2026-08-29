from __future__ import annotations

from app.evaluation.answer_metrics import (
    PerQuestionAnswer,
    abstention_accuracy,
    citation_metrics,
    conflict_handling_rate,
    groundedness_proxy,
)


def _answer(**overrides) -> PerQuestionAnswer:
    defaults = dict(
        question_id="q1",
        category="direct_factual",
        answerability="answerable",
        expected_answer="",
        abstained=False,
        llm_error="",
        answer_text="",
        citations_valid_count=0,
        citations_invalid_count=0,
        cited_chunk_texts=[],
        conflict_signal=False,
        end_to_end_latency_ms=100.0,
    )
    defaults.update(overrides)
    return PerQuestionAnswer(**defaults)


def test_correct_abstention_only_credits_the_code_level_gate():
    records = [
        _answer(answerability="unanswerable", abstained=True),
        _answer(answerability="unanswerable", abstained=False, answer_text="60 days, per the policy."),
    ]
    result = abstention_accuracy(records)
    assert result["correct_abstention_rate"] == 0.5
    assert result["unanswerable_n"] == 2


def test_correct_non_fabrication_also_credits_a_soft_refusal_in_the_generated_text():
    records = [
        _answer(answerability="unanswerable", abstained=True),
        _answer(
            answerability="unanswerable",
            abstained=False,
            answer_text="The evidence sources do not mention this topic.",
        ),
        _answer(answerability="unanswerable", abstained=False, answer_text="It is 60 days."),
    ]
    result = abstention_accuracy(records)
    assert result["correct_abstention_rate"] == round(1 / 3, 4)
    assert result["correct_non_fabrication_rate"] == round(2 / 3, 4)


def test_over_abstention_measured_only_on_answerable_questions():
    records = [
        _answer(answerability="answerable", abstained=True),
        _answer(answerability="answerable", abstained=False),
    ]
    result = abstention_accuracy(records)
    assert result["over_abstention_rate_on_answerable"] == 0.5


def test_citation_correctness_counts_valid_vs_invalid_tags():
    records = [
        _answer(abstained=False, citations_valid_count=2, citations_invalid_count=1),
        _answer(abstained=False, citations_valid_count=1, citations_invalid_count=0),
    ]
    result = citation_metrics(records)
    assert result["citation_correctness"] == 3 / 4
    assert result["citation_coverage"] == 1.0


def test_groundedness_proxy_measures_word_overlap_with_cited_sources():
    records = [
        _answer(
            abstained=False,
            answer_text="Logs are retained for sixty days per policy.",
            cited_chunk_texts=["Log data must be retained for sixty days from generation."],
        )
    ]
    result = groundedness_proxy(records)
    assert result["n"] == 1
    assert 0.0 < result["lexical_overlap_score"] <= 1.0


def test_conflict_handling_rate_only_scoped_to_contradictory_evidence_category():
    records = [
        _answer(category="contradictory_evidence", answer_text="The sources disagree on this."),
        _answer(category="contradictory_evidence", answer_text="It is definitely 500 per hour."),
        _answer(category="direct_factual", answer_text="disagree disagree disagree"),
    ]
    result = conflict_handling_rate(records)
    assert result["n"] == 2
    assert result["conflict_handling_rate"] == 0.5
