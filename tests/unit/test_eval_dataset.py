"""Loads the actual committed evaluation dataset — doubles as a regression
check that it stays valid and meets the assignment's category minimums."""

from __future__ import annotations

from pathlib import Path

from app.evaluation.dataset import load_eval_questions

DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "evaluation" / "eval_questions.jsonl"

_REQUIRED_MINIMUMS = {
    "direct_factual": 6,
    "multi_chunk_synthesis": 6,
    "multi_document_comparison": 5,
    "version_recency": 4,
    "contradictory_evidence": 4,
    "ambiguous_query": 4,
    "unanswerable": 6,
    "conversation_follow_up": 5,
    "adversarial_injection": 3,
}


def test_dataset_meets_required_category_minimums():
    questions = load_eval_questions(DATASET_PATH)
    assert len(questions) >= 43

    counts: dict[str, int] = {}
    for q in questions:
        counts[q.category] = counts.get(q.category, 0) + 1

    for category, minimum in _REQUIRED_MINIMUMS.items():
        assert counts.get(category, 0) >= minimum, f"{category} has {counts.get(category, 0)}, needs >= {minimum}"


def test_every_question_has_an_id_and_valid_answerability():
    questions = load_eval_questions(DATASET_PATH)
    seen_ids = set()
    for q in questions:
        assert q.id not in seen_ids, f"duplicate question id {q.id}"
        seen_ids.add(q.id)
        assert q.answerability in {"answerable", "unanswerable"}
        assert q.question.strip()


def test_conversation_follow_up_questions_carry_history():
    questions = load_eval_questions(DATASET_PATH)
    follow_ups = [q for q in questions if q.category == "conversation_follow_up"]
    assert follow_ups
    for q in follow_ups:
        assert len(q.history) >= 1
