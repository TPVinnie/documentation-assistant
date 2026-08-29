"""Loads and validates the labeled evaluation dataset (D4)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.retrieval.query_processing import ConversationTurn

REQUIRED_FIELDS = {"id", "category", "question", "answerability", "expected_answer", "expected_sources"}


@dataclass
class EvalQuestion:
    id: str
    category: str
    question: str
    answerability: str  # "answerable" | "unanswerable"
    expected_answer: str
    expected_sources: list[str]
    notes: str = ""
    history: list[ConversationTurn] = field(default_factory=list)


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = REQUIRED_FIELDS - record.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing required fields: {missing}")
            history = [ConversationTurn(role=t["role"], content=t["content"]) for t in record.get("history", [])]
            questions.append(
                EvalQuestion(
                    id=record["id"],
                    category=record["category"],
                    question=record["question"],
                    answerability=record["answerability"],
                    expected_answer=record["expected_answer"],
                    expected_sources=record["expected_sources"],
                    notes=record.get("notes", ""),
                    history=history,
                )
            )
    return questions
