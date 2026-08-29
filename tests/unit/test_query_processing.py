from __future__ import annotations

from app.retrieval.query_processing import ConversationTurn, process_query


def test_normalizes_whitespace():
    result = process_query("  What   is\tthe   retention  period?  ")
    assert result.normalized == "What is the retention period?"
    assert result.original != result.normalized


def test_no_history_leaves_retrieval_query_unchanged():
    result = process_query("What is the retention period?")
    assert result.retrieval_query == result.normalized
    assert result.used_conversation_context is False


def test_short_follow_up_pulls_in_last_user_question():
    history = [
        ConversationTurn(role="user", content="What is the current log retention period?"),
        ConversationTurn(role="assistant", content="60 days."),
    ]
    result = process_query("What about backups?", history)

    assert result.used_conversation_context is True
    assert "log retention period" in result.retrieval_query
    assert result.original == "What about backups?"


def test_long_self_contained_question_does_not_pull_in_history():
    history = [
        ConversationTurn(role="user", content="What is the current log retention period?"),
        ConversationTurn(role="assistant", content="60 days."),
    ]
    long_question = "What is the minimum password length required by the access control policy?"
    result = process_query(long_question, history)

    assert result.used_conversation_context is False
    assert result.retrieval_query == long_question
