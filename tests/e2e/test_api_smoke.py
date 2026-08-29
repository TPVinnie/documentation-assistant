"""End-to-end smoke test through the actual FastAPI app (D8, FR-15), using
the Mock LLM provider and an isolated tmp_path index so it never touches the
real corpus/index and needs no network access.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.conftest import write_doc


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    write_doc(
        documents_dir,
        "policy.md",
        """---
title: Retention Policy
category: policy
version: 1.0
effective_date: 2024-01-01
---

# Retention

Log data must be retained for 60 days.
""",
    )

    monkeypatch.setenv("DOCS_ASSISTANT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCS_ASSISTANT_DOCUMENTS_DIR", str(documents_dir))
    monkeypatch.setenv("DOCS_ASSISTANT_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("DOCS_ASSISTANT_LLM_PROVIDER", "mock")

    from app.api import main as api_main

    with TestClient(api_main.app) as client:
        yield client


def test_health_endpoint_reports_ok(api_client: TestClient):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["llm"]["ok"] is True


def test_ingest_then_ask_then_evidence_then_feedback(api_client: TestClient):
    ingest_response = api_client.post("/ingest", json={})
    assert ingest_response.status_code == 200
    assert ingest_response.json()["summary"]["indexed"] == 1

    status_response = api_client.get("/status")
    assert status_response.json()["file_count"] == 1

    ask_response = api_client.post("/ask", json={"question": "How long is log data retained?"})
    assert ask_response.status_code == 200
    body = ask_response.json()
    assert body["abstained"] is False
    assert len(body["citations"]) >= 1

    chunk_id = body["citations"][0]["chunk_id"]
    evidence_response = api_client.get(f"/evidence/{chunk_id}")
    assert evidence_response.status_code == 200
    assert "60 days" in evidence_response.json()["text"]

    feedback_response = api_client.post(
        "/feedback",
        json={"answer_id": body["answer_id"], "question": "How long is log data retained?", "useful": True},
    )
    assert feedback_response.status_code == 200

    facets_response = api_client.get("/facets")
    assert facets_response.status_code == 200
    facets = facets_response.json()
    assert facets["categories"] == ["policy"]
    assert facets["file_names"] == ["policy.md"]

    status_after = api_client.get("/status").json()
    assert status_after["feedback_summary"]["useful"] == 1


def test_ask_with_unknown_config_returns_400(api_client: TestClient):
    response = api_client.post("/ask", json={"question": "test", "config_name": "not_a_real_config"})
    assert response.status_code == 400


def test_evidence_for_unknown_chunk_returns_404(api_client: TestClient):
    response = api_client.get("/evidence/does-not-exist")
    assert response.status_code == 404
