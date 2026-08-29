from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    documents_dir = tmp_path / "documents"
    documents_dir.mkdir()
    return Settings(
        data_dir=tmp_path,
        documents_dir=documents_dir,
        index_dir=tmp_path / "index",
        llm_provider="mock",
        use_reranker=True,
        min_evidence_score=0.15,
        min_citation_coverage=0.2,
    )


def write_doc(directory: Path, relative_path: str, content: str) -> Path:
    path = directory / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path
