from __future__ import annotations

from app.config import Settings
from app.evaluation.configs import BASELINE, IMPROVED_A, IMPROVED_B
from app.indexing.lexical_index import LexicalIndex
from app.indexing.vector_store import VectorStore
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve
from tests.conftest import write_doc


def _seed_corpus(test_settings: Settings) -> None:
    docs_dir = test_settings.documents_dir
    write_doc(
        docs_dir,
        "retention-v1.md",
        """---
title: Retention Policy
category: policy
version: 1.0
effective_date: 2023-01-01
---

# Retention

Logs are kept for 30 days.
""",
    )
    write_doc(
        docs_dir,
        "retention-v2.md",
        """---
title: Retention Policy
category: policy
version: 2.0
effective_date: 2024-01-01
---

# Retention

Logs are kept for 60 days.
""",
    )
    write_doc(
        docs_dir,
        "onboarding.md",
        """---
category: procedure
---

# Onboarding

New hires receive a laptop on day one.
""",
    )
    IngestionPipeline(test_settings).run()


def test_retrieval_returns_current_version_ranked_above_superseded(test_settings: Settings):
    _seed_corpus(test_settings)
    vector_store = VectorStore(test_settings.chroma_dir)
    lexical_index = LexicalIndex(test_settings.bm25_path)

    outcome = retrieve(
        "how long are logs kept?", test_settings, IMPROVED_B, vector_store, lexical_index
    )

    file_names = [h.metadata["file_name"] for h in outcome.hits]
    assert "retention-v2.md" in file_names
    doc_statuses = {h.metadata["file_name"]: h.metadata["doc_status"] for h in outcome.hits}
    assert doc_statuses["retention-v2.md"] == "current"
    # current-version evidence should not rank below the superseded version for the same fact
    v2_rank = file_names.index("retention-v2.md")
    if "retention-v1.md" in file_names:
        assert v2_rank < file_names.index("retention-v1.md")


def test_category_filter_excludes_other_categories(test_settings: Settings):
    _seed_corpus(test_settings)
    vector_store = VectorStore(test_settings.chroma_dir)
    lexical_index = LexicalIndex(test_settings.bm25_path)

    outcome = retrieve(
        "laptop",
        test_settings,
        IMPROVED_B,
        vector_store,
        lexical_index,
        filters=RetrievalFilters(category="policy"),
    )

    assert all(h.metadata["category"] == "policy" for h in outcome.hits)


def test_baseline_config_uses_dense_only_no_reranker(test_settings: Settings):
    _seed_corpus(test_settings)
    vector_store = VectorStore(test_settings.chroma_dir)
    lexical_index = LexicalIndex(test_settings.bm25_path)

    outcome = retrieve("how long are logs kept?", test_settings, BASELINE, vector_store, lexical_index)

    assert outcome.reranker_used is False
    assert all(h.lexical_score is None for h in outcome.hits)


def test_improved_a_uses_hybrid_without_reranker(test_settings: Settings):
    _seed_corpus(test_settings)
    vector_store = VectorStore(test_settings.chroma_dir)
    lexical_index = LexicalIndex(test_settings.bm25_path)

    outcome = retrieve("how long are logs kept?", test_settings, IMPROVED_A, vector_store, lexical_index)

    assert outcome.reranker_used is False
    assert any(h.lexical_score is not None for h in outcome.hits)
