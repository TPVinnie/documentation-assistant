#!/usr/bin/env python
"""CLI test client: ask a question directly against the pipeline (no API
server required).

Usage:
    python scripts/ask.py "How long must log data be retained?"
    python scripts/ask.py "What about backups?" --config baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.evaluation.configs import ALL_CONFIGS
from app.generation.llm_client import build_llm_client
from app.generation.service import answer_question
from app.indexing.lexical_index import LexicalIndex
from app.indexing.vector_store import VectorStore
from app.logging_config import configure_logging
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--config", default="improved_b", choices=list(ALL_CONFIGS))
    parser.add_argument("--category", default=None)
    parser.add_argument("--file-name", dest="file_name", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--show-evidence", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    vector_store = VectorStore(settings.chroma_dir)
    lexical_index = LexicalIndex(settings.bm25_path)
    llm_client = build_llm_client(settings.llm_provider, settings.ollama_host, settings.ollama_model, settings.llm_timeout_seconds)

    config = ALL_CONFIGS[args.config]
    filters = RetrievalFilters(file_name=args.file_name, category=args.category, version=args.version)

    outcome = retrieve(args.question, settings, config, vector_store, lexical_index, filters=filters)
    result = answer_question(outcome, settings, llm_client)

    print(f"Question: {outcome.processed_query.original}")
    print(f"Config: {config.name}\n")
    print(f"Answer: {result.answer}\n")
    print(f"Abstained: {result.abstained} ({result.abstention_reason})" if result.abstained else "Abstained: False")
    print(f"Evidence quality: {result.evidence_quality} | Conflict signal: {result.conflict_signal}")
    print(f"Reranker used: {result.reranker_used} ({result.reranker_fallback_reason})")
    print("\nCitations:")
    for c in result.citations:
        print(f"  [{c.tag}] {c.file_name} — {c.unit_label} ({c.doc_status})")

    if args.show_evidence:
        print("\nRetrieved evidence:")
        for hit in outcome.hits:
            print(f"  {hit.chunk_id} | dense={hit.dense_score} lexical={hit.lexical_score} "
                  f"fused={hit.fused_score} rerank={hit.rerank_score}")
            print(f"    {hit.text[:200]}...")

    print("\nStage timings (ms):", json.dumps(result.stage_timings_ms, indent=2))


if __name__ == "__main__":
    main()
