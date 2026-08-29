#!/usr/bin/env python
"""CLI: run baseline/improved_a/improved_b over the labeled evaluation
dataset and write machine-readable results (D5, D6).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --configs baseline improved_b
    python scripts/evaluate.py --dataset data/evaluation/eval_questions.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.evaluation.configs import ALL_CONFIGS
from app.evaluation.dataset import load_eval_questions
from app.evaluation.runner import run_config
from app.generation.llm_client import build_llm_client
from app.indexing.lexical_index import LexicalIndex
from app.indexing.vector_store import VectorStore
from app.logging_config import configure_logging


def build_comparison(results: dict[str, dict]) -> dict:
    comparison = {"configs": list(results.keys()), "by_config": {}}
    for name, result in results.items():
        r = result["retrieval_metrics"]["overall"]
        a = result["answer_metrics"]["overall"]
        comparison["by_config"][name] = {
            "hit_rate_at_k": r["hit_rate_at_k"],
            "mrr": r["mrr"],
            "citation_source_coverage": r["citation_source_coverage"],
            "retrieval_latency_avg_ms": r["latency"]["avg_ms"],
            "retrieval_latency_p95_ms": r["latency"]["p95_ms"],
            "correct_abstention_rate": a["abstention"]["correct_abstention_rate"],
            "correct_non_fabrication_rate": a["abstention"]["correct_non_fabrication_rate"],
            "over_abstention_rate_on_answerable": a["abstention"]["over_abstention_rate_on_answerable"],
            "citation_correctness": a["citations"]["citation_correctness"],
            "citation_coverage": a["citations"]["citation_coverage"],
            "groundedness_lexical_overlap": a["groundedness_proxy"]["lexical_overlap_score"],
            "answer_completeness_proxy": a["answer_completeness_proxy"]["key_fact_coverage"],
            "conflict_handling_rate": a["conflict_handling"]["conflict_handling_rate"],
            "version_recency_accuracy": a["version_recency"]["version_recency_accuracy"],
            "end_to_end_latency_avg_ms": a["latency"]["avg_ms"],
            "end_to_end_latency_p95_ms": a["latency"]["p95_ms"],
            "failure_rate": a["failure_rate"],
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/evaluation/eval_questions.jsonl"))
    parser.add_argument("--configs", nargs="+", default=list(ALL_CONFIGS), choices=list(ALL_CONFIGS))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    questions = load_eval_questions(args.dataset)
    print(f"Loaded {len(questions)} evaluation questions from {args.dataset}", file=sys.stderr)

    vector_store = VectorStore(settings.chroma_dir)
    lexical_index = LexicalIndex(settings.bm25_path)
    llm_client = build_llm_client(
        settings.llm_provider, settings.ollama_host, settings.ollama_model, settings.llm_timeout_seconds
    )

    results: dict[str, dict] = {}
    for config_name in args.configs:
        config = ALL_CONFIGS[config_name]
        print(f"Running config '{config_name}'...", file=sys.stderr)
        result = run_config(config, questions, settings, vector_store, lexical_index, llm_client)
        results[config_name] = result

        out_path = args.output_dir / f"results_{config_name}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"  wrote {out_path}", file=sys.stderr)

    comparison = build_comparison(results)
    comparison_path = args.output_dir / "results_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Wrote comparison to {comparison_path}", file=sys.stderr)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
