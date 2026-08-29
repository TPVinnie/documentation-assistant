#!/usr/bin/env python
"""One-off correction: recomputes `correct_non_fabrication_rate` in already-run
evaluation results from their saved `run_log` (answer_text + abstained +
answerability), without re-invoking retrieval or the LLM.

Why this exists: the original `_SOFT_REFUSAL_KEYWORDS` list missed the
llama3.2:3b phrasing "there is no direct information/statement...", so two
genuinely correct declines were miscounted as failures the first time
`scripts/evaluate.py` was run. The keyword list has since been fixed
(app/evaluation/answer_metrics.py). Since the fix is purely in how an
already-recorded answer is *scored*, not in retrieval or generation, it is
reproduced here from the existing run_log instead of re-running the full
(LLM-inference-bound) evaluation a second time. See TECHNICAL_REPORT.md
Section 5 for the full writeup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.evaluation.answer_metrics import _declined  # noqa: E402 (see sys.path insert above)


def _recompute_for_run_log(run_log: list[dict], answerability_key: str = "answerability") -> dict:
    unanswerable = [r for r in run_log if r[answerability_key] == "unanswerable"]
    if not unanswerable:
        return {"correct_non_fabrication_rate": 0.0, "unanswerable_n": 0}

    class _Row:
        def __init__(self, d: dict) -> None:
            self.abstained = d["abstained"]
            self.answer_text = d["answer_text"]

    declined = sum(1 for r in unanswerable if _declined(_Row(r)))
    return {
        "correct_non_fabrication_rate": round(declined / len(unanswerable), 4),
        "unanswerable_n": len(unanswerable),
    }


def main() -> None:
    artifacts_dir = Path("artifacts")
    comparison_path = artifacts_dir / "results_comparison.json"
    comparison = json.loads(comparison_path.read_text())

    for config_name in comparison["configs"]:
        result_path = artifacts_dir / f"results_{config_name}.json"
        result = json.loads(result_path.read_text())

        recomputed = _recompute_for_run_log(result["run_log"])
        old_rate = result["answer_metrics"]["overall"]["abstention"]["correct_non_fabrication_rate"]
        result["answer_metrics"]["overall"]["abstention"]["correct_non_fabrication_rate"] = recomputed[
            "correct_non_fabrication_rate"
        ]
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        comparison["by_config"][config_name]["correct_non_fabrication_rate"] = recomputed[
            "correct_non_fabrication_rate"
        ]
        print(
            f"{config_name}: correct_non_fabrication_rate {old_rate} -> "
            f"{recomputed['correct_non_fabrication_rate']}"
        )

    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(f"Patched {comparison_path}")


if __name__ == "__main__":
    main()
