# Handover Summary

## What's implemented

All functional requirements (FR-01 through FR-16) and the advanced technical requirements (4.1
transparent 7-stage pipeline, 4.2 three-configuration comparison, 4.3 responsible failure
handling, 4.4 local-first with a mock offline mode) have a working implementation. See
`ARCHITECTURE.md` for the component map and `TECHNICAL_REPORT.md` for the evaluation results,
error analysis, and design rationale.

Quick orientation for a new developer:

1. Read `README.md` (setup/run/evaluate) and `ARCHITECTURE.md` (how it fits together).
2. `app/retrieval/pipeline.py` and `app/generation/service.py` are the two files that show the
   whole request lifecycle end to end — start there.
3. `app/evaluation/configs.py` defines the three retrieval configurations; `scripts/evaluate.py`
   is the reproducible entry point for comparing them.
4. `data/evaluation/eval_questions.jsonl` is the labeled question set; `TECHNICAL_REPORT.md`
   Section "Error analysis" walks through specific failures found while using it.

## Known issues / limitations (see `TECHNICAL_REPORT.md` for the full list)

- **Answer-quality metrics are lexical-overlap heuristics**, not an LLM-judge or human evaluation.
  They're fast and fully local, but a paraphrased-correctly answer can score lower than a
  word-for-word one, and vice versa for a fluent-but-wrong answer. Do not quote them as
  ground truth without spot-checking the underlying `run_log` entries in `artifacts/results_*.json`.
- **Conflict detection (FR-13) is prompt-driven**, backed only by a coarse
  "evidence came from >1 document" signal (`app/generation/conflict.py`) for transparency
  metadata. There's no real semantic contradiction detector — the LLM is asked to notice and
  describe disagreement, which is a behavior, not a proof.
- **The injection guard is regex-based** (`app/generation/injection_guard.py`). It catches the
  phrasings exercised by the adversarial eval questions and is intentionally biased toward
  over-flagging, but it will miss novel phrasings a real attacker might use. A production version
  would want a dedicated classifier or a second LLM pass.
- **Duplicate-file detection is order-dependent**: when two files have byte-identical content, which
  one is kept as canonical and which is flagged "duplicate" depends on filesystem scan order, not
  file age or path. Functionally correct (exactly one is indexed), but not deterministic across
  machines — documented in `data/SAMPLE_CORPUS_NOTES.md`.
- **Version/family grouping** (`app/ingestion/doc_metadata.py`) groups documents by
  `(category, title)` where title comes from front-matter or a filename heuristic. A document
  family that doesn't share an exact title string across versions won't be linked — this depends
  on documents following a consistent naming/heading convention.
- **No authentication/authorization** on the API — out of scope per the assignment (SSO/enterprise
  deployment explicitly excluded), but a real deployment needs it before exposing `/ingest` or
  `/feedback`.
- **Single-process, single-machine**: no distributed indexing, no multi-tenant isolation. See
  "Production roadmap" in `TECHNICAL_REPORT.md`.

## Next priorities, roughly in order

1. Replace the lexical-overlap answer metrics with (or supplement them alongside) an LLM-judge
   pass for groundedness/completeness, now that the abstention-scale bug (see
   `TECHNICAL_REPORT.md` "Error analysis") is fixed and the baseline numbers are trustworthy.
2. Add a real semantic contradiction check (e.g., an NLI model or a dedicated LLM prompt pass
   over pairs of top candidates) rather than relying solely on the generation-time prompt rule.
3. Add authentication and per-request rate limiting before any non-local deployment.
4. Expand the injection-pattern set based on adversarial red-teaming beyond the 3 eval cases here.

## Where to find things

| Need | Location |
|---|---|
| Change a retrieval/chunking default | `app/config.py` / `.env` |
| Add a new retrieval configuration to compare | `app/evaluation/configs.py` |
| Add an eval question | `data/evaluation/eval_questions.jsonl` (see field docs in `app/evaluation/dataset.py`) |
| Change the system prompt / citation format | `app/generation/prompt.py` |
| Add an injection pattern | `app/generation/injection_guard.py` |
| Regenerate the synthetic corpus | `scripts/make_corpus.py` |
