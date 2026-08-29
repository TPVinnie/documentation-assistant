# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local-first RAG system: ingests documents, retrieves with hybrid (dense + BM25) search, reranks,
and generates cited, grounded answers with abstention when evidence is weak. Built for an
assignment — see `README.md` for full setup/usage, `ARCHITECTURE.md` for component/sequence
diagrams and design-decision rationale, and `TECHNICAL_REPORT.md` for evaluation results, a
classified error analysis (real bugs found and fixed during evaluation, with root causes), and
known limitations. Read those before making non-trivial changes — this document only covers what's
needed to start working productively.

## Commands

Environment is a local `.venv` (not system Python):

```bash
source .venv/bin/activate
```

**Setup** (idempotent): `bash scripts/setup.sh` — creates the venv, installs deps, copies `.env`,
generates the synthetic corpus.

**Corpus + index** (required before retrieval/API will return real results):
```bash
python scripts/make_corpus.py   # regenerates data/sample_documents/ (synthetic — no doc pack was supplied)
python scripts/ingest.py        # indexes it into data/index/ (Chroma + BM25 + SQLite); incremental by checksum
python scripts/ingest.py --path /other/dir   # index an arbitrary folder instead (CLI only — the API's
                                              # POST /ingest deliberately takes no path, see app/api/schemas.py)
```
Delete `data/index/` any time to force a full rebuild; it's fully derived from `data/sample_documents/`.

**Run**:
```bash
uvicorn app.api.main:app --reload      # API + UI at http://localhost:8000/ui
python scripts/ask.py "question" [--config baseline|improved_a|improved_b] [--show-evidence]
```
`DOCS_ASSISTANT_LLM_PROVIDER=mock` (env var) swaps Ollama for a deterministic, offline stub — use
it for fast iteration; this is also what the entire automated test suite runs under, so tests need
no running model or network access.

**Test**:
```bash
pytest                                          # all tests (unit + integration + e2e)
pytest tests/unit/test_fusion.py                # one file
pytest tests/unit/test_fusion.py::test_rrf_favors_items_ranked_highly_in_both_lists  # one test
pytest --cov=app --cov-report=term-missing      # coverage
```
`tests/integration/` and `tests/e2e/` build real (tmp-dir-isolated) Chroma/BM25/SQLite indexes and
load the embedding + cross-encoder models, so they're slower (~10-20s) than `tests/unit/` (~1s).

**Lint**: `ruff check app scripts tests` (config in `pyproject.toml`; `ruff check --fix` for
mechanical fixes). No separate formatter/type-checker is wired in.

**Evaluate** (compares the three retrieval configurations against the labeled question set):
```bash
python scripts/evaluate.py                              # all 3 configs, ~40-70 min (real LLM inference)
python scripts/evaluate.py --configs baseline improved_b # subset
```
Writes `artifacts/results_<config>.json` and `artifacts/results_comparison.json` — these are
committed deliverables (D6), not build output; re-running overwrites them in place. Set
`DOCS_ASSISTANT_LLM_PROVIDER=mock` first for a near-instant run of the pipeline mechanics
(retrieval metrics are unaffected either way — they don't depend on the LLM).

## Architecture

**Request flow** (`POST /ask`, `scripts/ask.py`, and `app/evaluation/runner.py` all drive this same
path): `retrieval.pipeline.retrieve()` runs stages 1-4 (query processing → dense+lexical candidates
→ RRF fusion → reranking) and returns a `RetrievalOutcome`; `generation.service.answer_question()`
takes that outcome through stages 5-7 (abstention gate → context building → generation → citation
validation). The two are separate calls in every entry point specifically so retrieval can be
inspected/evaluated independently of generation — don't collapse them into one function.

**Two chunking strategies are indexed simultaneously**, not swapped between runs. Every ingested
document is chunked both ways (`app/indexing/chunking.py`: `fixed_window` and `structure_aware`),
and `chunk_strategy` is just a field on a `RetrievalConfig` (`app/evaluation/configs.py`) that
selects which chunk set to query. This is what makes `scripts/evaluate.py` able to compare chunking
approaches without re-ingesting.

**Score scales are not interchangeable — this caused a real bug** (see `TECHNICAL_REPORT.md` §5,
error #1). RRF fused scores (~0.01-0.03), cross-encoder rerank scores (unbounded, ~±10), and cosine
similarity (0-1) coexist on the same `Candidate` object (`app/retrieval/candidates.py`).
`Candidate.final_score` picks whichever scoring stage ran last (for *ranking*); anything that needs
an absolute, comparable threshold (currently just the abstention gate,
`app/generation/abstention.py`) must use `Candidate.quality_score` (always `dense_score`) instead.
Don't compare `final_score` against a fixed threshold.

**Version/recency is a corpus-wide, post-hoc computation, not a per-file decision.**
`IngestionPipeline._recompute_version_status` (`app/ingestion/pipeline.py`) runs after every file
is processed, groups all files by `(category, title)`, and picks the "current" one — so ingesting
a new version of a document can flip the `doc_status` of an *unrelated, untouched* file from a
previous run. Superseded documents stay retrievable (ranked lower via a penalty in
`retrieval/fusion.py`), never filtered out — version/conflict questions need them.

**Metadata (category/title/version/effective_date) comes from document front matter first**, with
filename/folder-name fallbacks only when front matter is absent (`app/ingestion/doc_metadata.py`,
`app/ingestion/parsers.py`). PDFs are the one format where front matter isn't guaranteed to have a
blank-line/`---` separator after it (PDF text extraction drops truly blank lines) — see
`parse_pdf`'s handling of `body_start`. If a file's category looks wrong, check whether it has
front matter at all before assuming the fallback logic is broken.

**Everything hits an interface, never a concrete client directly**: `app.generation.llm_client`
(`OllamaClient` / `MockClient`), and reranking always goes through
`app.reranking.fallback.apply_reranking`, which catches exceptions and falls back to the fused
order rather than failing the request — `reranker_used` and `reranker_fallback_reason` are threaded
all the way to the API response so this is never silent.

**Citations are validated, not trusted**: the model is asked to cite `[S<n>]` tags matching the
context blocks it was actually given (`app/generation/context_builder.py` builds them,
`app/generation/prompt.py` explains the format); `app/generation/citations.py` strips any tag that
doesn't correspond to a real block before the answer reaches the caller. If citations look wrong,
check `invalid_citation_count` before assuming the model or the prompt is at fault.

**The evaluation harness reuses the exact same retrieval/generation code paths** as the API
(`app/evaluation/runner.py` calls `retrieval.pipeline.retrieve()` and
`generation.service.answer_question()` directly) — there is no separate "eval mode" implementation
to keep in sync.

## Working in this repo

- The synthetic corpus (`data/sample_documents/`, generated by `scripts/make_corpus.py`) has
  deliberately-built scenarios: a superseded-version pair, a genuine cross-document contradiction
  (two different, both-current documents disagreeing), and two differently-styled embedded
  prompt-injection attempts. See `data/SAMPLE_CORPUS_NOTES.md` before adding fixture content — it's
  easy to accidentally defeat one of these by editing the wrong file.
- `data/evaluation/eval_questions.jsonl` (43 labeled questions) has required minimum counts per
  category (enforced by `tests/unit/test_eval_dataset.py`) — don't remove questions without
  checking those minimums still hold.
- `app/ui/index.html` is a single static file (fetch-based, no build step) mounted directly by
  FastAPI's `StaticFiles`; there's no bundler/framework to wire up.
