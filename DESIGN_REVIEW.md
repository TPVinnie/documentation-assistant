# Design Review

A single-document, offline-readable companion for design review. `ARCHITECTURE.md` owns the
authoritative component/sequence diagrams and per-decision rationale; `TECHNICAL_REPORT.md` owns
evaluation results and the error analysis. This document adds the four lenses those two don't
cover: **design principles**, **design patterns**, **cost**, and **token economics** — plus the
optimization measures that follow from them. Numbers below are pulled directly from the code and
from `TECHNICAL_REPORT.md`'s measured results, not estimated from scratch.

---

## 1. System overview (condensed)

A local-first RAG pipeline in seven stages, split into two independently-callable halves:

```
Ingestion → Indexing  ⎫
Retrieval (1–4)        ⎬  retrieval.pipeline.retrieve()      → RetrievalOutcome
  query proc → dense+lexical search → RRF fusion → rerank  ⎭
Generation (5–7)       ⎫
  abstention gate → context build → prompt → LLM → citation validation  ⎬  generation.service.answer_question()
```

| Layer | Module(s) | Backend | Local? |
|---|---|---|---|
| Embeddings | `app/indexing/embeddings.py` | `sentence-transformers/all-MiniLM-L6-v2` | Yes |
| Vector index | `app/indexing/vector_store.py` | ChromaDB (persistent, on disk) | Yes |
| Lexical index | `app/indexing/lexical_index.py` | BM25 (`rank_bm25`, pickled) | Yes |
| Metadata | `app/indexing/metadata_store.py` | SQLite | Yes |
| Fusion | `app/retrieval/fusion.py` | Reciprocal Rank Fusion | N/A (pure Python) |
| Reranking | `app/reranking/cross_encoder.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Yes |
| Generation | `app/generation/llm_client.py` | Ollama (`llama3.2:3b`) or `mock` | Yes / N/A |

Full component and sequence diagrams: `ARCHITECTURE.md` §1–2. This document assumes that layout
and layers straight into principles, patterns, cost, and tokens.

---

## 2. Design principles

Principles actually embodied in the code, not aspirational ones — each row names where to verify it.

| Principle | What it looks like here | Evidence |
|---|---|---|
| **Local-first, zero mandatory external dependency** | Every component (embeddings, vector store, lexical index, reranker) runs on-device; generation is the one component with an external process (Ollama), and even that is swappable for a deterministic offline stub. | `README.md` "Local-first design" table; `DOCS_ASSISTANT_LLM_PROVIDER=mock` |
| **Separation of retrieval and generation** | Two distinct entry points, never fused into one function, so retrieval can be evaluated (Hit Rate@K, MRR) independently of generation quality. | `ARCHITECTURE.md` "Request flow"; `app/evaluation/runner.py` calls both separately |
| **Interface over implementation (dependency inversion)** | Nothing in the pipeline talks to a concrete LLM/reranker client directly — everything goes through a `Protocol` or a fallback-wrapped call, so backends swap without touching callers. | `LLMClient` Protocol (`app/generation/llm_client.py`); `apply_reranking` (`app/reranking/fallback.py`) |
| **Explicit over implicit, especially around scale mismatches** | Three incompatible score scales (RRF ~0.01–0.03, cross-encoder ~±10, cosine 0–1) coexist on one `Candidate`; the codebase names *which one* is safe for threshold comparisons (`quality_score`) rather than letting callers guess. | `app/retrieval/candidates.py`; this was a real bug (`TECHNICAL_REPORT.md` §5 error #1) |
| **Fail-safe degradation over hard failure** | Reranker errors → fall back to fused order. LLM unreachable → return evidence with a clear message, not a 500. Weak evidence → abstain before ever calling the model. | `ARCHITECTURE.md` §5 "Failure paths" |
| **Defense in depth for untrusted content** | Documents are treated as adversarial input at two independent layers: a regex guard redacts injection-shaped sentences *before* they reach the model, and the system prompt separately instructs the model to treat all evidence as data, never instructions. Either layer alone is known-incomplete; together they're the mitigation. | `app/generation/injection_guard.py`; `SYSTEM_PROMPT` rule 6 in `app/generation/prompt.py` |
| **Validate on the way out, not just on the way in** | The model is *asked* to cite real sources, but nothing trusts that it did — every `[S<n>]` tag is checked against blocks actually sent before the answer reaches the caller. | `app/generation/citations.py` |
| **No calibrated-probability claims** | Evidence quality is surfaced as a named bucket (`low`/`medium`/`high`), never as a percentage or confidence score, because the underlying signals (cosine similarity, cross-encoder logits) aren't calibrated for that. | `app/generation/abstention.py` docstring; `TECHNICAL_REPORT.md` §6 |
| **Evaluation harness reuses production code paths** | `scripts/evaluate.py` and `app/evaluation/runner.py` call the exact same `retrieve()` / `answer_question()` functions the API uses — there is no separate "eval mode" implementation that can silently drift from what's shipped. | `ARCHITECTURE.md` §1.2 |
| **Idempotent, incremental by default** | Re-running ingestion is a checksum diff, not a rebuild; unchanged files cost nothing on a re-run. | `app/ingestion/pipeline.py`; `README.md` "Indexing" |
| **Single source of truth for tunables** | Every threshold, top-k, model name, and budget lives in one `pydantic-settings` module, not scattered across call sites. | `app/config.py` |

---

## 3. Design patterns

| Pattern | Where | Why it's the right shape here |
|---|---|---|
| **Strategy** | `RetrievalConfig` presets (`baseline` / `improved_a` / `improved_b` in `app/evaluation/configs.py`); `chunk_strategy` (`fixed_window` / `structure_aware`) | Retrieval behavior (chunking, hybrid on/off, reranking on/off, top-k) is swapped by passing a different config object into the same pipeline function — no branching inside the pipeline itself. |
| **Protocol / Adapter** | `LLMClient` (`OllamaClient`, `MockClient`) in `app/generation/llm_client.py` | Generation backend is fully interchangeable behind one two-method interface; the mock adapter exists purely to make the rest of the system's behavior testable without the real dependency. |
| **Pipeline (staged transformation)** | `retrieval.pipeline.retrieve()` (stages 1–4), `generation.service.answer_question()` (stages 5–7) | Each stage takes the previous stage's typed output and returns a new typed output (`ProcessedQuery` → `list[Candidate]` → `RetrievalOutcome` → `ContextBundle` → `AnswerResult`) — inspectable at every boundary, which is exactly what the evaluation harness needs. |
| **Fallback chain / graceful-degradation wrapper** | `apply_reranking()` (`app/reranking/fallback.py`) catches reranker failures and substitutes the fused order; `generation/service.py` catches `LLMUnavailableError` and substitutes an evidence-only response | Both wrap "the risky call" and unconditionally produce *a* valid response, reporting what happened (`reranker_used`, `reranker_fallback_reason`) rather than raising past the caller. |
| **Repository** | `VectorStore`, `LexicalIndex`, `MetadataStore` (`app/indexing/`) | Each owns persistence for one concern (vectors / lexical postings / relational metadata) behind a small method surface — callers never touch ChromaDB, pickle, or SQLite directly. |
| **Guard clause as a cost gate** | `decide_abstention()` (`app/generation/abstention.py`), called *before* `LLMClient.generate()` | Doubles as both a correctness mechanism (don't answer without evidence) and the system's single largest cost/latency optimization — see §5.1. |
| **Sanitize-then-trust (output validation)** | `app/generation/citations.py` strips any invented `[S<n>]` tag before the answer is returned | The model's output is treated the same way documents are: useful but not trusted until checked against ground truth the code actually controls. |
| **Factory** | `build_llm_client(provider, ...)` (`app/generation/llm_client.py`) | One function decides which concrete `LLMClient` to construct from a config string; call sites never branch on provider name themselves. |
| **Value objects / typed DTOs at every stage boundary** | `Candidate`, `ContextBlock`, `ContextBundle`, `RetrievalOutcome`, `AnswerResult` (dataclasses throughout `app/retrieval/`, `app/generation/`) | Stage boundaries pass structured, named fields instead of dicts — the score-scale bug (§2) was catchable precisely because `Candidate` names its fields instead of hiding them behind one generic `score`. |
| **Template-style prompt assembly** | `app/generation/prompt.py` separates a fixed `SYSTEM_PROMPT` (behavior contract) from a per-request `build_user_prompt()` (history + evidence + question) | Keeps the part of the prompt that should never change auditable and separate from the part that varies every request — also the natural cache-prefix boundary if this were ever swapped to a hosted, cache-billed API (§5.2). |

---

## 4. Cost analysis

### 4.1 Current cost model: compute, not currency

Every backend is local, so there is no per-token, per-request, or per-document API bill today.
"Cost" here is **wall-clock compute time on the operator's own hardware**, which is the number that
actually matters for a local deployment. Measured on CPU-only hardware, from `TECHNICAL_REPORT.md`
§3.1 (43-question eval set, `llama3.2:3b`):

| Stage | baseline | improved_a | improved_b | What drives it |
|---|---|---|---|---|
| Retrieval latency avg / p95 | 108 / 61 ms\* | 31 / 48 ms | 347 / 728 ms | dense-only vs. hybrid+RRF vs. hybrid+RRF+**reranker** |
| End-to-end latency avg / p95 | 22.0 / 40.2 s | 25.1 / 50.8 s | 18.0 / 36.9 s | dominated by LLM generation, not retrieval |

\* baseline's avg exceeds its own p95 because the very first query in the process pays a one-time
embedding-model cold-load cost later queries don't.

Two things fall out of this table directly:

- **Generation dominates cost by roughly two orders of magnitude over retrieval.** Even
  `improved_b`'s worst-case 728 ms retrieval is under 4% of its ~18 s average end-to-end time. Any
  cost-reduction effort should target generation first — which is exactly what the abstention gate
  does (§4.3).
- **Reranking's 3–10x retrieval-latency cost is real but cheap in absolute terms** relative to
  generation — see `TECHNICAL_REPORT.md` error #11. It's a correctness-for-latency trade that's
  clearly worth it here (citation-source coverage 0.90 → 0.986) and would only need revisiting at a
  much larger corpus size.

### 4.2 Hypothetical cost if generation moved to a hosted API

`LLMClient` is deliberately the one interface designed to make this swap contained
(`ARCHITECTURE.md` §3, `TECHNICAL_REPORT.md` §8 "Production roadmap"). Using the token budget
worked out in §5 (~2,000–2,200 input tokens, ~150–400 output tokens per question) and current
first-party Claude API pricing:

| Model | Input $/1M | Output $/1M | Est. cost per question\* | Est. cost per 1,000 questions |
|---|---|---|---|---|
| Claude Haiku 4.5 | $1.00 | $5.00 | ~$0.003 | ~$3 |
| Claude Sonnet 5 | $2.00 | $10.00 | ~$0.006 | ~$6 |
| Claude Opus 5 | $5.00 | $25.00 | ~$0.016 | ~$16 |

\* Assumes ~2,100 input tokens + ~250 output tokens per question, **no prompt caching** applied
(baseline case). Pricing as of this review; re-check current rates before quoting.

This is the number that matters for the production-roadmap decision in `TECHNICAL_REPORT.md` §8:
at this corpus/traffic scale (43 eval questions, a small document set), a hosted API is inexpensive
in absolute terms — the trade being made by staying local isn't about avoiding a large bill, it's
about avoiding *any* external dependency, data egress, and per-request network latency, at the cost
of the 15–40 s CPU-inference latency shown in §4.1. That trade-off should be named explicitly rather
than assumed: local wins on privacy/dependency-freedom, hosted wins on latency and (at this scale)
trivial dollar cost.

**If a hosted adapter were added**, the system prompt (§5.1, ~485 tokens, byte-identical every
request) is a textbook prompt-caching candidate — cached reads run at a fraction of input pricing,
so caching it would cut the dominant fixed cost per request by most of ~485 tokens' worth of input
billing, compounding over volume. The evidence context (§5.1, up to ~1,500 tokens) is *not* a good
caching candidate as currently built, since `build_context()` produces different text per query.

### 4.3 What actually controls cost here

| Lever | Effect | Where |
|---|---|---|
| **Abstention gate** | Skips the LLM call entirely — the single biggest cost/latency lever in the system, local or hosted. On the 43-question eval set, the code-level gate fired on 22–33% of unanswerable-labeled questions before any generation cost was incurred. | `app/generation/abstention.py`, called in `generation/service.py` before `LLMClient.generate()` |
| **`max_context_chars`** | Directly bounds evidence-context token count sent to the model; this is the one knob that would move a hosted-API bill the most, since it's the largest variable component of input tokens. | `app/config.py` (`max_context_chars: int = 6000`) |
| **`top_k_final`** | Bounds how many chunks *could* fill that budget; identical (5) across all three named configs, so — see §5.4 — LLM-side token cost is flat across `baseline`/`improved_a`/`improved_b` even though retrieval compute cost isn't. | `app/evaluation/configs.py` |
| **`use_reranker`** | Zero dollar cost locally (it's a local model); pure latency/compute cost, and only at the retrieval stage, not generation. | `app/config.py`, `app/reranking/fallback.py` |
| **Model size choice** (MiniLM embeddings, MiniLM cross-encoder, `llama3.2:3b`) | Every model in the stack was picked as "smallest model that clears the quality bar for this corpus size," not defaulted to the largest available — see `TECHNICAL_REPORT.md` §2. | `README.md` "Local-first design" table |

---

## 5. Token analysis

### 5.1 How tokens are approximated

`context_builder.py` budgets the evidence context in **characters**, not model tokens, by explicit
design choice — documented in its own module docstring as "a simple, dependency-free proxy for a
token budget... an approximation, not an exact tokenizer count." No tokenizer dependency is pulled
in anywhere in the pipeline. This section uses the common ~4 characters/token heuristic for English
prose (a reasonable approximation for both the Ollama/Llama tokenizer and Claude's tokenizer) to
translate the codebase's character budgets into token estimates — treat these as estimates, not
measured counts.

### 5.2 Per-request token budget breakdown

| Component | Size (chars) | Est. tokens (÷4) | Fixed or variable? |
|---|---|---|---|
| System prompt (`SYSTEM_PROMPT`) | 1,939 | ~485 | **Fixed** — byte-identical every request |
| Evidence context (`max_context_chars`) | ≤ 6,000 | ≤ ~1,500 | Variable, hard-capped |
| Per-block header overhead (`[S<n>] file — unit_label (superseded)`) | ~40–70/block × ≤5 blocks | ~50–90 | Variable, **not** counted against the 6,000-char budget (see §7.2) |
| Conversation history (≤ 6 turns, `_format_history`) | varies | typically 100–400 | Variable, only present on follow-up turns |
| User question | varies | typically 10–50 | Variable |
| **Total input, typical case** | ~8,000–8,700 | **~2,000–2,200** | |
| Output (answer text) | unbounded — see §7.2 | typically ~150–400 in eval runs | Variable, **not client-capped** |

### 5.3 Where the budget actually goes

The evidence context (up to ~1,500 tokens) and the system prompt (~485 tokens, fixed) together
account for roughly 90-95% of a typical request's input tokens — history and the question itself
are minor by comparison except on long conversation threads. This means:

- The system prompt is the best prompt-caching target if the LLM backend ever becomes hosted (§4.2).
- The evidence-context budget (`max_context_chars`) is the one setting an operator would tune to
  trade answer completeness against token/latency cost — raising it lets more/larger chunks in
  (`build_context()`'s `dropped_for_budget` counter reports exactly how many chunks got cut), at
  the cost of a longer, more expensive prompt.

### 5.4 Chunking strategy changes retrieval cost, not generation cost

This is a non-obvious result worth stating explicitly for review: `structure_aware` chunking
(`improved_b`) produces roughly 2.7x as many chunks as `fixed_window` (38 vs. 14, per
`TECHNICAL_REPORT.md` error #11) — but `top_k_final` is **5 in every one of the three named
configs** (`app/evaluation/configs.py`). So:

- **Retrieval/rerank compute cost scales with chunk count** — more, smaller chunks means more
  candidates for the cross-encoder to score, which is exactly why `improved_b`'s retrieval latency
  (347 ms avg) is far higher than `improved_a`'s (31 ms avg) despite both using the same `top_k_final`.
- **Generation token cost does not** — whatever gets retrieved, only the top 5 reranked/fused
  chunks are ever packed into the context, capped again at 6,000 characters. The LLM-facing token
  bill is effectively constant across all three configs; the entire cost differential between them
  lives in the retrieval stage, which (§4.1) is cheap relative to generation anyway.

### 5.5 A real gap worth flagging: uncapped output tokens

`OllamaClient.generate()` (`app/generation/llm_client.py`) sends no `options.num_predict` (or
equivalent) to the Ollama `/api/chat` call — output length is bounded only by the model's own
stopping behavior and the outer `llm_timeout_seconds` (90 s) wall-clock timeout, not by a token
count. This is invisible locally (no per-token bill), but it's the one thing in this pipeline that
would need to change before a hosted-LLM swap: an unbounded `max_tokens` on a paid API is an
unbounded worst-case bill per request. Flagged here as a concrete pre-condition for the
production-roadmap hosted-adapter work already named in `TECHNICAL_REPORT.md` §8, not something
broken in the current local-only setup.

---

## 6. Optimization measures

### 6.1 Already implemented

| Optimization | Mechanism | Effect |
|---|---|---|
| **Abstention as a pre-generation gate** | `decide_abstention()` runs on retrieval results before any LLM call | Skips the single most expensive step (15–40 s local generation; §4.2's $/question if hosted) whenever evidence is too weak to be worth generating from |
| **Funnel-shaped retrieval** | `top_k_dense`/`top_k_lexical` (20) → RRF fuse → `top_k_fused` (10) → rerank → `top_k_final` (5) | Each stage narrows the candidate set before the next, more expensive stage runs — the cross-encoder (the one per-candidate-expensive step) only ever scores `top_k_fused` items, never the full index |
| **Rank-only fusion (RRF)** | `app/retrieval/fusion.py` | O(n) over candidate ranks; no score-normalization computation, and immune to the scale-mismatch bug class described in §3 |
| **Character-budgeted context** | `build_context()` hard-stops at `max_context_chars`, reporting `dropped_for_budget` | Prompt size — and therefore generation latency/cost — is bounded regardless of how many or how large the retrieved chunks are |
| **Reranker/LLM fallback instead of retry-and-fail** | `apply_reranking()`, `LLMUnavailableError` handling in `generation/service.py` | Avoids the cost of a failed request needing a second full pipeline run — one pass either succeeds fully or degrades gracefully |
| **Process-wide model caching** | `get_embedder()` (`app/indexing/embeddings.py`) is `@lru_cache`-wrapped | The embedding model loads once per process, not once per call — this is also why `baseline`'s *first* query in a fresh process is disproportionately slow (§4.1's footnote) while every later query is fast |
| **Batched embedding calls** | `embed_texts()` always encodes a list through `SentenceTransformer.encode()` in one call, both at ingestion and query time | One batched forward pass instead of N sequential ones per document/query |
| **Incremental ingestion by checksum** | `app/ingestion/pipeline.py` diffs against `MetadataStore` | Re-running ingestion after a small edit costs one embed pass for the changed file(s), not a full corpus re-embed |
| **Deliberately small local models throughout** | MiniLM embeddings (~22M params), MiniLM cross-encoder, `llama3.2:3b` | Smallest model that meets the quality bar for this corpus size at every layer — not defaulting to the largest available model "to be safe" |
| **Mock LLM provider for dev/test loop** | `DOCS_ASSISTANT_LLM_PROVIDER=mock` | Removes the dominant cost (generation latency) from the entire automated test suite and iterative development — tests run in ~1–20 s instead of minutes |

### 6.2 Not yet implemented (roadmap-level)

These are gaps, not defects — named here so the cost/token picture above is complete, and cross-
referenced to `TECHNICAL_REPORT.md` §8 where they already overlap the production roadmap:

| Gap | Why it matters | Where it would go |
|---|---|---|
| No response cache (exact or semantic) for repeated/near-duplicate questions | Every question pays the full ~18–25 s generation cost even if asked before | `generation/service.py`, keyed on `(config_name, normalized_query, filters)` |
| No output-token cap sent to the LLM backend (§5.5) | Fine locally; becomes an unbounded-bill risk the moment a paid hosted backend is added | `OllamaClient.generate()` / its hosted-adapter equivalent |
| No prompt caching (N/A today, relevant the moment a hosted adapter exists) | The system prompt is a ~485-token fixed prefix repeated on every single request — the textbook caching case | Hosted `LLMClient` implementation, `app/generation/prompt.py`'s system/user split already gives it a clean boundary |
| `top_k_fused` not revisited for corpus scale | Reranker cost is linear in candidate count; fine at 12-document scale, would need tuning before a much larger corpus (already flagged as expected in `TECHNICAL_REPORT.md` error #11) | `app/config.py`, `app/evaluation/configs.py` |

---

## 7. How to read this alongside the other two documents

- **`ARCHITECTURE.md`** — the authoritative component/sequence diagrams, the full design-decision
  table, and every documented failure path. Read it first for "how does a request actually flow."
- **`TECHNICAL_REPORT.md`** — measured evaluation results across the three configurations, how each
  metric is computed and what it doesn't prove, and the 11-item classified error analysis this
  document's cost/latency numbers are sourced from. Read it for "does this actually work, and what
  broke during evaluation."
- **This document** — the principles/patterns/cost/token lens across both, for a design review
  that needs those questions answered explicitly rather than inferred from the code.
