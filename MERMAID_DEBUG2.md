# Sequence diagram diagnostic (temporary — will be deleted)

## A. Dotted participant alias

```mermaid
sequenceDiagram
    participant A
    participant B as retrieval.pipeline
    A->>B: hello
    B-->>A: hi
```

## B. Self-message loop

```mermaid
sequenceDiagram
    participant A
    A->>A: self call
```

## C. alt/else block

```mermaid
sequenceDiagram
    participant A
    participant B
    A->>B: request
    alt condition true
        B-->>A: yes
    else condition false
        B-->>A: no
    end
```

## D. Many participants (8, matching the real diagram)

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Retrieval as retrieval.pipeline
    participant Vec as VectorStore
    participant Lex as LexicalIndex
    participant Rerank as reranking
    participant Gen as generation.service
    participant LLM as LLMClient
    Client->>API: test
```

## E. The real diagram, unmodified, for a fresh baseline read

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI
    participant Retrieval as retrieval.pipeline
    participant Vec as VectorStore
    participant Lex as LexicalIndex
    participant Rerank as reranking
    participant Gen as generation.service
    participant LLM as LLMClient

    Client->>API: POST /ask {question, history, filters, config_name}
    API->>Retrieval: retrieve(question, config, filters, history)
    Retrieval->>Retrieval: process_query (normalize + follow-up heuristic)
    Retrieval->>Vec: dense_search(retrieval_query, filters)
    Retrieval->>Lex: lexical_search(retrieval_query, filters)  (if hybrid)
    Retrieval->>Retrieval: reciprocal_rank_fusion() + superseded penalty
    Retrieval->>Rerank: apply_reranking(fused_top)  (if enabled)
    Rerank-->>Retrieval: reranked hits (or fused order on fallback)
    Retrieval-->>API: RetrievalOutcome (hits + stage timings)
    API->>Gen: answer_question(outcome)
    Gen->>Gen: decide_abstention(hits)
    alt insufficient evidence
        Gen-->>API: AnswerResult (abstained=true, no LLM call)
    else evidence sufficient
        Gen->>Gen: build_context() (injection scan + token-budget pack)
        Gen->>LLM: generate(system_prompt, user_prompt, context_blocks)
        LLM-->>Gen: raw answer text with [S1], [S2], ... tags
        Gen->>Gen: assemble_and_validate_citations() (drop hallucinated tags)
        Gen-->>API: AnswerResult (answer, citations, evidence_quality, conflict_signal)
    end
    API-->>Client: AskResponse (answer, citations, retrieved_evidence, stage_timings_ms)
```
