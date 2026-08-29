"""FastAPI application (FR-15): ingestion, status, questions, evidence,
health, and feedback endpoints.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.schemas import (
    AskRequest,
    AskResponse,
    CitationOut,
    ComponentHealth,
    EvidenceOut,
    FeedbackRequest,
    HealthResponse,
    IngestRequest,
)
from app.config import Settings, get_settings
from app.evaluation.configs import ALL_CONFIGS
from app.generation.llm_client import build_llm_client
from app.generation.service import answer_question
from app.indexing.embeddings import embed_query
from app.indexing.lexical_index import LexicalIndex
from app.indexing.metadata_store import MetadataStore
from app.indexing.vector_store import VectorStore
from app.ingestion.pipeline import IngestionPipeline
from app.logging_config import configure_logging, get_correlation_id, new_correlation_id
from app.reranking.cross_encoder import score_pairs
from app.retrieval.filters import RetrievalFilters
from app.retrieval.pipeline import retrieve
from app.retrieval.query_processing import ConversationTurn

logger = logging.getLogger(__name__)


class AppState:
    settings: Settings
    vector_store: VectorStore
    lexical_index: LexicalIndex
    metadata_store: MetadataStore
    ingestion_pipeline: IngestionPipeline
    llm_client: object


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    state.settings = settings
    state.vector_store = VectorStore(settings.chroma_dir)
    state.lexical_index = LexicalIndex(settings.bm25_path)
    state.metadata_store = MetadataStore(settings.metadata_db_path)
    state.ingestion_pipeline = IngestionPipeline(settings)
    state.llm_client = build_llm_client(
        settings.llm_provider, settings.ollama_host, settings.ollama_model, settings.llm_timeout_seconds
    )
    logger.info("app_startup", extra={"llm_provider": settings.llm_provider})
    yield


app = FastAPI(title="Local Documentation Intelligence Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_UI_DIR = Path(__file__).resolve().parent.parent / "ui"
if _UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_UI_DIR), html=True), name="ui")


@app.post("/ingest")
def ingest(request: IngestRequest | None = None):
    new_correlation_id()
    result = state.ingestion_pipeline.run()  # always the configured documents_dir — see IngestRequest
    return {
        "correlation_id": get_correlation_id(),
        "duration_ms": result.duration_ms,
        **result.report.to_dict(),
    }


@app.get("/status")
def status():
    files = state.metadata_store.all_files()
    chunk_counts = {
        strategy: state.metadata_store.chunk_count_for_strategy(strategy)
        for strategy in ("fixed_window", "structure_aware")
    }
    return {
        "file_count": len(files),
        "files_by_status": {
            status_value: sum(1 for f in files if f.doc_status == status_value)
            for status_value in {f.doc_status for f in files}
        },
        "chunk_counts_by_strategy": chunk_counts,
        "vector_store_count": state.vector_store.count(),
        "feedback_summary": state.metadata_store.feedback_summary(),
    }


@app.get("/facets")
def facets():
    """Distinct filter values currently available in the index (FR-08), so a
    client can offer a category/file-name picker instead of free-text entry."""
    files = state.metadata_store.all_files()
    return {
        "categories": sorted({f.category for f in files if f.category}),
        "file_names": sorted({f.file_name for f in files if f.file_name}),
    }


@app.get("/health", response_model=HealthResponse)
def health():
    settings = state.settings
    vs_ok, vs_msg = True, "ok"
    try:
        state.vector_store.count()
    except Exception as exc:  # local persistent store — failure means a real problem
        vs_ok, vs_msg = False, str(exc)

    emb_ok, emb_msg = True, "ok"
    try:
        embed_query(settings.embedding_model, "health check")
    except Exception as exc:
        emb_ok, emb_msg = False, str(exc)

    rerank_ok, rerank_msg = True, "ok"
    if settings.use_reranker:
        try:
            score_pairs(settings.reranker_model, "health check", ["health check"])
        except Exception as exc:
            rerank_ok, rerank_msg = False, str(exc)
    else:
        rerank_msg = "reranker disabled by configuration"

    llm_ok, llm_msg = state.llm_client.health_check()

    overall = "ok" if (vs_ok and emb_ok and llm_ok) else "degraded"
    return HealthResponse(
        status=overall,
        vector_store=ComponentHealth(ok=vs_ok, message=vs_msg),
        embedding_model=ComponentHealth(ok=emb_ok, message=emb_msg),
        reranker=ComponentHealth(ok=rerank_ok, message=rerank_msg),
        llm=ComponentHealth(ok=llm_ok, message=llm_msg),
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    correlation_id = new_correlation_id()
    settings = state.settings

    config = ALL_CONFIGS.get(request.config_name)
    if config is None:
        raise HTTPException(400, f"Unknown config_name '{request.config_name}'. Known: {list(ALL_CONFIGS)}")

    filters = RetrievalFilters(file_name=request.file_name, category=request.category, version=request.version)
    history = [ConversationTurn(role=t.role, content=t.content) for t in request.history]

    try:
        outcome = retrieve(
            request.question,
            settings,
            config,
            state.vector_store,
            state.lexical_index,
            filters=filters,
            history=history,
        )
    except Exception as exc:
        logger.exception("retrieval_failed", extra={"correlation_id": correlation_id})
        raise HTTPException(503, f"Retrieval is currently unavailable: {exc}") from exc

    result = answer_question(outcome, settings, state.llm_client)

    evidence = [
        EvidenceOut(
            chunk_id=hit.chunk_id,
            file_name=hit.metadata.get("file_name", ""),
            unit_label=hit.metadata.get("unit_label", ""),
            category=hit.metadata.get("category", ""),
            version=hit.metadata.get("version", ""),
            doc_status=hit.metadata.get("doc_status", "unknown"),
            chunk_strategy=hit.metadata.get("chunk_strategy", ""),
            dense_score=hit.dense_score,
            lexical_score=hit.lexical_score,
            fused_score=hit.fused_score,
            rerank_score=hit.rerank_score,
            text=hit.text,
        )
        for hit in outcome.hits
    ]

    return AskResponse(
        answer_id=correlation_id,
        original_question=outcome.processed_query.original,
        retrieval_query=outcome.processed_query.retrieval_query,
        used_conversation_context=outcome.processed_query.used_conversation_context,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        abstained=result.abstained,
        abstention_reason=result.abstention_reason,
        evidence_quality=result.evidence_quality,
        conflict_signal=result.conflict_signal,
        reranker_used=result.reranker_used,
        reranker_fallback_reason=result.reranker_fallback_reason,
        config_used=config.name,
        retrieved_evidence=evidence,
        stage_timings_ms=result.stage_timings_ms,
    )


@app.get("/evidence/{chunk_id}")
def get_evidence(chunk_id: str):
    chunk = state.metadata_store.get_chunk(chunk_id)
    if chunk is None:
        raise HTTPException(404, f"No chunk found with id '{chunk_id}'")
    return chunk


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    state.metadata_store.add_feedback(request.answer_id, request.question, request.useful, request.reason)
    return {"status": "recorded"}
