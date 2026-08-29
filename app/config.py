"""Centralized, environment-driven configuration.

Every tunable named in the assignment (model names, paths, chunk parameters,
top-k, thresholds, feature switches) lives here so no component hardcodes it.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCS_ASSISTANT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    data_dir: Path = Path("./data")
    documents_dir: Path = Path("./data/sample_documents")
    index_dir: Path = Path("./data/index")

    # --- Embeddings ---
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Generation backend ---
    llm_provider: str = "ollama"  # "ollama" | "mock"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = 90.0

    # --- Reranking ---
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # --- Security / resource limits ---
    max_file_size_mb: float = 25.0

    # --- Chunking ---
    chunk_strategy: str = "structure_aware"  # "fixed_window" | "structure_aware"
    fixed_chunk_size_chars: int = 900
    fixed_chunk_overlap_chars: int = 150
    structure_max_chunk_chars: int = 1200

    # --- Retrieval defaults ---
    top_k_dense: int = 20
    top_k_lexical: int = 20
    top_k_fused: int = 10
    top_k_final: int = 5
    use_hybrid: bool = True
    fusion_rrf_k: int = 60

    # --- Context / generation budget ---
    max_context_chars: int = 6000

    # --- Abstention thresholds (heuristic; documented as non-probabilistic) ---
    min_evidence_score: float = 0.15
    min_citation_coverage: float = 0.3

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def chroma_dir(self) -> Path:
        return self.index_dir / "chroma"

    @property
    def bm25_path(self) -> Path:
        return self.index_dir / "bm25_index.pkl"

    @property
    def metadata_db_path(self) -> Path:
        return self.index_dir / "metadata.sqlite3"

    @property
    def feedback_db_path(self) -> Path:
        return self.index_dir / "feedback.sqlite3"


def get_settings() -> Settings:
    """Fresh settings instance (re-reads env/.env) — avoids stale singletons in tests."""
    return Settings()
