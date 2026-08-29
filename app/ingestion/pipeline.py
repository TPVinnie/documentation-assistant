"""Ingestion orchestration (FR-01..FR-05): discover -> diff -> parse -> chunk
-> embed -> index, producing a `ProcessingReport`.

Every supported file is chunked with *both* configured strategies so the
evaluation harness (4.2) can compare chunking approaches without a second
ingestion pass. Version/recency status (FR-12) is recomputed corpus-wide
after each run since adding a newer document can supersede an older one
that wasn't itself touched this run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings
from app.indexing.chunking import CHUNKERS, Chunk
from app.indexing.embeddings import embed_texts
from app.indexing.lexical_index import LexicalIndex
from app.indexing.metadata_store import FileRecord, MetadataStore
from app.indexing.vector_store import VectorStore
from app.ingestion.checksum import sha256_of_file
from app.ingestion.doc_metadata import derive_doc_metadata, doc_order_key
from app.ingestion.parsers import (
    SUPPORTED_EXTENSIONS,
    CorruptedDocumentError,
    EmptyDocumentError,
    EncryptedDocumentError,
    UnsupportedFileTypeError,
    parse_document,
)
from app.ingestion.report import FileStatus, ProcessingReport

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    report: ProcessingReport
    duration_ms: float


class IngestionPipeline:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._metadata = MetadataStore(settings.metadata_db_path)
        self._vector_store = VectorStore(settings.chroma_dir)
        self._lexical_index = LexicalIndex(settings.bm25_path)

    @property
    def metadata_store(self) -> MetadataStore:
        return self._metadata

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store

    def run(self, documents_dir: Path | None = None) -> IngestionResult:
        start = time.perf_counter()
        documents_dir = documents_dir or self._settings.documents_dir
        report = ProcessingReport()

        disk_files = self._discover_files(documents_dir, report)
        seen_checksums: dict[str, str] = {}  # checksum -> first file_path seen in this run
        touched_paths: set[str] = set()

        for path in disk_files:
            self._process_file(path, documents_dir, report, seen_checksums, touched_paths)

        self._handle_deletions(disk_files, touched_paths, report)
        self._recompute_version_status()
        self._rebuild_lexical_indexes()

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "ingestion_complete",
            extra={"duration_ms": duration_ms, "summary": report.summary()},
        )
        return IngestionResult(report=report, duration_ms=duration_ms)

    # --- discovery ---

    def _discover_files(self, documents_dir: Path, report: ProcessingReport) -> list[Path]:
        if not documents_dir.exists():
            logger.warning("documents_dir_missing", extra={"documents_dir": str(documents_dir)})
            return []
        supported: list[Path] = []
        for path in sorted(documents_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                report.add(str(path), FileStatus.SKIPPED, message=f"unsupported file type '{path.suffix}'")
                continue
            supported.append(path)
        return supported

    # --- per-file processing ---

    def _process_file(
        self,
        path: Path,
        documents_dir: Path,
        report: ProcessingReport,
        seen_checksums: dict[str, str],
        touched_paths: set[str],
    ) -> None:
        file_path_str = str(path)
        touched_paths.add(file_path_str)

        max_bytes = self._settings.max_file_size_mb * 1024 * 1024
        file_size = path.stat().st_size
        if file_size > max_bytes:
            report.add(
                file_path_str,
                FileStatus.SKIPPED,
                message=f"file size {file_size / (1024 * 1024):.1f} MB exceeds the "
                f"{self._settings.max_file_size_mb:.0f} MB limit",
            )
            return

        try:
            checksum = sha256_of_file(path)
        except OSError as exc:
            report.add(file_path_str, FileStatus.FAILED, message=f"could not read file: {exc}")
            return

        mtime = path.stat().st_mtime
        existing = self._metadata.get_file(file_path_str)

        if existing is not None and existing.checksum == checksum:
            report.add(
                file_path_str,
                FileStatus.UNCHANGED,
                chunk_count=self._chunk_count_for_path(file_path_str),
            )
            return

        duplicate_of = seen_checksums.get(checksum)
        if duplicate_of is None:
            duplicate_of = self._find_duplicate_elsewhere(checksum, file_path_str)
        if duplicate_of is not None:
            report.add(file_path_str, FileStatus.SKIPPED, message=f"duplicate content of {duplicate_of}")
            self._metadata.delete_file(file_path_str)
            self._vector_store.delete_by_file_path(file_path_str)
            return
        seen_checksums[checksum] = file_path_str

        try:
            raw_doc = parse_document(path)
        except UnsupportedFileTypeError as exc:
            report.add(file_path_str, FileStatus.SKIPPED, message=str(exc))
            return
        except EmptyDocumentError as exc:
            report.add(file_path_str, FileStatus.SKIPPED, message=f"empty document: {exc}")
            return
        except EncryptedDocumentError as exc:
            report.add(file_path_str, FileStatus.FAILED, message=f"encrypted document: {exc}")
            return
        except CorruptedDocumentError as exc:
            report.add(file_path_str, FileStatus.FAILED, message=f"corrupted document: {exc}")
            return
        except Exception as exc:  # last-resort guard so one bad file never aborts the whole run
            report.add(file_path_str, FileStatus.FAILED, message=f"unexpected parse error: {exc}")
            logger.exception("unexpected_parse_error", extra={"file_path": file_path_str})
            return

        doc_meta = derive_doc_metadata(path, documents_dir, raw_doc.header)
        doc_meta["doc_status"] = "unknown"  # resolved in _recompute_version_status

        all_chunks: list[Chunk] = []
        for strategy, chunker in CHUNKERS.items():
            if strategy == "fixed_window":
                chunks = chunker(
                    raw_doc,
                    checksum,
                    doc_meta,
                    self._settings.fixed_chunk_size_chars,
                    self._settings.fixed_chunk_overlap_chars,
                )
            else:
                chunks = chunker(raw_doc, checksum, doc_meta, self._settings.structure_max_chunk_chars)
            all_chunks.extend(chunks)

        if not all_chunks:
            report.add(file_path_str, FileStatus.SKIPPED, message="no chunks produced (empty content)")
            return

        self._index_chunks(all_chunks)
        self._metadata.replace_chunks_for_file(file_path_str, all_chunks)
        self._metadata.upsert_file(
            FileRecord(
                file_path=file_path_str,
                file_name=path.name,
                file_type=raw_doc.file_type,
                checksum=checksum,
                mtime=mtime,
                category=doc_meta["category"],
                title=doc_meta["title"],
                version=doc_meta["version"],
                effective_date=doc_meta["effective_date"],
                doc_status="unknown",
                supersedes=doc_meta["supersedes"],
                ingested_at=datetime.now(UTC).isoformat(),
            )
        )
        status = FileStatus.UPDATED if existing is not None else FileStatus.INDEXED
        report.add(file_path_str, status, chunk_count=len(all_chunks))

    def _find_duplicate_elsewhere(self, checksum: str, file_path_str: str) -> str | None:
        for record in self._metadata.all_files():
            if record.checksum == checksum and record.file_path != file_path_str:
                return record.file_path
        return None

    def _chunk_count_for_path(self, file_path: str) -> int:
        return len(self._metadata.chunks_for_file(file_path))

    def _index_chunks(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = embed_texts(self._settings.embedding_model, texts)
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "file_path": c.file_path,
                "file_name": c.file_name,
                "file_type": c.file_type,
                "category": c.category,
                "version": c.version,
                "effective_date": c.effective_date,
                "doc_status": c.doc_status,
                "chunk_strategy": c.chunk_strategy,
                "unit_label": c.unit_label,
            }
            for c in chunks
        ]
        self._vector_store.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=texts)

    # --- deletions ---

    def _handle_deletions(self, disk_files: list[Path], touched_paths: set[str], report: ProcessingReport) -> None:
        disk_paths = {str(p) for p in disk_files}
        for record in self._metadata.all_files():
            if record.file_path in disk_paths:
                continue
            self._vector_store.delete_by_file_path(record.file_path)
            self._metadata.delete_file(record.file_path)
            report.add(record.file_path, FileStatus.REMOVED)

    # --- version/recency resolution (FR-12) ---

    def _recompute_version_status(self) -> None:
        files = [f for f in self._metadata.all_files()]
        families: dict[tuple[str, str], list[FileRecord]] = {}
        for f in files:
            families.setdefault((f.category, f.title), []).append(f)

        for members in families.values():
            if len(members) == 1:
                self._apply_doc_status(members[0], "current")
                continue
            ranked = sorted(members, key=lambda f: doc_order_key(f.effective_date, f.version, f.mtime))
            current = ranked[-1]
            for member in ranked[:-1]:
                self._apply_doc_status(member, "superseded")
            self._apply_doc_status(current, "current")

    def _apply_doc_status(self, record: FileRecord, doc_status: str) -> None:
        if record.doc_status == doc_status:
            return
        self._metadata.set_doc_status(record.file_path, doc_status)
        chunk_ids = [row["chunk_id"] for row in self._metadata.chunks_for_file(record.file_path)]
        if chunk_ids:
            self._vector_store.update_metadata(
                ids=chunk_ids, metadatas=[{"doc_status": doc_status}] * len(chunk_ids)
            )

    # --- lexical index rebuild ---

    def _rebuild_lexical_indexes(self) -> None:
        for strategy in CHUNKERS:
            rows = self._metadata.all_chunks_for_strategy(strategy)
            self._lexical_index.rebuild_strategy(
                strategy,
                chunk_ids=[r["chunk_id"] for r in rows],
                texts=[r["text"] for r in rows],
                metadatas=[
                    {
                        "file_path": r["file_path"],
                        "file_name": r["file_name"],
                        "file_type": r["file_type"],
                        "category": r["category"],
                        "version": r["version"],
                        "effective_date": r["effective_date"],
                        "doc_status": r["doc_status"],
                        "chunk_strategy": r["chunk_strategy"],
                        "unit_label": r["unit_label"],
                    }
                    for r in rows
                ],
            )
