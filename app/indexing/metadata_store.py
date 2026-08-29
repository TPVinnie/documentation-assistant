"""SQLite-backed metadata store.

This is the source of truth for incremental indexing (FR-03): it tracks each
file's checksum/mtime so re-running ingestion only touches files that were
added, changed, or removed. It also stores chunk metadata (for
filtering/version-awareness, FR-08/FR-12) and user feedback (FR-16).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.indexing.chunking import Chunk

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    file_path TEXT PRIMARY KEY,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    checksum TEXT NOT NULL,
    mtime REAL NOT NULL,
    category TEXT,
    title TEXT,
    version TEXT,
    effective_date TEXT,
    doc_status TEXT,
    supersedes TEXT,
    ingested_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL,
    category TEXT,
    version TEXT,
    effective_date TEXT,
    doc_status TEXT,
    supersedes TEXT,
    checksum TEXT NOT NULL,
    chunk_strategy TEXT NOT NULL,
    unit_label TEXT,
    chunk_index INTEGER,
    text TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_path ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(chunk_strategy);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    answer_id TEXT NOT NULL,
    question TEXT NOT NULL,
    useful INTEGER NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL
);
"""


@dataclass
class FileRecord:
    file_path: str
    file_name: str
    file_type: str
    checksum: str
    mtime: float
    category: str
    title: str
    version: str
    effective_date: str
    doc_status: str
    supersedes: str
    ingested_at: str


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # --- file registry (incremental indexing) ---

    def get_file(self, file_path: str) -> FileRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM files WHERE file_path = ?", (file_path,)).fetchone()
        if row is None:
            return None
        return FileRecord(**{k: row[k] for k in row.keys()})

    def all_files(self) -> list[FileRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM files").fetchall()
        return [FileRecord(**{k: r[k] for k in r.keys()}) for r in rows]

    def upsert_file(self, record: FileRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO files (file_path, file_name, file_type, checksum, mtime, category,
                                    title, version, effective_date, doc_status, supersedes, ingested_at)
                VALUES (:file_path, :file_name, :file_type, :checksum, :mtime, :category,
                        :title, :version, :effective_date, :doc_status, :supersedes, :ingested_at)
                ON CONFLICT(file_path) DO UPDATE SET
                    file_name=excluded.file_name, file_type=excluded.file_type,
                    checksum=excluded.checksum, mtime=excluded.mtime, category=excluded.category,
                    title=excluded.title, version=excluded.version, effective_date=excluded.effective_date,
                    doc_status=excluded.doc_status, supersedes=excluded.supersedes,
                    ingested_at=excluded.ingested_at
                """,
                record.__dict__,
            )

    def delete_file(self, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

    def set_doc_status(self, file_path: str, doc_status: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE files SET doc_status = ? WHERE file_path = ?", (doc_status, file_path))
            conn.execute("UPDATE chunks SET doc_status = ? WHERE file_path = ?", (doc_status, file_path))

    # --- chunks ---

    def replace_chunks_for_file(self, file_path: str, chunks: list[Chunk]) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
            conn.executemany(
                """
                INSERT INTO chunks (chunk_id, file_path, file_name, file_type, category,
                                     version, effective_date, doc_status, supersedes, checksum,
                                     chunk_strategy, unit_label, chunk_index, text)
                VALUES (:chunk_id, :file_path, :file_name, :file_type, :category,
                        :version, :effective_date, :doc_status, :supersedes, :checksum,
                        :chunk_strategy, :unit_label, :chunk_index, :text)
                """,
                [
                    {
                        "chunk_id": c.chunk_id,
                        "file_path": c.file_path,
                        "file_name": c.file_name,
                        "file_type": c.file_type,
                        "category": c.category,
                        "version": c.version,
                        "effective_date": c.effective_date,
                        "doc_status": c.doc_status,
                        "supersedes": c.supersedes,
                        "checksum": c.checksum,
                        "chunk_strategy": c.chunk_strategy,
                        "unit_label": c.unit_label,
                        "chunk_index": c.chunk_index,
                        "text": c.text,
                    }
                    for c in chunks
                ],
            )

    def get_chunk(self, chunk_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return dict(row) if row else None

    def chunks_for_file(self, file_path: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE file_path = ?", (file_path,)).fetchall()
        return [dict(r) for r in rows]

    def all_chunks_for_strategy(self, chunk_strategy: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE chunk_strategy = ?", (chunk_strategy,)).fetchall()
        return [dict(r) for r in rows]

    def chunk_count_for_strategy(self, chunk_strategy: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE chunk_strategy = ?", (chunk_strategy,)
            ).fetchone()
        return int(row["n"])

    # --- feedback (FR-16) ---

    def add_feedback(self, answer_id: str, question: str, useful: bool, reason: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback (answer_id, question, useful, reason, created_at) VALUES (?, ?, ?, ?, ?)",
                (answer_id, question, int(useful), reason, datetime.now(UTC).isoformat()),
            )

    def feedback_summary(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT useful, COUNT(*) AS n FROM feedback GROUP BY useful").fetchall()
        summary = {"useful": 0, "not_useful": 0}
        for r in rows:
            summary["useful" if r["useful"] else "not_useful"] = r["n"]
        return summary
