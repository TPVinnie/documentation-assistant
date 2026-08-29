"""Processing report models (FR-02): what ingestion did to every file it saw."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class FileStatus(StrEnum):
    INDEXED = "indexed"
    UPDATED = "updated"
    UNCHANGED = "unchanged"
    REMOVED = "removed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class FileOutcome:
    file_path: str
    status: FileStatus
    message: str = ""
    chunk_count: int = 0


@dataclass
class ProcessingReport:
    outcomes: list[FileOutcome] = field(default_factory=list)

    def add(self, file_path: str, status: FileStatus, message: str = "", chunk_count: int = 0) -> None:
        self.outcomes.append(FileOutcome(file_path=file_path, status=status, message=message, chunk_count=chunk_count))

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {status.value: 0 for status in FileStatus}
        for outcome in self.outcomes:
            counts[outcome.status.value] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "files": [
                {
                    "file_path": o.file_path,
                    "status": o.status.value,
                    "message": o.message,
                    "chunk_count": o.chunk_count,
                }
                for o in self.outcomes
            ],
        }
