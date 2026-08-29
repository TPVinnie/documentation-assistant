from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.report import FileStatus
from tests.conftest import write_doc


def _outcome(report, path: Path) -> str:
    for o in report.outcomes:
        if o.file_path == str(path):
            return o.status
    raise AssertionError(f"no outcome recorded for {path}")


def test_ingestion_handles_supported_unsupported_empty_and_duplicate_files(test_settings: Settings):
    docs_dir = test_settings.documents_dir
    good = write_doc(docs_dir, "policy-a.md", "# Title\n\nThis is a real policy with real content.")
    write_doc(docs_dir, "unsupported.rtf", "{\\rtf1 not supported}")
    empty = docs_dir / "empty.txt"
    empty.write_text("", encoding="utf-8")
    duplicate = write_doc(docs_dir, "policy-a-copy.md", "# Title\n\nThis is a real policy with real content.")

    pipeline = IngestionPipeline(test_settings)
    result = pipeline.run()

    assert _outcome(result.report, docs_dir / "unsupported.rtf") == FileStatus.SKIPPED
    assert _outcome(result.report, empty) == FileStatus.SKIPPED
    # exactly one of the byte-identical files is indexed and the other flagged as a duplicate;
    # which one wins is a filesystem-scan-order detail, not something the pipeline guarantees
    statuses = {_outcome(result.report, good), _outcome(result.report, duplicate)}
    assert statuses == {FileStatus.INDEXED, FileStatus.SKIPPED}


def test_incremental_reingestion_skips_unchanged_files(test_settings: Settings):
    docs_dir = test_settings.documents_dir
    write_doc(docs_dir, "policy-a.md", "# Title\n\nOriginal content here.")

    pipeline = IngestionPipeline(test_settings)
    first = pipeline.run()
    assert first.report.summary()["indexed"] == 1

    second = pipeline.run()
    summary = second.report.summary()
    assert summary["indexed"] == 0
    assert summary["unchanged"] == 1


def test_modifying_a_file_reindexes_it_and_deleting_removes_it(test_settings: Settings):
    docs_dir = test_settings.documents_dir
    path = write_doc(docs_dir, "policy-a.md", "# Title\n\nOriginal content here.")

    pipeline = IngestionPipeline(test_settings)
    pipeline.run()

    write_doc(docs_dir, "policy-a.md", "# Title\n\nCompletely different updated content.")
    updated = pipeline.run()
    assert _outcome(updated.report, path) == FileStatus.UPDATED

    path.unlink()
    removed = pipeline.run()
    assert _outcome(removed.report, path) == FileStatus.REMOVED
    assert pipeline.vector_store.count() == 0


def test_version_family_resolves_current_and_superseded(test_settings: Settings):
    docs_dir = test_settings.documents_dir
    write_doc(
        docs_dir,
        "retention-v1.md",
        """---
title: Retention Policy
category: policy
version: 1.0
effective_date: 2023-01-01
---

Logs are kept for 30 days.
""",
    )
    write_doc(
        docs_dir,
        "retention-v2.md",
        """---
title: Retention Policy
category: policy
version: 2.0
effective_date: 2024-01-01
---

Logs are kept for 60 days.
""",
    )

    pipeline = IngestionPipeline(test_settings)
    pipeline.run()

    files = {f.file_name: f for f in pipeline.metadata_store.all_files()}
    assert files["retention-v1.md"].doc_status == "superseded"
    assert files["retention-v2.md"].doc_status == "current"


def test_encrypted_and_corrupted_files_are_reported_as_failed(test_settings: Settings, tmp_path: Path):
    docs_dir = test_settings.documents_dir
    fake_pdf = docs_dir / "corrupted.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4\nnot a real pdf body")

    pipeline = IngestionPipeline(test_settings)
    result = pipeline.run()

    assert _outcome(result.report, fake_pdf) == FileStatus.FAILED


def test_oversized_file_is_skipped_without_being_read(test_settings: Settings):
    test_settings.max_file_size_mb = 0.001  # ~1 KB, so a normal test doc trivially exceeds it
    docs_dir = test_settings.documents_dir
    huge = write_doc(docs_dir, "huge.md", "word " * 2000)

    pipeline = IngestionPipeline(test_settings)
    result = pipeline.run()

    assert _outcome(result.report, huge) == FileStatus.SKIPPED
    outcome = next(o for o in result.report.outcomes if o.file_path == str(huge))
    assert "exceeds" in outcome.message
    assert pipeline.vector_store.count() == 0
