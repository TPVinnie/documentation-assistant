from __future__ import annotations

from pathlib import Path

from app.ingestion.doc_metadata import derive_doc_metadata, doc_order_key


def test_header_fields_take_precedence_over_filename():
    root = Path("/docs")
    path = root / "policies" / "some-file-v9.md"
    header = {"title": "Real Title", "category": "policy", "version": "2.0", "effective_date": "2024-01-01"}

    meta = derive_doc_metadata(path, root, header)

    assert meta["title"] == "real title"
    assert meta["category"] == "policy"
    assert meta["version"] == "2.0"
    assert meta["effective_date"] == "2024-01-01"


def test_falls_back_to_folder_and_filename_when_no_header():
    root = Path("/docs")
    path = root / "release-notes" / "release-notes-2.4.md"

    meta = derive_doc_metadata(path, root, header={})

    assert meta["category"] == "release-notes"
    assert meta["version"] == "2.4"
    assert "release" in meta["title"]


def test_doc_order_key_prefers_effective_date_over_version():
    older_but_higher_version = doc_order_key(effective_date="", version="9.0", mtime=100.0)
    newer_with_date = doc_order_key(effective_date="2024-01-01", version="1.0", mtime=50.0)

    assert newer_with_date > older_but_higher_version


def test_doc_order_key_falls_back_to_version_then_mtime():
    v1 = doc_order_key(effective_date="", version="1.0", mtime=100.0)
    v2 = doc_order_key(effective_date="", version="2.0", mtime=1.0)
    assert v2 > v1

    no_metadata_older = doc_order_key(effective_date="", version="", mtime=1.0)
    no_metadata_newer = doc_order_key(effective_date="", version="", mtime=2.0)
    assert no_metadata_newer > no_metadata_older
