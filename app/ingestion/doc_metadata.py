"""Derives category/title/version/effective_date for a document (FR-01, FR-12).

Front-matter header fields (parsed by `parsers.py`) always win. Anything
missing falls back to the folder name (category) and filename patterns
(title/version), so plain files without a header still get usable metadata.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

_VERSION_SUFFIX_RE = re.compile(r"[-_ ]v?(\d+(?:\.\d+)*)$", re.IGNORECASE)
_DATE_SUFFIX_RE = re.compile(r"[-_ ](\d{4}-\d{2}-\d{2})$")


def derive_doc_metadata(file_path: Path, documents_root: Path, header: dict[str, str]) -> dict[str, str]:
    try:
        rel_parent = file_path.resolve().relative_to(documents_root.resolve()).parent
    except ValueError:
        rel_parent = file_path.parent
    category = header.get("category") or (rel_parent.parts[0] if rel_parent.parts else "uncategorized")

    stem = file_path.stem
    stem_no_date = _DATE_SUFFIX_RE.sub("", stem)
    filename_version_match = _VERSION_SUFFIX_RE.search(stem_no_date)
    stem_no_version = _VERSION_SUFFIX_RE.sub("", stem_no_date)

    title = header.get("title") or stem_no_version.replace("-", " ").replace("_", " ").strip()
    version = header.get("version") or (filename_version_match.group(1) if filename_version_match else "")
    date_match = _DATE_SUFFIX_RE.search(stem)
    effective_date = header.get("effective_date") or (date_match.group(1) if date_match else "")
    supersedes = header.get("supersedes", "")
    status_hint = header.get("status", "")

    return {
        "category": category.lower(),
        "title": title.lower(),
        "version": version,
        "effective_date": effective_date,
        "supersedes": supersedes,
        "status_hint": status_hint,
    }


def _parse_date(value: str) -> date | None:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _parse_version_tuple(value: str) -> tuple[int, ...]:
    digits = re.findall(r"\d+", value or "")
    return tuple(int(d) for d in digits)


def doc_order_key(effective_date: str, version: str, mtime: float) -> tuple:
    """Sort key such that the *applicable latest* document is the maximum.

    Prefers effective_date when present (it's the actual real-world
    recency signal), then falls back to a parsed version number, then to
    file modification time as a last resort.
    """
    parsed_date = _parse_date(effective_date)
    version_tuple = _parse_version_tuple(version)
    return (
        1 if parsed_date else 0,
        parsed_date or date.min,
        1 if version_tuple else 0,
        version_tuple,
        mtime,
    )
