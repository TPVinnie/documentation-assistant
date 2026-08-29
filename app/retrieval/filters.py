"""Metadata filters (FR-08): document, version/date, and category.

Filters are applied at query time against both the dense and lexical
indexes, rather than after fusion, so a narrow filter (e.g. one document)
can't be starved out by a top-k cut before it ever gets a chance to match —
the trade-off, versus filtering strictly after fusion, is documented in
ARCHITECTURE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalFilters:
    file_name: str | None = None
    category: str | None = None
    version: str | None = None


def as_plain_dict(filters: RetrievalFilters) -> dict[str, Any]:
    """Flat field->value dict, AND semantics — used directly by the BM25 index."""
    where: dict[str, Any] = {}
    if filters.file_name:
        where["file_name"] = filters.file_name
    if filters.category:
        where["category"] = filters.category
    if filters.version:
        where["version"] = filters.version
    return where


def as_chroma_where(filters: RetrievalFilters) -> dict[str, Any] | None:
    """Chroma requires `$and` once more than one equality condition is present."""
    plain = as_plain_dict(filters)
    if not plain:
        return None
    if len(plain) == 1:
        (key, value), = plain.items()
        return {key: value}
    return {"$and": [{key: value} for key, value in plain.items()]}
