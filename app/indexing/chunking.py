"""Two configurable chunking strategies (FR-04).

fixed_window
    Splits the full document text into fixed-size character windows with a
    configurable overlap. Simple, predictable chunk sizes, cheap to compute.
    Trade-off: windows can cut across sentence/section boundaries, which can
    split a claim from the condition that qualifies it (bad for
    multi-chunk-synthesis questions) and produces citations at a coarser
    "somewhere in the document" granularity since a window may span units.

structure_aware
    Chunks within each `ContentUnit` (a page or a heading-delimited section),
    splitting on paragraph boundaries and packing paragraphs up to a max
    chunk size instead of cutting mid-paragraph. Trade-off: chunk sizes are
    less uniform and very long single paragraphs still get hard-split, but
    each chunk stays inside one page/section, so citations are precise and
    a chunk never silently mixes unrelated sections.

Both strategies can be run over the same corpus (chunk_strategy is stored on
every chunk) so the evaluation harness can compare them directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.parsers import RawDocument


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    file_name: str
    file_type: str
    category: str
    version: str
    effective_date: str
    doc_status: str  # "current" | "superseded" | "unknown"
    supersedes: str
    checksum: str  # content hash — doubles as the stable identity of the source document's content
    chunk_strategy: str
    unit_label: str
    chunk_index: int
    text: str


def _make_chunk_id(checksum: str, strategy: str, index: int) -> str:
    return f"{checksum[:16]}:{strategy}:{index}"


def chunk_fixed_window(
    doc: RawDocument,
    checksum: str,
    doc_meta: dict[str, str],
    chunk_size_chars: int = 900,
    overlap_chars: int = 150,
) -> list[Chunk]:
    full_text = doc.full_text
    if not full_text:
        return []
    step = max(1, chunk_size_chars - overlap_chars)
    chunks: list[Chunk] = []
    index = 0
    pos = 0
    n = len(full_text)
    while pos < n:
        window = full_text[pos : pos + chunk_size_chars].strip()
        if window:
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(checksum, "fixed_window", index),
                    file_path=str(doc.file_path),
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    category=doc_meta.get("category", "unknown"),
                    version=doc_meta.get("version", ""),
                    effective_date=doc_meta.get("effective_date", ""),
                    doc_status=doc_meta.get("doc_status", "unknown"),
                    supersedes=doc_meta.get("supersedes", ""),
                    checksum=checksum,
                    chunk_strategy="fixed_window",
                    unit_label="offset "
                    + f"{pos}-{min(pos + chunk_size_chars, n)}",
                    chunk_index=index,
                    text=window,
                )
            )
            index += 1
        if pos + chunk_size_chars >= n:
            break
        pos += step
    return chunks


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    return [p for p in parts if p]


def chunk_structure_aware(
    doc: RawDocument,
    checksum: str,
    doc_meta: dict[str, str],
    max_chunk_chars: int = 1200,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for unit in doc.units:
        paragraphs = _split_paragraphs(unit.text) or [unit.text]
        current: list[str] = []
        current_len = 0

        # unit_label is bound as a default arg (evaluated now, not at call time) so flush()
        # always reports the label of the unit it was defined for, not whatever `unit` is
        # by the time it's called on the next loop iteration.
        def flush(current_ref: list[str], unit_label: str = unit.label) -> None:
            nonlocal index
            text = "\n\n".join(current_ref).strip()
            if not text:
                return
            chunks.append(
                Chunk(
                    chunk_id=_make_chunk_id(checksum, "structure_aware", index),
                    file_path=str(doc.file_path),
                    file_name=doc.file_name,
                    file_type=doc.file_type,
                    category=doc_meta.get("category", "unknown"),
                    version=doc_meta.get("version", ""),
                    effective_date=doc_meta.get("effective_date", ""),
                    doc_status=doc_meta.get("doc_status", "unknown"),
                    supersedes=doc_meta.get("supersedes", ""),
                    checksum=checksum,
                    chunk_strategy="structure_aware",
                    unit_label=unit_label,
                    chunk_index=index,
                    text=text,
                )
            )
            index += 1

        for para in paragraphs:
            if len(para) > max_chunk_chars:
                flush(current)
                current, current_len = [], 0
                for start in range(0, len(para), max_chunk_chars):
                    hard_slice = para[start : start + max_chunk_chars]
                    flush([hard_slice])
                continue
            if current_len + len(para) + 2 > max_chunk_chars and current:
                flush(current)
                current, current_len = [], 0
            current.append(para)
            current_len += len(para) + 2
        flush(current)
    return chunks


CHUNKERS = {
    "fixed_window": chunk_fixed_window,
    "structure_aware": chunk_structure_aware,
}
