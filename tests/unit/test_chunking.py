from __future__ import annotations

from pathlib import Path

from app.indexing.chunking import chunk_fixed_window, chunk_structure_aware
from app.ingestion.parsers import ContentUnit, RawDocument

DOC_META = {"category": "policy", "title": "test doc", "version": "1.0", "effective_date": "2024-01-01", "doc_status": "current", "supersedes": ""}


def _doc(units: list[ContentUnit]) -> RawDocument:
    return RawDocument(file_path=Path("doc.md"), file_name="doc.md", file_type="md", header={}, units=units)


def test_fixed_window_respects_size_and_overlap():
    text = "word " * 500  # 2500 chars
    doc = _doc([ContentUnit(index=0, label="document body", text=text.strip())])
    chunks = chunk_fixed_window(doc, checksum="abc123", doc_meta=DOC_META, chunk_size_chars=900, overlap_chars=150)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert len(chunk.text) <= 900
    assert all(c.chunk_strategy == "fixed_window" for c in chunks)
    # chunk ids are stable and ordered
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_fixed_window_overlap_repeats_tail_of_previous_chunk():
    text = "".join(f"{i:04d}-" for i in range(400))  # deterministic, indexable content
    doc = _doc([ContentUnit(index=0, label="document body", text=text)])
    chunks = chunk_fixed_window(doc, checksum="abc123", doc_meta=DOC_META, chunk_size_chars=200, overlap_chars=50)

    assert len(chunks) > 1
    tail_of_first = chunks[0].text[-50:]
    assert tail_of_first in chunks[1].text


def test_structure_aware_keeps_chunks_within_unit_boundaries():
    unit_a = ContentUnit(index=0, label="Section A", text="Short paragraph one.\n\nShort paragraph two.")
    unit_b = ContentUnit(index=1, label="Section B", text="Another section entirely.")
    doc = _doc([unit_a, unit_b])
    chunks = chunk_structure_aware(doc, checksum="abc123", doc_meta=DOC_META, max_chunk_chars=1200)

    assert {c.unit_label for c in chunks} == {"Section A", "Section B"}
    # no chunk mixes text from both sections
    for c in chunks:
        if c.unit_label == "Section A":
            assert "Another section" not in c.text


def test_structure_aware_hard_splits_oversized_paragraph():
    long_paragraph = "x" * 3000
    doc = _doc([ContentUnit(index=0, label="document body", text=long_paragraph)])
    chunks = chunk_structure_aware(doc, checksum="abc123", doc_meta=DOC_META, max_chunk_chars=1000)

    assert len(chunks) == 3
    assert all(len(c.text) <= 1000 for c in chunks)


def test_structure_aware_packs_multiple_short_paragraphs_into_one_chunk():
    text = "\n\n".join(["Paragraph."] * 5)
    doc = _doc([ContentUnit(index=0, label="document body", text=text)])
    chunks = chunk_structure_aware(doc, checksum="abc123", doc_meta=DOC_META, max_chunk_chars=1200)

    assert len(chunks) == 1
