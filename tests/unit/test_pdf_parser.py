"""Tests for PDF front-matter extraction (app.ingestion.parsers.parse_pdf).

Uses fpdf2 (already a project dependency, used by scripts/make_corpus.py) to
generate small real PDFs rather than mocking pypdf internals.
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

from app.ingestion.parsers import parse_pdf


def _make_pdf(path: Path, pages: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for page_text in pages:
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        pdf.multi_cell(0, 8, page_text)
    pdf.output(str(path))


def test_front_matter_on_page_one_is_extracted_and_stripped_from_body(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    _make_pdf(
        path,
        pages=[
            "Title: Deployment Guide\n"
            "Category: technical-guide\n"
            "Version: 1.3\n"
            "Effective_Date: 2024-04-10\n"
            "Status: current\n"
            "\n"
            "Overview\n\n"
            "This guide describes how to deploy the platform.",
            "Second page content with no header at all.",
        ],
    )

    doc = parse_pdf(path)

    assert doc.header["title"] == "Deployment Guide"
    assert doc.header["category"] == "technical-guide"
    assert doc.header["version"] == "1.3"
    assert doc.header["effective_date"] == "2024-04-10"
    assert doc.header["status"] == "current"

    page_one = next(u for u in doc.units if u.label == "page 1")
    assert "Title:" not in page_one.text
    assert "Category:" not in page_one.text
    assert "Overview" in page_one.text
    assert "This guide describes" in page_one.text

    page_two = next(u for u in doc.units if u.label == "page 2")
    assert page_two.text == "Second page content with no header at all."


def test_pdf_without_front_matter_has_empty_header_and_full_page_one_text(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    _make_pdf(path, pages=["Just a plain page with no header lines at all."])

    doc = parse_pdf(path)

    assert doc.header == {}
    assert doc.units[0].text == "Just a plain page with no header lines at all."


def test_page_that_becomes_empty_after_stripping_header_is_dropped(tmp_path: Path):
    path = tmp_path / "doc.pdf"
    _make_pdf(
        path,
        pages=[
            "Title: Header Only\nCategory: technical-guide",
            "Real content on the second page.",
        ],
    )

    doc = parse_pdf(path)

    assert all(u.label != "page 1" for u in doc.units)
    assert doc.header["title"] == "Header Only"
    assert doc.units[0].label == "page 2"
