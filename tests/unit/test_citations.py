from __future__ import annotations

from app.generation.citations import assemble_and_validate_citations
from app.generation.context_builder import ContextBlock, ContextBundle


def _bundle() -> ContextBundle:
    blocks = [
        ContextBlock(
            tag="S1", chunk_id="c1", file_name="a.md", unit_label="Intro",
            category="policy", version="1.0", doc_status="current", text="text a", injection_flagged=False,
        ),
        ContextBlock(
            tag="S2", chunk_id="c2", file_name="b.md", unit_label="Body",
            category="policy", version="1.0", doc_status="current", text="text b", injection_flagged=False,
        ),
    ]
    return ContextBundle(blocks=blocks, dropped_for_budget=0)


def test_valid_citations_are_kept_and_mapped():
    answer = "Fact one [S1]. Fact two [S2]."
    result = assemble_and_validate_citations(answer, _bundle())

    assert [c.tag for c in result.citations] == ["S1", "S2"]
    assert result.citations[0].file_name == "a.md"
    assert result.invalid_tags_found == []
    assert "[S1]" in result.cleaned_answer and "[S2]" in result.cleaned_answer


def test_hallucinated_citation_tag_is_stripped():
    answer = "Fact one [S1]. A made-up fact [S5]."
    result = assemble_and_validate_citations(answer, _bundle())

    assert [c.tag for c in result.citations] == ["S1"]
    assert result.invalid_tags_found == ["S5"]
    assert "[S5]" not in result.cleaned_answer


def test_multi_tag_bracket_keeps_only_valid_tags():
    answer = "Combined fact [S1, S5]."
    result = assemble_and_validate_citations(answer, _bundle())

    assert [c.tag for c in result.citations] == ["S1"]
    assert "[S1]" in result.cleaned_answer
    assert "S5" not in result.cleaned_answer


def test_non_citation_brackets_are_left_untouched():
    answer = "See the [glossary] for definitions [S1]."
    result = assemble_and_validate_citations(answer, _bundle())

    assert "[glossary]" in result.cleaned_answer
    assert [c.tag for c in result.citations] == ["S1"]
