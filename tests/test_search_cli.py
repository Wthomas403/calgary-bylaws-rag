from cli.search import format_result
from src.retriever import SearchResult


def test_format_result_shows_bylaw_section_and_page():
    result = SearchResult(
        bylaw_name="pet-bylaw",
        section="2",
        page=3,
        text="Residents may keep backyard chickens under permit conditions.",
        score=0.91,
        bm25_score=0.5,
        dense_score=0.95,
    )

    formatted = format_result(1, result)

    assert "pet-bylaw" in formatted
    assert "Section 2" in formatted
    assert "page 3" in formatted
    assert "0.910" in formatted
    assert "backyard chickens" in formatted


def test_format_result_handles_no_section():
    result = SearchResult(
        bylaw_name="pet-bylaw",
        section=None,
        page=1,
        text="Preamble text.",
        score=0.5,
        bm25_score=0.0,
        dense_score=0.5,
    )

    formatted = format_result(1, result)

    assert "Preamble" in formatted


def test_format_result_truncates_long_text():
    long_text = "word " * 200
    result = SearchResult(
        bylaw_name="pet-bylaw",
        section="1",
        page=1,
        text=long_text,
        score=0.5,
        bm25_score=0.0,
        dense_score=0.5,
    )

    formatted = format_result(1, result)

    assert formatted.rstrip().endswith("...")
    assert len(formatted) < len(long_text)


def test_format_result_collapses_internal_whitespace():
    result = SearchResult(
        bylaw_name="pet-bylaw",
        section="1",
        page=1,
        text="Line one.\nLine   two.",
        score=0.5,
        bm25_score=0.0,
        dense_score=0.5,
    )

    formatted = format_result(1, result)

    assert "Line one. Line two." in formatted
