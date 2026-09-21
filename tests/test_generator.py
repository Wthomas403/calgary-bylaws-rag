from src.generator import format_excerpts, generate_answer
from src.retriever import SearchResult
from tests.fake_anthropic import FakeAnthropicClient

RESULT = SearchResult(
    bylaw_name="pet-bylaw",
    section="2",
    page=3,
    text="Residents may keep backyard chickens under permit conditions.",
    score=0.9,
    bm25_score=0.5,
    dense_score=0.9,
)


def test_format_excerpts_includes_source_metadata():
    formatted = format_excerpts([RESULT])
    assert "pet-bylaw" in formatted
    assert "Section 2" in formatted
    assert "page 3" in formatted
    assert "backyard chickens" in formatted


def test_format_excerpts_handles_no_section():
    result = SearchResult(
        bylaw_name="b", section=None, page=1, text="t", score=0.0, bm25_score=0.0, dense_score=0.0
    )
    formatted = format_excerpts([result])
    assert "Preamble" in formatted


def test_format_excerpts_joins_multiple_results():
    result_2 = SearchResult(
        bylaw_name="parking-bylaw",
        section="1",
        page=1,
        text="No parking on boulevard grass.",
        score=0.8,
        bm25_score=0.5,
        dense_score=0.8,
    )
    formatted = format_excerpts([RESULT, result_2])
    assert "pet-bylaw" in formatted
    assert "parking-bylaw" in formatted


def test_generate_answer_returns_model_text():
    client = FakeAnthropicClient(["Yes, with a permit. (pet-bylaw, Section 2)"])
    answer = generate_answer(client, "Can I keep chickens?", [RESULT])
    assert answer == "Yes, with a permit. (pet-bylaw, Section 2)"


def test_generate_answer_handles_no_results_without_calling_the_model():
    client = FakeAnthropicClient([])  # would raise StopIteration if called
    answer = generate_answer(client, "Can I keep chickens?", [])
    assert "couldn't find" in answer.lower()
