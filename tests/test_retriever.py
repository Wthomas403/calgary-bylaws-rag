import numpy as np
import pytest

from src.retriever import Retriever

CHUNKS = [
    {
        "bylaw_name": "parking-bylaw",
        "section": "1",
        "page": 1,
        "text": "No person shall park a vehicle on boulevard grass.",
        "embedding": [1.0, 0.0],
    },
    {
        "bylaw_name": "pet-bylaw",
        "section": "2",
        "page": 3,
        "text": "Residents may keep backyard chickens under permit conditions.",
        "embedding": [0.0, 1.0],
    },
    {
        "bylaw_name": "noise-bylaw",
        "section": "5",
        "page": 2,
        "text": "Loud construction noise is prohibited before 7am.",
        "embedding": [0.7, 0.7],
    },
]


def make_embed_fn(vector):
    return lambda query: np.array(vector)


def test_keyword_match_ranks_first_with_bm25_only():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 0.0]), bm25_weight=1.0)
    results = retriever.search("park vehicle boulevard", top_k=2)
    assert results[0].bylaw_name == "parking-bylaw"


def test_dense_match_ranks_first_with_dense_only():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0)
    results = retriever.search("something with no shared keywords", top_k=1)
    assert results[0].bylaw_name == "pet-bylaw"


def test_hybrid_combines_both_signals():
    # Keywords point at parking-bylaw, embedding points at pet-bylaw;
    # an even split should still surface both in the top 2.
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.5)
    results = retriever.search("park vehicle boulevard", top_k=2)
    names = {r.bylaw_name for r in results}
    assert names == {"parking-bylaw", "pet-bylaw"}


def test_top_k_limits_results():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([1.0, 1.0]))
    results = retriever.search("bylaw", top_k=2)
    assert len(results) == 2


def test_result_carries_section_and_page_metadata():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0)
    result = retriever.search("chickens", top_k=1)[0]
    assert result.section == "2"
    assert result.page == 3


def test_invalid_bm25_weight_raises():
    with pytest.raises(ValueError):
        Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 0.0]), bm25_weight=1.5)


def test_empty_chunks_raises():
    with pytest.raises(ValueError):
        Retriever([], embed_fn=make_embed_fn([0.0, 0.0]))
