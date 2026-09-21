import numpy as np

from scripts.eval import compute_faithfulness, compute_recall_at_k, is_source_hit, judge_faithfulness
from src.retriever import Retriever
from tests.fake_anthropic import FakeAnthropicClient

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
]


def make_embed_fn(vector):
    return lambda query: np.array(vector)


def test_is_source_hit_matches_bylaw_and_section():
    result = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0).search(
        "keep chickens", top_k=1
    )[0]
    qa = {"bylaw_name": "pet-bylaw", "section": "2"}
    assert is_source_hit(qa, result)


def test_is_source_hit_rejects_wrong_section():
    result = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0).search(
        "keep chickens", top_k=1
    )[0]
    qa = {"bylaw_name": "pet-bylaw", "section": "99"}
    assert not is_source_hit(qa, result)


def test_compute_recall_at_k_counts_hits():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0)
    qa_pairs = [
        {"question": "chickens?", "bylaw_name": "pet-bylaw", "section": "2"},
        {"question": "chickens again?", "bylaw_name": "parking-bylaw", "section": "1"},
    ]

    recall, details = compute_recall_at_k(qa_pairs, retriever, k=1)

    assert recall == 0.5
    assert details[0]["hit"] is True
    assert details[1]["hit"] is False


def test_judge_faithfulness_parses_yes_no():
    client = FakeAnthropicClient(["YES", "NO"])
    qa = {"question": "q", "answer": "a"}
    assert judge_faithfulness(client, qa, "some text") is True
    assert judge_faithfulness(client, qa, "some text") is False


def test_compute_faithfulness_averages_over_sample():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0)
    qa_pairs = [
        {"question": "q1", "answer": "a1", "bylaw_name": "pet-bylaw", "section": "2"},
        {"question": "q2", "answer": "a2", "bylaw_name": "pet-bylaw", "section": "2"},
    ]
    client = FakeAnthropicClient(["YES", "NO"])

    faithfulness, graded = compute_faithfulness(qa_pairs, retriever, client, sample_size=10, seed=0)

    assert graded == 2
    assert faithfulness == 0.5


def test_compute_faithfulness_respects_sample_size():
    retriever = Retriever(CHUNKS, embed_fn=make_embed_fn([0.0, 1.0]), bm25_weight=0.0)
    qa_pairs = [
        {"question": f"q{i}", "answer": "a", "bylaw_name": "pet-bylaw", "section": "2"}
        for i in range(5)
    ]
    client = FakeAnthropicClient(["YES", "YES"])

    faithfulness, graded = compute_faithfulness(qa_pairs, retriever, client, sample_size=2, seed=0)

    assert graded == 2
    assert faithfulness == 1.0
