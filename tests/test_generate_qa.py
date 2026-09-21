from scripts.generate_qa import generate_qa_pair, sample_chunks, strip_code_fence
from tests.fake_anthropic import FakeAnthropicClient

CHUNKS = [
    {"bylaw_name": "a", "section": "1", "page": 1, "text": "x" * 200},
    {"bylaw_name": "a", "section": "2", "page": 1, "text": "x" * 200},
    {"bylaw_name": "a", "section": "3", "page": 2, "text": "x" * 200},
    {"bylaw_name": "b", "section": "1", "page": 1, "text": "x" * 200},
    {"bylaw_name": "c", "section": "1", "page": 1, "text": "too short"},
]


def test_sample_chunks_skips_short_chunks():
    selected = sample_chunks(CHUNKS, n=10, max_per_bylaw=10)
    assert all(len(c["text"]) >= 120 for c in selected)
    assert "c" not in {c["bylaw_name"] for c in selected}


def test_sample_chunks_skips_sectionless_preamble_chunks():
    chunks = CHUNKS + [{"bylaw_name": "d", "section": None, "page": 1, "text": "x" * 200}]
    selected = sample_chunks(chunks, n=10, max_per_bylaw=10)
    assert "d" not in {c["bylaw_name"] for c in selected}


def test_sample_chunks_caps_per_bylaw():
    selected = sample_chunks(CHUNKS, n=10, max_per_bylaw=1)
    counts = {}
    for c in selected:
        counts[c["bylaw_name"]] = counts.get(c["bylaw_name"], 0) + 1
    assert all(count <= 1 for count in counts.values())


def test_sample_chunks_respects_n():
    selected = sample_chunks(CHUNKS, n=2, max_per_bylaw=10)
    assert len(selected) == 2


def test_strip_code_fence_removes_json_fence():
    fenced = '```json\n{"question": "q", "answer": "a"}\n```'
    assert strip_code_fence(fenced) == '{"question": "q", "answer": "a"}'


def test_strip_code_fence_removes_bare_fence():
    fenced = '```\n{"question": "q", "answer": "a"}\n```'
    assert strip_code_fence(fenced) == '{"question": "q", "answer": "a"}'


def test_strip_code_fence_leaves_unfenced_json_alone():
    raw = '{"question": "q", "answer": "a"}'
    assert strip_code_fence(raw) == raw


def test_generate_qa_pair_parses_fenced_response():
    client = FakeAnthropicClient(['```json\n{"question": "Can I do X?", "answer": "No."}\n```'])
    pair = generate_qa_pair(client, CHUNKS[0])
    assert pair["question"] == "Can I do X?"


def test_generate_qa_pair_parses_valid_response():
    client = FakeAnthropicClient(['{"question": "Can I do X?", "answer": "No, per section 1."}'])
    chunk = CHUNKS[0]

    pair = generate_qa_pair(client, chunk)

    assert pair == {
        "question": "Can I do X?",
        "answer": "No, per section 1.",
        "bylaw_name": "a",
        "section": "1",
        "page": 1,
    }


def test_generate_qa_pair_returns_none_on_invalid_json():
    client = FakeAnthropicClient(["not json"])
    pair = generate_qa_pair(client, CHUNKS[0])
    assert pair is None


def test_generate_qa_pair_returns_none_on_missing_fields():
    client = FakeAnthropicClient(['{"question": "Can I do X?"}'])
    pair = generate_qa_pair(client, CHUNKS[0])
    assert pair is None
