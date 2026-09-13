from src.chunker import chunk_pages


def test_chunk_pages_splits_on_section_numbers():
    pages = [
        "1. Short Title\nThis bylaw may be cited as the Test Bylaw.\n"
        "1.1 Definitions\nIn this bylaw, \"dwelling\" means a building used for residence.",
        "2. General Provisions\nNo person shall park a vehicle on boulevard grass.",
    ]

    chunks = chunk_pages("test-bylaw", pages)

    assert [c.section for c in chunks] == ["1", "1.1", "2"]
    assert all(c.bylaw_name == "test-bylaw" for c in chunks)
    assert chunks[0].page == 1
    assert chunks[2].page == 2
    assert "dwelling" in chunks[1].text


def test_chunk_pages_ignores_blank_lines():
    pages = ["1. Title\n\n\nSome text here."]

    chunks = chunk_pages("test-bylaw", pages)

    assert len(chunks) == 1
    assert "Some text here." in chunks[0].text


def test_chunk_pages_handles_preamble_with_no_section():
    pages = ["Preamble text before any numbered section.\n1. Title\nBody text."]

    chunks = chunk_pages("test-bylaw", pages)

    assert chunks[0].section is None
    assert chunks[1].section == "1"
