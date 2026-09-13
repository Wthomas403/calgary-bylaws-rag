# Calgary Bylaws RAG

A retrieval-augmented generation system that answers questions about Calgary
bylaws, with retrieval and answer quality measured by an automated eval suite
that runs in CI on every commit.

## Architecture

```
PDFs (data/bylaws/)
        │
        ▼
  src/chunker.py        section-aware chunking (splits on numbered
        │                sections/subsections, keeps bylaw name +
        │                section number + page as metadata)
        ▼
  scripts/ingest.py      embeds chunks with sentence-transformers
        │                (all-MiniLM-L6-v2) and writes them to
        │                data/chunks.json (chunk text + metadata +
        │                embedding vector)
        ▼
  src/retriever.py       hybrid search over data/chunks.json:
        │                  - BM25 keyword search
        │                  - dense cosine similarity search
        │                combined into a single top-5 ranking
        │
        ├──► cli/search.py       `python -m cli.search "question"`
        │                        prints top-3 results with source
        │                        bylaw + section
        │
        └──► FastAPI service     serves retrieval (and eventually
             (planned)           generation) over HTTP
```

Eval loop:

```
scripts/generate_qa.py   uses the Claude API to synthesize ~200 Q&A
                          pairs from bylaw text -> data/synthetic_qa.json
                          (hand-validated by a human before trusted)

scripts/eval.py           runs retrieval for each question and reports:
                             - recall@5: did a chunk containing the
                               right answer appear in the top 5?
                             - faithfulness: does the retrieved text
                               actually support the answer?
                           exits non-zero if either metric is below
                           threshold

.github/workflows/eval.yml   runs scripts/eval.py on every push and
                              fails the build if recall@5 < 0.70 or
                              faithfulness < 0.80
```

## Project layout

```
src/            core library code (chunker, retriever, ...)
scripts/        one-off / pipeline scripts (ingest, generate_qa, eval)
cli/            Typer CLI entry points
tests/          pytest test suite
data/bylaws/    source bylaw PDFs (not committed; add your own)
data/           generated artifacts: chunks.json, synthetic_qa.json
```

## Status

- [x] Phase 1: project scaffolding
- [x] Phase 2: chunking + ingestion pipeline
- [ ] Phase 3: hybrid retrieval
- [ ] Phase 4: synthetic eval set + CI gate
- [ ] Phase 5: CLI

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add bylaw PDFs to `data/bylaws/`, then run:

```bash
python -m scripts.ingest
```

This writes chunked, embedded bylaw text to `data/chunks.json`.

## Tests

```bash
pytest
```
