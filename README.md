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
                              faithfulness < 0.75
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

## Searching from the command line

```bash
python -m cli.search "can I keep chickens in my backyard"
```

Prints the top 3 matching bylaw passages with their bylaw name, section,
and page number. Use `--top-k N` for more/fewer results.

## Eval set

Generating the synthetic Q&A eval set calls the Claude API and needs an
API key:

```bash
export ANTHROPIC_API_KEY=sk-...   # or put it in a .env file
python -m scripts.generate_qa
```

This samples chunks (capped per bylaw so no single bylaw dominates),
asks Claude for one question+answer grounded in each chunk's text, and
writes the result to `data/synthetic_qa.json`. **Hand-validate a sample
of these** before trusting the set -- a bad synthetic question will
just make the eval noisy.

Once you're happy with it, commit `data/synthetic_qa.json` (it's
tracked in git, unlike the source PDFs) so CI can evaluate against it.

Then run the eval itself:

```bash
python -m scripts.eval
```

This reports `recall@5` (did the right bylaw section show up in the
top 5 retrieved results?) and `faithfulness` (does the same top-5
evidence actually support the expected answer, judged by Claude across
the whole eval set for a stable, non-flaky number) and exits non-zero
if either falls below threshold (0.70 / 0.75 by default -- these were
set from the real measured baseline of this dataset: 0.720 / 0.755).

`.github/workflows/eval.yml` runs the test suite and this eval on every
push. It needs an `ANTHROPIC_API_KEY` secret set on the GitHub repo
(Settings -> Secrets and variables -> Actions) for the faithfulness
judge to work.

## Tests

```bash
pytest
```
