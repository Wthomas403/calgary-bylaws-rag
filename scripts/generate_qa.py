"""Generate synthetic Q&A pairs from bylaw chunks using the Claude API.

Usage:
    python -m scripts.generate_qa

Requires ANTHROPIC_API_KEY in the environment (or a .env file in the
project root). Output is written to data/synthetic_qa.json and is meant
to be hand-validated on a sample before being trusted as an eval set --
scripts/eval.py treats it as ground truth.
"""

import json
import random
import re
from pathlib import Path

import typer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "claude-haiku-4-5-20251001"
MIN_CHUNK_CHARS = 120  # skip boilerplate/title chunks too short to ask a real question about

PROMPT_TEMPLATE = """You are building an evaluation set for a bylaw search system.

Below is one excerpt from a City of Calgary bylaw. Write exactly one \
realistic question a resident might ask that is answered by this excerpt, \
and a concise answer based only on the excerpt's text.

Rules:
- The question must be answerable using only the excerpt below -- do not \
assume outside knowledge of the bylaw.
- Phrase the question the way an ordinary resident would ask it (plain \
language), not by quoting the section number.
- The answer must be 1-3 sentences and grounded only in the excerpt.
- Respond with ONLY a JSON object of the form: {{"question": "...", "answer": "..."}}

Bylaw: {bylaw_name}
Excerpt:
\"\"\"
{text}
\"\"\"
"""

CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

app = typer.Typer()


def strip_code_fence(text: str) -> str:
    """Claude sometimes wraps JSON in a ```json ... ``` fence despite being
    told not to. Strip it so json.loads() sees a bare object."""
    return CODE_FENCE_PATTERN.sub("", text).strip()


def sample_chunks(chunks: list[dict], n: int, max_per_bylaw: int) -> list[dict]:
    """Sample up to n chunks, capping how many come from any one bylaw
    so a handful of long bylaws don't dominate the eval set."""
    # section is None only for the one title/preamble chunk each bylaw PDF
    # starts with ("OFFICE CONSOLIDATION / BYLAW NUMBER ..."); it's boilerplate,
    # not a specific, answerable provision, so questions grounded in it tend to
    # come out generic ("What is this bylaw about?").
    eligible = [
        c for c in chunks if c["section"] is not None and len(c["text"]) >= MIN_CHUNK_CHARS
    ]
    random.shuffle(eligible)

    per_bylaw_count: dict[str, int] = {}
    selected: list[dict] = []
    for chunk in eligible:
        if len(selected) >= n:
            break
        count = per_bylaw_count.get(chunk["bylaw_name"], 0)
        if count >= max_per_bylaw:
            continue
        selected.append(chunk)
        per_bylaw_count[chunk["bylaw_name"]] = count + 1

    return selected


def generate_qa_pair(client: Anthropic, chunk: dict) -> dict | None:
    """Ask Claude for one question+answer grounded in a chunk. Returns
    None if the response isn't parseable JSON with the expected fields."""
    prompt = PROMPT_TEMPLATE.format(bylaw_name=chunk["bylaw_name"], text=chunk["text"])
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = strip_code_fence(response.content[0].text.strip())

    try:
        parsed = json.loads(raw)
        question = parsed["question"].strip()
        answer = parsed["answer"].strip()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return None

    if not question or not answer:
        return None

    return {
        "question": question,
        "answer": answer,
        "bylaw_name": chunk["bylaw_name"],
        "section": chunk.get("section"),
        "page": chunk["page"],
    }


@app.command()
def generate(
    chunks_path: Path = typer.Option(Path("data/chunks.json"), help="Chunks + embeddings produced by scripts/ingest.py."),
    output_path: Path = typer.Option(Path("data/synthetic_qa.json")),
    n: int = typer.Option(200, help="Number of Q&A pairs to generate."),
    max_per_bylaw: int = typer.Option(15, help="Cap per bylaw so no single bylaw dominates the eval set."),
    seed: int = typer.Option(0),
) -> None:
    random.seed(seed)
    chunks = json.loads(chunks_path.read_text())
    selected = sample_chunks(chunks, n=n, max_per_bylaw=max_per_bylaw)
    bylaw_count = len({c["bylaw_name"] for c in selected})
    typer.echo(f"Sampled {len(selected)} chunks across {bylaw_count} bylaws")

    client = Anthropic()
    qa_pairs = []
    for i, chunk in enumerate(selected, start=1):
        pair = generate_qa_pair(client, chunk)
        if pair is None:
            typer.echo(f"  [{i}/{len(selected)}] skipped (unparseable response)")
            continue
        qa_pairs.append(pair)
        typer.echo(f"  [{i}/{len(selected)}] {pair['question']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(qa_pairs, indent=2))
    typer.echo(f"Wrote {len(qa_pairs)} Q&A pairs to {output_path}")
    typer.echo("Hand-validate a sample of these before trusting them as an eval set.")


if __name__ == "__main__":
    app()
