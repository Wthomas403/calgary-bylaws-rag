"""Evaluate retrieval quality against the synthetic Q&A eval set.

Usage:
    python -m scripts.eval

Computes:
  - recall@5: for each question, did a chunk from its source bylaw
    section appear anywhere in the top-5 retrieved results?
  - faithfulness: for a sample of questions, is the top-5 retrieved
    evidence (the same evidence a downstream answer step would be
    given) sufficient to give the expected answer? Judged by Claude.

Exits with a non-zero status if either metric falls below its
threshold, so this is safe to wire up as a CI gate.
"""

import json
import random
from pathlib import Path

import typer
from anthropic import Anthropic
from dotenv import load_dotenv

from src.retriever import Retriever, SearchResult

load_dotenv()

JUDGE_MODEL = "claude-haiku-4-5-20251001"

RECALL_THRESHOLD_DEFAULT = 0.70
FAITHFULNESS_THRESHOLD_DEFAULT = 0.75
# Grading a random subset makes this a flaky CI gate -- the same threshold
# check can pass or fail run to run depending on which questions got drawn
# (measured: one 50-question draw scored 0.60, another scored 0.78, on the
# same retriever). Cost per call is ~$0.0007 with claude-haiku-4-5, so
# grading the full ~200-question set (~$0.13/run) buys a stable, reproducible
# number instead. Lower this via --faithfulness-sample-size if the eval set
# grows much larger and cost becomes a real concern.
FAITHFULNESS_SAMPLE_SIZE_DEFAULT = 200
FAITHFULNESS_TOP_K = 5  # judge the same evidence set recall@5 checks, not just rank 1

FAITHFULNESS_PROMPT = """You are grading a retrieval system for a bylaw search tool.

Question: {question}
Expected answer: {expected_answer}

Retrieved passage:
\"\"\"
{retrieved_text}
\"\"\"

Does the retrieved passage contain enough information to correctly \
answer the question, consistent with the expected answer? Respond with \
only "YES" or "NO".
"""

app = typer.Typer()


def is_source_hit(qa: dict, result: SearchResult) -> bool:
    return result.bylaw_name == qa["bylaw_name"] and result.section == qa["section"]


def compute_recall_at_k(qa_pairs: list[dict], retriever: Retriever, k: int = 5) -> tuple[float, list[dict]]:
    details = []
    for qa in qa_pairs:
        results = retriever.search(qa["question"], top_k=k)
        hit = any(is_source_hit(qa, r) for r in results)
        details.append({
            "question": qa["question"],
            "hit": hit,
            "top_result": results[0].to_dict() if results else None,
        })
    recall = sum(d["hit"] for d in details) / len(details) if details else 0.0
    return recall, details


def judge_faithfulness(client: Anthropic, qa: dict, retrieved_text: str) -> bool:
    prompt = FAITHFULNESS_PROMPT.format(
        question=qa["question"], expected_answer=qa["answer"], retrieved_text=retrieved_text
    )
    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=5,
        messages=[{"role": "user", "content": prompt}],
    )
    verdict = response.content[0].text.strip().upper()
    return verdict.startswith("YES")


def compute_faithfulness(
    qa_pairs: list[dict],
    retriever: Retriever,
    client: Anthropic,
    sample_size: int,
    seed: int,
) -> tuple[float, int]:
    rng = random.Random(seed)
    sample = qa_pairs if len(qa_pairs) <= sample_size else rng.sample(qa_pairs, sample_size)

    passed = 0
    graded = 0
    for qa in sample:
        results = retriever.search(qa["question"], top_k=FAITHFULNESS_TOP_K)
        if not results:
            continue
        graded += 1
        combined_text = "\n\n---\n\n".join(r.text for r in results)
        if judge_faithfulness(client, qa, combined_text):
            passed += 1

    faithfulness = passed / graded if graded else 0.0
    return faithfulness, graded


@app.command()
def evaluate(
    qa_path: Path = typer.Option(Path("data/synthetic_qa.json")),
    chunks_path: Path = typer.Option(Path("data/chunks.json")),
    recall_threshold: float = typer.Option(RECALL_THRESHOLD_DEFAULT),
    faithfulness_threshold: float = typer.Option(FAITHFULNESS_THRESHOLD_DEFAULT),
    faithfulness_sample_size: int = typer.Option(
        FAITHFULNESS_SAMPLE_SIZE_DEFAULT,
        help="Number of questions to LLM-judge for faithfulness (bounds API cost).",
    ),
    seed: int = typer.Option(0),
) -> None:
    if not qa_path.exists():
        typer.echo(
            f"{qa_path} not found. Generate it first with "
            f"'python -m scripts.generate_qa' (requires ANTHROPIC_API_KEY)."
        )
        raise typer.Exit(code=1)

    qa_pairs = json.loads(qa_path.read_text())
    if not qa_pairs:
        typer.echo("No Q&A pairs found -- nothing to evaluate.")
        raise typer.Exit(code=1)

    retriever = Retriever.from_json(chunks_path)

    recall, details = compute_recall_at_k(qa_pairs, retriever, k=5)
    hits = sum(d["hit"] for d in details)
    typer.echo(f"recall@5: {recall:.3f} ({hits}/{len(details)}) [threshold {recall_threshold}]")

    client = Anthropic()
    faithfulness, graded = compute_faithfulness(
        qa_pairs, retriever, client, faithfulness_sample_size, seed
    )
    typer.echo(f"faithfulness: {faithfulness:.3f} (sampled {graded} questions) [threshold {faithfulness_threshold}]")

    failures = []
    if recall < recall_threshold:
        failures.append(f"recall@5 {recall:.3f} < {recall_threshold}")
    if faithfulness < faithfulness_threshold:
        failures.append(f"faithfulness {faithfulness:.3f} < {faithfulness_threshold}")

    if failures:
        typer.echo("FAILED: " + "; ".join(failures))
        raise typer.Exit(code=1)

    typer.echo("PASSED")


if __name__ == "__main__":
    app()
