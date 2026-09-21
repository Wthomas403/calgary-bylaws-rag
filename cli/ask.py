"""Ask a natural-language question about Calgary bylaws and get a
natural-language answer, grounded in retrieved bylaw text.

Usage:
    python -m cli.ask "can I keep chickens in my backyard"

Requires ANTHROPIC_API_KEY in the environment (or a .env file).
"""

from pathlib import Path

import typer
from anthropic import Anthropic
from dotenv import load_dotenv

from src.generator import generate_answer
from src.retriever import Retriever

load_dotenv()

app = typer.Typer()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to ask."),
    chunks_path: Path = typer.Option(Path("data/chunks.json"), help="Chunks + embeddings produced by scripts/ingest.py."),
    top_k: int = typer.Option(5, help="Number of passages to retrieve as context for the answer."),
) -> None:
    if not chunks_path.exists():
        typer.echo(
            f"{chunks_path} not found. Add bylaw PDFs to data/bylaws/ and run "
            f"'python -m scripts.ingest' first."
        )
        raise typer.Exit(code=1)

    retriever = Retriever.from_json(chunks_path)
    results = retriever.search(question, top_k=top_k)

    client = Anthropic()
    answer = generate_answer(client, question, results)

    typer.echo(answer)

    if results:
        typer.echo("")
        typer.echo("Sources:")
        for r in results:
            section = f"Section {r.section}" if r.section else "Preamble"
            typer.echo(f"  - {r.bylaw_name}, {section}, page {r.page}")


if __name__ == "__main__":
    app()
