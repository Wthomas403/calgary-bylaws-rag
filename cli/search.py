"""Search Calgary bylaws from the command line.

Usage:
    python -m cli.search "can I keep chickens in my backyard"
"""

from pathlib import Path

import typer

from src.retriever import Retriever, SearchResult

app = typer.Typer()


def format_result(rank: int, result: SearchResult) -> str:
    section = f"Section {result.section}" if result.section else "Preamble"
    snippet = " ".join(result.text.split())
    if len(snippet) > 280:
        snippet = snippet[:280].rsplit(" ", 1)[0] + "..."
    return (
        f"{rank}. {result.bylaw_name} -- {section}, page {result.page} "
        f"(score {result.score:.3f})\n"
        f"   {snippet}"
    )


@app.command()
def search(
    question: str = typer.Argument(..., help="The question to search bylaws for."),
    chunks_path: Path = typer.Option(Path("data/chunks.json"), help="Chunks + embeddings produced by scripts/ingest.py."),
    top_k: int = typer.Option(3, help="Number of results to show."),
) -> None:
    if not chunks_path.exists():
        typer.echo(
            f"{chunks_path} not found. Add bylaw PDFs to data/bylaws/ and run "
            f"'python -m scripts.ingest' first."
        )
        raise typer.Exit(code=1)

    retriever = Retriever.from_json(chunks_path)
    results = retriever.search(question, top_k=top_k)

    if not results:
        typer.echo("No results found.")
        raise typer.Exit(code=1)

    for rank, result in enumerate(results, start=1):
        typer.echo(format_result(rank, result))
        typer.echo("")


if __name__ == "__main__":
    app()
