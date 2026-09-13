"""Chunk all bylaw PDFs, embed them, and save chunks + embeddings to JSON.

Usage:
    python -m scripts.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_all

MODEL_NAME = "all-MiniLM-L6-v2"

app = typer.Typer()


@app.command()
def ingest(
    bylaws_dir: Path = typer.Option(Path("data/bylaws"), help="Directory of source PDFs."),
    output_path: Path = typer.Option(Path("data/chunks.json"), help="Where to write chunks + embeddings."),
) -> None:
    chunks = chunk_all(bylaws_dir)
    if not chunks:
        raise typer.BadParameter(f"No chunks produced from PDFs in {bylaws_dir}")

    typer.echo(f"Chunked {len(chunks)} sections from {bylaws_dir}")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    records = [
        {**chunk.to_dict(), "embedding": embedding.tolist()}
        for chunk, embedding in zip(chunks, embeddings)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    typer.echo(f"Wrote {len(records)} chunks with embeddings to {output_path}")


if __name__ == "__main__":
    app()
