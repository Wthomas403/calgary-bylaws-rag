"""Hybrid retrieval over pre-embedded bylaw chunks.

Combines BM25 keyword search (good for exact terms, bylaw numbers,
legal phrasing) with dense embedding search (good for paraphrases and
semantic matches) into a single ranked list.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

EmbedFn = Callable[[str], np.ndarray]


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class SearchResult:
    bylaw_name: str
    section: str | None
    page: int
    text: str
    score: float
    bm25_score: float
    dense_score: float

    def to_dict(self) -> dict:
        return {
            "bylaw_name": self.bylaw_name,
            "section": self.section,
            "page": self.page,
            "text": self.text,
            "score": self.score,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
        }


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    low, high = float(scores.min()), float(scores.max())
    if high - low < 1e-12:
        return np.zeros_like(scores)
    return (scores - low) / (high - low)


def _default_embed_fn() -> EmbedFn:
    model = SentenceTransformer(MODEL_NAME)

    def embed(text: str) -> np.ndarray:
        return model.encode([text], normalize_embeddings=True)[0]

    return embed


class Retriever:
    """Hybrid BM25 + dense embedding search over a fixed set of chunks.

    Chunk embeddings are expected to already exist (e.g. from
    ``scripts/ingest.py``) and to be normalized, so dense similarity is
    a plain dot product against the query embedding.
    """

    def __init__(
        self,
        chunks: list[dict],
        embed_fn: EmbedFn | None = None,
        # Measured against the synthetic eval set: residents paraphrase in
        # plain language that shares few keywords with legal bylaw text, so
        # dense search alone outperforms a 50/50 split (0.750 vs 0.690
        # recall@5). A small BM25 weight is kept for exact-term queries
        # (bylaw numbers, specific legal terms) rather than dropping it to 0.
        bm25_weight: float = 0.2,
    ):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        if not 0.0 <= bm25_weight <= 1.0:
            raise ValueError("bm25_weight must be between 0 and 1")

        self.chunks = chunks
        self.bm25_weight = bm25_weight
        self.dense_weight = 1.0 - bm25_weight

        self._bm25 = BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])
        self._embeddings = np.array([chunk["embedding"] for chunk in chunks], dtype=np.float32)
        self._embed_fn = embed_fn or _default_embed_fn()

    @classmethod
    def from_json(cls, path: Path, **kwargs) -> "Retriever":
        chunks = json.loads(Path(path).read_text())
        return cls(chunks, **kwargs)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        bm25_scores = np.array(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

        query_embedding = np.asarray(self._embed_fn(query), dtype=np.float32)
        dense_scores = self._embeddings @ query_embedding

        combined = (
            self.bm25_weight * _min_max_normalize(bm25_scores)
            + self.dense_weight * _min_max_normalize(dense_scores)
        )

        top_indices = np.argsort(-combined)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append(
                SearchResult(
                    bylaw_name=chunk["bylaw_name"],
                    section=chunk.get("section"),
                    page=chunk["page"],
                    text=chunk["text"],
                    score=float(combined[idx]),
                    bm25_score=float(bm25_scores[idx]),
                    dense_score=float(dense_scores[idx]),
                )
            )
        return results
