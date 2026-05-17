"""Cosine retrieval with parent FAQ deduplication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from devcolor_rag.corpus import FAQEntry
from devcolor_rag.index import VectorIndex

STRICT_CONFIDENCE_THRESHOLD = 0.35


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    chunk_id: str
    parent_id: int
    score: float
    snippet: str
    parent_question: str


@dataclass(frozen=True)
class RetrievalResult:
    chunks: list[RetrievedChunk]
    parent_entries: list[FAQEntry]
    best_score: float
    passes_strict: bool


def cosine_scores(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)
    return (matrix @ query_vec.T).flatten()


def retrieve(
    index: VectorIndex,
    query: str,
    *,
    top_k: int = 5,
    max_sources: int = 3,
) -> RetrievalResult:
    query_vec = index.embedder.encode_query(query)
    scores = cosine_scores(query_vec, index.embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    retrieved_chunks: list[RetrievedChunk] = []
    seen_parents: list[int] = []
    parent_entries: list[FAQEntry] = []
    lookup = index.entries_by_id

    for rank, idx in enumerate(top_indices, start=1):
        chunk = index.chunks[int(idx)]
        score = float(scores[int(idx)])
        retrieved_chunks.append(
            RetrievedChunk(
                rank=rank,
                chunk_id=chunk.chunk_id,
                parent_id=chunk.parent_id,
                score=score,
                snippet=chunk.text[:120] + ("…" if len(chunk.text) > 120 else ""),
                parent_question=chunk.parent_question,
            )
        )
        if chunk.parent_id not in seen_parents:
            seen_parents.append(chunk.parent_id)
            entry = lookup.get(chunk.parent_id)
            if entry:
                parent_entries.append(entry)
        if len(parent_entries) >= max_sources:
            break

    best_score = float(scores[top_indices[0]]) if len(top_indices) else 0.0
    return RetrievalResult(
        chunks=retrieved_chunks,
        parent_entries=parent_entries,
        best_score=best_score,
        passes_strict=best_score >= STRICT_CONFIDENCE_THRESHOLD,
    )
