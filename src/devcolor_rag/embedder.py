"""Embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class Embedder(ABC):
    name: str

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return (n, dim) float32 matrix."""

    @abstractmethod
    def encode_query(self, text: str) -> np.ndarray:
        """Return (dim,) float32 vector."""


class MiniLMEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self._model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(vecs.astype(np.float32))

    def encode_query(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class TfidfEmbedder(Embedder):
    def __init__(self) -> None:
        self.name = "tfidf"
        self._vectorizer: TfidfVectorizer | None = None
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
        self._vectorizer.fit(texts)
        self._fitted = True

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted or self._vectorizer is None:
            self.fit(texts)
        matrix = self._vectorizer.transform(texts)
        dense = matrix.toarray().astype(np.float32)
        return _l2_normalize(dense)

    def encode_query(self, text: str) -> np.ndarray:
        if not self._fitted or self._vectorizer is None:
            raise RuntimeError("TfidfEmbedder must be fit before encoding queries")
        vec = self._vectorizer.transform([text]).toarray().astype(np.float32)
        return _l2_normalize(vec)[0]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return normalize(matrix, norm="l2", axis=1).astype(np.float32)


def create_embedder(embedder_type: str, embed_model: str) -> Embedder:
    if embedder_type == "tfidf":
        return TfidfEmbedder()
    return MiniLMEmbedder(embed_model)
