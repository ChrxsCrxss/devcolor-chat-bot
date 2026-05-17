"""Hardware-friendly model profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["light", "balanced", "quality"]


@dataclass(frozen=True)
class Profile:
    name: ProfileName
    embedder: str  # "tfidf" | "minilm"
    embed_model: str
    llm_model: str
    description: str


PROFILES: dict[ProfileName, Profile] = {
    "light": Profile(
        name="light",
        embedder="tfidf",
        embed_model="tfidf",
        llm_model="llama3.2:1b",
        description="Low RAM — TF-IDF retrieval + smallest Ollama model",
    ),
    "balanced": Profile(
        name="balanced",
        embedder="minilm",
        embed_model="all-MiniLM-L6-v2",
        llm_model="llama3.2:3b",
        description="Default — semantic MiniLM + llama3.2:3b",
    ),
    "quality": Profile(
        name="quality",
        embedder="minilm",
        embed_model="all-MiniLM-L6-v2",
        llm_model="phi3:mini",
        description="Better phrasing — MiniLM + phi3:mini",
    ),
}


def get_profile(name: str) -> Profile:
    key = name.lower().strip()
    if key not in PROFILES:
        valid = ", ".join(PROFILES)
        raise ValueError(f"Unknown profile '{name}'. Choose: {valid}")
    return PROFILES[key]  # type: ignore[index]
