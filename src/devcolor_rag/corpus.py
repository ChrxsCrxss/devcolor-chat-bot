"""Parse FAQ corpus into parent entries and sentence chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FAQEntry:
    id: int
    question: str
    answer: str

    @property
    def text(self) -> str:
        return f"Q: {self.question}\nA: {self.answer}"


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    parent_id: int
    sentence_index: int
    text: str
    parent_question: str

    @property
    def embed_text(self) -> str:
        return f"Q: {self.parent_question}\n{self.text}"


_ENTRY_RE = re.compile(
    r"^\s*(\d+)\.\s+\*\*(.+?)\*\*\s*\n\s*(.+?)(?=\n\s*\d+\.\s+\*\*|\Z)",
    re.MULTILINE | re.DOTALL,
)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def load_faq(path: Path) -> list[FAQEntry]:
    raw = path.read_text(encoding="utf-8")
    entries: list[FAQEntry] = []
    for match in _ENTRY_RE.finditer(raw):
        num = int(match.group(1))
        question = match.group(2).strip()
        answer = match.group(3).strip()
        entries.append(FAQEntry(id=num, question=question, answer=answer))
    if not entries:
        raise ValueError(f"No FAQ entries parsed from {path}")
    return entries


def build_chunks(entries: list[FAQEntry]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for entry in entries:
        sentences = _split_sentences(entry.answer)
        if not sentences:
            sentences = [entry.answer]
        for idx, sentence in enumerate(sentences):
            chunks.append(
                Chunk(
                    chunk_id=f"{entry.id}-{idx}",
                    parent_id=entry.id,
                    sentence_index=idx,
                    text=sentence,
                    parent_question=entry.question,
                )
            )
    return chunks


def entries_by_id(entries: list[FAQEntry]) -> dict[int, FAQEntry]:
    return {e.id: e for e in entries}
