"""Build and cache embedding index."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from devcolor_rag.corpus import Chunk, FAQEntry, build_chunks, load_faq
from devcolor_rag.embedder import Embedder, TfidfEmbedder, create_embedder
from devcolor_rag.profiles import Profile

CHUNK_SCHEMA_VERSION = "1"
CACHE_DIR_NAME = ".cache/index"


@dataclass
class IndexManifest:
    corpus_hash: str
    embedder_type: str
    embed_model: str
    chunk_schema_version: str
    profile: str
    num_chunks: int
    num_entries: int
    created_at: str


@dataclass
class VectorIndex:
    manifest: IndexManifest
    entries: list[FAQEntry]
    chunks: list[Chunk]
    embeddings: np.ndarray
    embedder: Embedder

    @property
    def entries_by_id(self) -> dict[int, FAQEntry]:
        return {e.id: e for e in self.entries}


def corpus_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def cache_dir(project_root: Path) -> Path:
    return project_root / CACHE_DIR_NAME


def build_index(
    corpus_path: Path,
    profile: Profile,
    project_root: Path,
    *,
    wipe: bool = False,
) -> VectorIndex:
    cdir = cache_dir(project_root)
    if wipe and cdir.exists():
        shutil.rmtree(cdir)

    c_hash = corpus_hash(corpus_path)
    embedder = create_embedder(profile.embedder, profile.embed_model)
    entries = load_faq(corpus_path)
    chunks = build_chunks(entries)
    texts = [c.embed_text for c in chunks]

    if isinstance(embedder, TfidfEmbedder):
        embedder.fit(texts)
    embeddings = embedder.encode(texts)

    manifest = IndexManifest(
        corpus_hash=c_hash,
        embedder_type=profile.embedder,
        embed_model=profile.embed_model,
        chunk_schema_version=CHUNK_SCHEMA_VERSION,
        profile=profile.name,
        num_chunks=len(chunks),
        num_entries=len(entries),
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    save_index(cdir, manifest, chunks, embeddings)
    return VectorIndex(
        manifest=manifest,
        entries=entries,
        chunks=chunks,
        embeddings=embeddings,
        embedder=embedder,
    )


def save_index(
    cdir: Path,
    manifest: IndexManifest,
    chunks: list[Chunk],
    embeddings: np.ndarray,
) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2), encoding="utf-8"
    )
    np.save(cdir / "embeddings.npy", embeddings)
    chunk_data = [
        {
            "chunk_id": c.chunk_id,
            "parent_id": c.parent_id,
            "sentence_index": c.sentence_index,
            "text": c.text,
            "parent_question": c.parent_question,
        }
        for c in chunks
    ]
    (cdir / "chunks.json").write_text(json.dumps(chunk_data, indent=2), encoding="utf-8")


def manifest_matches(
    manifest: IndexManifest,
    corpus_path: Path,
    profile: Profile,
) -> bool:
    return (
        manifest.corpus_hash == corpus_hash(corpus_path)
        and manifest.embedder_type == profile.embedder
        and manifest.embed_model == profile.embed_model
        and manifest.chunk_schema_version == CHUNK_SCHEMA_VERSION
    )


def load_index(
    corpus_path: Path,
    profile: Profile,
    project_root: Path,
) -> VectorIndex | None:
    cdir = cache_dir(project_root)
    manifest_path = cdir / "manifest.json"
    if not manifest_path.exists():
        return None

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = IndexManifest(**data)
    if not manifest_matches(manifest, corpus_path, profile):
        return None

    embeddings = np.load(cdir / "embeddings.npy")
    chunk_data = json.loads((cdir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in chunk_data]
    embedder = create_embedder(profile.embedder, profile.embed_model)
    if isinstance(embedder, TfidfEmbedder):
        embedder.fit([c.embed_text for c in chunks])

    entries = load_faq(corpus_path)
    return VectorIndex(
        manifest=manifest,
        entries=entries,
        chunks=chunks,
        embeddings=embeddings,
        embedder=embedder,
    )


def get_or_build_index(
    corpus_path: Path,
    profile: Profile,
    project_root: Path,
    *,
    force_reindex: bool = False,
    wipe: bool = False,
) -> VectorIndex:
    if not force_reindex and not wipe:
        cached = load_index(corpus_path, profile, project_root)
        if cached is not None:
            return cached
    return build_index(corpus_path, profile, project_root, wipe=wipe)


def check_cache(
    corpus_path: Path,
    profile: Profile,
    project_root: Path,
) -> str:
    cdir = cache_dir(project_root)
    manifest_path = cdir / "manifest.json"
    if not manifest_path.exists():
        return "missing"
    manifest = IndexManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
    if manifest_matches(manifest, corpus_path, profile):
        return "valid"
    return "stale"
