"""Human-readable and JSON observability logs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from devcolor_rag.config import SessionConfig
from devcolor_rag.prompt import PROMPT_VERSION
from devcolor_rag.retriever import RetrievalResult


@dataclass
class TurnLog:
    session_id: str
    turn: int
    timestamp: str
    prompt_version: str
    profile: str
    embed_model: str
    llm_model: str
    query: str
    config: dict
    retrieval: dict
    system_prompt: str
    user_prompt: str
    response: str
    latency_ms: float
    skipped_generation: bool
    skip_reason: str | None = None


def write_turn_log(
    project_root: Path,
    turn_log: TurnLog,
) -> tuple[Path, Path]:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = turn_log.timestamp.replace(":", "-").replace(".", "-")
    base = f"rag_{ts}_t{turn_log.turn}"

    log_path = logs_dir / f"{base}.log"
    json_path = logs_dir / f"{base}.json"

    json_path.write_text(json.dumps(asdict(turn_log), indent=2), encoding="utf-8")
    log_path.write_text(_format_human_log(turn_log), encoding="utf-8")
    return log_path, json_path


def _format_human_log(t: TurnLog) -> str:
    lines = [
        "=" * 72,
        "DEVCOLOR RAG — TURN LOG",
        "=" * 72,
        "",
        f"Timestamp:      {t.timestamp}",
        f"Session:        {t.session_id}",
        f"Turn:           {t.turn}",
        "",
        "-" * 72,
        "CHECKPOINTS",
        "-" * 72,
        f"Prompt version: {t.prompt_version}",
        f"Profile:        {t.profile}",
        f"Embed model:    {t.embed_model}",
        f"LLM model:      {t.llm_model}",
        "",
        "-" * 72,
        "SESSION CONFIG (effective this turn)",
        "-" * 72,
    ]
    for key, val in t.config.items():
        lines.append(f"  {key:16} {val}")
    lines.extend(
        [
            "",
            "-" * 72,
            "USER QUERY",
            "-" * 72,
            t.query,
            "",
            "-" * 72,
            "RETRIEVAL",
            "-" * 72,
            f"Best score:     {t.retrieval.get('best_score', 0):.4f}",
            f"Strict pass:    {t.retrieval.get('passes_strict')}",
            f"Parent FAQ ids: {t.retrieval.get('parent_ids', [])}",
            "",
        ]
    )
    for chunk in t.retrieval.get("chunks", []):
        lines.append(
            f"  #{chunk['rank']}  score={chunk['score']:.4f}  "
            f"faq={chunk['parent_id']}  {chunk['snippet'][:80]}"
        )
    lines.extend(
        [
            "",
            "-" * 72,
            "PROMPT (system)",
            "-" * 72,
            t.system_prompt[:4000],
            "",
            "-" * 72,
            "PROMPT (user)",
            "-" * 72,
            t.user_prompt,
            "",
            "-" * 72,
            "RESPONSE",
            "-" * 72,
            f"Skipped LLM:    {t.skipped_generation}",
        ]
    )
    if t.skip_reason:
        lines.append(f"Skip reason:    {t.skip_reason}")
    lines.extend(
        [
            f"Latency (ms):   {t.latency_ms:.1f}",
            "",
            t.response,
            "",
            "=" * 72,
        ]
    )
    return "\n".join(lines)


def build_turn_log(
    *,
    session_id: str,
    turn: int,
    config: SessionConfig,
    embed_model: str,
    llm_model: str,
    query: str,
    retrieval: RetrievalResult,
    system_prompt: str,
    user_prompt: str,
    response: str,
    latency_ms: float,
    skipped_generation: bool,
    skip_reason: str | None = None,
) -> TurnLog:
    return TurnLog(
        session_id=session_id,
        turn=turn,
        timestamp=datetime.now(timezone.utc).isoformat(),
        prompt_version=PROMPT_VERSION,
        profile=config.profile,
        embed_model=embed_model,
        llm_model=llm_model,
        query=query,
        config={
            "top_k": config.top_k,
            "max_sources": config.max_sources,
            "strict": config.strict,
            "sources": config.sources,
            "debug": config.debug,
            "profile": config.profile,
        },
        retrieval={
            "best_score": retrieval.best_score,
            "passes_strict": retrieval.passes_strict,
            "parent_ids": [e.id for e in retrieval.parent_entries],
            "chunks": [
                {
                    "rank": c.rank,
                    "chunk_id": c.chunk_id,
                    "parent_id": c.parent_id,
                    "score": c.score,
                    "snippet": c.snippet,
                }
                for c in retrieval.chunks
            ],
        },
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response=response,
        latency_ms=latency_ms,
        skipped_generation=skipped_generation,
        skip_reason=skip_reason,
    )
