"""RAG orchestration."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from devcolor_rag.config import SessionConfig
from devcolor_rag.formatting import plain_terminal_text
from devcolor_rag.index import VectorIndex, get_or_build_index
from devcolor_rag.llm import EchoLLM, LLM, OllamaLLM
from devcolor_rag.observability import build_turn_log, write_turn_log
from devcolor_rag.profiles import Profile, get_profile
from devcolor_rag.prompt import build_prompts
from devcolor_rag.retriever import RetrievalResult, retrieve


STRICT_LOW_CONFIDENCE_MSG = (
    "I don't have enough matching FAQ context to answer confidently. "
    "Try rephrasing your question, or visit https://devcolor.org to learn more."
)


@dataclass
class QueryResult:
    answer: str
    retrieval: RetrievalResult
    skipped_generation: bool
    skip_reason: str | None
    log_paths: tuple[Path, Path] | None


class RAGPipeline:
    def __init__(
        self,
        index: VectorIndex,
        llm: LLM,
        project_root: Path,
        *,
        use_echo_fallback: bool = True,
    ) -> None:
        self.index = index
        self.llm = llm
        self.project_root = project_root
        self.use_echo_fallback = use_echo_fallback
        self.session_id = uuid.uuid4().hex[:8]
        self.turn = 0
        self._last_retrieval: RetrievalResult | None = None

    @classmethod
    def from_profile(
        cls,
        corpus_path: Path,
        project_root: Path,
        profile: Profile,
        *,
        llm_model_override: str | None = None,
        force_reindex: bool = False,
        wipe: bool = False,
        use_echo: bool = False,
    ) -> RAGPipeline:
        index = get_or_build_index(
            corpus_path, profile, project_root, force_reindex=force_reindex, wipe=wipe
        )
        model = llm_model_override or profile.llm_model
        if use_echo:
            llm: LLM = EchoLLM(model)
        else:
            llm = OllamaLLM(model)
        return cls(index, llm, project_root, use_echo_fallback=not use_echo)

    def reload_index(self, profile: Profile, *, wipe: bool = False) -> None:
        corpus = self.project_root / "data" / "devcolorfaq.txt"
        self.index = get_or_build_index(
            corpus, profile, self.project_root, force_reindex=True, wipe=wipe
        )

    def update_llm(self, profile: Profile, override: str | None = None) -> None:
        model = override or profile.llm_model
        if isinstance(self.llm, EchoLLM):
            self.llm = EchoLLM(model)
        else:
            self.llm = OllamaLLM(model)

    @property
    def last_retrieval(self) -> RetrievalResult | None:
        return self._last_retrieval

    def query(self, text: str, config: SessionConfig) -> QueryResult:
        self.turn += 1
        retrieval = retrieve(
            self.index,
            text,
            top_k=config.top_k,
            max_sources=config.max_sources,
        )
        self._last_retrieval = retrieval

        skipped = False
        skip_reason: str | None = None
        response = ""
        system_prompt = ""
        user_prompt = text
        start = time.perf_counter()

        if config.strict and not retrieval.passes_strict:
            skipped = True
            skip_reason = "strict_confidence_gate"
            response = STRICT_LOW_CONFIDENCE_MSG
        elif not retrieval.parent_entries:
            skipped = True
            skip_reason = "no_entries"
            response = STRICT_LOW_CONFIDENCE_MSG
        else:
            system_prompt, user_prompt = build_prompts(text, retrieval.parent_entries)
            try:
                response = self.llm.generate(system_prompt, user_prompt)
            except RuntimeError:
                if self.use_echo_fallback:
                    echo = EchoLLM("echo-fallback")
                    response = echo.generate(system_prompt, user_prompt)
                    skip_reason = "ollama_fallback_echo"
                else:
                    raise

        response = plain_terminal_text(response)

        latency_ms = (time.perf_counter() - start) * 1000
        profile = get_profile(config.profile)
        turn_log = build_turn_log(
            session_id=self.session_id,
            turn=self.turn,
            config=config,
            embed_model=profile.embed_model,
            llm_model=self.llm.model_name,
            query=text,
            retrieval=retrieval,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response=response,
            latency_ms=latency_ms,
            skipped_generation=skipped,
            skip_reason=skip_reason,
        )
        log_paths = write_turn_log(self.project_root, turn_log)

        return QueryResult(
            answer=response,
            retrieval=retrieval,
            skipped_generation=skipped,
            skip_reason=skip_reason,
            log_paths=log_paths,
        )
