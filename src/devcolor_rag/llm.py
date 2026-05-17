"""LLM backends."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod

import requests

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class LLM(ABC):
    model_name: str

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        ...


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str,
        host: str | None = None,
        timeout: int = 120,
    ) -> None:
        self.model_name = model
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip(
            "/"
        )
        self.timeout = timeout

    def generate(self, system: str, user: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. "
                f"Start Ollama and run: ollama pull {self.model_name}"
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"Ollama error ({resp.status_code}): {resp.text[:200]}. "
                f"Try: ollama pull {self.model_name}"
            ) from exc

        data = resp.json()
        return data.get("message", {}).get("content", "").strip()


def _first_sentences(text: str, count: int = 2) -> str:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return " ".join(parts[:count]).strip()


class EchoLLM(LLM):
    """Offline stub — formatted summary when Ollama is unavailable."""

    def __init__(self, model: str = "echo") -> None:
        self.model_name = model

    def generate(self, system: str, user: str) -> str:
        context = system
        if "Context:\n" in system:
            context = system.split("Context:\n", 1)[1].strip()

        if not context or context == "(no context retrieved)":
            return (
                "✨ I don't have enough FAQ context for that one.\n\n"
                "Try rephrasing, or visit https://devcolor.org to learn more about /dev/color. 🙌"
            )

        blocks = [b.strip() for b in context.split("\n---\n") if b.strip()]
        snippets: list[str] = []
        for block in blocks[:2]:
            if block.startswith("Q:"):
                _, _, answer = block.partition("\nA:")
                snippet = _first_sentences(answer.strip(), count=3)
                if snippet:
                    snippets.append(snippet)

        body = "\n\n".join(snippets) if snippets else "Please see devcolor.org for more."

        return (
            "💡 Demo mode — start Ollama for fuller, emoji-rich answers.\n\n"
            "✨ Here's what the FAQ says:\n\n"
            f"{body}\n\n"
            "🙌 You've got this — explore more at https://devcolor.org"
        )


def check_ollama(host: str | None = None) -> tuple[bool, list[str]]:
    base = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    try:
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        return True, models
    except (requests.RequestException, KeyError):
        return False, []
