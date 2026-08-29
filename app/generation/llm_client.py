"""Generation backends behind one interface (4.4 local-first requirement).

`OllamaClient` talks to a local Ollama server — no API key, no external
network call. `MockClient` is a deterministic, offline stand-in used by
default in tests/CI (`DOCS_ASSISTANT_LLM_PROVIDER=mock`) so ingestion,
retrieval, and automated tests never require a running model.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from app.generation.context_builder import ContextBlock


class LLMUnavailableError(RuntimeError):
    """Raised when the configured generation backend cannot produce an answer."""


class LLMClient(Protocol):
    def generate(
        self, system_prompt: str, user_prompt: str, context_blocks: list[ContextBlock] | None = None
    ) -> str: ...

    def health_check(self) -> tuple[bool, str]: ...


class OllamaClient:
    def __init__(self, host: str, model: str, timeout_seconds: float) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    def generate(
        self, system_prompt: str, user_prompt: str, context_blocks: list[ContextBlock] | None = None
    ) -> str:
        try:
            response = httpx.post(
                f"{self._host}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"]
        except httpx.HTTPError as exc:
            raise LLMUnavailableError(f"Ollama request failed ({self._host}, model={self._model}): {exc}") from exc
        except (KeyError, ValueError) as exc:
            raise LLMUnavailableError(f"Unexpected Ollama response shape: {exc}") from exc

    def health_check(self) -> tuple[bool, str]:
        try:
            response = httpx.get(f"{self._host}/api/tags", timeout=5.0)
            response.raise_for_status()
            return True, "ollama reachable"
        except httpx.HTTPError as exc:
            return False, f"ollama unreachable: {exc}"


class MockClient:
    """Deterministic offline stand-in. Synthesizes an answer directly from the
    structured context blocks (not by parsing the prompt string) so its
    behavior is stable regardless of prompt wording — used to exercise the
    citation-validation and abstention plumbing in tests without a model.
    """

    def generate(
        self, system_prompt: str, user_prompt: str, context_blocks: list[ContextBlock] | None = None
    ) -> str:
        blocks = context_blocks or []
        if not blocks:
            return "The available evidence does not answer this question."
        sentences = []
        for block in blocks[:3]:
            first_sentence = block.text.strip().split(". ")[0].rstrip(".")
            sentences.append(f"{first_sentence} [{block.tag}].")
        return " ".join(sentences)

    def health_check(self) -> tuple[bool, str]:
        return True, "mock provider always available"


def build_llm_client(provider: str, ollama_host: str, ollama_model: str, timeout_seconds: float) -> LLMClient:
    if provider == "mock":
        return MockClient()
    if provider == "ollama":
        return OllamaClient(ollama_host, ollama_model, timeout_seconds)
    raise ValueError(f"Unknown LLM provider '{provider}' (expected 'ollama' or 'mock')")
