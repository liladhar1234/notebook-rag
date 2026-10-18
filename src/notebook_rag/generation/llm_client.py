"""LLMClient protocol and concrete implementations.

Supports:
- OpenAI-compatible endpoints (Gemini, Groq, Ollama, local)
- Anthropic (Claude)
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_TRANSIENT_NAMES = {"APIConnectionError", "APITimeoutError"}  # same names in the openai and anthropic SDKs


def _is_transient(exc: BaseException) -> bool:
    """Retry rate limits, server errors and network problems; fail fast on bad keys or requests."""
    return getattr(exc, "status_code", None) in _TRANSIENT_STATUS or type(exc).__name__ in _TRANSIENT_NAMES


_retry_transient = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal protocol for text generation."""

    def complete(self, system: str, user: str) -> str:
        """Return a completion string given system and user prompts."""
        ...


class OpenAICompatClient:
    """Client for any OpenAI-compatible endpoint (Gemini, Groq, Ollama…)."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @_retry_transient
    def complete(self, system: str, user: str) -> str:
        log.debug("Calling OpenAI-compat model: %s", self._model)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        text = resp.choices[0].message.content
        if not text:
            raise RuntimeError("The LLM returned an empty response.")
        return text


class AnthropicClient:
    """Client for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-3-5-haiku-20241022",
        api_key: str = "",
        max_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> None:
        import anthropic

        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._client = anthropic.Anthropic(api_key=api_key)

    @_retry_transient
    def complete(self, system: str, user: str) -> str:
        log.debug("Calling Anthropic model: %s", self._model)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


def build_llm_client(
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    anthropic_api_key: str,
) -> LLMClient:
    """Factory: instantiate the correct LLM client from config."""
    if provider == "anthropic":
        return AnthropicClient(model=model, api_key=anthropic_api_key)
    # Default: OpenAI-compatible
    return OpenAICompatClient(model=model, api_key=api_key, base_url=base_url)
