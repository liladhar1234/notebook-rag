"""Optional OpenAI-compatible dense embedder.

Drop-in replacement for DenseEmbedder when you need a hosted model.
"""
from __future__ import annotations

from openai import OpenAI


class OpenAIDenseEmbedder:
    """Dense embedder via OpenAI (or compatible) API."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]
