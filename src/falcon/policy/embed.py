"""Embedding client behind one small interface, so retrieval does not care who computes vectors.

- openrouter: OpenAI-compatible `/embeddings` endpoint on OpenRouter (EMBEDDING_MODEL is an id listed at
  https://openrouter.ai/models?output_modalities=embeddings; default baai/bge-m3, multilingual, 1024 dims).
- none: no embeddings; `search_policy` reports itself unavailable and the agent offers a human (no lexical fallback).
"""

from __future__ import annotations

from falcon.core.config import settings


class Embedder:
    def __init__(self):
        self.provider = settings.embedding_provider
        self.model = settings.embedding_model
        self._client = None

    @property
    def enabled(self) -> bool:
        if self.provider == "openrouter":
            return bool(settings.openrouter_api_key)
        return False

    def _get_client(self):
        if self._client is None and self.provider == "openrouter":
            from openai import OpenAI

            # a search tool cannot wait the client's default 10 minutes: one retry, 20 s, then search reports itself unavailable
            self._client = OpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key, timeout=20, max_retries=1,
                                  default_headers={"HTTP-Referer": settings.runtime_public_url, "X-Title": "Falcon Airline agent"})
        return self._client

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self.enabled:
            raise RuntimeError("No embedding provider configured (EMBEDDING_PROVIDER / keys).")
        client = self._get_client()
        out: list[list[float]] = []
        for i in range(0, len(texts), 64):
            batch = [t[:8000] for t in texts[i:i + 64]]
            resp = client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in sorted(resp.data, key=lambda d: d.index)])
        return out
