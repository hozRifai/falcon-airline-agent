"""Model factory. The model is a config switch; nothing else in the runtime knows which provider is behind it.

- openrouter: any model on OpenRouter through its OpenAI-compatible API (the default; one key, many models,
  so the cost/accuracy experiment is a one-line change of LLM_MODEL).
- anthropic: direct Anthropic API.
- fake: a scripted stand-in used by the tests and by the demo when no key is configured.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from falcon.core.config import settings


def make_chat_model(model: str | None = None, *, temperature: float = 0.0) -> BaseChatModel:
    provider = settings.llm_provider
    name = model or settings.llm_model
    if provider == "fake" or (provider == "openrouter" and not settings.openrouter_api_key) or (provider == "anthropic" and not settings.anthropic_api_key):
        from .fake_llm import RuleBasedFakeModel

        return RuleBasedFakeModel()
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=name,
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            temperature=temperature,
            # A reply never needs more. Without a cap, a degenerate generation (most likely DeepSeek repeating tool calls)
            # can run to the 60 s timeout on each of three attempts: two eval turns took ~220 s (FAILURE_DIARY #24).
            max_tokens=1500,
            timeout=60,
            max_retries=2,
            default_headers={"HTTP-Referer": settings.runtime_public_url, "X-Title": "Falcon Airline agent"},
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=name, api_key=settings.anthropic_api_key, temperature=temperature, max_tokens=1500, timeout=60, max_retries=2)
    raise ValueError(f"unknown LLM_PROVIDER {provider}")


def model_label() -> str:
    if settings.llm_provider == "fake" or (settings.llm_provider == "openrouter" and not settings.openrouter_api_key):
        return "fake (no API key configured)"
    return f"{settings.llm_provider}:{settings.llm_model}"
