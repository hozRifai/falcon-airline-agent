"""The chat model. OpenRouter (OpenAI-compatible, any model id) when a key is configured; otherwise the scripted stand-in
that follows keyword rules through the real tools, so every flow can be seen without spending a token."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from falcon.core.config import settings


def use_fake_model() -> bool:
    return settings.llm_provider == "fake" or not settings.openrouter_api_key


def make_chat_model(model: str | None = None, *, timeout: float = 60, max_retries: int = 2) -> BaseChatModel:
    if use_fake_model():
        from .fake_llm import RuleBasedFakeModel

        return RuleBasedFakeModel()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or settings.llm_model,
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        temperature=0.0,
        # A reply never needs more. Without a cap a degenerate generation ran to the 60 s timeout on each of three attempts.
        max_tokens=1500,
        timeout=timeout,
        max_retries=max_retries,
        default_headers={"HTTP-Referer": settings.runtime_public_url, "X-Title": "Falcon Airline agent"},
    )


def model_label() -> str:
    return "fake (no API key configured)" if use_fake_model() else f"openrouter:{settings.llm_model}"
