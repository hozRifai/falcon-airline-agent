"""Central configuration. Every service imports `settings` from here.

Why one settings object: the panel will ask "how do the pieces find each other and how is
anything secret handled?" The answer is: one `.env` locally (a secret store in production), and no service ever reads a
credential from anywhere else.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(os.environ.get("FALCON_ROOT") or Path(__file__).resolve().parents[3])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    # Clock
    falcon_now: str = "2026-10-20T09:00:00+04:00"

    # Postgres
    pghost: str = "127.0.0.1"
    pgport: int = 5433
    pgdatabase: str = "falcon"
    pgadmin_user: str = "postgres"
    pgadmin_password: str = "falcon_super"
    pg_res_password: str = "res_rw_pw"
    pg_opsapi_password: str = "ops_api_pw"
    pg_policy_password: str = "policy_ro_pw"
    pg_agent_password: str = "agent_rw_pw"
    pg_audit_password: str = "audit_w_pw"

    # Auth
    auth_mode: str = "dev"  # dev | entra
    falcon_dev_secret: str = "change-me-dev-secret-32-chars-minimum"
    entra_tenant_id: str = ""
    entra_runtime_client_id: str = ""
    entra_aud_reservations: str = "api://falcon-reservations"
    entra_aud_flightops: str = "api://falcon-flightops"
    entra_aud_policy: str = "api://falcon-policy"

    # Services
    legacy_api_url: str = "http://127.0.0.1:8010"
    legacy_api_key: str = "legacy-dev-key"
    mcp_reservations_url: str = "http://127.0.0.1:8001/mcp"
    mcp_flightops_url: str = "http://127.0.0.1:8002/mcp"
    mcp_policy_url: str = "http://127.0.0.1:8003/mcp"
    runtime_public_url: str = "http://127.0.0.1:8000"

    # Models
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_provider: str = "openrouter"  # openrouter | anthropic | fake
    llm_model: str = "deepseek/deepseek-v4.1-flash-1017"  # DeepSeek V4.1 Flash (1017) on OpenRouter: tool calling, 1M context; Hozayfa's choice 19 Oct (decision #44)
    llm_model_cheap: str = "deepseek/deepseek-v4.1-flash-1017"
    query_rewrite_model: str = ""  # policy query rewriting; empty = llm_model_cheap
    policy_query_rewrite: str = "auto"  # off | auto | always: LLM query rewriting + expansion; auto = only when the first pass is low-confidence (decision #31)
    policy_rewrite_threshold: float = 0.6
    policy_query_aliases: bool = True  # customer words -> contract words appended before embedding (src/falcon/policy/aliases.py, decision #39)  # cosine similarity of the best first-pass hit below which "auto" rewrites (chosen on the gold set, docs/EVALUATION.md)
    llm_price_in_per_m: float = 0.15  # USD per million input tokens, for the cost metrics (DeepSeek V4.1 Flash 1017 on OpenRouter, 19 Oct 2026)
    llm_price_out_per_m: float = 0.60  # USD per million output tokens
    simulator_model: str = "deepseek/deepseek-chat"  # the simulated customer stays on V3: a different model from the agent, and a constant across the runs
    embedding_provider: str = "openrouter"  # openrouter | none
    embedding_model: str = "baai/bge-m3"  # multilingual (Arabic + English in one space), 1024 dims, 8k context
    anthropic_api_key: str = ""

    # Demo protection
    demo_access_code: str = ""   # empty: no access code asked (the demo picks a customer from a list); set one to protect a public deployment
    rate_limit_per_minute: int = 30
    max_turns_per_session: int = 40
    max_tool_calls_per_turn: int = 8
    # Reviewer loop (decision #26): every draft reply is checked against the conversation's tool trace before the customer
    # sees it; a failing draft goes back to the model with the exact problem, up to review_max_rounds times, then a safe
    # "could not resolve" message is shown. The deterministic checks always run; review_llm adds a model reviewer (one more
    # model call per reply) when true.
    review_max_rounds: int = 3
    # Decision #43: after this many unresolved replies in a row (held by the reviewer or cut off by the tool cap) the runtime logs a
    # handover case itself and tells the customer the case number; 0 turns the rule off.
    escalate_after_unresolved: int = 2
    # Decision #46: the guard's classifier names the topic of each message; the agent gets that topic's prompt section and tools.
    # Off, or no topic (opening message, classifier off or unavailable, scripted model): the whole prompt and every tool.
    topic_routing: bool = True
    review_llm: bool = False
    review_model: str = "deepseek/deepseek-v4.1-flash-1017"
    # Decision #37: every clause a policy answer cites is checked against the clause text by the review model (one call per reply
    # that cites, ~2 s); a sentence the text does not support sends the draft back. Deterministic checks stay the floor.
    review_citations: bool = True
    review_citation_timeout_s: float = 25.0
    # Generation evaluation judge (decision #35): a different model family from the agent, so DeepSeek never grades DeepSeek;
    # prices are USD per million tokens (OpenRouter models API, 18 Oct 2026), for the cost line of the report only.
    judge_model: str = "qwen/qwen3-235b-a22b-2507"
    judge_price_in_per_m: float = 0.09
    judge_price_out_per_m: float = 0.55
    # Input guardrail (decision #29): personal data is always redacted and the pattern layer always runs; guardrail_llm adds the
    # model classifier (one small call per customer message) for everything the patterns do not settle.
    guardrail_llm: bool = True
    guardrail_model: str = ""       # empty = llm_model_cheap
    guardrail_timeout_s: float = 8.0
    daily_token_budget: int = 2_000_000
    admin_token: str = "change-me-admin"

    # Tracing (decision #27): OpenTelemetry once, any backend. Empty endpoint = tracing off (no-op provider).
    otel_exporter_otlp_endpoint: str = ""       # http://phoenix:6006 (local Phoenix) or https://cloud.langfuse.com/api/public/otel
    otel_exporter_otlp_headers: str = ""        # key=value,key2=value2 (Langfuse: Authorization=Basic <base64(public:secret)>)
    trace_ui_url: str = ""                      # shown to the presenter as a link next to each reply's trace id (http://localhost:6006)
    environment: str = "demo"

    # ---- derived DSNs, one per database role ----
    def dsn(self, user: str, password: str) -> str:
        return f"postgresql://{user}:{password}@{self.pghost}:{self.pgport}/{self.pgdatabase}"

    @property
    def dsn_admin(self) -> str:
        return self.dsn(self.pgadmin_user, self.pgadmin_password)

    @property
    def dsn_reservations(self) -> str:
        return self.dsn("res_rw", self.pg_res_password)

    @property
    def dsn_flightops(self) -> str:
        return self.dsn("ops_api", self.pg_opsapi_password)

    @property
    def dsn_policy(self) -> str:
        return self.dsn("policy_ro", self.pg_policy_password)

    @property
    def dsn_agent(self) -> str:
        return self.dsn("agent_rw", self.pg_agent_password)

    @property
    def dsn_audit(self) -> str:
        return self.dsn("audit_w", self.pg_audit_password)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
