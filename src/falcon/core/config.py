"""Central configuration. Every service imports `settings` from here; one `.env` holds every secret and URL."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(os.environ.get("FALCON_ROOT") or Path(__file__).resolve().parents[3])


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(REPO_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore")

    # The frozen clock every rule, seed and evaluation reads (core/clock.py).
    falcon_now: str = "2026-10-23T09:00:00+04:00"

    # Postgres: one instance, one role per system (db/schema.sql).
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
    # Every query is cancelled by Postgres after this long, so a slow or locked query becomes an error instead of a hung turn.
    db_statement_timeout_ms: int = 15000

    # Tokens between the browser, the runtime and the MCP servers (core/auth.py).
    falcon_dev_secret: str = "change-me-dev-secret-32-chars-minimum"
    admin_token: str = "change-me-admin"  # the evaluation harness: full tool results in the done event, trace tags

    # Services.
    legacy_api_url: str = "http://127.0.0.1:8010"
    legacy_api_key: str = "legacy-dev-key"
    mcp_reservations_url: str = "http://127.0.0.1:8001/mcp"
    mcp_flightops_url: str = "http://127.0.0.1:8002/mcp"
    mcp_policy_url: str = "http://127.0.0.1:8003/mcp"
    runtime_public_url: str = "http://127.0.0.1:8000"

    # Models, all through OpenRouter (OpenAI-compatible). No key: the scripted stand-in model runs everything.
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_provider: str = "openrouter"  # openrouter | fake
    llm_model: str = "deepseek/deepseek-v4.1-flash-1017"  # the agent, the guard classifier, the query rewriter and the citation check (decision #44)
    llm_price_in_per_m: float = 0.15  # USD per million tokens, for the cost metrics only
    llm_price_out_per_m: float = 0.60
    simulator_model: str = "deepseek/deepseek-chat"  # the simulated customer: a different model from the agent
    judge_model: str = "qwen/qwen3-235b-a22b-2507"  # the generation judge: a different family, so DeepSeek never grades DeepSeek
    judge_price_in_per_m: float = 0.09
    judge_price_out_per_m: float = 0.55
    embedding_provider: str = "openrouter"  # openrouter | none
    embedding_model: str = "baai/bge-m3"  # multilingual (Arabic + English in one space), 1024 dims

    # Policy retrieval (policy/search.py): rewrite the question only when the first pass is low-confidence (decision #31).
    policy_query_rewrite: str = "auto"  # off | auto | always
    policy_rewrite_threshold: float = 0.6
    policy_query_aliases: bool = True  # customer words -> contract words appended before embedding (policy/aliases.py)

    # The agent loop (runtime/graph.py).
    max_tool_calls_per_turn: int = 8  # a looping model is cut off instead of burning budget
    review_max_rounds: int = 3  # a draft the reviewer rejects goes back to the model this many times, then the reply is held
    review_citations: bool = True  # every cited clause is checked against its text by one model call (decision #39)
    review_citation_timeout_s: float = 25.0
    escalate_after_unresolved: int = 2  # held or cut-off replies in a row before the runtime logs a handover case (decision #43); 0 = off
    topic_routing: bool = True  # the classifier's topic picks the prompt section and the tool menu (decision #46)
    guardrail_llm: bool = True  # the model classifier behind the pattern layer (decision #29)
    guardrail_timeout_s: float = 8.0
    daily_token_budget: int = 2_000_000  # the demo's spend cap: /api/chat answers 429 once the day's tokens are spent

    # Tracing: OpenTelemetry to whatever listens at the endpoint; empty = off (decision #27).
    otel_exporter_otlp_endpoint: str = ""  # http://phoenix:6006 with scripts/up.sh --trace
    otel_exporter_otlp_headers: str = ""  # key=value,key2=value2 for a hosted collector
    trace_ui_url: str = ""  # turns the trace id under each reply into a link (http://localhost:6006)

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


settings = Settings()
