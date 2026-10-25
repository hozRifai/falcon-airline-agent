"""Tracing: OpenTelemetry once, any backend.

Every service calls `setup_tracing("<service>")` at start-up. With no endpoint configured it does nothing beyond
installing a no-op provider, so the code path is the same in tests and in the demo. With `OTEL_EXPORTER_OTLP_ENDPOINT`
set, spans go over OTLP/HTTP to whatever listens there:

- Arize Phoenix locally (`docker compose --profile observability up -d phoenix`, endpoint http://phoenix:6006),
- Langfuse (endpoint https://cloud.langfuse.com/api/public/otel with basic-auth headers, or a self-hosted one),
- any other OTLP collector (a production backend is a URL and a header, never a code change).

What is instrumented:
- the agent runtime: LangGraph/LangChain (openinference: every node, model call with tokens, tool call), the FastAPI app,
  outgoing httpx (the MCP calls carry W3C `traceparent`, so the servers' spans join the agent's trace), psycopg;
- the MCP servers and the legacy flight API: ASGI (one server span per request, continuing the caller's trace), psycopg,
  outgoing httpx (reservations -> flight ops), plus a span per tool call from `audited()`;
- hand-made spans where the agent's own logic lives: `chat.turn` (the root of a customer turn, with session, thread,
  persona, language and the reviewer's verdict), `reviewer.round`, `reviewer.hold`, `confirmation.decision`,
  `customer.feedback` (linked to the turn it rates).

The trace id is returned to the UI with every reply and stored on `agent.turns` and `agent.audit_log`, so a turn can
be found in the trace UI, and a tool call in the audit log can be matched to its trace without any backend at all.
"""

from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Link, SpanContext, TraceFlags, format_span_id, format_trace_id

from .config import settings

log = logging.getLogger("falcon.telemetry")
_configured: dict[str, Any] = {"provider": None, "service": None}


def enabled() -> bool:
    return bool(_configured.get("forced") or settings.otel_exporter_otlp_endpoint)


def _otlp_traces_endpoint(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/v1/traces") else base + "/v1/traces"


def _headers(raw: str) -> dict[str, str]:
    """OTEL_EXPORTER_OTLP_HEADERS format: key=value,key2=value2 (Langfuse: Authorization=Basic <base64(pk:sk)>)."""
    out = {}
    for part in (raw or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def setup_tracing(service: str, app=None, *, langchain: bool = False, exporter=None) -> None:
    """Idempotent per process. `app` is a FastAPI/Starlette app to wrap with the ASGI instrumentation.
    `exporter` (tests) replaces the OTLP exporter and turns tracing on regardless of settings."""
    if exporter is not None:
        _configured["forced"] = True
    if _configured["provider"] is None:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        resource = Resource.create({"service.name": service, "service.namespace": "falcon"})
        provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(provider)
        if exporter is not None:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor

            provider.add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[union-attr]
            _configured.setdefault("exporters", []).append(exporter)
        elif settings.otel_exporter_otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            otlp = OTLPSpanExporter(endpoint=_otlp_traces_endpoint(settings.otel_exporter_otlp_endpoint), headers=_headers(settings.otel_exporter_otlp_headers) or None)
            provider.add_span_processor(BatchSpanProcessor(otlp))  # type: ignore[union-attr]
        _configured["provider"] = provider
        _configured["service"] = service
    elif exporter is not None and not any(e is exporter for e in _configured.setdefault("exporters", [])):
        # tracing was already set up in this process (importing the runtime app does it) without the test's exporter: attach it to
        # the existing provider, once, or the test sees no spans at all (21 Oct: the suite failed once a test imported the app first)
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        _configured["provider"].add_span_processor(SimpleSpanProcessor(exporter))  # type: ignore[union-attr]
        _configured["exporters"].append(exporter)
        if enabled():
            _instrument_libraries(langchain=langchain)
            log.info("tracing on for %s -> %s", service, settings.otel_exporter_otlp_endpoint or "test exporter")
    if app is not None and enabled():
        _instrument_app(app)


def _instrument_libraries(*, langchain: bool) -> None:
    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()  # injects traceparent into every outgoing call: MCP servers, legacy API, OpenRouter
    except Exception as exc:  # noqa: BLE001
        log.warning("httpx instrumentation unavailable: %s", exc)
    try:
        from opentelemetry.instrumentation.psycopg import PsycopgInstrumentor

        PsycopgInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        log.warning("psycopg instrumentation unavailable: %s", exc)
    if langchain:
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor

            LangChainInstrumentor().instrument(tracer_provider=_configured["provider"])
        except Exception as exc:  # noqa: BLE001
            log.warning("LangChain instrumentation unavailable: %s", exc)


def _instrument_app(app) -> None:
    try:
        from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

        app.add_middleware(OpenTelemetryMiddleware)  # one server span per request, continuing the caller's traceparent
    except Exception as exc:  # noqa: BLE001
        log.warning("ASGI instrumentation unavailable: %s", exc)


def tracer(name: str = "falcon"):
    return trace.get_tracer(name)


def current_ids() -> tuple[str | None, str | None]:
    """(trace_id, span_id) of the current span as hex, or (None, None) when there is no recording span."""
    ctx = trace.get_current_span().get_span_context()
    if not ctx or not ctx.is_valid:
        return None, None
    return format_trace_id(ctx.trace_id), format_span_id(ctx.span_id)


def link_to(trace_id: str | None, span_id: str | None) -> list[Link]:
    """A span link back to an earlier trace (the feedback span points at the turn it rates)."""
    if not trace_id or not span_id:
        return []
    try:
        ctx = SpanContext(trace_id=int(trace_id, 16), span_id=int(span_id, 16), is_remote=True, trace_flags=TraceFlags(TraceFlags.SAMPLED))
        return [Link(ctx)]
    except ValueError:
        return []
