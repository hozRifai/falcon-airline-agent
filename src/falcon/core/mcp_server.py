"""Shared plumbing for the three MCP servers: auth, user context, audit log, server factory.

Each server is an OAuth resource server (the MCP spec's own authorization model). The
bearer token is verified by `FalconTokenVerifier`; the per-request user assertion is read
by `UserContextMiddleware` and exposed to tools through `current_user()`.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import time
from typing import Any, Callable

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import USER_HEADER, UserContext, build_verifier, verify_user_assertion
from .config import settings
from .db import Database
from .telemetry import current_ids, setup_tracing, tracer

log = logging.getLogger("falcon.mcp")

_user_ctx: contextvars.ContextVar[UserContext | None] = contextvars.ContextVar("falcon_user", default=None)


class FalconTokenVerifier(TokenVerifier):
    def __init__(self, audience: str):
        self.inner = build_verifier(audience)

    async def verify_token(self, token: str) -> AccessToken | None:
        verified = self.inner.verify(token)
        if verified is None:
            return None
        return AccessToken(token=token, client_id=verified.subject, scopes=verified.roles, expires_at=verified.expires_at, claims=verified.claims)


class UserContextMiddleware:
    """Reads the signed user assertion (if any) and stores it for the duration of the request."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        user: UserContext | None = None
        for name, value in scope.get("headers", []):
            if name.decode().lower() == USER_HEADER:
                try:
                    user = verify_user_assertion(value.decode())
                except Exception:  # invalid assertion => no user; tools that need one will refuse
                    user = None
                break
        token = _user_ctx.set(user)
        try:
            await self.app(scope, receive, send)
        finally:
            _user_ctx.reset(token)


def current_user() -> UserContext:
    user = _user_ctx.get()
    if user is None:
        raise ToolError("No signed-in user context on this request.")
    return user


def optional_user() -> UserContext | None:
    return _user_ctx.get()


def require_role(role: str) -> None:
    """Token-level guardrail: the calling service must hold the app role for this tool."""
    token = get_access_token()
    if token is None or role not in token.scopes:
        raise ToolError(f"Forbidden: this tool requires the '{role}' role, which the calling service does not have.")


def require_user_role(*roles: str) -> UserContext:
    """User-level guardrail: the signed-in person must hold one of these roles (today only "customer" exists)."""
    user = current_user()
    if user.role not in roles:
        raise ToolError(f"Forbidden: this action is only available to {', '.join(roles)} users.")
    return user


def redact(args: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in args.items():
        if any(s in k.lower() for s in ("email", "phone", "passport", "card")):
            out[k] = "***"
        else:
            out[k] = v
    return out


class Audit:
    """Append-only audit log. Every tool call on every server lands here."""

    def __init__(self, service: str, db: Database):
        self.service = service
        self.db = db

    async def record(self, tool: str, args: dict[str, Any], *, ok: bool, latency_ms: int, summary: str) -> None:
        user = optional_user()
        trace_id, _ = current_ids()  # the agent's trace, carried in by traceparent: audit row <-> trace without any backend
        try:
            await self.db.execute(
                "INSERT INTO agent.audit_log(service, session_id, actor, tool, args, ok, latency_ms, summary, trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (self.service, user.session_id if user else None, user.user_id if user else None, tool, json.dumps(redact(args), default=str), ok, latency_ms, summary[:500], trace_id),
            )
        except Exception as exc:  # the audit log must never break the business call
            log.warning("audit write failed: %s", exc)


def audited(audit: Audit, name: str):
    """Decorator: times the tool, records success or failure, re-raises errors as ToolError. Each call is a span
    (`tool.<name>`) inside the agent's trace, carried in by the traceparent header the runtime sends."""

    def wrap(fn: Callable):
        @functools.wraps(fn)
        async def inner(*args, **kwargs):
            started = time.perf_counter()
            with tracer("falcon.mcp").start_as_current_span(f"tool.{name}", attributes={"falcon.tool": name, "falcon.service": audit.service}) as span:
                try:
                    result = await fn(*args, **kwargs)
                except ToolError as exc:
                    span.set_attribute("falcon.ok", False)
                    span.set_attribute("falcon.error", str(exc)[:300])
                    await audit.record(name, kwargs, ok=False, latency_ms=int((time.perf_counter() - started) * 1000), summary=str(exc))
                    raise
                except Exception as exc:
                    span.set_attribute("falcon.ok", False)
                    span.set_attribute("falcon.error", f"error: {exc}"[:300])
                    await audit.record(name, kwargs, ok=False, latency_ms=int((time.perf_counter() - started) * 1000), summary=f"error: {exc}")
                    raise ToolError(f"{name} failed: {exc}") from exc
                summary = result if isinstance(result, str) else json.dumps(result, default=str)
                span.set_attribute("falcon.ok", True)
                await audit.record(name, kwargs, ok=True, latency_ms=int((time.perf_counter() - started) * 1000), summary=summary)
                return result

        return inner

    return wrap


def build_server(name: str, *, audience: str, public_url: str, host: str = "0.0.0.0", port: int, lifespan=None, instructions: str = "") -> FastMCP:
    """FastMCP configured as a stateless Streamable HTTP resource server with bearer auth."""
    auth = AuthSettings(
        issuer_url="https://login.microsoftonline.com/" + settings.entra_tenant_id + "/v2.0" if settings.auth_mode == "entra" else "http://falcon-dev-issuer.local",
        resource_server_url=public_url,
        required_scopes=[],
        validate_token_resource=False,  # audience is checked inside the verifier
    )
    transport = TransportSecuritySettings(enable_dns_rebinding_protection=False)  # containers call us by service name
    server = FastMCP(
        name,
        instructions=instructions,
        token_verifier=FalconTokenVerifier(audience),
        auth=auth,
        host=host,
        port=port,
        stateless_http=True,   # one request = one unit of work: no sticky sessions, scales out behind any load balancer
        json_response=True,
        transport_security=transport,
        lifespan=lifespan,
    )
    return server


def wrap_app(server: FastMCP):
    app = server.streamable_http_app()
    app.add_middleware(UserContextMiddleware)
    setup_tracing(server.name, app)  # server spans continue the agent's trace (decision #27); a no-op without an endpoint
    return app
