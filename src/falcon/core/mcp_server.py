"""Shared plumbing for the three MCP servers: bearer-token verification, the signed user context, the audit log, the server factory.

Each server is a stateless Streamable HTTP resource server. The bearer token is checked by `FalconTokenVerifier` (audience =
the server's name, roles inside); the per-request user assertion is read by `UserContextMiddleware` and reached by tools through
`current_user()`. Every tool is wrapped by `audited()`: one audit row and one trace span per call.
"""

from __future__ import annotations

import contextvars
import functools
import json
import logging
import time
from collections.abc import Callable
from typing import Any

import jwt
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import ASGIApp, Receive, Scope, Send

from .auth import SERVICE_ISSUER, USER_HEADER, UserContext, verify_service_token, verify_user_assertion
from .db import Database
from .telemetry import current_ids, setup_tracing, tracer

log = logging.getLogger("falcon.mcp")

_user_ctx: contextvars.ContextVar[UserContext | None] = contextvars.ContextVar("falcon_user", default=None)


class FalconTokenVerifier(TokenVerifier):
    def __init__(self, audience: str):
        self.audience = audience

    async def verify_token(self, token: str) -> AccessToken | None:
        verified = verify_service_token(token, self.audience)
        if verified is None:
            return None
        return AccessToken(token=token, client_id=verified.subject, scopes=verified.roles, expires_at=verified.expires_at)


class UserContextMiddleware:
    """Reads the signed user assertion (if any) and keeps it for the duration of the request."""

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
                except (jwt.PyJWTError, KeyError, ValueError):  # an invalid assertion means no user: tools that need one refuse
                    user = None
                break
        token = _user_ctx.set(user)
        try:
            await self.app(scope, receive, send)
        finally:
            _user_ctx.reset(token)


def current_user() -> UserContext:
    """The signed-in customer this request is made for. Tools never take a customer id: this is the only source."""
    user = _user_ctx.get()
    if user is None:
        raise ToolError("No signed-in user context on this request.")
    return user


def require_role(role: str) -> None:
    """Token-level guardrail: the calling service must hold the app role for this tool."""
    token = get_access_token()
    if token is None or role not in token.scopes:
        raise ToolError(f"Forbidden: this tool requires the '{role}' role, which the calling service does not have.")


def redact(args: dict[str, Any]) -> dict[str, Any]:
    """Tool arguments as the audit log and the UI may show them."""
    return {k: ("***" if any(s in k.lower() for s in ("email", "phone", "passport", "card")) else v) for k, v in args.items()}


class Audit:
    """Append-only audit log: every tool call on every server lands in agent.audit_log, with the agent's trace id."""

    def __init__(self, service: str, db: Database):
        self.service = service
        self.db = db

    async def record(self, tool: str, args: dict[str, Any], *, ok: bool, latency_ms: int, summary: str) -> None:
        user = _user_ctx.get()
        trace_id, _ = current_ids()
        try:
            await self.db.execute(
                "INSERT INTO agent.audit_log(service, session_id, actor, tool, args, ok, latency_ms, summary, trace_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (self.service, user.session_id if user else None, user.user_id if user else None, tool, json.dumps(redact(args), default=str), ok, latency_ms, summary[:500], trace_id),
            )
        except Exception as exc:  # noqa: BLE001 - the audit log must never break the business call
            log.warning("audit write failed: %s", exc)


def audited(audit: Audit, name: str):
    """Decorator: times the tool, records success or failure, re-raises errors as ToolError, one span per call."""

    def wrap(fn: Callable):
        @functools.wraps(fn)
        async def inner(*args, **kwargs):
            started = time.perf_counter()
            with tracer("falcon.mcp").start_as_current_span(f"tool.{name}", attributes={"falcon.tool": name, "falcon.service": audit.service}) as span:
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    error = str(exc) if isinstance(exc, ToolError) else f"error: {exc}"
                    span.set_attribute("falcon.ok", False)
                    span.set_attribute("falcon.error", error[:300])
                    await audit.record(name, kwargs, ok=False, latency_ms=int((time.perf_counter() - started) * 1000), summary=error)
                    if isinstance(exc, ToolError):
                        raise
                    raise ToolError(f"{name} failed: {exc}") from exc
                span.set_attribute("falcon.ok", True)
                await audit.record(name, kwargs, ok=True, latency_ms=int((time.perf_counter() - started) * 1000),
                                   summary=result if isinstance(result, str) else json.dumps(result, default=str))
                return result

        return inner

    return wrap


def build_server(name: str, *, public_url: str, port: int, instructions: str = "") -> FastMCP:
    """FastMCP as a stateless Streamable HTTP resource server with bearer auth; the audience is the server's short name."""
    return FastMCP(
        f"falcon-{name}",
        instructions=instructions,
        token_verifier=FalconTokenVerifier(name),
        auth=AuthSettings(issuer_url=f"http://{SERVICE_ISSUER}.local", resource_server_url=public_url, required_scopes=[], validate_token_resource=False),
        host="0.0.0.0",
        port=port,
        stateless_http=True,  # one request = one unit of work: no sticky sessions
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),  # containers call us by service name
    )


def wrap_app(server: FastMCP):
    app = server.streamable_http_app()
    app.add_middleware(UserContextMiddleware)
    setup_tracing(server.name, app)  # server spans continue the agent's trace; a no-op without an endpoint
    return app
