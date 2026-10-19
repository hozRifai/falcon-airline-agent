"""Runtime-side MCP client factory: one client per signed-in session, identity in the headers.

Every tool call carries:
  Authorization: Bearer <service token with the roles this persona is allowed>
  X-Falcon-User:  <signed assertion: who the person is, their customer id, role, language>
The LangChain tool objects returned here have schemas without any customer id, so the model
never sees or chooses one.
"""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

from .auth import USER_HEADER, TokenProvider, UserContext, mint_user_assertion
from .config import settings

SERVERS = {
    "reservations": ("mcp_reservations_url", "entra_aud_reservations"),
    "flightops": ("mcp_flightops_url", "entra_aud_flightops"),
    "policy": ("mcp_policy_url", "entra_aud_policy"),
}

_token_provider = TokenProvider()


def connections_for(user: UserContext, roles: list[str], servers: list[str] | None = None) -> dict:
    assertion = mint_user_assertion(user)
    conns = {}
    for name in servers or list(SERVERS):
        url_attr, aud_attr = SERVERS[name]
        token = _token_provider.service_token(getattr(settings, aud_attr), roles)
        conns[name] = {
            "transport": "streamable_http",
            "url": getattr(settings, url_attr),
            "headers": {"Authorization": f"Bearer {token}", USER_HEADER: assertion},
            "timeout": 30,
        }
    return conns


def client_for(user: UserContext, roles: list[str], servers: list[str] | None = None) -> MultiServerMCPClient:
    return MultiServerMCPClient(connections_for(user, roles, servers))
