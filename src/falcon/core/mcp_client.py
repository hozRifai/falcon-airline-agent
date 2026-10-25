"""Runtime-side MCP client: one per signed-in session, identity in the headers.

Every tool call carries a bearer token with the roles the customer persona is allowed and the signed `X-Falcon-User`
assertion. The LangChain tool objects returned have schemas without any customer id, so the model never sees or chooses one.
"""

from __future__ import annotations

from langchain_mcp_adapters.client import MultiServerMCPClient

from .auth import USER_HEADER, UserContext, mint_service_token, mint_user_assertion
from .config import settings

SERVER_URLS = {"reservations": settings.mcp_reservations_url, "flightops": settings.mcp_flightops_url, "policy": settings.mcp_policy_url}


def client_for(user: UserContext, roles: list[str], servers: list[str] | None = None) -> MultiServerMCPClient:
    assertion = mint_user_assertion(user)
    return MultiServerMCPClient({
        name: {"transport": "streamable_http", "url": SERVER_URLS[name], "timeout": 30,
               "headers": {"Authorization": f"Bearer {mint_service_token(name, roles)}", USER_HEADER: assertion}}
        for name in servers or list(SERVER_URLS)
    })
