"""Authentication between the agent runtime and the MCP servers.

Two things travel with every MCP request, and they are deliberately separate:

1. `Authorization: Bearer <service token>` proves WHICH SERVICE is calling and WHAT IT MAY DO.
   The token carries roles such as `Bookings.Write` or `Flights.Read`; every tool checks the
   role it needs. In `dev` mode the runtime mints these itself (HS256, shared secret).
   In `entra` mode they are Microsoft Entra ID app-only tokens validated against the tenant's
   JWKS; roles come from app-role assignments on the runtime's managed identity.

2. `X-Falcon-User: <user assertion>` says ON WHOSE BEHALF the call is made: the signed-in
   customer with their customer id, role and language. The runtime
   signs it after the person signs in; the MCP server verifies it and filters every query by
   it. No tool has a `customer_id` parameter, so the model cannot name a customer at all.

Production upgrade (one line on a slide): replace the user assertion with OAuth token
exchange, so the reservations API receives a user-scoped token and enforces access itself.
"""

from __future__ import annotations

import time
from typing import Literal

import jwt
from pydantic import BaseModel

from .config import settings

Role = Literal["customer"]  # one persona by design (decision #25): staff work happens outside the agent

# Service roles that tools check. Names follow Entra app-role conventions.
BOOKINGS_READ = "Bookings.Read"
BOOKINGS_WRITE = "Bookings.Write"
FLIGHTS_READ = "Flights.Read"
POLICY_READ = "Policy.Read"

DEV_ISSUER = "http://falcon-dev-issuer.local"
USER_HEADER = "x-falcon-user"


class UserContext(BaseModel):
    user_id: str
    display_name: str
    role: Role
    customer_id: str | None = None
    lang: str = "en"
    session_id: str = ""


# ---------------------------------------------------------------- minting (runtime side)

def mint_service_token(audience: str, roles: list[str], *, subject: str = "agent-runtime", ttl_seconds: int = 3600) -> str:
    """Dev-mode service token. In entra mode `TokenProvider` asks Entra ID instead."""
    now = int(time.time())
    payload = {
        "iss": DEV_ISSUER,
        "sub": subject,
        "aud": audience,
        "iat": now,
        "exp": now + ttl_seconds,
        "roles": roles,
    }
    return jwt.encode(payload, settings.falcon_dev_secret, algorithm="HS256")


def mint_user_assertion(user: UserContext, *, ttl_seconds: int = 3600) -> str:
    now = int(time.time())
    payload = {"iss": "falcon-runtime", "iat": now, "exp": now + ttl_seconds, "user": user.model_dump()}
    return jwt.encode(payload, settings.falcon_dev_secret, algorithm="HS256")


def verify_user_assertion(token: str) -> UserContext:
    data = jwt.decode(token, settings.falcon_dev_secret, algorithms=["HS256"], issuer="falcon-runtime")
    return UserContext.model_validate(data["user"])


# ---------------------------------------------------------------- web session tokens (browser <-> runtime)

def mint_session_token(user: UserContext, *, ttl_seconds: int = 12 * 3600) -> str:
    now = int(time.time())
    payload = {"iss": "falcon-web", "iat": now, "exp": now + ttl_seconds, "user": user.model_dump()}
    return jwt.encode(payload, settings.falcon_dev_secret, algorithm="HS256")


def verify_session_token(token: str) -> UserContext:
    data = jwt.decode(token, settings.falcon_dev_secret, algorithms=["HS256"], issuer="falcon-web")
    return UserContext.model_validate(data["user"])


# ---------------------------------------------------------------- verifying (MCP server side)

class VerifiedToken(BaseModel):
    subject: str
    roles: list[str]
    expires_at: int | None = None
    claims: dict = {}


class DevTokenVerifier:
    def __init__(self, audience: str):
        self.audience = audience

    def verify(self, token: str) -> VerifiedToken | None:
        try:
            data = jwt.decode(token, settings.falcon_dev_secret, algorithms=["HS256"], audience=self.audience, issuer=DEV_ISSUER)
        except jwt.PyJWTError:
            return None
        return VerifiedToken(subject=data.get("sub", ""), roles=list(data.get("roles", [])), expires_at=data.get("exp"), claims=data)


class EntraTokenVerifier:
    """Validates Microsoft Entra ID access tokens with the tenant's published signing keys."""

    def __init__(self, audience: str, tenant_id: str):
        self.audience = audience
        self.tenant_id = tenant_id
        self._jwks = jwt.PyJWKClient(f"https://login.microsoftonline.com/{tenant_id}/discovery/v2.0/keys", cache_keys=True)
        self.issuers = [
            f"https://login.microsoftonline.com/{tenant_id}/v2.0",
            f"https://sts.windows.net/{tenant_id}/",
        ]

    def verify(self, token: str) -> VerifiedToken | None:
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            data = jwt.decode(token, key, algorithms=["RS256"], audience=self.audience, issuer=self.issuers)
        except jwt.PyJWTError:
            return None
        roles = list(data.get("roles", []))
        if "scp" in data:  # delegated scopes, if a user-scoped token is ever used
            roles += data["scp"].split()
        return VerifiedToken(subject=data.get("sub") or data.get("azp", ""), roles=roles, expires_at=data.get("exp"), claims=data)


def build_verifier(audience: str):
    if settings.auth_mode == "entra":
        if not settings.entra_tenant_id:
            raise RuntimeError("AUTH_MODE=entra requires ENTRA_TENANT_ID")
        return EntraTokenVerifier(audience, settings.entra_tenant_id)
    return DevTokenVerifier(audience)


class TokenProvider:
    """Runtime-side: obtains the service token for a given MCP server audience."""

    def __init__(self):
        self._credential = None

    def service_token(self, audience: str, roles: list[str]) -> str:
        if settings.auth_mode == "entra":
            # Roles come from the app-role assignments of this identity, not from the request.
            from azure.identity import DefaultAzureCredential

            if self._credential is None:
                self._credential = DefaultAzureCredential()
            return self._credential.get_token(f"{audience}/.default").token
        return mint_service_token(audience, roles)
