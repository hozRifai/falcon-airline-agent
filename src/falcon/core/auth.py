"""Who is calling, and on whose behalf. Every token here is HS256 with one shared secret (FALCON_DEV_SECRET): the demo
runs on one laptop. In production the service token would come from the platform's identity provider (app roles) and the
user assertion would be an OAuth token exchange; the checks the servers make stay the same.

Two things travel with every MCP request, deliberately separate:

1. `Authorization: Bearer <service token>` says WHICH SERVICE is calling and WHAT IT MAY DO. The token carries roles
   such as `Bookings.Write`; every tool checks the role it needs. The audience is the server's name, so a token for the
   policy server is refused by the reservations server.
2. `X-Falcon-User: <user assertion>` says ON WHOSE BEHALF: the signed-in customer with their customer id and language.
   The runtime signs it after sign-in; the MCP server verifies it and filters every query by it. No tool has a
   `customer_id` parameter, so the model cannot name a customer at all.

The browser holds a third token, the web session, with the same claims as the user assertion.
"""

from __future__ import annotations

import time

import jwt
from pydantic import BaseModel

from .config import settings

BOOKINGS_READ = "Bookings.Read"
BOOKINGS_WRITE = "Bookings.Write"
FLIGHTS_READ = "Flights.Read"
POLICY_READ = "Policy.Read"

USER_HEADER = "x-falcon-user"
SERVICE_ISSUER = "falcon-runtime"
WEB_ISSUER = "falcon-web"


class UserContext(BaseModel):
    user_id: str
    display_name: str
    customer_id: str | None = None
    lang: str = "en"
    session_id: str = ""


class VerifiedToken(BaseModel):
    subject: str
    roles: list[str]
    expires_at: int | None = None


def _mint(payload: dict, ttl_seconds: int) -> str:
    now = int(time.time())
    return jwt.encode({**payload, "iat": now, "exp": now + ttl_seconds}, settings.falcon_dev_secret, algorithm="HS256")


def mint_service_token(audience: str, roles: list[str]) -> str:
    return _mint({"iss": SERVICE_ISSUER, "sub": "agent-runtime", "aud": audience, "roles": roles}, ttl_seconds=3600)


def verify_service_token(token: str, audience: str) -> VerifiedToken | None:
    try:
        data = jwt.decode(token, settings.falcon_dev_secret, algorithms=["HS256"], audience=audience, issuer=SERVICE_ISSUER)
    except jwt.PyJWTError:
        return None
    return VerifiedToken(subject=data.get("sub", ""), roles=list(data.get("roles", [])), expires_at=data.get("exp"))


def mint_user_assertion(user: UserContext) -> str:
    return _mint({"iss": SERVICE_ISSUER, "user": user.model_dump()}, ttl_seconds=3600)


def verify_user_assertion(token: str) -> UserContext:
    data = jwt.decode(token, settings.falcon_dev_secret, algorithms=["HS256"], issuer=SERVICE_ISSUER)
    return UserContext.model_validate(data["user"])


def mint_session_token(user: UserContext) -> str:
    return _mint({"iss": WEB_ISSUER, "user": user.model_dump()}, ttl_seconds=12 * 3600)


def verify_session_token(token: str) -> UserContext:
    data = jwt.decode(token, settings.falcon_dev_secret, algorithms=["HS256"], issuer=WEB_ISSUER)
    return UserContext.model_validate(data["user"])
