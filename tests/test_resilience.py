"""When a dependency fails, the rest keeps working, and the failure is visible.

- One MCP server down or refusing the token removes its tools only; the agent is told what is down (it used to fail every turn).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from falcon.core.auth import UserContext
from falcon.services.runtime import app as runtime_app
from falcon.services.runtime.graph import unavailable_note

USER = UserContext(user_id="u-fatima", display_name="Fatima", customer_id="C-1001", lang="en", session_id="resilience-test")
TOOLS = {"reservations": ["list_my_bookings", "quote_cancellation", "cancel_booking"], "flightops": ["get_flight_status"], "policy": ["search_policy"]}


def fake_clients(down: set[str]):
    def client_for(user, roles, servers):
        (name,) = servers  # discovery asks one server at a time

        async def get_tools():
            if name in down:
                raise RuntimeError(f"{name}: 401 Unauthorized")
            return [SimpleNamespace(name=t) for t in TOOLS[name]]
        return SimpleNamespace(get_tools=get_tools)
    return client_for


async def test_one_server_down_removes_only_its_tools(monkeypatch):
    runtime_app._tool_cache.clear()
    monkeypatch.setattr(runtime_app, "client_for", fake_clients({"flightops"}))
    tools, unavailable = await runtime_app.tools_for(USER)
    assert {t.name for t in tools} == {"list_my_bookings", "quote_cancellation", "cancel_booking", "search_policy"}
    assert unavailable == ["flightops"]
    note = unavailable_note(unavailable)
    assert "flight operations system" in note and "never guess" in note
    assert unavailable_note([]) is None


async def test_a_degraded_tool_list_is_retried_soon(monkeypatch):
    runtime_app._tool_cache.clear()
    monkeypatch.setattr(runtime_app, "client_for", fake_clients({"flightops"}))
    await runtime_app.tools_for(USER)
    stamp, tools, unavailable = runtime_app._tool_cache[USER.session_id]
    runtime_app._tool_cache[USER.session_id] = (stamp - runtime_app.DEGRADED_TTL_S - 1, tools, unavailable)  # 31 s later
    monkeypatch.setattr(runtime_app, "client_for", fake_clients(set()))  # flight ops is back
    _, unavailable = await runtime_app.tools_for(USER)
    assert unavailable == []


async def test_no_server_reachable_refuses_the_turn(monkeypatch):
    runtime_app._tool_cache.clear()
    monkeypatch.setattr(runtime_app, "client_for", fake_clients(set(TOOLS)))
    with pytest.raises(runtime_app.NoSystemsReachable):
        await runtime_app.tools_for(USER)


# ---------------------------------------------------------------- the database: slow queries end, dropped connections are replaced
def _db_up() -> bool:
    import psycopg
    from falcon.core.config import settings
    try:
        with psycopg.connect(settings.dsn_admin, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


@pytest.mark.skipif(not _db_up(), reason="database not running")
async def test_a_slow_query_is_cancelled_by_the_statement_timeout():
    import psycopg
    from falcon.core.config import settings
    from falcon.core.db import Database
    db = Database(settings.dsn_agent, max_size=1, statement_timeout_ms=300)
    try:
        with pytest.raises(psycopg.errors.QueryCanceled):
            await db.fetch_one("SELECT pg_sleep(2)")
        assert (await db.fetch_one("SELECT 1 AS ok"))["ok"] == 1  # the pool still serves after a cancelled query
    finally:
        await db.close()


@pytest.mark.skipif(not _db_up(), reason="database not running")
async def test_a_connection_the_server_dropped_is_replaced():
    import psycopg
    from falcon.core.config import settings
    from falcon.core.db import Database
    db = Database(settings.dsn_agent, min_size=1, max_size=1)
    try:
        pid = (await db.fetch_one("SELECT pg_backend_pid() AS pid"))["pid"]
        with psycopg.connect(settings.dsn_admin, autocommit=True) as admin:  # what a database restart or failover does to it
            admin.execute("SELECT pg_terminate_backend(%s)", (pid,))
        row = await db.fetch_one("SELECT pg_backend_pid() AS pid")  # the very next request works, on a new connection
        assert row["pid"] != pid
    finally:
        await db.close()


# ---------------------------------------------------------------- readiness: says what is down (health said "ok" with a server down)
class _Pool:
    def __init__(self, ok: bool):
        self.ok = ok

    def connection(self):
        pool = self

        class _Conn:
            async def __aenter__(self):
                if not pool.ok:
                    raise OSError("connection refused")
                return self

            async def __aexit__(self, *exc):
                return False

            async def execute(self, sql):
                return None
        return _Conn()


async def test_ready_names_what_is_down(monkeypatch):
    import json

    async def db_ok(sql, params=None):
        return {"ok": 1}
    monkeypatch.setattr(runtime_app.db, "fetch_one", db_ok)
    monkeypatch.setitem(runtime_app.state, "saver_pool", _Pool(ok=True))
    monkeypatch.setattr(runtime_app, "client_for", fake_clients({"flightops"}))
    r = await runtime_app.ready()
    body = json.loads(r.body)
    assert r.status_code == 503 and body["ready"] is False
    assert body["checks"] == {"database": "ok", "checkpoints": "ok", "reservations": "ok", "flightops": "down (RuntimeError)", "policy": "ok"}
    assert "401" not in r.body.decode()  # error types only, never the details

    monkeypatch.setattr(runtime_app, "client_for", fake_clients(set()))
    r = await runtime_app.ready()
    assert r.status_code == 200 and json.loads(r.body)["ready"] is True
