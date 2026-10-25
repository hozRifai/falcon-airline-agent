"""db/migrate.py: idempotent, and it leaves the columns the code writes to.

Runs against the database the live tests use (skipped when it is not reachable). It cannot rebuild a database from an old
schema here, so the promise it checks is the one that matters day to day: running the migration on a current database, twice,
changes nothing and fails nowhere, and every column the runtime and the servers insert into exists afterwards.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import psycopg
import pytest

from falcon.core.config import REPO_ROOT, settings


def _db_up() -> bool:
    try:
        with psycopg.connect(settings.dsn_admin, connect_timeout=2):
            return True
    except psycopg.Error:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="database not running")


def _load_migrate():
    spec = importlib.util.spec_from_file_location("falcon_migrate", Path(REPO_ROOT) / "db" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _columns(conn, table: str) -> set[str]:
    schema, name = table.split(".")
    rows = conn.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = %s", (schema, name)
    ).fetchall()
    return {r[0] for r in rows}


def test_migration_is_idempotent_and_matches_what_the_code_writes(capsys):
    migrate = _load_migrate()
    with psycopg.connect(settings.dsn_admin) as conn:
        before = {t: _columns(conn, t) for t in ("agent.turns", "agent.audit_log", "agent.feedback", "agent.users")}
    for _ in range(2):
        migrate.main()
        assert capsys.readouterr().out.rstrip().endswith("schema is current")
    with psycopg.connect(settings.dsn_admin) as conn:
        after = {t: _columns(conn, t) for t in before}
        assert after == before  # a current database is left exactly as it was
        assert {"thread", "verification", "trace_id", "span_id"} <= after["agent.turns"]
        assert "trace_id" in after["agent.audit_log"]
        assert {"rating", "turn_id", "message"} <= after["agent.feedback"]
        assert conn.execute("SELECT count(*) FROM agent.users WHERE role <> 'customer'").fetchone()[0] == 0
        assert conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'ops_views'").fetchone() is None
    # every statement in the list is written to be re-runnable: the names say what they do, none is a one-shot
    assert len(migrate.MIGRATIONS) >= 6 and all(name and sql.strip() for name, sql in migrate.MIGRATIONS)


def test_migration_refuses_an_empty_database():
    migrate = _load_migrate()
    with psycopg.connect(settings.dsn_admin, autocommit=True) as conn:
        conn.execute("DROP DATABASE IF EXISTS falcon_migrate_empty")
        conn.execute("CREATE DATABASE falcon_migrate_empty")
    try:
        empty_dsn = settings.dsn_admin.rsplit("/", 1)[0] + "/falcon_migrate_empty"
        with pytest.raises(SystemExit) as exc:
            migrate.main(empty_dsn)
        assert "reset.sh" in str(exc.value)  # points at the first-time setup instead of half-migrating nothing
    finally:
        with psycopg.connect(settings.dsn_admin, autocommit=True) as conn:
            conn.execute("DROP DATABASE IF EXISTS falcon_migrate_empty")
