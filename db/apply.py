"""Create the database, apply db/schema.sql, set role passwords, then seed.

    python db/apply.py            # schema + seed (destructive: drops all Falcon schemas)
    python db/apply.py --seed-only
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falcon.core.config import settings

HERE = Path(__file__).resolve().parent


def ensure_database() -> None:
    admin = f"postgresql://{settings.pgadmin_user}:{settings.pgadmin_password}@{settings.pghost}:{settings.pgport}/postgres"
    with psycopg.connect(admin, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.pgdatabase,)).fetchone()
        if not exists:
            conn.execute(f'CREATE DATABASE "{settings.pgdatabase}"')
            print(f"created database {settings.pgdatabase}")


def apply_schema() -> None:
    sql = (HERE / "schema.sql").read_text(encoding="utf-8")
    with psycopg.connect(settings.dsn_admin, autocommit=True) as conn:
        conn.execute(sql)
        for role, pw in (
            ("res_rw", settings.pg_res_password),
            ("ops_api", settings.pg_opsapi_password),
            ("policy_ro", settings.pg_policy_password),
            ("agent_rw", settings.pg_agent_password),
            ("audit_w", settings.pg_audit_password),
        ):
            # ALTER ROLE cannot take a bind parameter; quote the literal safely instead.
            from psycopg import sql

            conn.execute(sql.SQL("ALTER ROLE {} WITH PASSWORD {}").format(sql.Identifier(role), sql.Literal(pw)))
    print("schema applied, roles configured")


if __name__ == "__main__":
    ensure_database()
    if "--seed-only" not in sys.argv:
        apply_schema()
    from seed import run_seed

    run_seed()
