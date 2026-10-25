"""Idempotent schema migrations: bring an existing database up to the current code without touching its data.

    python db/migrate.py            # safe to run any number of times; scripts/up.sh and scripts/dev.sh start run it for you

`db/apply.py` (used by scripts/reset.sh) recreates everything from schema.sql; this file is the path for a running demo
whose code moved on. Every statement is written so that applying it twice is a no-op, and the list only grows: a change to
schema.sql that an existing database must pick up gets a line here too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from falcon.core.config import settings

MIGRATIONS: list[tuple[str, str]] = [
    # 2026-10-17: one persona; staff users, the ops views and the ops_ro role go
    ("agent.users: customers only", """
        DELETE FROM agent.users WHERE role <> 'customer';
        ALTER TABLE agent.users DROP CONSTRAINT IF EXISTS users_role_check;
        ALTER TABLE agent.users ADD CONSTRAINT users_role_check CHECK (role IN ('customer'));
    """),
    ("drop ops_views schema", "DROP SCHEMA IF EXISTS ops_views CASCADE;"),
    ("drop ops_ro role (best effort: roles are cluster-wide, another database may still reference it)", """
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ops_ro') THEN
            EXECUTE 'REVOKE ALL ON SCHEMA public FROM ops_ro';
            EXECUTE 'DROP OWNED BY ops_ro';
            EXECUTE 'DROP ROLE ops_ro';
          END IF;
        END $$;
    """),
    # 2026-10-17: reviewer loop and thumbs feedback
    ("agent.turns: thread, verification", """
        ALTER TABLE agent.turns ADD COLUMN IF NOT EXISTS thread text;
        ALTER TABLE agent.turns ADD COLUMN IF NOT EXISTS verification jsonb;
    """),
    ("agent.feedback", """
        CREATE TABLE IF NOT EXISTS agent.feedback (
          id          bigserial PRIMARY KEY,
          ts          timestamptz NOT NULL DEFAULT now(),
          session_id  text NOT NULL,
          user_id     text,
          thread      text,
          turn_id     bigint REFERENCES agent.turns(id) ON DELETE SET NULL,
          rating      text NOT NULL CHECK (rating IN ('up','down')),
          comment     text,
          message     text
        );
        CREATE INDEX IF NOT EXISTS feedback_rating_ts_idx ON agent.feedback (rating, ts);
        GRANT ALL ON agent.feedback TO agent_rw;
        GRANT USAGE, SELECT ON SEQUENCE agent.feedback_id_seq TO agent_rw;
    """),
    # 2026-10-18: tracing
    ("trace ids", """
        ALTER TABLE agent.turns ADD COLUMN IF NOT EXISTS trace_id text;
        ALTER TABLE agent.turns ADD COLUMN IF NOT EXISTS span_id text;
        ALTER TABLE agent.audit_log ADD COLUMN IF NOT EXISTS trace_id text;
    """),
    # 2026-10-18: dense-only policy search dropped the lexical columns from schema.sql; a laptop database built
    # before that still had text_norm NOT NULL, so the new ingest failed on insert
    ("policy.clauses: lexical columns gone", """
        DROP INDEX IF EXISTS policy.clauses_tsv_idx;
        ALTER TABLE policy.clauses DROP COLUMN IF EXISTS tsv;
        ALTER TABLE policy.clauses DROP COLUMN IF EXISTS text_norm;
    """),
    # 2026-10-18: input guardrail
    ("agent.turns: guardrail", "ALTER TABLE agent.turns ADD COLUMN IF NOT EXISTS guardrail jsonb;"),
    # 2026-10-19: time to first token per turn (the latency a voice channel would feel)
    ("agent.turns: first_token_ms", "ALTER TABLE agent.turns ADD COLUMN IF NOT EXISTS first_token_ms int;"),
    # 2026-10-18: new bookings: a booking quote has no PNR until it is executed
    ("reservations.quotes: pnr nullable", "ALTER TABLE reservations.quotes ALTER COLUMN pnr DROP NOT NULL;"),
    # 2026-10-22: per-passenger cancellation and the five use cases (decision #60)
    ("reservations.passengers: cancelled_at", "ALTER TABLE reservations.passengers ADD COLUMN IF NOT EXISTS cancelled_at timestamptz;"),
    ("agent.users: shown", "ALTER TABLE agent.users ADD COLUMN IF NOT EXISTS shown boolean NOT NULL DEFAULT true;"),
    # 2026-10-18: itinerary by email
    ("reservations.email_outbox", """
        CREATE TABLE IF NOT EXISTS reservations.email_outbox (
          id serial PRIMARY KEY, pnr text REFERENCES reservations.bookings(pnr), customer_id text NOT NULL, to_email text NOT NULL,
          subject text NOT NULL, body text NOT NULL, status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','sent','failed')),
          created_at timestamptz NOT NULL);
        GRANT SELECT, INSERT, UPDATE, DELETE ON reservations.email_outbox TO res_rw;
        GRANT USAGE, SELECT ON SEQUENCE reservations.email_outbox_id_seq TO res_rw;
    """),
    # 2026-10-21: documents are no longer attached in the chat (the care team requests them by email); the table goes
    ("reservations.attachments: removed", "DROP TABLE IF EXISTS reservations.attachments;"),
]


def main(dsn: str | None = None) -> None:
    with psycopg.connect(dsn or settings.dsn_admin, autocommit=True) as conn:
        exists = conn.execute("SELECT 1 FROM information_schema.schemata WHERE schema_name = 'agent'").fetchone()
        if not exists:
            raise SystemExit("No Falcon schema in this database yet: run db/apply.py (scripts/reset.sh) first.")
        for name, sql in MIGRATIONS:
            try:
                conn.execute(sql)
                print(f"ok  {name}")
            except psycopg.Error as exc:
                if "best effort" in name:
                    print(f"--  {name}: left as is ({str(exc).splitlines()[0]})")
                else:
                    raise
    print("schema is current")


if __name__ == "__main__":
    main()
