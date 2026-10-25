-- Falcon Airline demo database.
-- One Postgres instance, FOUR SYSTEMS as four schemas, each with its own role and no cross-schema grants.
-- In production these are different vendors (a PSS, an ops system, a knowledge base); the agent would not
-- notice, because the MCP servers are the seam. Here the seam is the role: res_rw cannot SELECT from
-- flightops, ops_api cannot SELECT from reservations, policy_ro can only read policy, and the agent's own
-- role only sees the agent schema.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS unaccent;

DROP SCHEMA IF EXISTS agent CASCADE;
DROP SCHEMA IF EXISTS policy CASCADE;
DROP SCHEMA IF EXISTS flightops CASCADE;
DROP SCHEMA IF EXISTS reservations CASCADE;

CREATE SCHEMA reservations;
CREATE SCHEMA flightops;
CREATE SCHEMA policy;
CREATE SCHEMA agent;

-- ---------------------------------------------------------------- roles (passwords set by db/apply.py)
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'res_rw')    THEN CREATE ROLE res_rw LOGIN;    END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ops_api')   THEN CREATE ROLE ops_api LOGIN;   END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'policy_ro') THEN CREATE ROLE policy_ro LOGIN; END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_rw')  THEN CREATE ROLE agent_rw LOGIN;  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'audit_w')   THEN CREATE ROLE audit_w LOGIN;   END IF;
END $$;

-- ================================================================ RESERVATIONS (the booking system)
CREATE TABLE reservations.airports (
  code           char(3) PRIMARY KEY,
  name_en        text NOT NULL,
  name_ar        text NOT NULL,
  city_en        text NOT NULL,
  city_ar        text NOT NULL,
  country        text NOT NULL,
  tz             text NOT NULL,                     -- IANA zone for local times ("Europe/London": the schedule crosses the DST change)
  serves_cities  text[] NOT NULL DEFAULT '{}',      -- lower-case city names a customer may type ("dubai" -> DXB)
  road_alt_code  char(3),                            -- the other Dubai airport reachable by road (DXB <-> DWC)
  road_km        int,
  road_minutes   int
);

CREATE TABLE reservations.fare_families (
  code                   text PRIMARY KEY,           -- SPECIAL | SAVER | FLEX
  name_en                text NOT NULL,
  name_ar                text NOT NULL,
  bag_kg                 int NOT NULL,
  free_changes           int NOT NULL,
  max_changes            int NOT NULL,
  change_fee             numeric(10,2) NOT NULL,
  refund_credit_allowed  boolean NOT NULL,
  refund_credit_fee      numeric(10,2) NOT NULL,
  refund_cash_allowed    boolean NOT NULL,
  refund_cash_fee        numeric(10,2) NOT NULL,
  deadline_hours         int NOT NULL,
  price_multiplier       numeric(6,3) NOT NULL,
  sort_order             int NOT NULL
);

CREATE TABLE reservations.flights (                   -- the timetable
  flight_number  text PRIMARY KEY,
  origin         char(3) NOT NULL REFERENCES reservations.airports(code),
  destination    char(3) NOT NULL REFERENCES reservations.airports(code),
  dep_time       time NOT NULL,                       -- local time at origin
  duration_min   int NOT NULL,
  days_of_week   int[] NOT NULL                       -- 0=Mon .. 6=Sun
);

CREATE TABLE reservations.flight_instances (          -- one row per flight per day
  id             text PRIMARY KEY,                    -- FA123-2026-10-21
  flight_number  text NOT NULL REFERENCES reservations.flights(flight_number),
  service_date   date NOT NULL,
  origin         char(3) NOT NULL,
  destination    char(3) NOT NULL,
  dep_at         timestamptz NOT NULL,
  arr_at         timestamptz NOT NULL,
  seats_total    int NOT NULL,
  seats_sold     int NOT NULL DEFAULT 0,
  base_price     numeric(10,2) NOT NULL               -- SPECIAL fare per passenger; other families = base * multiplier
);
CREATE INDEX ON reservations.flight_instances (origin, destination, service_date);

CREATE TABLE reservations.customers (
  customer_id    text PRIMARY KEY,                    -- C-1001
  first_name     text NOT NULL,
  last_name      text NOT NULL,
  name_ar        text,
  email          text NOT NULL,
  phone          text,
  preferred_lang text NOT NULL DEFAULT 'en'
);

CREATE TABLE reservations.bookings (
  pnr            text PRIMARY KEY,                    -- 6 chars
  customer_id    text NOT NULL REFERENCES reservations.customers(customer_id),
  status         text NOT NULL CHECK (status IN ('confirmed','cancelled','completed')),
  fare_family    text NOT NULL REFERENCES reservations.fare_families(code),
  currency       char(3) NOT NULL DEFAULT 'AED',
  total_paid     numeric(10,2) NOT NULL,
  changes_used   int NOT NULL DEFAULT 0,
  created_at     timestamptz NOT NULL,
  cancelled_at   timestamptz,
  cancel_reason  text,
  refund_method  text,
  refund_amount  numeric(10,2)
);
CREATE INDEX ON reservations.bookings (customer_id);

CREATE TABLE reservations.booking_segments (
  id                 serial PRIMARY KEY,
  pnr                text NOT NULL REFERENCES reservations.bookings(pnr),
  seq                int NOT NULL,
  flight_instance_id text NOT NULL REFERENCES reservations.flight_instances(id),
  fare_per_pax       numeric(10,2) NOT NULL,
  UNIQUE (pnr, seq)
);
CREATE INDEX ON reservations.booking_segments (flight_instance_id);

CREATE TABLE reservations.passengers (
  id               serial PRIMARY KEY,
  pnr              text NOT NULL REFERENCES reservations.bookings(pnr),
  first_name       text NOT NULL,
  last_name        text NOT NULL,
  pax_type         text NOT NULL CHECK (pax_type IN ('adult','child','infant')),
  special_request  text,                               -- free text typed by customers: untrusted data (see eval injection task)
  cancelled_at     timestamptz                         -- cancelled on a booking that stays for the others (decision #60)
);

CREATE TABLE reservations.booking_extras (
  id            serial PRIMARY KEY,
  pnr           text NOT NULL REFERENCES reservations.bookings(pnr),
  passenger_id  int NOT NULL REFERENCES reservations.passengers(id),
  kind          text NOT NULL,
  kg            int NOT NULL,
  price         numeric(10,2) NOT NULL,
  created_at    timestamptz NOT NULL
);

CREATE TABLE reservations.transactions (
  id          serial PRIMARY KEY,
  pnr         text NOT NULL REFERENCES reservations.bookings(pnr),
  kind        text NOT NULL CHECK (kind IN ('payment','refund_cash','refund_credit','fee')),
  amount      numeric(10,2) NOT NULL,
  currency    char(3) NOT NULL DEFAULT 'AED',
  reference   text,
  created_at  timestamptz NOT NULL
);

CREATE TABLE reservations.email_outbox (              -- what the agent emailed the customer (decision #32); the demo's mail service is this table
  id             serial PRIMARY KEY,
  pnr            text REFERENCES reservations.bookings(pnr),
  customer_id    text NOT NULL,
  to_email       text NOT NULL,                       -- always the profile's address; the agent never takes one from the chat
  subject        text NOT NULL,
  body           text NOT NULL,
  status         text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued','sent','failed')),
  created_at     timestamptz NOT NULL
);
CREATE TABLE reservations.quotes (                    -- every write action must present one of these
  quote_id       text PRIMARY KEY,
  pnr            text REFERENCES reservations.bookings(pnr),  -- NULL for a booking quote until create_booking fills it in
  customer_id    text NOT NULL,
  kind           text NOT NULL,                       -- cancel | change | bag | waiver | book
  request        jsonb NOT NULL,
  result         jsonb NOT NULL,
  amount         numeric(10,2) NOT NULL,
  created_at     timestamptz NOT NULL,
  expires_at     timestamptz NOT NULL,
  consumed_at    timestamptz,
  consumed_result jsonb
);

CREATE TABLE reservations.approval_requests (         -- the human in the loop
  id             serial PRIMARY KEY,
  pnr            text REFERENCES reservations.bookings(pnr),
  customer_id    text NOT NULL,
  kind           text NOT NULL CHECK (kind IN ('waiver','handover','uncovered')),
  reason         text NOT NULL,
  proposed       jsonb,                               -- the quote that would execute if approved
  status         text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','approved','rejected')),
  created_at     timestamptz NOT NULL,
  decided_at     timestamptz,
  decided_by     text,
  decision_note  text,
  executed       jsonb
);

-- ================================================================ FLIGHT OPS (the operations system: its own copy of the timetable)
CREATE TABLE flightops.flights (
  flight_instance_id text PRIMARY KEY,
  flight_number      text NOT NULL,
  service_date       date NOT NULL,
  origin             char(3) NOT NULL,
  destination        char(3) NOT NULL,
  dep_at             timestamptz NOT NULL,
  arr_at             timestamptz NOT NULL,
  aircraft           text NOT NULL DEFAULT 'B777-300ER',
  seats_total        int NOT NULL
);
CREATE INDEX ON flightops.flights (flight_number, service_date);

CREATE TABLE flightops.flight_status (
  flight_instance_id text PRIMARY KEY REFERENCES flightops.flights(flight_instance_id),
  status             text NOT NULL CHECK (status IN ('scheduled','on_time','delayed','boarding','departed','landed','cancelled')),
  delay_minutes      int NOT NULL DEFAULT 0,
  reason             text,
  gate               text,
  updated_at         timestamptz NOT NULL
);

CREATE TABLE flightops.disruption_events (
  id                 serial PRIMARY KEY,
  flight_instance_id text NOT NULL REFERENCES flightops.flights(flight_instance_id),
  event_type         text NOT NULL,                   -- delay | cancellation | reinstated
  details            jsonb,
  created_by         text,
  created_at         timestamptz NOT NULL
);

-- ================================================================ POLICY (the knowledge base)
CREATE TABLE policy.documents (
  id            text PRIMARY KEY,                     -- gcoc-en | gcoc-ar
  lang          text NOT NULL,
  title         text NOT NULL,
  source_url    text,
  version_date  date,
  extraction    text,                                 -- how the text was obtained: web-text | pdf-text | ocr-tesseract | ocr-azure-di
  loaded_at     timestamptz NOT NULL
);

CREATE TABLE policy.clauses (
  id             serial PRIMARY KEY,
  doc_id         text NOT NULL REFERENCES policy.documents(id),
  clause_id      text NOT NULL,                       -- "10.2.1"; the key that links EN and AR
  article_no     int NOT NULL,
  article_title  text NOT NULL,
  lang           text NOT NULL,
  text           text NOT NULL,
  embedding      vector,                              -- bge-m3 (1024 dims) via scripts/ingest_policy.py; NULL for title-only rows
  char_len       int NOT NULL,
  UNIQUE (doc_id, clause_id)
);
CREATE INDEX ON policy.clauses (clause_id);

CREATE TABLE policy.glossary (
  term_en   text PRIMARY KEY,
  term_ar   text NOT NULL,
  note      text
);

-- ================================================================ AGENT (runtime state: users, memory, audit)
CREATE TABLE agent.users (                             -- demo identity provider
  user_id        text PRIMARY KEY,
  display_name   text NOT NULL,
  display_name_ar text,
  role           text NOT NULL CHECK (role IN ('customer')),  -- one persona (decision #25)
  customer_id    text,                                -- identity resolution: web user -> reservations customer
  lang           text NOT NULL DEFAULT 'en',
  persona_note_en text,
  persona_note_ar text,
  sort_order     int NOT NULL DEFAULT 100,
  shown          boolean NOT NULL DEFAULT true        -- in the app's use-case list; hidden users still sign in (the evaluation's personas)
);

CREATE TABLE agent.audit_log (
  id          bigserial PRIMARY KEY,
  ts          timestamptz NOT NULL DEFAULT now(),
  service     text NOT NULL,
  session_id  text,
  actor       text,
  tool        text NOT NULL,
  args        jsonb,
  ok          boolean NOT NULL,
  latency_ms  int,
  summary     text,
  trace_id    text                                    -- the agent's trace (decision #27): audit row <-> trace without a backend
);
CREATE INDEX ON agent.audit_log (session_id);

CREATE TABLE agent.turns (                              -- per model turn: tokens, latency, tool calls ("What I checked"), reviewer outcome
  id             bigserial PRIMARY KEY,
  ts             timestamptz NOT NULL DEFAULT now(),
  session_id     text NOT NULL,
  user_id        text,
  thread         text,
  lang_in        text,
  lang_out       text,
  model          text,
  input_tokens   int,
  output_tokens  int,
  cached_tokens  int,
  latency_ms     int,
  first_token_ms int,                                   -- time to the first streamed token (null when nothing streamed: a card, a hold, a refusal)
  tool_calls     jsonb,
  finish         text,
  verification   jsonb,                                 -- {status: verified|retried|held|cut_off, rounds: [...], amounts, evidence} (decision #26)
  guardrail      jsonb,                                 -- {label, action, source, redactions, ...} from the input guardrail (decision #29)
  trace_id       text,                                  -- OpenTelemetry trace of this turn (decision #27)
  span_id        text
);
CREATE INDEX ON agent.turns (session_id);

CREATE TABLE agent.feedback (                           -- thumbs up/down on a reply, as the customer saw it
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
CREATE INDEX ON agent.feedback (rating, ts);

CREATE TABLE agent.token_budget (                       -- daily spend cap for the public demo
  day           date PRIMARY KEY,
  input_tokens  bigint NOT NULL DEFAULT 0,
  output_tokens bigint NOT NULL DEFAULT 0
);
-- LangGraph's checkpointer creates its own tables in this schema (checkpoints, checkpoint_blobs, checkpoint_writes, checkpoint_migrations).

-- ================================================================ GRANTS: least privilege per system
GRANT USAGE ON SCHEMA reservations TO res_rw;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA reservations TO res_rw;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA reservations TO res_rw;

GRANT USAGE ON SCHEMA flightops TO ops_api;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA flightops TO ops_api;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA flightops TO ops_api;

GRANT USAGE ON SCHEMA policy TO policy_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA policy TO policy_ro;

GRANT USAGE ON SCHEMA agent TO agent_rw;
GRANT ALL ON ALL TABLES IN SCHEMA agent TO agent_rw;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agent TO agent_rw;
GRANT CREATE ON SCHEMA agent TO agent_rw;              -- LangGraph creates its checkpoint tables here
ALTER ROLE agent_rw SET search_path = agent, public;

-- the audit writer: INSERT on the audit log and nothing else. Every MCP server uses it for its audit
-- trail, so no business role needs any access to the agent schema.
GRANT USAGE ON SCHEMA agent TO audit_w;
GRANT INSERT ON agent.audit_log TO audit_w;
GRANT USAGE, SELECT ON SEQUENCE agent.audit_log_id_seq TO audit_w;

REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO res_rw, ops_api, policy_ro, agent_rw, audit_w;  -- for the vector type
