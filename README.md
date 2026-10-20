# Falcon Airline agent

A customer-service agent for a fictional Dubai-based airline, built for the Salesforce FDE panel (23 Oct 2026). Bilingual
(English / Modern Standard Arabic), deterministic money and rules, identity enforced below the model, one MCP server
per airline system, a human in the loop for exceptions, and an evaluation harness that reports pass^k.

The design notes (architecture, the decision log, the failure diary and the status page) are working documents kept outside this
repository; this snapshot carries the code, the data, the evaluation sets and the scripts.

## Run it locally (no Docker)

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -e . pytest pytest-asyncio
cp .env.example .env                      # defaults work locally; add OPENROUTER_API_KEY for a real model
# Postgres 16 with pgvector on port 5433 (see scripts/dev.sh for the local cluster; or `docker compose up db`)
python db/apply.py                        # schema + roles + deterministic seed (frozen clock 2026-10-20 09:00 Gulf)
python scripts/ingest_policy.py           # Conditions of Carriage (Emirates, EN + AR PDFs), clause-linked, embedded with bge-m3 when the key is set
(cd web && npm install && npm run build)  # the web app the runtime serves
scripts/dev.sh start                      # legacy API 8010, MCP servers 8001-8003, runtime 8000
open http://127.0.0.1:8000  (access code: falcon)
```

Without a model key the runtime uses a scripted stand-in model, so every flow can be seen working (bookings, quotes,
confirmation button, logged cases). With `OPENROUTER_API_KEY` set, `LLM_MODEL` is used.

## Tests

```bash
python -m pytest tests/test_rules.py tests/test_policy_parse.py -q   # no services needed
python -m pytest tests/test_reservations_live.py -q                    # MCP servers + legacy API (services running)
python -m pytest tests/test_runtime_live.py -q                         # end to end through the HTTP API, scripted model
python -m falcon.eval.runner --fake --k 1                              # evaluation harness self-test
python -m falcon.eval.runner --k 4                                     # the real evaluation (needs a model key)
```

## Run it with Docker (recommended on Windows)

Needs Docker Desktop running. From the project folder (Git Bash, WSL or any shell):

```bash
scripts/reset.sh      # first time, or to put the demo data back: wipe DB, build, schema + seed + policy, start
scripts/up.sh         # every other time, including after pulling new code: rebuild if needed, migrate the schema in place, start
scripts/up.sh --trace # the same, plus the Phoenix trace UI on http://localhost:6006
```

Then open http://localhost:8000 and pick a customer in the left sidebar (no access code). Only the runtime is published (port 8000); the MCP servers
and the legacy API are internal to the compose network. Postgres is also published on 5433 for a DB tool.

- `scripts/reset.sh --data` resets only the data (no rebuild); `--clean` also rebuilds the image from zero
- `scripts/up.sh --logs` starts and follows the runtime log; `--no-build` never rebuilds
- `scripts/eval.sh --k 1 --filter cancel_saver` runs the evaluation harness against the running stack (reports in `eval/results/`)
- Stop: `docker compose down` (data is kept; add `-v` to delete the database volume too)
- Real model: put `OPENROUTER_API_KEY` in `.env`, then `scripts/up.sh`

The scripts are thin wrappers over compose; the underlying commands are `docker compose build`,
`docker compose run --rm init` (schema + roles + seed + policy index) and `docker compose up -d`.

Cloud deployment is not part of the demo (decision #42): the same image runs anywhere a container, Postgres with pgvector
and an OTLP collector are available; identity between services is the dev secret locally or Entra ID (`AUTH_MODE=entra`).

## Voice agent

The same agent on a call (Pipecat): the browser's microphone goes over WebRTC to a voice bot that transcribes with Deepgram Nova-3
(Arabic or English, one per call, the app's language), sends each finished turn to the runtime's `/api/chat` with the caller's own
session token, and speaks the reviewed reply with ElevenLabs. It is a front end, not a second agent: the guard, the tools, the
reviewer, the confirmation gate and the traces are the chat's. What is spoken follows four fixed rules: only the reviewed reply
(never the token stream); a fixed progress phrase when the reply is slow; the confirmation card read aloud, answered with a bare
"yes" / «نعم» (anything else repeats it once, then declines); codes and amounts normalised for speech. In the app, the
conversation has a Chat | Voice switch; the voice view is a call screen with live meters, and every spoken turn also appears in the
chat with its badges and trace id.

```bash
# DEEPGRAM_API_KEY and ELEVENLABS_API_KEY in .env (see .env.example), then:
scripts/up.sh --voice   # the stack plus the voice bot on the laptop (http://localhost:7860; WebRTC media cannot enter Docker Desktop on Windows)
```

Then pick a customer in the app, switch to Voice and press the microphone (Chrome; allow the microphone). Tools: `scripts/voice_talk.py`
(the voice logic over text, no audio), `python -m falcon.eval.voice_tasks` (five tasks through the voice path, graded like the chat
tasks), `scripts/voice_latency.py` and `scripts/voice_stt_compare.py` (latency and word error rate from a call's log).

## Layout

```
db/            schema.sql (4 schemas, least-privilege roles), seed.py (Dubai network, personas), apply.py
data/policy/   the two Conditions of Carriage PDFs (EN, AR)
src/falcon/core        config, clock, db, auth, rules engine, Arabic helpers, MCP server plumbing
src/falcon/services    legacy_flight_api, mcp_reservations, mcp_flightops, mcp_policy, runtime (FastAPI + LangGraph), voice (Pipecat bot)
src/falcon/policy      parse_coc (EN/AR PDFs, ligature repair), search (dense, pgvector), rewrite (measured, off), embeddings
src/falcon/eval        tasks, simulator, runner, scoring (pass^k)
eval/tasks/            task YAML (EN, AR, adversarial)
scripts/               reset.sh, up.sh, eval.sh, talk.py (Docker), dev.sh (no Docker), ingest_policy.py
web/                   React/Vite/Tailwind app (EN/AR, RTL; Chat and Voice views)
tests/                 unit and live tests
```

Falcon Airline is fictional. The policy text is Emirates' public Conditions of Carriage for Passengers and Baggage
(`data/policy/coc_en.pdf`, `coc_ar.pdf`), used with credit for demonstration only; the fare rules, network and data
are Falcon's own and synthetic.
