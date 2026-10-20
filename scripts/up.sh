#!/usr/bin/env bash
# Falcon for a normal working session: start the containers against the data you already have.
# Code changes are picked up (the image is rebuilt from cache, seconds when nothing changed) and the database schema is
# brought up to date in place (db/migrate.py, idempotent): a code change never needs scripts/reset.sh. Reset is for
# wiping the data (or the first run).
#
#   scripts/up.sh               # start (or restart) everything, rebuild if the code changed
#   scripts/up.sh --no-build    # start only, never rebuild
#   scripts/up.sh --logs        # start, then follow the runtime log (Ctrl-C leaves it running)
#   scripts/up.sh --trace       # also start the trace UI (Phoenix, http://localhost:6006) and export every service's spans to it
#   scripts/up.sh --voice       # also start the voice bot on the laptop (scripts/voice.sh, http://localhost:7860/client; decision #49)
#
# If the database has never been seeded (first time, or after `docker compose down -v`), this script
# says so and points you to scripts/reset.sh instead of starting a broken stack.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_docker_common.sh"

BUILD=1
LOGS=0
TRACE=0
VOICE=0
for arg in "$@"; do
  case "$arg" in
    --no-build) BUILD=0 ;;
    --logs) LOGS=1 ;;
    --trace) TRACE=1 ;;
    --voice) VOICE=1 ;;
    *) echo "usage: scripts/up.sh [--no-build] [--logs] [--trace] [--voice]"; exit 2 ;;
  esac
done
need_docker
ensure_env
if [ "$TRACE" = 1 ]; then
  # decision #27: the services read OTEL_EXPORTER_OTLP_ENDPOINT; inside the compose network Phoenix is "phoenix:6006".
  # After need_docker: it sets $DC, and the script runs with set -u (FAILURE_DIARY #52).
  export OTEL_EXPORTER_OTLP_ENDPOINT="${OTEL_EXPORTER_OTLP_ENDPOINT:-http://phoenix:6006}"
  export TRACE_UI_URL="${TRACE_UI_URL:-http://localhost:6006}"
  DC="$DC --profile observability"
fi

step "Starting Postgres"
$DC up -d db
wait_for_db

if ! db_is_seeded; then
  echo
  echo "The database has no demo data yet (or it was wiped). Run the first-time setup instead:"
  echo "    scripts/reset.sh"
  exit 1
fi

if [ "$BUILD" = 1 ]; then
  step "Building the image if the code changed"
  $DC build
fi
step "Bringing the database schema up to date (idempotent; data is kept)"
# --build: `init` is behind a compose profile, so `$DC build` above skips it and `run` would reuse a stale image
# (on the laptop it still had the 17 Oct ingest and no db/migrate.py; FAILURE_DIARY #40).
$DC run --rm --build --no-deps init python db/migrate.py
step "Starting the services"
# A container that was stopped while Docker Desktop recreated the compose network keeps the old network id and fails to start with
# "network ... not found" (FAILURE_DIARY #54). Stopped containers are free to recreate (data lives in volumes), so drop them first.
$DC ps -a --status exited -q | xargs -r docker rm -f >/dev/null 2>&1 || true
$DC up -d
wait_for_runtime

echo
echo "Ready."
echo "  App:    http://localhost:8000   $( [ -n "$(access_code)" ] && echo "(access code: $(access_code))" || echo "(no access code: pick a customer)")"
echo "  Logs:   $DC logs -f runtime"
echo "  Stop:   $DC down                (data is kept)"
if [ "$TRACE" = 1 ]; then echo "  Traces: http://localhost:6006   (Phoenix; every turn is a trace named chat.turn)"; fi
if [ "$VOICE" = 1 ]; then
  # outside Docker (decision #49); a bot from an earlier up.sh --voice is stopped first so the new code runs
  "$ROOT/scripts/voice.sh" --stop >/dev/null
  # appended, not overwritten: a restart must not erase the previous calls (scripts/voice_stt_compare.py and the latency summary read it)
  echo "===== voice bot started $(date -Iseconds) =====" >> "$ROOT/.voice.log"
  nohup "$ROOT/scripts/voice.sh" >> "$ROOT/.voice.log" 2>&1 &
  echo $! > "$ROOT/.voice.pid"
  echo "  Voice:  http://localhost:7860/client   (log: .voice.log; stop: scripts/voice.sh --stop)"
fi
if [ "$LOGS" = 1 ]; then
  echo
  $DC logs -f runtime
fi
