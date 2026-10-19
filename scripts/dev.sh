#!/usr/bin/env bash
# Start / stop / status for all Falcon services locally (no Docker needed).
#   scripts/dev.sh start [service...]   scripts/dev.sh stop [service...]   scripts/dev.sh status   scripts/dev.sh restart [service...]
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
mkdir -p .run

declare -A CMD PORT
CMD[legacy]="uvicorn falcon.services.legacy_flight_api.app:app --host 0.0.0.0 --port 8010"; PORT[legacy]=8010
CMD[reservations]="python -m falcon.services.mcp_reservations.server";                      PORT[reservations]=8001
CMD[flightops]="python -m falcon.services.mcp_flightops.server";                            PORT[flightops]=8002
CMD[policy]="python -m falcon.services.mcp_policy.server";                                  PORT[policy]=8003
CMD[runtime]="uvicorn falcon.services.runtime.app:app --host 0.0.0.0 --port 8000";          PORT[runtime]=8000
ORDER=(legacy reservations flightops policy runtime)
PGDATA_LOCAL=${FALCON_PGDATA:-/var/lib/postgresql/falcon}
PGBIN=/usr/lib/postgresql/16/bin

ensure_postgres() {
  # Local cluster (workspace without Docker). With Docker, `docker compose up db` replaces this.
  if [ -d "$PGDATA_LOCAL" ] && ! sudo -u postgres $PGBIN/pg_ctl -D "$PGDATA_LOCAL" status >/dev/null 2>&1; then
    sudo -u postgres $PGBIN/pg_ctl -D "$PGDATA_LOCAL" -l "$PGDATA_LOCAL/server.log" start >/dev/null 2>&1 && echo "started postgres (:5433)"
    sleep 2
  fi
}

stop_one() { local p=${PORT[$1]}; fuser -k -n tcp "$p" >/dev/null 2>&1 && echo "stopped $1 (:$p)" || true; }
start_one() { stop_one "$1"; nohup ${CMD[$1]} > ".run/$1.log" 2>&1 & echo "started $1 (:${PORT[$1]}) log .run/$1.log"; }
status_one() { if fuser -n tcp "${PORT[$1]}" >/dev/null 2>&1; then echo "$1: up (:${PORT[$1]})"; else echo "$1: down"; fi; }
migrate() {
  # schema changes land as idempotent statements in db/migrate.py (FAILURE_DIARY #38): a code change never needs a reset
  local out
  if out=$(python db/migrate.py 2>&1); then echo "${out##*$'\n'}"; else echo "$out" | tail -3; echo "schema migration failed (see above); starting the services anyway"; fi
}

action=${1:-status}; shift || true
targets=("$@"); [ ${#targets[@]} -eq 0 ] && targets=("${ORDER[@]}")
case "$action" in
  start)   ensure_postgres; migrate; for s in "${targets[@]}"; do start_one "$s"; done; sleep 3; for s in "${targets[@]}"; do status_one "$s"; done ;;
  stop)    for s in "${targets[@]}"; do stop_one "$s"; done ;;
  restart) ensure_postgres; migrate; for s in "${targets[@]}"; do start_one "$s"; done; sleep 3; for s in "${targets[@]}"; do status_one "$s"; done ;;
  status)  for s in "${targets[@]}"; do status_one "$s"; done ;;
  *) echo "usage: $0 start|stop|restart|status [service...]"; exit 1 ;;
esac
