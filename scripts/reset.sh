#!/usr/bin/env bash
# Falcon from scratch: wipe the database, build the images, create schema + roles + demo data, load the
# policy, start everything. Use it the first time, and whenever you want the demo data back to its
# designed state (persona bookings untouched, live-demo flight scheduled again).
#
#   scripts/reset.sh            # full reset (keeps Docker's build cache: fast when the code hasn't changed)
#   scripts/reset.sh --clean    # also rebuild the images from zero (slow: re-downloads node + python deps)
#   scripts/reset.sh --data     # only reset the data (schema + seed + policy); no rebuild, services restarted
#
# Needs Docker Desktop running. Works from Git Bash or WSL on Windows, and from any shell on macOS/Linux.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_docker_common.sh"

MODE=full
case "${1:-}" in
  --clean) MODE=clean ;;
  --data) MODE=data ;;
  "") ;;
  *) echo "usage: scripts/reset.sh [--clean|--data]"; exit 2 ;;
esac

need_docker
ensure_env

if [ "$MODE" = data ]; then
  step "Stopping the services (database kept)"
  $DC stop runtime reservations flightops policy legacy >/dev/null 2>&1 || true
else
  step "Removing containers and the database volume"
  $DC down -v --remove-orphans
  if [ "$MODE" = clean ]; then
    step "Building the image from zero (5-10 min the first time)"
    $DC build --no-cache
  else
    step "Building the image (cached layers are reused)"
    $DC build
  fi
fi

step "Starting Postgres"
$DC up -d db
wait_for_db

step "Schema + roles + deterministic seed + Conditions of Carriage (EN/AR)"
$DC run --rm --build init   # --build: the init image sits behind a profile and is not rebuilt by `$DC build` (FAILURE_DIARY #40)

step "Starting the services"
$DC up -d
wait_for_runtime

echo
echo "Ready."
echo "  App:          http://localhost:8000   $( [ -n "$(access_code)" ] && echo "(access code: $(access_code))" || echo "(no access code: pick a customer)")"
echo "  Postgres:     localhost:5433 (user postgres / password from .env)"
echo "  Logs:         $DC logs -f runtime"
echo "  Stop:         $DC down          (keeps the data; next time just run scripts/up.sh)"
echo "  Reset data:   scripts/reset.sh --data"
