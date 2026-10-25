# Shared helpers for scripts/reset.sh and scripts/up.sh (sourced, not executed).

step() { printf '\n==> %s\n' "$*"; }

need_docker() {
  if docker compose version >/dev/null 2>&1; then
    DC="docker compose"
  elif command -v docker-compose >/dev/null 2>&1; then
    DC="docker-compose"
  else
    echo "Docker Compose was not found. Install Docker Desktop and start it, then run this again."; exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "Docker is installed but the daemon is not reachable. Start Docker Desktop and wait until it says it is running."; exit 1
  fi
}

ensure_env() {
  if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env from .env.example (defaults work locally; add OPENROUTER_API_KEY to use a real model)."
  fi
}

wait_for_db() {
  local i
  for i in $(seq 1 40); do
    if $DC exec -T db pg_isready -U postgres >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "Postgres did not become ready in 80 s. Check: $DC logs db"; exit 1
}

db_is_seeded() {
  local n
  n=$($DC exec -T db psql -U postgres -d falcon -tAc "select count(*) from agent.users" 2>/dev/null | tr -d '[:space:]' || true)
  [ -n "$n" ] && [ "$n" != "0" ]
}

wait_for_runtime() {
  local i
  for i in $(seq 1 45); do
    if curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; then return 0; fi
    sleep 2
  done
  echo "The runtime did not answer on http://localhost:8000 within 90 s. Check: $DC logs runtime"; exit 1
}
