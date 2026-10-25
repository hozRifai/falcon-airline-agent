#!/usr/bin/env bash
# Run the evaluation harness against the running Docker stack. Arguments go to falcon.eval.runner.
#
#   scripts/eval.sh --fake --k 1                      # harness self-test, scripted model, no cost
#   scripts/eval.sh --k 1 --filter cancel_saver       # first paid smoke test: two short conversations (cents)
#   scripts/eval.sh --k 1                             # all tasks once (cents on DeepSeek)
#   scripts/eval.sh --k 1 --lang ar                   # Arabic tasks only
#
# Reports land in eval/results/<timestamp>/report.md (+ runs.jsonl with every transcript).
# The runner resets the demo data before each task through the runtime's admin endpoint, so run it
# when nobody is clicking through the app.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/_docker_common.sh"
need_docker
ensure_env
mkdir -p eval/results
if ! curl -fsS http://localhost:8000/api/health >/dev/null 2>&1; then
  echo "The stack is not running. Start it first: scripts/up.sh"; exit 1
fi
# --build: pick up scorer/task edits (the eval image is otherwise built once and reused stale)
# --no-deps: without it compose compares the running services with its own view of their config (no OTEL endpoint when run from
# eval.sh) and silently recreates them: tracing off, conversations reset (FAILURE_DIARY #55).
$DC run --rm --build --no-deps eval python -m falcon.eval.runner "$@"
latest=$(ls -1dt eval/results/*/ 2>/dev/null | head -1 || true)
if [ -n "$latest" ]; then
  echo
  echo "Report: ${latest}report.md"
fi
