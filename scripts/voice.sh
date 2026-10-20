#!/usr/bin/env bash
# The voice channel (docs/VOICE_PLAN.md): the Pipecat bot, run on the laptop next to the Docker stack (decision #49: WebRTC media
# cannot reach a container through Docker Desktop on Windows, so this one process lives outside it and calls the runtime on :8000).
#
#   scripts/voice.sh            # first run creates .venv-voice (Python 3.11+, Pipecat 1.13); then http://localhost:7860/client
#   scripts/voice.sh --stop     # stop a bot started by scripts/up.sh --voice
#
# Settings come from .env: DEEPGRAM_API_KEY and ELEVENLABS_API_KEY (required), VOICE_LANG (ar | en, default ar),
# VOICE_DEMO_USER (default u-fatima) and VOICE_RESET (1: put the demo data back when a call starts from Pipecat's page),
# ELEVENLABS_VOICE_AR / ELEVENLABS_VOICE_EN, VOICE_PROBE=1 (log the microphone level and every transcript).
# VOICE_PYTHON picks the interpreter that creates the venv (default: the first of py -3.13 / -3.12 / -3.11, python3).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VENV="$ROOT/.venv-voice"
PIDFILE="$ROOT/.voice.pid"

if [ "${1:-}" = "--stop" ]; then
  if [ -f "$PIDFILE" ]; then kill "$(cat "$PIDFILE")" 2>/dev/null || true; rm -f "$PIDFILE"; echo "voice bot stopped"; else echo "no voice bot running"; fi
  exit 0
fi

for key in DEEPGRAM_API_KEY ELEVENLABS_API_KEY; do
  if ! grep -Eq "^${key}=.+" .env; then echo "$key is missing in .env (docs/VOICE_PLAN.md)"; exit 1; fi
done

PY="$VENV/Scripts/python"; [ -x "$PY" ] || PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  BASE="${VOICE_PYTHON:-}"
  if [ -z "$BASE" ]; then
    for v in 3.13 3.12 3.11; do if py -"$v" -c "" >/dev/null 2>&1; then BASE="py -$v"; break; fi; done
  fi
  BASE="${BASE:-python3}"
  echo "Creating .venv-voice with $BASE (Pipecat needs Python 3.11 or newer)"
  $BASE -m venv "$VENV"
  PY="$VENV/Scripts/python"; [ -x "$PY" ] || PY="$VENV/bin/python"
  "$PY" -m pip install -q --upgrade pip
  # the voice extra of pyproject.toml, plus the two libraries the voice modules import from the main dependency list
  "$PY" -m pip install -q "pipecat-ai[webrtc,deepgram,elevenlabs,runner]==1.13.0" httpx num2words
fi

# PYTHONUTF8: the Windows console is cp1252 and Pipecat's start banner crashes it (FAILURE_DIARY #64).
# PYTHONPATH instead of an install: the project needs Python 3.12+, the voice venv may be 3.11, and only falcon.core and
# falcon.services.voice are imported here (no LangGraph).
export PYTHONUTF8=1 PYTHONPATH="$ROOT/src"
exec "$PY" -m falcon.services.voice.bot -t webrtc "$@"
