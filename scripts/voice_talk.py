"""Talk to the running agent through the voice channel's text path: the same CallSession the Pipecat bot uses, minus the audio.

Each utterance goes through the four rules (done-only speech, progress phrase, spoken card, normaliser) and the script prints what
the voice would say, with the time from the end of the utterance to the first spoken line (what the caller waits through, before
STT finalisation and TTS start are added).

    .venv-voice/Scripts/python scripts/voice_talk.py --as u-fatima --lang ar "أريد إلغاء حجزي K3P7QZ" "نعم"

PYTHONPATH=src is set by the script itself. No audio, no Deepgram or ElevenLabs spend; the agent's model is spent as in the chat.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402

from falcon.services.voice.call import CallSession, RuntimeClient  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--as", dest="user", default="u-fatima")
    ap.add_argument("--lang", default="ar")
    ap.add_argument("--url", default=os.getenv("VOICE_RUNTIME_URL", "http://localhost:8000"))
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument("utterances", nargs="+")
    a = ap.parse_args()
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as http:
        r = await http.post(f"{a.url}/api/demo/switch", json={"user_id": a.user, "reset": not a.no_reset})
        r.raise_for_status()
        thread = f"voice-{uuid.uuid4().hex[:8]}"
        state = {"t0": 0.0, "first": None}

        async def speak(text: str, kind: str) -> None:
            dt = time.perf_counter() - state["t0"]
            if state["first"] is None:
                state["first"] = dt
            print(f"  [{kind} +{dt:.1f}s] {text}")

        async def observe(kind: str, payload: dict) -> None:
            if kind == "done":
                print(f"  (done: turn {payload.get('turn_id')}, trace {payload.get('trace_id')}, pending {bool(payload.get('pending'))}, "
                      f"verification {(payload.get('verification') or {}).get('status')})")

        call = CallSession(RuntimeClient(a.url, r.json()["token"], thread, a.lang, http), a.lang, speak, observe)
        print(f"{a.user} on thread {thread} ({a.lang})")
        for u in a.utterances:
            print(f"> {u}")
            state["t0"], state["first"] = time.perf_counter(), None
            await call.handle(u)
            print(f"  end of utterance -> first spoken line: {state['first'] if state['first'] is not None else float('nan'):.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
