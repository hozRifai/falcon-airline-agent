"""What the caller waits through, per turn and as p50 / p95, from the voice bot's own log (docs/VOICE_PLAN.md §4).

For each turn of each call in .voice.log: end of the customer's speech to the first audio (Pipecat's UserBotLatencyObserver total,
which includes the end-of-turn pause, STT and TTS start), and to the start of the answer itself (the first audio plus the time
between the first spoken line and the reply, both measured from the turn being sent). The plan's bars: first audio below 4 s is
showable; policy turns above 8 s stay in chat.

    .venv-voice/Scripts/python scripts/voice_latency.py            # every call in the log
    .venv-voice/Scripts/python scripts/voice_latency.py --last 1   # the last call only
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
HEARD = re.compile(r"heard \((\w+)\): (.*)$")
TIMING = re.compile(r"voice timing: (\w+) at \+([\d.]+)s")
TOTAL = re.compile(r"voice latency:\s+([\d.]+)s\s+TOTAL$")
CALL = re.compile(r"voice stt: ")


def turns(log: Path, last: int | None) -> list[dict]:
    calls: list[list[dict]] = []
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw).rstrip()
        if CALL.search(line):
            calls.append([])
        elif (m := HEARD.search(line)) and calls:
            calls[-1].append({"text": m.group(2), "first_audio": None, "lines": []})
        elif (m := TOTAL.search(line)) and calls and calls[-1] and calls[-1][-1]["first_audio"] is None:
            calls[-1][-1]["first_audio"] = float(m.group(1))
        elif (m := TIMING.search(line)) and calls and calls[-1]:
            calls[-1][-1]["lines"].append((m.group(1), float(m.group(2))))
    picked = calls[-last:] if last else calls
    out = []
    for t in (t for c in picked for t in c):
        if t["first_audio"] is None or not t["lines"]:
            continue
        first_line = t["lines"][0][1]
        reply = next((after for kind, after in t["lines"] if kind in ("reply", "card", "error")), None)
        t["answer"] = round(t["first_audio"] + (reply - first_line), 2) if reply is not None else None
        out.append(t)
    return out


def pct(xs: list[float], p: int) -> float:
    xs = sorted(xs)
    return xs[0] if len(xs) == 1 else statistics.quantiles(xs, n=100, method="inclusive")[p - 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / ".voice.log"))
    ap.add_argument("--last", type=int, help="only the last N calls")
    a = ap.parse_args()
    ts = turns(Path(a.log), a.last)
    if not ts:
        raise SystemExit("no measured turn in the log yet (a call with the current bot writes them)")
    for t in ts:
        print(f"  first audio {t['first_audio']:5.2f} s   answer {t['answer'] if t['answer'] is not None else '-':>5} s   {t['text'][:70]}")
    fa = [t["first_audio"] for t in ts]
    an = [t["answer"] for t in ts if t["answer"] is not None]
    print(f"\n{len(ts)} turns.  End of speech -> first audio: p50 {pct(fa, 50):.2f} s, p95 {pct(fa, 95):.2f} s (bar: below 4 s).")
    if an:
        print(f"End of speech -> answer starts: p50 {pct(an, 50):.2f} s, p95 {pct(an, 95):.2f} s (policy turns above 8 s stay in chat).")


if __name__ == "__main__":
    main()
