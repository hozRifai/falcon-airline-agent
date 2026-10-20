"""Compare the speech-to-text services of one voice call: the one that drove the call and the shadow (decision #55).

Reads the bot's log (.voice.log), takes the last call (or --call N, counting from 1), groups the final transcripts by service, and
scores each service's whole transcript against the sentences that were read (eval/voice/stt_script.yaml) with WER and CER after
the same normalisation for everyone (falcon.services.voice.wer). No audio is sent anywhere: the two services already heard the call.

    .venv-voice/Scripts/python scripts/voice_stt_compare.py --lang ar
    .venv-voice/Scripts/python scripts/voice_stt_compare.py --lang en --call 2

Start the bot with a shadow first:  VOICE_STT_SHADOW=elevenlabs scripts/voice.sh   (or put it in .env)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from falcon.services.voice.wer import error_rates  # noqa: E402

CALL = re.compile(r"voice stt: (\S+?)(?:, shadow (\S+))? \((\w+)\)")
LINE = re.compile(r"^(\S+ \S+) .*voice transcript \[([^\]]+)\]: confidence mean (\S+) lowest (\S+): (.*)$")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def calls(log: Path) -> list[dict]:
    out: list[dict] = []
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        line = ANSI.sub("", raw)
        if m := CALL.search(line):
            out.append({"main": m.group(1), "shadow": m.group(2), "lang": m.group(3), "lines": []})
        elif (m := LINE.match(line)) and out:
            out[-1]["lines"].append({"at": m.group(1), "stt": m.group(2), "mean": m.group(3), "low": m.group(4), "text": m.group(5)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=str(ROOT / ".voice.log"))
    ap.add_argument("--lang", choices=["ar", "en"], help="which sentence list was read (default: the call's language)")
    ap.add_argument("--call", type=int, help="1-based call number in the log (default: the last one)")
    a = ap.parse_args()
    found = calls(Path(a.log))
    if not found:
        sys.exit("no call in the log yet")
    call = found[(a.call or len(found)) - 1]
    lang = a.lang or call["lang"]
    script = yaml.safe_load((ROOT / "eval" / "voice" / "stt_script.yaml").read_text(encoding="utf-8"))[lang]
    reference = " ".join(script)
    print(f"call {a.call or len(found)} of {len(found)}: main {call['main']}, shadow {call['shadow'] or '-'}, language {lang}\n")
    for ln in call["lines"]:
        print(f"  {ln['at'][-12:]}  [{ln['stt']:<18}] {ln['text']}")
    print()
    for stt in dict.fromkeys(ln["stt"] for ln in call["lines"]):
        text = " ".join(ln["text"] for ln in call["lines"] if ln["stt"] == stt)
        wer, cer = error_rates(reference, text, lang)
        n = sum(ln["stt"] == stt for ln in call["lines"])
        print(f"  {stt:<20} WER {wer:6.1%}   CER {cer:6.1%}   ({n} final transcripts)")
    print("\nWER counts whole words, CER characters; both include anything said that is not in the script (the agent's own answers are")
    print("not transcribed, but remarks of yours are). The plan's bar: Arabic WER above 15% means voice is English-only or not shown.")


if __name__ == "__main__":
    main()
