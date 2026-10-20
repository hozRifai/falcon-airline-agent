"""Voice task set (docs/VOICE_PLAN.md §4, phase 3): five of the existing Arabic tasks run through the voice channel's text path.

Each task's fixed script goes, utterance by utterance, through the same CallSession the Pipecat bot uses, so the four voice rules
are in the loop: only the reviewed reply is spoken, at most two timed progress phrases per turn, the confirmation card is spoken
and a spoken "نعم" presses confirm. Grading is the harness's own `score()` (the same database checks, mentions and metrics as the
chat run), plus the voice rules checked on what would have been spoken. Audio is not synthesised here (STT accuracy is measured on
Hozayfa's reading calls, scripts/voice_stt_compare.py); the latencies are from the turn being sent to each spoken line.

    docker compose run --rm --no-deps eval python -m falcon.eval.voice_tasks --filter cancel_saver   # smoke one first
    docker compose run --rm --no-deps eval python -m falcon.eval.voice_tasks                          # all five; eval/results/voice-<ts>/
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

from falcon.core.config import settings
from falcon.eval.runner import RESULTS, reset_db
from falcon.eval.scoring import score
from falcon.eval.tasks import load_tasks
from falcon.services.voice.call import CallSession, RuntimeClient
from falcon.services.voice.speech import normalise

YES = {"ar": "نعم", "en": "yes"}
VOICE_TASKS = ["cancel_saver_credit_ar", "cancel_special_wants_cash_ar", "airline_cancelled_full_refund_ar", "pregnant_38th_month_ar",
               "prompt_injection_direct_ar"]


async def run_one(task, http: httpx.AsyncClient) -> dict:
    reset_db()
    base = settings.runtime_public_url
    r = await http.post(f"{base}/api/auth/demo-login", json={"user_id": task.persona, "access_code": settings.demo_access_code},
                        headers={"X-Admin-Token": settings.admin_token})
    r.raise_for_status()
    headers = {"X-Admin-Token": settings.admin_token, "X-Falcon-Tags": f"eval,voice,task:{task.id}"}
    runtime = RuntimeClient(base, r.json()["token"], f"voice-eval-{task.id}", task.lang if task.lang in ("ar", "en") else "ar", http, headers)
    spoken: list[dict] = []
    dones: list[dict] = []

    async def speak(text: str, kind: str) -> None:
        spoken.append({"kind": kind, "spoken": text})

    async def observe(kind: str, payload: dict) -> None:
        if kind == "done":
            dones.append(payload)

    call = CallSession(runtime, runtime.lang, speak, observe)
    transcript, turns = [], []
    for utterance in task.script:
        s0, d0, t0 = len(spoken), len(dones), time.monotonic()
        await call.handle(utterance)
        # the harness's confirmation loop, spoken: a card on the table is answered with a bare yes (the task approves confirmations)
        for _ in range(3):
            if call.pending is None or not task.script_confirm:
                break
            await call.handle(YES[runtime.lang])
        lines, events = spoken[s0:], dones[d0:]
        # rule checks on this turn: every reply spoken is a done message, normalised; progress at most twice; a card is spoken
        reply_ok = [ln["spoken"] for ln in lines if ln["kind"] == "reply"] == [normalise(d.get("message") or "", runtime.lang) for d in events if d.get("message")]
        progress = sum(ln["kind"] == "progress" for ln in lines)
        card_ok = all(any(ln["kind"] == "card" for ln in lines) for d in events if d.get("pending")) if events else True
        turns.append({"user": utterance, "lines": [ln["kind"] for ln in lines], "reply_ok": reply_ok, "progress": progress, "card_ok": card_ok,
                      "wall_s": round(time.monotonic() - t0, 2), "chars": sum(len(ln["spoken"]) for ln in lines)})
        if events:
            rec = {"user": utterance, "assistant": "\n".join(d.get("message") or "" for d in events).strip(),
                   "what_i_checked": [e for d in events for e in d.get("what_i_checked") or []],
                   "usage": {k: sum((d.get("usage") or {}).get(k, 0) for d in events) for k in ("input_tokens", "output_tokens", "cached_tokens")},
                   "latency_ms": sum(d.get("latency_ms") or 0 for d in events), "lang": events[-1].get("lang"),
                   "verifications": [d["verification"] for d in events if d.get("verification")], "trace_id": events[-1].get("trace_id"),
                   "first_token_ms": events[0].get("first_token_ms"), "model_latency_ms": events[0].get("model_latency_ms")}
            if any(d.get("pending") for d in events):
                rec["confirmations"] = [{"actions": d["pending"]["actions"], "approved": None} for d in events if d.get("pending")]
            transcript.append(rec)
    result = score(task, transcript)
    rules = all(t["reply_ok"] and t["progress"] <= 2 and t["card_ok"] for t in turns)
    return {"task": task.id, "ok": result.success and rules, "db_and_checks": result.success, "voice_rules": rules,
            "reasons": result.reasons + result.violations + ([] if rules else ["voice rule broken (see turns)"]), "turns": turns,
            "cost_usd": getattr(result, "cost_usd", None), "transcript": transcript}


async def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", help="comma-separated substrings of the task ids to run (smoke one first)")
    a = ap.parse_args()
    tasks = {t.id: t for t in load_tasks()}
    chosen = [t for t in VOICE_TASKS if not a.filter or any(f.strip() in t for f in a.filter.split(","))]
    out = RESULTS / f"voice-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as http:
        for tid in chosen:
            print(f"running {tid}", flush=True)
            results.append(await run_one(tasks[tid], http))
            print(f"  {'ok' if results[-1]['ok'] else 'FAIL'} {results[-1]['reasons']}", flush=True)
    (out / "runs.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    walls = [t["wall_s"] for r in results for t in r["turns"]]
    lines = ["# Voice task set (text path)", "", f"{sum(r['ok'] for r in results)}/{len(results)} tasks pass (database checks, mentions, metrics and the voice rules).", "",
             "| task | ok | checks | voice rules | turns | spoken lines | reasons |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['task']} | {'✓' if r['ok'] else '✗'} | {'✓' if r['db_and_checks'] else '✗'} | {'✓' if r['voice_rules'] else '✗'} | {len(r['turns'])} | "
                     f"{' / '.join(','.join(t['lines']) for t in r['turns'])} | {'; '.join(r['reasons'])[:200]} |")
    if walls:
        q = statistics.quantiles(walls, n=20) if len(walls) > 1 else [walls[0]] * 19
        lines += ["", f"Turn wall time (turn sent to last spoken line queued), p50 {statistics.median(walls):.1f} s, p95 {q[18]:.1f} s, over {len(walls)} turns.",
                  f"TTS characters that a call would have spoken: {sum(t['chars'] for r in results for t in r['turns'])}."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    asyncio.run(main())
