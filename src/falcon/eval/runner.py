"""Run the evaluation: for each task and each of k runs, reset the database, sign in as the persona, let
the simulated user and the agent talk through the runtime API, then score.

    python -m falcon.eval.runner --fake --k 1            # harness self-test with the scripted model
    python -m falcon.eval.runner --k 4 --lang en         # real run (needs OPENROUTER_API_KEY)
    python -m falcon.eval.runner --k 4 --filter policy --concurrency 4

Output: eval/results/<timestamp>/runs.jsonl and report.md. The frozen clock and the deterministic seed make
runs comparable across days; the only sources of variance are the models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

from falcon.core.config import REPO_ROOT, settings

from .metrics import render_scorecard
from .scoring import RunResult, score, summarize, to_json
from .simulator import STOP, make_simulator
from .tasks import Task, load_tasks

RESULTS = REPO_ROOT / "eval" / "results"
PROVIDER_ERROR_RE = re.compile(r"run error: .*(timed out|timeout|504|502|503|overloaded|rate limit|Connection error|peer closed)", re.IGNORECASE)


def reset_db() -> None:
    subprocess.run([sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], check=True, capture_output=True)


async def sse_events(client: httpx.AsyncClient, path: str, headers: dict, body: dict) -> list[tuple[str, dict]]:
    events, ev = [], None
    async with client.stream("POST", f"{settings.runtime_public_url}{path}", headers=headers, json=body, timeout=180) as resp:
        if resp.status_code != 200:
            text = await resp.aread()
            raise RuntimeError(f"{path} -> {resp.status_code}: {text[:200]}")
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                events.append((ev, json.loads(line[5:].strip())))
    return events


async def run_task(task: Task, run: int, fake: bool, lock: asyncio.Lock) -> RunResult:
    sim = make_simulator(task, fake)
    transcript: list[dict] = []
    history: list[tuple[str, str]] = []
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{settings.runtime_public_url}/api/auth/demo-login", json={"user_id": task.persona, "access_code": settings.demo_access_code}, headers={"X-Admin-Token": settings.admin_token})
        r.raise_for_status()
        headers = {"Authorization": f"Bearer {r.json()['token']}", "X-Admin-Token": settings.admin_token,
                   "X-Falcon-Tags": f"eval,task:{task.id},run:{run}"}  # traces of evaluation turns are filterable in the trace UI
        assistant_text = ""
        stopped = False
        for turn in range(task.max_turns):
            user_msg = await sim.next_message(assistant_text, history)
            if not user_msg or user_msg.strip() == STOP:
                stopped = True
                break
            if STOP in user_msg:
                # The marker is for the harness, never for the agent: sent verbatim, "نعم. ###STOP###" was classified as an injection
                # ("stop marker addressed to AI") and a cancellation was lost (19 Oct). The message is the last one of the conversation.
                user_msg = user_msg.replace(STOP, "").strip()
                last_turn = True
            else:
                last_turn = False
            if not user_msg:
                stopped = True
                break
            history.append(("user", user_msg))
            for att in task.attachments:  # the document the customer "attaches with the paperclip" (decision #33)
                if int(att.get("before_turn", 0)) == turn + 1:
                    path = REPO_ROOT / "eval" / "fixtures" / att["file"]
                    ctype = "application/pdf" if path.suffix.lower() == ".pdf" else "image/png"
                    up = await client.post(f"{settings.runtime_public_url}/api/attachments", headers=headers, files={"file": (path.name, path.read_bytes(), ctype)})
                    up.raise_for_status()
            events = await sse_events(client, "/api/chat", headers, {"message": user_msg, "thread": f"eval-{run}"})
            done = next(d for e, d in events if e == "done")
            record = {"user": user_msg, "assistant": done["message"], "what_i_checked": done["what_i_checked"], "usage": done["usage"], "latency_ms": done["latency_ms"], "lang": done["lang"],
                      "model_latency_ms": done.get("model_latency_ms"), "first_token_ms": done.get("first_token_ms"),
                      "verifications": [done["verification"]] if done.get("verification") else [], "trace_id": done.get("trace_id")}
            # confirmation loop: the simulated person decides
            while done.get("pending"):
                approved = await sim.decide(done["pending"]["actions"])
                record.setdefault("confirmations", []).append({"actions": done["pending"]["actions"], "approved": approved})
                events = await sse_events(client, "/api/chat/confirm", headers, {"approved": approved, "note": "", "thread": f"eval-{run}"})
                done = next(d for e, d in events if e == "done")
                record["assistant"] = (record["assistant"] + "\n" + done["message"]).strip()
                record["what_i_checked"] = record["what_i_checked"] + done["what_i_checked"]
                record["usage"] = {k: record["usage"].get(k, 0) + done["usage"].get(k, 0) for k in set(record["usage"]) | set(done["usage"])}
                record["latency_ms"] += done["latency_ms"]
                if done.get("verification"):
                    record["verifications"].append(done["verification"])
            transcript.append(record)
            assistant_text = record["assistant"]
            history.append(("assistant", assistant_text))
            if last_turn:
                stopped = True
                break
    result = score(task, transcript)
    result.run = run
    result.simulator_stopped = stopped
    return result


async def main_async(args) -> None:
    tasks = load_tasks(args.filter, fake_only=args.fake)
    if args.lang:
        tasks = [t for t in tasks if t.lang == args.lang]
    if not tasks:
        raise SystemExit("no tasks selected")
    out_dir = RESULTS / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    lock = asyncio.Lock()
    started = time.time()
    # Runs share one database, so they are executed sequentially per reset; concurrency applies across tasks
    # only when their personas and bookings do not overlap. Keep it simple and safe: sequential by default.
    from falcon.policy.embed import Embedder

    if not Embedder().enabled:
        # a disclosure task is answered by the deterministic check_travel_eligibility (decision #34), so it runs without retrieval too
        needs_retrieval = [t for t in tasks if (t.grounding or "policy" in t.tags or "rag" in t.tags) and not any("check_travel_eligibility" in step for step in t.expected_tools)]
        for t in needs_retrieval:
            print(f"SKIP {t.id}: needs policy retrieval and no embedding provider is configured (decision #24: no lexical fallback)", flush=True)
        tasks = [t for t in tasks if t not in needs_retrieval]
        if not tasks:
            raise SystemExit("every selected task needs policy retrieval; set OPENROUTER_API_KEY")
    for task in tasks:
        for run in range(args.k):
            reset_db()
            try:
                result = await run_task(task, run, args.fake, lock)
            except Exception as exc:
                result = RunResult(task_id=task.id, run=run, success=False, reasons=[f"run error: {exc}"])
            if not result.success and any(PROVIDER_ERROR_RE.search(r) for r in result.reasons):
                # The model provider failed before the conversation started (OpenRouter 504s took four tasks in one run, FAILURE_DIARY #47):
                # that measures the provider, not the agent. One retry after a pause, recorded as such.
                print(f"     provider error on {task.id}; retrying once in 20 s", flush=True)
                await asyncio.sleep(20)
                reset_db()
                try:
                    result = await run_task(task, run, args.fake, lock)
                    result.reasons.append("retried once after a provider error")
                except Exception as exc:
                    result = RunResult(task_id=task.id, run=run, success=False, reasons=[f"run error after one retry: {exc}"])
            results.append(result)
            mark = "PASS" if result.success else "FAIL"
            print(f"{mark} {task.id} run {run + 1}/{args.k} turns={result.turns} tools={result.tool_calls} {'; '.join(result.reasons + result.violations)}", flush=True)
            with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as f:
                f.write(to_json(result) + "\n")
    summary = summarize(results, k_values=tuple(k for k in (1, 2, 4) if k <= args.k))
    report = render_report(summary, args, time.time() - started)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("\n" + report)
    print(f"written to {out_dir}")


def render_report(summary: dict, args, seconds: float) -> str:
    ks = [k for k in (1, 2, 4) if f"pass^{k}" in summary["overall"]]
    lines = [f"# Evaluation report", "", f"model: {settings.llm_provider}:{settings.llm_model} ({'fake' if args.fake else 'real'}) · simulator: {settings.simulator_model} · k={args.k} · runs={summary['runs']} · {seconds:.0f}s", "",
             "| task | n | ok | " + " | ".join(f"pass^{k}" for k in ks) + " | violations | turns | tools | latency ms |", "|---|---|---|" + "---|" * len(ks) + "---|---|---|---|"]
    for row in summary["tasks"]:
        lines.append(f"| {row['task_id']} | {row['n']} | {row['successes']} | " + " | ".join(f"{row[f'pass^{k}']:.2f}" for k in ks) + f" | {row['violations']} | {row['avg_turns']:.1f} | {row['avg_tool_calls']:.1f} | {row['avg_latency_ms']:.0f} |")
    lines += ["", "**Overall:** " + " · ".join(f"pass^{k} = {summary['overall'][f'pass^{k}']:.2f}" for k in ks) + f" · policy violations = {summary['violations']}", ""]
    lines += [render_scorecard(summary["scorecard"]), ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--fake", action="store_true", help="scripted model + scripted user (harness self-test)")
    ap.add_argument("--filter", default=None, help="substring of task ids, or a comma-separated list of substrings")
    ap.add_argument("--lang", default=None, help="en | ar | mixed")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
