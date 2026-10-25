"""Scoring: did the agent do the right things (and only those), say what it must, in the right language?

pass^k (from τ-bench): the probability that all k independent runs of a task succeed, estimated from
n runs with c successes as C(c, k) / C(n, k). pass^1 is plain success rate; pass^4 punishes flakiness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from math import comb

import psycopg

from falcon.core.arabic import DIGITS, reply_script
from falcon.core.config import settings

from . import metrics as agent_metrics
from .tasks import Task

# a clause id cited in a reply: lookarounds on digits and dots rather than \b, because "البند 7.4.1 و7.4.3" glues the conjunction to the id
# and \b then starts the match at "4.3" (the scripted run of pregnant_38th_month_ar, 18 Oct); digits are folded before matching
CLAUSE_ID_RE = re.compile(r"(?<![\d.])(\d{1,2}(?:\.\d{1,2}){1,3})(?![\d.])")


@dataclass
class RunResult:
    task_id: str
    run: int
    success: bool
    reasons: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    language_ok: bool = True
    grounding_ok: bool = True
    transcript: list[dict] = field(default_factory=list)
    simulator_stopped: bool = False
    # agent-level metrics (falcon.eval.metrics); amount faithfulness and false claims also decide success
    sequence_ok: bool | None = None
    tool_errors: int = 0
    redundant_calls: int = 0
    efficiency: float | None = None
    amount_faithful: bool = True
    amounts_checked: bool = True
    unsupported_amounts: list[str] = field(default_factory=list)
    false_claim: bool = False
    handover: bool = False
    escalated: bool = False
    contained: bool = True
    turn_latencies: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    tool_sequence: list[str] = field(default_factory=list)
    gate_retried: int = 0
    gate_held: int = 0


def _text_lang(text: str) -> str:
    """The reply's script, by the same rule the runtime reviewer enforces (falcon.core.arabic.reply_script)."""
    return reply_script(text)


_ARABIC_DIGITS = {**DIGITS, **str.maketrans("٫’‘“”", ".''\"\"")}  # digits, plus the Arabic decimal point and curly quotes
_THOUSANDS = re.compile(r"(?<=\d)[,٬ ](?=\d{3}(?!\d))")


def _normalise(text: str) -> str:
    """'1,291', '1 291', '١٬٢٩١' all read as 1291, and "couldn’t" as "couldn't", so must_mention checks the words
    and amounts, not their typography."""
    return _THOUSANDS.sub("", text.translate(_ARABIC_DIGITS))


def executed_writes(transcript: list[dict]) -> list[dict]:
    out = []
    for turn in transcript:
        for e in turn.get("what_i_checked", []):
            if e.get("ok") and not e.get("declined"):
                out.append(e)
    return out


def score(task: Task, transcript: list[dict]) -> RunResult:
    r = RunResult(task_id=task.id, run=0, success=True, transcript=transcript)
    writes = executed_writes(transcript)
    assistant_text = "\n".join(t.get("assistant", "") for t in transcript)

    for want in task.write_actions:
        hit = any(w["tool"] == want["tool"] and all(str(w["args"].get(k)) == str(v) for k, v in (want.get("args") or {}).items()) for w in writes)
        if not hit:
            r.success = False
            r.reasons.append(f"missing action {want['tool']} {want.get('args', {})}")
    for forbidden in task.forbidden_actions:
        if any(w["tool"] == forbidden for w in writes):
            r.success = False
            r.violations.append(f"forbidden action executed: {forbidden}")
    assistant_text = _normalise(assistant_text)
    for pattern in task.must_mention:
        if not re.search(pattern, assistant_text, re.IGNORECASE):
            r.success = False
            r.reasons.append(f"did not mention /{pattern}/")
    for pattern in task.must_not_mention:
        if re.search(pattern, assistant_text, re.IGNORECASE):
            r.success = False
            r.violations.append(f"mentioned forbidden /{pattern}/")

    # language: every assistant reply must be in the language of the user message it answers
    for turn in transcript:
        expected = task.language if task.language != "follow" else _text_lang(turn.get("user", ""))
        got = _text_lang(turn.get("assistant", ""))
        if expected in ("ar", "en") and got in ("ar", "en") and expected != got:
            r.language_ok = False
            r.reasons.append(f"replied in {got} to a {expected} message")
    if not r.language_ok:
        r.success = False

    # grounding: cited clause ids must have been retrieved in the conversation so far (this turn or an earlier one: the runtime reviewer
    # accepts the same, and a correct second-turn reply once failed for re-citing the clause it had retrieved the turn before)
    if task.grounding:
        retrieved: set[str] = set()
        for turn in transcript:
            retrieved |= {cid for e in turn.get("what_i_checked", []) for cid in e.get("clause_ids", [])}
            cited = set(CLAUSE_ID_RE.findall(turn.get("assistant", "").translate(_ARABIC_DIGITS)))
            if cited and not cited <= retrieved:
                r.grounding_ok = False
                r.reasons.append(f"cited {sorted(cited - retrieved)} without retrieving them")
        if not r.grounding_ok:
            r.success = False

    # database checks
    if task.final_state:
        with psycopg.connect(settings.dsn_admin) as conn:
            for check in task.final_state:
                row = conn.execute(check["sql"]).fetchone()
                value = row[0] if row else None
                if str(value) != str(check["equals"]):
                    r.success = False
                    r.reasons.append(f"state check failed: {check['sql']} -> {value!r} != {check['equals']!r}")

    apply_metrics(r, task)
    r.turns = len(transcript)
    r.tool_calls = sum(len(t.get("what_i_checked", [])) for t in transcript)
    r.input_tokens = sum((t.get("usage") or {}).get("input_tokens", 0) for t in transcript)
    r.output_tokens = sum((t.get("usage") or {}).get("output_tokens", 0) for t in transcript)
    r.latency_ms = sum(t.get("latency_ms", 0) for t in transcript)
    return r


def apply_metrics(r: RunResult, task: Task) -> None:
    """Trajectory, quality and cost metrics from the transcript. Two of them are failures in this domain: an amount
    the tools never returned, and a claim that something was done when no write succeeded."""
    m = agent_metrics.compute(r.transcript, task.expected_tools)
    for k, v in m.__dict__.items():
        setattr(r, k, v)
    if not m.amount_faithful:
        r.success = False
        r.violations.append(f"amount not in any tool result: {', '.join(m.unsupported_amounts)}")
    if m.false_claim:
        r.success = False
        r.violations.append("claimed an action that did not happen")


def pass_at_k(successes: int, n: int, k: int) -> float:
    if k > n:
        return float("nan")
    return comb(successes, k) / comb(n, k) if comb(n, k) else 0.0


def summarize(results: list[RunResult], k_values=(1, 2, 4)) -> dict:
    by_task: dict[str, list[RunResult]] = {}
    for r in results:
        by_task.setdefault(r.task_id, []).append(r)
    rows = []
    for task_id, runs in by_task.items():
        n, c = len(runs), sum(1 for r in runs if r.success)
        rows.append({"task_id": task_id, "n": n, "successes": c, **{f"pass^{k}": pass_at_k(c, n, k) for k in k_values},
                     "violations": sum(len(r.violations) for r in runs), "avg_turns": sum(r.turns for r in runs) / n, "avg_tool_calls": sum(r.tool_calls for r in runs) / n,
                     "avg_latency_ms": sum(r.latency_ms for r in runs) / n, "tokens": sum(r.input_tokens + r.output_tokens for r in runs)})
    def mean(key):
        vals = [row[key] for row in rows if row[key] == row[key]]  # drop NaN
        return sum(vals) / len(vals) if vals else float("nan")
    return {"tasks": rows, "overall": {f"pass^{k}": mean(f"pass^{k}") for k in k_values}, "violations": sum(row["violations"] for row in rows), "runs": len(results),
            "scorecard": agent_metrics.scorecard([r.__dict__ for r in results])}


def to_json(r: RunResult) -> str:
    return json.dumps(r.__dict__, ensure_ascii=False, default=str)
