"""Agent-level metrics computed from a run's transcript: what the agent did, not only whether the task passed.

Outcome metrics (task success, forbidden actions, final database state) live in scoring.py. This module adds the
layers a service owner asks about once the pass rate is known:

Trajectory (did it work the right way?)
- sequence_ok        the task's `expected_tools` appear in order among the tools the agent called (a|b = either)
- tool_errors        tool calls that came back as errors (wrong arguments, unknown booking, refused by a role)
- redundant_calls    the same tool with the same arguments called again in the same run
- efficiency         len(expected_tools) / tool calls made, capped at 1 (1.0 = no wasted call)

Answer quality (did it say only true things?)
- amount_faithful    every amount of money in the replies appears in a tool result or in the customer's own words
- false_claim        the reply says something was done (cancelled, refunded, changed, sent) with no successful write
- gate_retried/held  how often the runtime's verification gate (same checks, falcon.core.faithfulness) intervened

Operations and business (what does it cost to run?)
- turn_latencies     wall-clock per turn, for p50/p95 across a run set
- first_token_ms     time to the first streamed token per turn (turns that streamed), and model time per turn: where the wait goes
- cost_usd           input/output tokens at the configured per-million prices
- handover / escalated   the conversation logged a case for a human agent / an exception case for the care team (by design)
- contained          resolved by the agent alone: no handover and no escalation

All of it is computed from the transcript the harness already records, so stored runs can be re-scored
(`python -m falcon.eval.rescore <results dir>`) without spending a token.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass, field

from falcon.core.config import settings

from falcon.core.faithfulness import DONE_CLAIM_RE, EVIDENCE_TOOLS, MONEY_RE, WRITE_TOOLS, amounts_in, numbers_in  # noqa: E402,F401 - one definition for the gate and the scorer

HANDOVER_TOOLS = {"handover_to_human"}
ESCALATION_TOOLS = {"request_exception"}


@dataclass
class AgentMetrics:
    sequence_ok: bool | None = None       # None when the task defines no expected_tools
    tool_errors: int = 0
    redundant_calls: int = 0
    efficiency: float | None = None
    amount_faithful: bool = True
    amounts_checked: bool = True          # False for transcripts recorded without full tool results
    unsupported_amounts: list[str] = field(default_factory=list)
    false_claim: bool = False
    handover: bool = False
    escalated: bool = False
    contained: bool = True
    turn_latencies: list[int] = field(default_factory=list)
    first_tokens: list[int] = field(default_factory=list)
    model_latencies: list[int] = field(default_factory=list)
    cost_usd: float = 0.0
    tool_sequence: list[str] = field(default_factory=list)
    gate_retried: int = 0                 # replies the verification gate sent back to the model once
    gate_held: int = 0                    # replies the gate held back (safe message shown instead)


def numbers_in_results(entries: list[dict]) -> set[str]:
    """Every number that appears anywhere in the tool results of a run, normalised to two decimals."""
    found: set[str] = set()
    for e in entries:
        blob = json.dumps(e.get("result_json"), ensure_ascii=False) if e.get("result_json") is not None else (e.get("summary") or "")
        found |= numbers_in(blob)
    return found


def sequence_matches(expected: list[str], actual: list[str]) -> bool:
    """expected steps appear in order in actual (not necessarily adjacent); a step 'a|b' is satisfied by either."""
    pos = 0
    for step in expected:
        options = set(step.split("|"))
        while pos < len(actual) and actual[pos] not in options:
            pos += 1
        if pos == len(actual):
            return False
        pos += 1
    return True


def compute(transcript: list[dict], expected_tools: list[str] | None = None) -> AgentMetrics:
    m = AgentMetrics()
    entries = [e for t in transcript for e in t.get("what_i_checked", [])]
    executed = [e for e in entries if not e.get("declined")]
    m.tool_sequence = [e["tool"] for e in executed]
    m.tool_errors = sum(1 for e in executed if not e.get("ok"))
    seen: set[tuple] = set()
    for e in executed:
        key = (e["tool"], json.dumps(e.get("args") or {}, sort_keys=True, ensure_ascii=False))
        if key in seen:
            m.redundant_calls += 1
        seen.add(key)
    if expected_tools:
        m.sequence_ok = sequence_matches(expected_tools, [e["tool"] for e in executed])
        m.efficiency = min(1.0, len(expected_tools) / len(executed)) if executed else 0.0

    # answer quality (amounts need the full tool results; older transcripts only carry 400-character summaries)
    m.amounts_checked = any(e.get("result_json") is not None for e in entries) or not entries
    allowed = numbers_in_results(entries)
    customer_said = set()
    for t in transcript:
        customer_said |= amounts_in(t.get("user", ""))
    said = set()
    for t in transcript:
        said |= amounts_in(t.get("assistant", ""))
    unsupported = sorted(a for a in said if a not in allowed and a not in customer_said) if m.amounts_checked else []
    m.unsupported_amounts = unsupported
    m.amount_faithful = not unsupported
    writes_ok_so_far = False
    for t in transcript:
        if DONE_CLAIM_RE.search(t.get("assistant", "")) and not writes_ok_so_far and not any(
                e.get("ok") and e["tool"] in EVIDENCE_TOOLS for e in t.get("what_i_checked", [])):
            m.false_claim = True
        if any(e.get("ok") and e["tool"] in EVIDENCE_TOOLS for e in t.get("what_i_checked", [])):
            writes_ok_so_far = True

    for t in transcript:
        for v in t.get("verifications", []) or ([t["verification"]] if t.get("verification") else []):
            if v.get("status") == "retried":
                m.gate_retried += 1
            elif v.get("status") == "held":
                m.gate_held += 1

    # operations
    m.handover = any(e.get("ok") and e["tool"] in HANDOVER_TOOLS for e in executed)
    m.escalated = any(e.get("ok") and e["tool"] in ESCALATION_TOOLS for e in executed)
    m.contained = not (m.handover or m.escalated)
    m.turn_latencies = [int(t.get("latency_ms", 0)) for t in transcript]
    m.first_tokens = [int(t["first_token_ms"]) for t in transcript if t.get("first_token_ms") is not None]
    m.model_latencies = [int(t["model_latency_ms"]) for t in transcript if t.get("model_latency_ms") is not None]
    tokens_in = sum((t.get("usage") or {}).get("input_tokens", 0) for t in transcript)
    tokens_out = sum((t.get("usage") or {}).get("output_tokens", 0) for t in transcript)
    m.cost_usd = tokens_in / 1e6 * settings.llm_price_in_per_m + tokens_out / 1e6 * settings.llm_price_out_per_m
    return m


def percentile(values: list[int | float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def scorecard(runs: list[dict]) -> dict:
    """Aggregate over run records (dicts with the AgentMetrics fields merged in, as runs.jsonl stores them)."""
    n = len(runs) or 1
    with_seq = [r for r in runs if r.get("sequence_ok") is not None]
    latencies = [l for r in runs for l in (r.get("turn_latencies") or [])]
    first_tokens = [l for r in runs for l in (r.get("first_tokens") or [])]
    model_lat = [l for r in runs for l in (r.get("model_latencies") or [])]
    tool_calls = sum(r.get("tool_calls", 0) for r in runs)
    successes = [r for r in runs if r.get("success")]
    cost_total = sum(r.get("cost_usd", 0.0) for r in runs)
    return {
        "runs": len(runs),
        "outcome": {
            "task_success_rate": len(successes) / n,
            "policy_violations": sum(len(r.get("violations", [])) for r in runs),
            "runs_with_violations": sum(1 for r in runs if r.get("violations")),
        },
        "trajectory": {
            "tool_sequence_correct_rate": (sum(1 for r in with_seq if r["sequence_ok"]) / len(with_seq)) if with_seq else float("nan"),
            "runs_with_expected_tools": len(with_seq),
            "tool_error_rate": (sum(r.get("tool_errors", 0) for r in runs) / tool_calls) if tool_calls else 0.0,
            "redundant_calls_per_run": sum(r.get("redundant_calls", 0) for r in runs) / n,
            "mean_efficiency": statistics.fmean([r["efficiency"] for r in runs if r.get("efficiency") is not None]) if any(r.get("efficiency") is not None for r in runs) else float("nan"),
            "avg_tool_calls": tool_calls / n,
            "avg_turns": sum(r.get("turns", 0) for r in runs) / n,
        },
        "quality": {
            "amount_faithful_rate": (sum(1 for r in runs if r.get("amounts_checked", True) and r.get("amount_faithful", True)) / len([r for r in runs if r.get("amounts_checked", True)])) if any(r.get("amounts_checked", True) for r in runs) else float("nan"),
            "amounts_checked_runs": sum(1 for r in runs if r.get("amounts_checked", True)),
            "runs_with_unsupported_amounts": [(r["task_id"], r.get("unsupported_amounts")) for r in runs if r.get("unsupported_amounts")],
            "false_claim_rate": sum(1 for r in runs if r.get("false_claim")) / n,
            "gate_retried_replies": sum(r.get("gate_retried", 0) for r in runs),
            "gate_held_replies": sum(r.get("gate_held", 0) for r in runs),
            "language_ok_rate": sum(1 for r in runs if r.get("language_ok", True)) / n,
            "grounding_ok_rate": sum(1 for r in runs if r.get("grounding_ok", True)) / n,
        },
        "operations": {
            "containment_rate": sum(1 for r in runs if r.get("contained", True)) / n,
            "escalation_rate": sum(1 for r in runs if r.get("escalated")) / n,
            "handover_rate": sum(1 for r in runs if r.get("handover")) / n,
            "turn_latency_p50_ms": percentile(latencies, 0.5),
            "turn_latency_p95_ms": percentile(latencies, 0.95),
            "first_token_p50_ms": percentile(first_tokens, 0.5) if first_tokens else float("nan"),
            "first_token_p95_ms": percentile(first_tokens, 0.95) if first_tokens else float("nan"),
            "model_share_of_turn": (sum(model_lat) / sum(latencies)) if model_lat and sum(latencies) else float("nan"),
            "tokens_per_conversation": sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in runs) / n,
            "cost_per_conversation_usd": cost_total / n,
            "cost_per_successful_conversation_usd": (cost_total / len(successes)) if successes else float("nan"),
            "projected_cost_per_1000_usd": cost_total / n * 1000,
        },
    }


def render_scorecard(card: dict) -> str:
    o, t, q, p = card["outcome"], card["trajectory"], card["quality"], card["operations"]
    f = lambda v, pct=True: ("n/a" if v != v else (f"{v:.0%}" if pct else f"{v:.2f}"))  # noqa: E731 - NaN check
    lines = [
        "## Agent scorecard", "",
        f"{card['runs']} conversations.", "",
        "| Layer | Metric | Value |", "|---|---|---|",
        f"| Outcome | task success | {f(o['task_success_rate'])} |",
        f"| Outcome | policy violations (forbidden action / forbidden statement) | {o['policy_violations']} in {o['runs_with_violations']} conversations |",
        f"| Trajectory | expected tools called in the right order | {f(t['tool_sequence_correct_rate'])} of {t['runs_with_expected_tools']} conversations with an expected sequence |",
        f"| Trajectory | tool calls that returned an error | {f(t['tool_error_rate'])} |",
        f"| Trajectory | redundant tool calls per conversation | {t['redundant_calls_per_run']:.2f} |",
        f"| Trajectory | efficiency (expected calls / calls made) | {f(t['mean_efficiency'], False)} |",
        f"| Trajectory | tool calls / turns per conversation | {t['avg_tool_calls']:.1f} / {t['avg_turns']:.1f} |",
        f"| Quality | every amount traceable to a tool result | {f(q['amount_faithful_rate'])} ({q["amounts_checked_runs"]} assessable conversations) |",
        f"| Quality | claimed an action that did not happen | {f(q['false_claim_rate'])} |",
        f"| Quality | verification gate: replies corrected by one retry / held back | {q['gate_retried_replies']} / {q['gate_held_replies']} |",
        f"| Quality | replied in the customer's language | {f(q['language_ok_rate'])} |",
        f"| Quality | citations grounded in retrieved clauses | {f(q['grounding_ok_rate'])} |",
        f"| Operations | contained (no human needed) | {f(p['containment_rate'])} |",
        f"| Operations | exception case logged for the care team (by design) | {f(p['escalation_rate'])} |",
        f"| Operations | case logged for a human agent | {f(p['handover_rate'])} |",
        f"| Operations | turn latency p50 / p95 | {p['turn_latency_p50_ms']:.0f} ms / {p['turn_latency_p95_ms']:.0f} ms |",
        f"| Operations | time to first token p50 / p95 (turns that streamed) | {p['first_token_p50_ms']:.0f} ms / {p['first_token_p95_ms']:.0f} ms |",
        f"| Operations | share of turn time spent in the model | {f(p['model_share_of_turn'])} |",
        f"| Operations | tokens per conversation | {p['tokens_per_conversation']:.0f} |",
        f"| Operations | model cost per conversation / per 1,000 | ${p['cost_per_conversation_usd']:.4f} / ${p['projected_cost_per_1000_usd']:.2f} |",
    ]
    if q["runs_with_unsupported_amounts"]:
        lines += ["", "Amounts not found in any tool result: " + "; ".join(f"`{tid}` {amts}" for tid, amts in q["runs_with_unsupported_amounts"])]
    lines += ["", f"Prices used: ${settings.llm_price_in_per_m}/M input, ${settings.llm_price_out_per_m}/M output (`LLM_PRICE_IN_PER_M`, `LLM_PRICE_OUT_PER_M`)."]
    return "\n".join(lines)
