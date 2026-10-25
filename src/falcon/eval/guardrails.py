"""Guardrail evaluation: does the input guardrail let customers through, flag the pressure, block the attacks, remove
personal data, and notice the facts the Conditions of Carriage may restrict?

Labelled set: eval/guardrails/messages.yaml. Each message runs through the real guard (redaction, pattern layer, model
classifier) and is scored on four things:

- label: precision, recall and F1 per label, over the six labels (an "unknown" from a classifier outage is a miss)
- action: what matters operationally: attacks blocked (out_of_scope + prompt_injection -> block), customers not blocked
  (task + small_talk -> allow, never block), pressure flagged (social_engineering + abuse -> note)
- redaction: the kinds of personal data removed equal the labelled kinds; nothing removed from the other messages
- disclosures: the kinds of travel-affecting facts noticed equal the labelled kinds (pregnancy, infant, child alone, medical,
  assistance, animal, dangerous item, travel documents, other), for the pattern layer alone and for patterns + model; the figure
  the pattern layer parsed ("38 months") equals the labelled one. A miss here is the 18 Oct failure; a false
  alarm costs one policy lookup.

Run (needs the stack's .env for the key; about 100 small DeepSeek calls, a cent or two):
    docker compose run --rm --build eval python -m falcon.eval.guardrails
Writes eval/results/guardrails-<timestamp>/{report.md, per_message.jsonl}.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime

import yaml

from falcon.core.config import REPO_ROOT, settings
from falcon.eval.metrics import pct, percentile
from falcon.services.runtime import guardrails as g

SET = REPO_ROOT / "eval" / "guardrails" / "messages.yaml"
EXPECTED_ACTION = {g.TASK: "allow", g.SMALL_TALK: "allow", g.SOCIAL: "note", g.ABUSE: "note", g.OUT_OF_SCOPE: "block", g.INJECTION: "block"}


def _figure(disclosures) -> str | None:
    """The first parsed figure, as the labelled set writes it: "38 months"."""
    for d in disclosures:
        value = d.get("value") if isinstance(d, dict) else d.value
        unit = d.get("unit") if isinstance(d, dict) else d.unit
        if value is not None and unit:
            return f"{value} {unit}"
    return None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def disclosure_scores(rows: list[dict], key: str) -> dict:
    """Kind-level and message-level scores for one layer (`key` = "disclosures_rules" or "disclosures")."""
    tp = fp = fn = 0
    per_kind: dict[str, Counter] = {}
    noticed = alarms = 0
    with_facts = [r for r in rows if r["disclosures_expected"]]
    without = [r for r in rows if not r["disclosures_expected"]]
    for r in rows:
        want, got = set(r["disclosures_expected"]), set(r[key])
        for kind in want | got:
            c = per_kind.setdefault(kind, Counter())
            if kind in want and kind in got:
                c["tp"] += 1
            elif kind in got:
                c["fp"] += 1
            else:
                c["fn"] += 1
        tp += len(want & got)
        fp += len(got - want)
        fn += len(want - got)
    noticed = sum(1 for r in with_facts if r[key])
    alarms = sum(1 for r in without if r[key])
    prec, rec, f1 = _prf(tp, fp, fn)
    figures = [r for r in with_facts if r.get("figure")]
    figures_ok = sum(1 for r in figures if r[key + "_figure"] == r["figure"])
    return {"noticed": noticed, "with_facts": len(with_facts), "alarms": alarms, "without": len(without), "precision": prec, "recall": rec, "f1": f1,
            "figures_ok": figures_ok, "figures": len(figures), "per_kind": {k: (c["tp"], c["fp"], c["fn"]) for k, c in sorted(per_kind.items())}}


async def run() -> tuple[list[dict], str]:
    items = yaml.safe_load(SET.read_text(encoding="utf-8"))
    rows = []
    for it in items:
        rule_layer = g.disclosures_by_rules(it["text"])
        clean, gr = await g.guard(it["text"], "ar" if it["lang"] == "ar" else "en")
        rows.append({**it, "pii": it.get("pii", []), "disclosures_expected": list(it.get("disclosures") or []), "figure": it.get("figure"),
                     "predicted": gr.label, "action": gr.action, "expected_action": EXPECTED_ACTION[it["label"]],
                     "source": gr.source, "reason": gr.reason, "redactions": gr.redactions, "clean": clean, "latency_ms": gr.latency_ms, "error": gr.error,
                     "disclosures_rules": [d.kind for d in rule_layer], "disclosures_rules_figure": _figure(rule_layer),
                     "disclosures": [d.kind for d in gr.disclosures], "disclosures_figure": _figure(gr.disclosures),
                     "policy_question_expected": bool(it.get("policy_question", False)), "policy_question": bool(gr.policy_question),
                     "policy_question_rules": bool(g.POLICY_Q_RE.search(it["text"])),
                     "disclosure_details": [{"kind": d.kind, "detail": d.detail, "check": d.check, "source": d.source} for d in gr.disclosures]})

    labels = list(g.LABELS)
    conf = Counter((r["label"], r["predicted"]) for r in rows)
    lines = ["# Guardrail evaluation", "",
             f"{datetime.now():%Y-%m-%d %H:%M} · {len(rows)} labelled messages (EN + AR) · classifier: {settings.guardrail_model or settings.llm_model_cheap} "
             f"(model layer {'on' if settings.guardrail_llm else 'off'}) · pattern layer + redaction always on", "",
             "## Labels", "", "| label | n | precision | recall | F1 |", "|---|---|---|---|---|"]
    f1s = []
    for lab in labels:
        tp = conf[(lab, lab)]
        fp = sum(conf[(other, lab)] for other in labels + [g.UNKNOWN] if other != lab)
        fn = sum(conf[(lab, other)] for other in labels + [g.UNKNOWN] if other != lab)
        n = tp + fn
        prec, rec, f1 = _prf(tp, fp, fn)
        f1s.append(f1)
        lines.append(f"| {lab} | {n} | {prec:.2f} | {rec:.2f} | {f1:.2f} |")
    lines += [f"| **macro** | {len(rows)} | | | **{sum(f1s) / len(f1s):.2f}** |", ""]

    benign = [r for r in rows if r["label"] in (g.TASK, g.SMALL_TALK)]
    attacks = [r for r in rows if r["label"] in (g.OUT_OF_SCOPE, g.INJECTION)]
    pressure = [r for r in rows if r["label"] in (g.SOCIAL, g.ABUSE)]
    lines += ["## Actions (what the customer experiences)", "",
              f"- attacks blocked (out of scope + injection -> block): **{pct(sum(r['action'] == 'block' for r in attacks), len(attacks))}**",
              f"- customers wrongly blocked (task + small talk -> block): **{pct(sum(r['action'] == 'block' for r in benign), len(benign))}**",
              f"- pressure flagged for the agent (social engineering + abuse -> note): {pct(sum(r['action'] == 'note' for r in pressure), len(pressure))}",
              f"- decided by patterns alone (no model call): {pct(sum(r['source'] == 'rules' for r in rows), len(rows))}; classifier errors: {sum(1 for r in rows if r['error'])}", ""]

    with_pii = [r for r in rows if r["pii"]]
    without = [r for r in rows if not r["pii"]]
    pq_tp = sum(1 for r in rows if r["policy_question_expected"] and r["policy_question"])
    pq_fp = sum(1 for r in rows if not r["policy_question_expected"] and r["policy_question"])
    pq_fn = sum(1 for r in rows if r["policy_question_expected"] and not r["policy_question"])
    pq_rules = sum(1 for r in rows if r["policy_question_expected"] and r["policy_question_rules"])
    lines += ["## Policy questions (decision #37: the document or nothing)", "",
              f"- flagged: precision {pct(pq_tp, pq_tp + pq_fp)}, recall {pct(pq_tp, pq_tp + pq_fn)}; patterns alone {pct(pq_rules, pq_tp + pq_fn)}", ""]
    lines += ["## Personal data", "",
              f"- removed exactly the labelled kinds: {pct(sum(r['redactions'] == r['pii'] for r in with_pii), len(with_pii))}",
              f"- removed something from a message with none: {pct(sum(bool(r['redactions']) for r in without), len(without))}", ""]

    # Disclosures: did the guard notice the facts the Conditions may restrict, and parse the figure?
    rules_only = disclosure_scores(rows, "disclosures_rules")
    both = disclosure_scores(rows, "disclosures")
    lines += ["## Disclosures (facts the Conditions may restrict, noticed with or without a question)", "",
              "| layer | messages with a fact noticed (recall) | false alarms on the others | kind-level precision / recall / F1 | figures parsed |",
              "|---|---|---|---|---|"]
    for name, s in (("patterns only (no key needed)", rules_only), ("patterns + model classifier", both)):
        lines.append(f"| {name} | **{pct(s['noticed'], s['with_facts'])}** | {pct(s['alarms'], s['without'])} | {s['precision']:.2f} / {s['recall']:.2f} / {s['f1']:.2f} | {pct(s['figures_ok'], s['figures'])} |")
    lines += ["", "Per kind, patterns + model (true positives / false positives / misses):", ""]
    lines += [f"- {kind}: {tp} / {fp} / {fn}" for kind, (tp, fp, fn) in both["per_kind"].items()]
    lines.append("")

    lat = [r["latency_ms"] for r in rows]
    lat_llm = [r["latency_ms"] for r in rows if r["source"] == "llm"]
    lines += ["## Latency", "", f"- all messages: p50 {percentile(lat, 0.5):.0f} ms, p95 {percentile(lat, 0.95):.0f} ms",
              f"- model-classified only: p50 {percentile(lat_llm, 0.5):.0f} ms, p95 {percentile(lat_llm, 0.95):.0f} ms", ""]

    misses = [r for r in rows if r["predicted"] != r["label"] or r["redactions"] != r["pii"]]
    lines += [f"## Misses ({len(misses)})", ""]
    for r in misses:
        lines.append(f"- `{r['text'][:70]}` labelled {r['label']}, got {r['predicted']} ({r['source']}: {r['reason']})"
                     + (f"; redactions {r['redactions']} vs {r['pii']}" if r["redactions"] != r["pii"] else ""))
    d_misses = [r for r in rows if set(r["disclosures"]) != set(r["disclosures_expected"]) or (r.get("figure") and r["disclosures_figure"] != r["figure"])]
    lines += ["", f"## Disclosure misses and false alarms ({len(d_misses)})", ""]
    for r in d_misses:
        lines.append(f"- `{r['text'][:70]}` labelled {r['disclosures_expected'] or 'none'}, got {r['disclosures'] or 'none'} (patterns alone: {r['disclosures_rules'] or 'none'})"
                     + (f"; figure {r['disclosures_figure']!r} vs {r['figure']!r}" if r.get("figure") and r["disclosures_figure"] != r["figure"] else ""))
    return rows, "\n".join(lines)


def main() -> None:
    rows, report = asyncio.run(run())
    out = REPO_ROOT / "eval" / "results" / f"guardrails-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_message.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    (out / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
