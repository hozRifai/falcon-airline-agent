"""Re-score stored runs with the current metrics, without spending a token.

    python -m falcon.eval.rescore eval/results/20261017-062930
    python -m falcon.eval.rescore eval/results/20261017-062930 --write   # also writes scorecard.md next to runs.jsonl

Outcome fields (success, reasons, violations) are kept as recorded: the database state they depended on is gone.
The trajectory, quality and operations metrics are recomputed from the transcripts. Runs recorded before the
runtime returned full tool results to the harness only expose 400-character summaries, so an amount deep in a long
result can be reported as unsupported there; the row says `summaries only` when that is the case.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import compute, render_scorecard, scorecard
from .tasks import load_tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    folder = Path(args.results_dir)
    tasks = {t.id: t for t in load_tasks()}
    runs = [json.loads(line) for line in (folder / "runs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    full_results = any(e.get("result_json") is not None for r in runs for t in r.get("transcript", []) for e in t.get("what_i_checked", []))
    for r in runs:
        task = tasks.get(r["task_id"])
        m = compute(r.get("transcript", []), task.expected_tools if task else None)
        r.update(m.__dict__)
    card = scorecard(runs)
    text = render_scorecard(card)
    header = f"# Re-scored: {folder.name} ({len(runs)} runs; tool results: {'full' if full_results else 'summaries only'})\n\n"
    rows = ["| task | success | sequence | errors | redundant | efficiency | amounts ok | false claim | contained | cost $ |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        seq = "n/a" if r.get("sequence_ok") is None else ("yes" if r["sequence_ok"] else "NO")
        rows.append(f"| {r['task_id']} | {'pass' if r.get('success') else 'FAIL'} | {seq} | {r.get('tool_errors', 0)} | {r.get('redundant_calls', 0)} | "
                    f"{'' if r.get('efficiency') is None else f'{r['efficiency']:.2f}'} | {'yes' if r.get('amount_faithful', True) else 'NO ' + str(r.get('unsupported_amounts'))} | "
                    f"{'YES' if r.get('false_claim') else 'no'} | {'yes' if r.get('contained', True) else 'no'} | {r.get('cost_usd', 0):.4f} |")
    out = header + text + "\n\n## Per run\n\n" + "\n".join(rows) + "\n"
    print(out)
    if args.write:
        (folder / "scorecard.md").write_text(out, encoding="utf-8")
        print(f"written to {folder / 'scorecard.md'}")


if __name__ == "__main__":
    main()
