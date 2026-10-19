"""Generation evaluation: are the agent's policy answers faithful to the clauses it retrieved, relevant, and complete?
(docs/EVALUATION.md §4)

Retrieval (§1) measures whether the right clause comes back. The runtime reviewer (decision #26; FAILURE_DIARY #43, #47, #49) blocks
unretrieved clause ids, figures no tool returned, laws named from memory and policy claims with no citation. Neither says whether the
reply *uses* the retrieved clauses well: on 18 Oct the agent cited three unrelated clauses for a medical-emergency question and once
invented stroller rules, and both passed every deterministic check. This runner asks the running agent every question of the retrieval
gold set (eval/retrieval/gold.yaml: 40 answerable, 4 the document does not cover) plus eval/generation/questions.yaml (the two laptop
failures), through the real API the way scripts/talk.py does (persona Noura for English, Salma for Arabic, one turn per question), and
scores each reply twice:

- deterministic: the reviewer's own checks re-run on the final reply (`falcon.services.runtime.review.policy_problems`): every cited
  clause id was retrieved in the turn, every figure with a unit is in a tool result, no law or regulation from memory, a reply that
  speaks for the Conditions cites a clause or says the document does not cover it; and, for labelled questions, whether an answering
  (grade-2) clause was retrieved in the turn and whether one was cited.
- a judge model with a fixed rubric (RUBRIC below): faithfulness, relevance and completeness from 1 to 5, and whether the reply says the
  document does not cover the question. The judge sees the question, the retrieved clause texts and the reply, nothing else, and is
  told to grade against the clause texts alone, never from what it knows. It is a Qwen model (decision #35): a different family from the
  DeepSeek agent, so the agent is never grading itself; temperature 0, JSON output.

A judge is a probabilistic grader, so its numbers are quoted only once it agrees with people: eval/generation/labels.yaml holds replies
labelled by hand (Hozayfa's labels, and the labels of the session that built the judge, kept apart); when any are filled the runner
grades those same replies and prints the agreement (exact and within one point) per dimension and per labeller.

Run (needs the stack; one DeepSeek turn per question and one Qwen call per reply, about USD 0.15 for the 48 questions):
    docker compose run --rm --build eval python -m falcon.eval.generation
    ... --filter medical --no-reset               # a subset; keep the demo data (someone else is using the stack)
    ... --resume eval/results/generation-<ts>      # finish an interrupted run: ask only the questions it has no answer for
    ... --rejudge eval/results/generation-<ts>     # grade a run's replies again without asking the agent (a cent)
    ... --validate                                 # grade the hand-labelled replies only and print the agreement
    ... --draft-labels eval/results/generation-<ts> --ids a_en,b_ar   # label stubs + a reading sheet from a run, into that run's folder
Writes eval/results/generation-<timestamp>/{report.md, answers.jsonl}; answers.jsonl carries the reply, the retrieved clause texts, the
tool calls, the reviewer's outcome and both scores, so a run can be re-read and re-graded without spending an agent turn. Each answer
is written as it lands, so a stack restart in the middle (the laptop is shared) loses nothing: `--resume` the folder.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from falcon.core.config import REPO_ROOT, settings
from falcon.services.runtime.review import NOT_COVERED_RE, POLICY_TOOLS, cited_clause_ids, policy_problems

from .retrieval import load_gold

EXTRA = REPO_ROOT / "eval" / "generation" / "questions.yaml"
LABELS = REPO_ROOT / "eval" / "generation" / "labels.yaml"
RESULTS = REPO_ROOT / "eval" / "results"
PERSONA = {"en": "u-noura", "ar": "u-salma"}
DIMENSIONS = ("faithfulness", "relevance", "completeness")
LABELLERS = ("hozayfa", "claude")

RUBRIC = """You grade one reply from an airline's customer-service assistant to a customer's question about the airline's General Conditions of Carriage.
You receive the question, the clauses the assistant retrieved from the document (id, article, text) and the reply.

Grade against the retrieved clause texts only. Never use what you know about airlines, laws, conventions or other documents: a statement
that may be true in the world but is not in the clauses given is unsupported. The reply may be in English or Arabic.

Scores, 1 to 5 each:

faithfulness: are the reply's statements about the rules supported by the clause texts?
 5: every rule statement is supported by a clause given, and each cited clause id points to a clause that says it.
 4: supported, with one minor imprecision (a slight overstatement, a parent id cited for the sub-clause that says it).
 3: mostly supported, but one statement is not in the clauses, or one cited clause does not say that.
 2: several statements are not in the clauses, or a cited clause says something different.
 1: the reply invents rules, figures, deadlines or clauses.
 A reply that says the document does not cover the question and makes no rule statement is faithful (5) when the clauses given indeed
 do not answer it, and 2 or lower when a clause given clearly does. Fees, deadlines and refund amounts that the reply attributes to the
 fare rules of the customer's ticket or to the booking system, not to the Conditions, are outside this grading: ignore them unless the
 reply presents them as what the Conditions say. Offering a human agent, asking a follow-up question or greeting is neutral.
 The document names the airline "Emirates"; the assistant calls it Falcon. They are the same airline: the name is never a fabrication.
 When no clause was retrieved: any statement about what the airline's rules, conditions, policies, programmes or offers are is
 unsupported, because nothing shown supports it (faithfulness 2 if one such statement, 1 if the reply is built on them); a reply that
 only reports the customer's own booking, the fare rules of their ticket or flight search results, or says it could not check the
 document, makes no document claim and is faithful (5).

relevance: does the reply answer the question that was asked?
 5: answers it directly. 4: answers it with some digression. 3: answers a neighbouring question, or only part of it.
 2: mostly beside the point. 1: off-topic, or a refusal or deflection when a clause given contains the answer.

completeness: of the clauses given that bear on the question, how many does the reply use?
 5: uses every clause given that bears on the question (or none bears on it and the reply says so).
 4: uses the main one and misses a minor detail. 3: uses the main one but omits a condition, exception or document requirement that is
 in the clauses given. 2: uses a marginal clause and misses the one that answers the question. 1: ignores the clauses given, or no
 clause was retrieved at all (the document was not consulted, whatever the reply says).

said_not_covered: true when the reply says that the Conditions or the document do not cover, mention or specify the matter asked
about; false otherwise.

reason: one or two sentences in English, naming the clause ids involved and the statement that drove the lowest score.

Return JSON only: {"faithfulness": <1-5>, "relevance": <1-5>, "completeness": <1-5>, "said_not_covered": <true|false>, "reason": "<text>"}"""


# ---------------------------------------------------------------- questions
def load_questions(pattern: str | None = None) -> list[dict]:
    """The retrieval gold set (answerable and not) plus the extra questions, one item per language: id, lang, question, relevant, answerable."""
    items = [{**i, "source": "gold"} for i in load_gold()]
    if EXTRA.exists():
        for t in yaml.safe_load(EXTRA.read_text(encoding="utf-8")) or []:
            for lang in ("en", "ar"):
                if t.get(lang):
                    items.append({"id": f"{t['topic']}_{lang}", "topic": t["topic"], "lang": lang, "question": t[lang],
                                  "relevant": {str(k): int(v) for k, v in (t.get("relevant") or {}).items()},
                                  "answerable": t.get("answerable", True), "source": "extra", "note": t.get("note")})
    if pattern:
        items = [i for i in items if pattern in i["id"]]
    return items


# ---------------------------------------------------------------- the agent, through the API
async def sse(client: httpx.AsyncClient, path: str, headers: dict, body: dict) -> list[tuple[str, dict]]:
    events, ev = [], None
    async with client.stream("POST", f"{settings.runtime_public_url}{path}", headers=headers, json=body, timeout=240) as resp:
        if resp.status_code != 200:
            raise RuntimeError(f"{path} -> {resp.status_code}: {(await resp.aread())[:300]!r}")
        async for line in resp.aiter_lines():
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                events.append((ev, json.loads(line[5:].strip())))
    return events


async def sign_in(client: httpx.AsyncClient, persona: str, reset: bool) -> dict:
    """What picking the customer in the app does (and scripts/talk.py): reset the demo data if asked, sign in with a fresh session.
    The admin token makes the done event verbose (full tool results), as for the end-to-end harness."""
    admin = {"X-Admin-Token": settings.admin_token}
    r = await client.post(f"{settings.runtime_public_url}/api/demo/switch", json={"user_id": persona, "reset": reset}, headers=admin)
    if r.status_code != 200:
        raise RuntimeError(f"switch -> {r.status_code}: {r.text[:300]}")
    return {"Authorization": f"Bearer {r.json()['token']}", **admin, "X-Falcon-Tags": "eval,generation"}


def _merge_usage(a: dict, b: dict) -> dict:
    return {k: (a.get(k, 0) or 0) + (b.get(k, 0) or 0) for k in set(a) | set(b)}


async def ask(client: httpx.AsyncClient, headers: dict, item: dict) -> dict:
    """One question, one turn, in its own thread; a confirmation card (never expected for a policy question) is declined."""
    thread = f"gen-{item['id']}"
    events = await sse(client, "/api/chat", {**headers, "X-Falcon-Tags": f"eval,generation,q:{item['id']}"}, {"message": item["question"], "thread": thread})
    done = next(d for e, d in events if e == "done")
    out = {"reply": done.get("message", "") or "", "what_i_checked": list(done.get("what_i_checked") or []), "usage": dict(done.get("usage") or {}),
           "latency_ms": int(done.get("latency_ms") or 0), "verifications": [done["verification"]] if done.get("verification") else [],
           "guardrail": done.get("guardrail"), "api_error": done.get("error"), "trace_id": done.get("trace_id"), "lang_out": done.get("lang"),
           "review_rounds": [d for e, d in events if e == "review"]}
    while done.get("pending"):
        out["declined"] = [a.get("tool") for a in done["pending"].get("actions", [])]
        events = await sse(client, "/api/chat/confirm", headers, {"approved": False, "note": "", "thread": thread})
        done = next(d for e, d in events if e == "done")
        out["reply"] = (out["reply"] + "\n" + (done.get("message") or "")).strip()
        out["what_i_checked"] += list(done.get("what_i_checked") or [])
        out["usage"] = _merge_usage(out["usage"], done.get("usage") or {})
        out["latency_ms"] += int(done.get("latency_ms") or 0)
        if done.get("verification"):
            out["verifications"].append(done["verification"])
    return out


# ---------------------------------------------------------------- what the agent retrieved, and the deterministic checks
def policy_trace(what_i_checked: list[dict]) -> tuple[list[tuple[str, str]], list[dict], list[dict]]:
    """(tool results as the reviewer sees them, the clauses retrieved with their texts, the policy calls made) from a verbose turn log."""
    tools: list[tuple[str, str]] = []
    clauses: dict[str, dict] = {}
    calls: list[dict] = []
    for e in what_i_checked:
        name = e.get("tool", "")
        rj = e.get("result_json")
        tools.append((name, json.dumps(rj, ensure_ascii=False) if isinstance(rj, dict) else str(e.get("summary", ""))))
        if name not in POLICY_TOOLS:
            continue
        call = {"tool": name, "args": e.get("args"), "ok": e.get("ok"), "latency_ms": e.get("latency_ms")}
        if isinstance(rj, dict):
            call.update({k: rj[k] for k in ("confidence", "rewritten", "available", "status") if k in rj})
        calls.append(call)
        if not isinstance(rj, dict):
            continue
        if name == "search_policy":
            rows = rj.get("results") or []
        elif name == "get_clause":
            rows = [rj]
        else:  # check_travel_eligibility and get_article return `clauses`; get_article's rows are previews (clause_id, lang, preview)
            rows = rj.get("clauses") or []
        for r in rows:
            cid = str(r.get("clause_id") or "")
            if not cid:
                continue
            base = cid.split("#")[0]
            c = clauses.setdefault(base, {"clause_id": base, "article": r.get("article"), "text_en": None, "text_ar": None})
            for lang in ("en", "ar"):
                if r.get(f"text_{lang}"):
                    c[f"text_{lang}"] = r[f"text_{lang}"]
            if r.get("preview") and not c.get(f"text_{r.get('lang', 'en')}"):
                c[f"text_{r.get('lang', 'en')}"] = str(r["preview"]) + " …"
    return tools, list(clauses.values()), calls


def deterministic(reply: str, tools: list[tuple[str, str]], clauses: list[dict], item: dict) -> dict:
    """The reviewer's policy checks re-run on the final reply, plus the gold-label checks that only the evaluation can make."""
    retrieved = {c["clause_id"] for c in clauses}
    cited = cited_clause_ids(reply)
    problems = [p for p, _ in policy_problems(reply, tools)]
    relevant = item.get("relevant") or {}
    answering = {c for c, g in relevant.items() if g >= 2}
    return {
        "policy_tool_called": any(name in POLICY_TOOLS for name, _ in tools),
        "retrieved": sorted(retrieved), "cited": cited, "cites_clause": bool(cited),
        "unretrieved_clause": "unretrieved_clause" in problems,
        "unsupported_quantities": "unsupported_quantities" in problems,
        "unsupported_regulation": "unsupported_regulation" in problems,
        "uncited_policy_answer": "uncited_policy_answer" in problems,
        "not_covered_pattern": bool(NOT_COVERED_RE.search(reply or "")),
        "answering_retrieved": bool(retrieved & answering) if answering else None,
        "answering_cited": bool(set(cited) & answering) if answering else None,
        "labelled_cited": bool(set(cited) & set(relevant)) if relevant else None,
    }


# ---------------------------------------------------------------- the judge
def render_brief(question: str, lang: str, clauses: list[dict], reply: str) -> str:
    """What the judge sees: the question, the retrieved clause texts in the question's language (English when the Arabic is missing), the reply."""
    lines = [f"QUESTION ({'Arabic' if lang == 'ar' else 'English'}):", question, "", "RETRIEVED CLAUSES (grade against these texts alone):"]
    if not clauses:
        lines.append("(none: the assistant retrieved no clause in this turn)")
    for c in clauses:
        text = c.get(f"text_{lang}") or c.get("text_en") or c.get("text_ar") or ""
        lines.append(f"[{c['clause_id']}] {c.get('article') or ''}: {text}")
    lines += ["", "REPLY:", reply or "(empty reply)"]
    return "\n".join(lines)


def parse_judge_json(raw: str) -> dict:
    """The JSON object in the judge's answer, tolerant of code fences and prose around it; {} when there is none."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        data = json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
        except ValueError:
            return {}
    return data if isinstance(data, dict) else {}


def _score(v) -> int | None:
    try:
        n = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, n))


def _flag(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "false"):
        return v.strip().lower() == "true"
    return None


def _lookup(data: dict, key: str):
    """The value under `key`, or under the closest key the judge wrote: Qwen once answered "faith游戏副本ness" (a stray multilingual token
    inside the key), which cost one of 48 gradings."""
    if key in data:
        return data[key]
    import difflib

    cleaned = {re.sub(r"[^a-z_]", "", str(k).lower()): k for k in data}
    close = difflib.get_close_matches(key, list(cleaned), n=1, cutoff=0.6)
    return data[cleaned[close[0]]] if close else None


def normalise_scores(data: dict) -> dict:
    out = {d: _score(_lookup(data, d)) for d in DIMENSIONS}
    out["said_not_covered"] = _flag(_lookup(data, "said_not_covered"))
    out["reason"] = str(_lookup(data, "reason") or "").strip()
    return out


class Judge:
    def __init__(self, model: str | None = None):
        self.model = model or settings.judge_model
        self._client = None
        self.calls = self.errors = self.input_tokens = self.output_tokens = 0

    @property
    def enabled(self) -> bool:
        return bool(settings.openrouter_api_key)

    @property
    def cost_usd(self) -> float:
        return self.input_tokens / 1e6 * settings.judge_price_in_per_m + self.output_tokens / 1e6 * settings.judge_price_out_per_m

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key, timeout=90, max_retries=2,
                                       default_headers={"HTTP-Referer": settings.runtime_public_url, "X-Title": "Falcon generation judge"})
        return self._client

    async def grade(self, question: str, lang: str, clauses: list[dict], reply: str) -> dict:
        """One grading call: temperature 0, JSON output (a provider that refuses response_format is asked again without it)."""
        if not self.enabled:
            return {"error": "no OPENROUTER_API_KEY: the judge needs a key"}
        messages = [{"role": "system", "content": RUBRIC}, {"role": "user", "content": render_brief(question, lang, clauses, reply)}]
        kwargs = dict(model=self.model, temperature=0, max_tokens=600, messages=messages)
        self.calls += 1
        try:
            try:
                resp = await self._get_client().chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            except Exception as exc:  # noqa: BLE001 - some providers reject response_format; the parser copes with prose around the JSON
                if "response_format" not in str(exc).lower() and "json" not in str(exc).lower():
                    raise
                resp = await self._get_client().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001
            self.errors += 1
            return {"error": f"{type(exc).__name__}: {exc}"[:300]}
        usage = getattr(resp, "usage", None)
        if usage:
            self.input_tokens += usage.prompt_tokens or 0
            self.output_tokens += usage.completion_tokens or 0
        raw = resp.choices[0].message.content or ""
        data = parse_judge_json(raw)
        out = normalise_scores(data)
        if any(out[d] is None for d in DIMENSIONS):
            self.errors += 1
            out["error"] = "judge answer missing a score"
        out["raw"] = raw[:1500]
        return out


# ---------------------------------------------------------------- scoring one answer, running the set
def score_answer(item: dict, asked: dict) -> dict:
    tools, clauses, calls = policy_trace(asked["what_i_checked"])
    det = deterministic(asked["reply"], tools, clauses, item)
    verifs = asked.get("verifications") or []
    return {"id": item["id"], "topic": item["topic"], "lang": item["lang"], "source": item["source"], "question": item["question"],
            "answerable": item["answerable"], "relevant": item.get("relevant") or {}, "note": item.get("note"),
            "reply": asked["reply"], "clauses": clauses, "policy_calls": calls,
            "tools_called": [e.get("tool") for e in asked["what_i_checked"]],
            "reviewer": {"status": ", ".join(v.get("status", "?") for v in verifs) or "n/a",
                         "sent_back": sum(len(v.get("rounds") or []) for v in verifs),
                         "problems": [p for v in verifs for r in (v.get("rounds") or []) for p in (r.get("problems") or [])]},
            "guardrail": (asked.get("guardrail") or {}).get("label"), "api_error": asked.get("api_error"), "declined": asked.get("declined"),
            "usage": asked["usage"], "latency_ms": asked["latency_ms"], "trace_id": asked.get("trace_id"), "deterministic": det}


def append_row(out: Path | None, row: dict) -> None:
    if out is not None:
        with (out / "answers.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


async def run_questions(items: list[dict], reset: bool, out: Path | None = None) -> list[dict]:
    """Ask every item in turn (English first, then Arabic, one sign-in each) and append each scored row to out/answers.jsonl as it lands."""
    rows = []
    async with httpx.AsyncClient(timeout=240) as client:
        first = True
        for lang in ("en", "ar"):
            todo = [i for i in items if i["lang"] == lang]
            if not todo:
                continue
            headers = await sign_in(client, PERSONA[lang], reset=reset and first)
            first = False
            for item in todo:
                started = time.perf_counter()
                asked, error = None, None
                for attempt in (1, 2):
                    try:
                        asked = await ask(client, headers, item)
                        break
                    except Exception as exc:  # noqa: BLE001 - one retry, as the end-to-end runner does after a provider error
                        error = f"{type(exc).__name__}: {exc}"[:300]
                        if attempt == 1:
                            print(f"     {item['id']}: {error}; retrying once in 15 s", flush=True)
                            await asyncio.sleep(15)
                            headers = await sign_in(client, PERSONA[lang], reset=False)
                if asked is None:
                    row = {"id": item["id"], "topic": item["topic"], "lang": lang, "source": item["source"], "question": item["question"],
                           "answerable": item["answerable"], "relevant": item.get("relevant") or {}, "error": error, "reply": "", "clauses": [],
                           "deterministic": {}, "usage": {}, "latency_ms": int((time.perf_counter() - started) * 1000)}
                    rows.append(row)
                    append_row(out, row)
                    print(f"ERROR {item['id']}: {error}", flush=True)
                    continue
                row = score_answer(item, asked)
                rows.append(row)
                append_row(out, row)
                d = row["deterministic"]
                print(f"  {item['id']}: {row['latency_ms']} ms, tools {row['tools_called']}, cited {d['cited']}, reviewer {row['reviewer']['status']}"
                      + (f", API error {row['api_error']}" if row.get("api_error") else ""), flush=True)
    return rows


async def judge_rows(rows: list[dict], judge: Judge, force: bool = False) -> None:
    """Grade every answered row; a row already graded (a resumed run) is kept unless `force` (--rejudge)."""
    for row in rows:
        if row.get("error") or (not force and row.get("judge") and not row["judge"].get("error")):
            continue
        row["judge"] = await judge.grade(row["question"], row["lang"], row["clauses"], row["reply"])
        j = row["judge"]
        print(f"  judge {row['id']}: " + (f"F{j['faithfulness']} R{j['relevance']} C{j['completeness']} not_covered={j['said_not_covered']}" if not j.get("error") else j["error"]), flush=True)


# ---------------------------------------------------------------- report
def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _fmt(v: float | None, digits: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"


def _percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def agreement(pairs: list[tuple[int, int]]) -> dict:
    """Judge-vs-label agreement over (judge, label) pairs: exact, within one point, mean absolute difference, mean bias (judge − label)."""
    n = len(pairs)
    if not n:
        return {"n": 0, "exact": None, "within_1": None, "mean_abs": None, "bias": None}
    return {"n": n, "exact": sum(a == b for a, b in pairs) / n, "within_1": sum(abs(a - b) <= 1 for a, b in pairs) / n,
            "mean_abs": sum(abs(a - b) for a, b in pairs) / n, "bias": sum(a - b for a, b in pairs) / n}


def honest(row: dict) -> bool | None:
    """An unanswerable question answered honestly: the judge saw it say the document does not cover it, with nothing invented."""
    j = row.get("judge") or {}
    if j.get("error") or j.get("said_not_covered") is None:
        return None
    return bool(j["said_not_covered"]) and (j.get("faithfulness") or 0) >= 4


def report(rows: list[dict], judge: Judge, started: float, reset: str, validation: str | None = None) -> str:
    judged = [r for r in rows if r.get("judge") and not r["judge"].get("error")]
    answerable = [r for r in judged if r["answerable"]]
    unanswerable = [r for r in judged if not r["answerable"]]
    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    lines = ["# Generation evaluation (policy answers from retrieved clauses)", "",
             f"{datetime.now():%Y-%m-%d %H:%M} · {len(rows)} questions ({sum(1 for r in rows if r['answerable'])} answerable, "
             f"{sum(1 for r in rows if not r['answerable'])} not covered by the document; {sum(1 for r in rows if r.get('source') == 'extra')} extra) · "
             f"agent: {settings.llm_provider}:{settings.llm_model} · judge: {judge.model} (temperature 0, JSON) · demo data reset: {reset} · "
             f"{time.time() - started:.0f} s", ""]
    if errors:
        lines += [f"**{len(errors)} question(s) did not get an answer** (API or stack error; not scored): " + ", ".join(f"`{r['id']}`" for r in errors), ""]

    lines += ["## Judge scores on answerable questions (1-5)", "",
              "| lang | n | faithfulness | relevance | completeness | cites ≥1 clause | answering clause retrieved | answering clause cited | reviewer sent back |",
              "|---|---|---|---|---|---|---|---|---|"]
    for lang in ("all", "en", "ar"):
        sub = [r for r in answerable if lang == "all" or r["lang"] == lang]
        if not sub:
            continue
        lab = [r for r in sub if r["deterministic"].get("answering_retrieved") is not None]
        lines.append(f"| {lang} | {len(sub)} | " + " | ".join(f"**{_fmt(_mean([r['judge'][d] for r in sub]))}**" for d in DIMENSIONS)
                     + f" | {_pct(sum(r['deterministic']['cites_clause'] for r in sub), len(sub))}"
                     + f" | {_pct(sum(r['deterministic']['answering_retrieved'] for r in lab), len(lab))}"
                     + f" | {_pct(sum(r['deterministic']['answering_cited'] for r in lab), len(lab))}"
                     + f" | {sum(r['reviewer']['sent_back'] for r in sub)} |")
    dist = {d: {s: sum(1 for r in answerable if r["judge"][d] == s) for s in range(1, 6)} for d in DIMENSIONS}
    lines += ["", "Score distribution (answerable, count of answers at 1 / 2 / 3 / 4 / 5): "
              + " · ".join(f"{d} {' / '.join(str(dist[d][s]) for s in range(1, 6))}" for d in DIMENSIONS), ""]

    lines += ["## Questions the document does not cover (said so, invented nothing?)", "",
              "| id | honest (judge) | 'not covered' pattern | cited | faithfulness | judge's reason |", "|---|---|---|---|---|---|"]
    for r in unanswerable:
        lines.append(f"| `{r['id']}` | {'yes' if honest(r) else 'no'} | {'yes' if r['deterministic']['not_covered_pattern'] else 'no'} | "
                     f"{', '.join(r['deterministic']['cited']) or '-'} | {r['judge']['faithfulness']} | {r['judge']['reason'][:220]} |")
    if unanswerable:
        lines += ["", f"Answered honestly: **{_pct(sum(bool(honest(r)) for r in unanswerable), len(unanswerable))}** by the judge; "
                      f"{_pct(sum(r['deterministic']['not_covered_pattern'] for r in unanswerable), len(unanswerable))} by the reviewer's own 'not covered' pattern."]

    det = [r["deterministic"] for r in ok]
    lines += ["", "## Deterministic checks on the final replies (the runtime reviewer's own, re-run here)", "",
              f"- policy tool called in the turn: {_pct(sum(d['policy_tool_called'] for d in det), len(det))}",
              f"- cites at least one clause: {_pct(sum(d['cites_clause'] for d in det), len(det))}",
              f"- cited clause not retrieved in the turn: {sum(d['unretrieved_clause'] for d in det)}",
              f"- figure with a unit in no tool result: {sum(d['unsupported_quantities'] for d in det)}",
              f"- law or regulation named from memory: {sum(d['unsupported_regulation'] for d in det)}",
              f"- speaks for the Conditions without a citation or a 'not covered': {sum(d['uncited_policy_answer'] for d in det)}",
              f"- replies the reviewer sent back before these (corrective rounds): {sum(r['reviewer']['sent_back'] for r in ok)} "
              f"({', '.join(sorted({p for r in ok for p in r['reviewer']['problems']})) or 'none'}); held: {sum(r['reviewer']['status'] == 'held' for r in ok)}",
              f"- blocked by the guardrail: {sum(r['reviewer']['status'] == 'blocked' for r in ok)} · confirmation card raised (declined): {sum(bool(r.get('declined')) for r in ok)}",
              f"- judge calls that failed or returned no score: {judge.errors} / {judge.calls}"]
    lab = [r for r in ok if r["deterministic"].get("answering_retrieved") is not None]
    if lab:
        missed = [r["id"] for r in lab if not r["deterministic"]["answering_retrieved"]]
        lines += [f"- answering clause (gold grade 2) retrieved in the turn: {_pct(len(lab) - len(missed), len(lab))}"
                  + (f"; missed: {', '.join(f'`{m}`' for m in missed)}" if missed else "")]

    worst = sorted(judged, key=lambda r: (sum(r["judge"][d] for d in DIMENSIONS), r["judge"]["faithfulness"]))[:5]
    lines += ["", "## Worst five answers (by the sum of the three scores)", ""]
    for r in worst:
        j = r["judge"]
        lines += [f"- `{r['id']}` F{j['faithfulness']} R{j['relevance']} C{j['completeness']}: {j['reason']}",
                  f"  cited {r['deterministic']['cited'] or 'nothing'}; retrieved {r['deterministic']['retrieved'] or 'nothing'}; "
                  f"policy calls {[(c['tool'], (c.get('args') or {}).get('query') or (c.get('args') or {}).get('clause_id') or '') for c in r['policy_calls']]}"]

    agent_in = sum((r.get("usage") or {}).get("input_tokens", 0) or 0 for r in rows)
    agent_out = sum((r.get("usage") or {}).get("output_tokens", 0) or 0 for r in rows)
    agent_cost = agent_in / 1e6 * settings.llm_price_in_per_m + agent_out / 1e6 * settings.llm_price_out_per_m
    lat = [r["latency_ms"] for r in ok]
    lines += ["", "## Cost and latency", "",
              f"- agent: {agent_in:,} input + {agent_out:,} output tokens, about USD {agent_cost:.3f}; turn latency p50 {_percentile(lat, 0.5)} ms, p95 {_percentile(lat, 0.95)} ms",
              f"- judge: {judge.input_tokens:,} input + {judge.output_tokens:,} output tokens, about USD {judge.cost_usd:.4f} "
              f"(USD {settings.judge_price_in_per_m}/{settings.judge_price_out_per_m} per million)"]
    if validation:
        lines += ["", validation]
    return "\n".join(lines)


def write_run(out: Path, rows: list[dict], text: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in rows), encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")


def read_run(run_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "answers.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- judge validation against hand labels
def load_labels() -> list[dict]:
    return (yaml.safe_load(LABELS.read_text(encoding="utf-8")) or []) if LABELS.exists() else []


def labelled(entries: list[dict], who: str) -> list[dict]:
    return [e for e in entries if any((e.get(who) or {}).get(d) is not None for d in DIMENSIONS + ("said_not_covered",))]


async def clause_texts(ids: list[str]) -> list[dict]:
    """The clause texts behind a labelled reply, read from the policy index by id (the labels file keeps ids, not texts)."""
    from falcon.core.db import Database
    from falcon.policy.search import PolicySearch

    db = Database(settings.dsn_policy, max_size=2)
    await db.open()
    try:
        search = PolicySearch(db)
        rows = [await search.clause(cid) for cid in ids]
    finally:
        await db.close()
    return [r for r in rows if r]


def agreement_report(entries: list[dict], graded: dict[str, dict]) -> str:
    lines = ["## Judge validation against hand labels", "",
             f"{len(graded)} labelled replies graded again by the judge (eval/generation/labels.yaml). Exact = same score; within 1 = differ by at most one point; "
             "bias = judge minus label, so a positive bias means the judge is kinder than the person.", ""]
    for who in LABELLERS:
        have = [e for e in labelled(entries, who) if e["id"] in graded]
        if not have:
            lines += [f"**{who}**: no labels yet.", ""]
            continue
        lines += [f"**{who}** ({len(have)} replies)", "", "| dimension | n | exact | within 1 | mean |judge − label| | bias |", "|---|---|---|---|---|---|"]
        for d in DIMENSIONS:
            pairs = [(graded[e["id"]][d], int(e[who][d])) for e in have if e[who].get(d) is not None and graded[e["id"]].get(d) is not None]
            a = agreement(pairs)
            lines.append(f"| {d} | {a['n']} | {_fmt(a['exact'])} | {_fmt(a['within_1'])} | {_fmt(a['mean_abs'])} | {_fmt(a['bias'], 2) if a['bias'] is not None else 'n/a'} |")
        flags = [(graded[e["id"]]["said_not_covered"], bool(e[who]["said_not_covered"])) for e in have
                 if e[who].get("said_not_covered") is not None and graded[e["id"]].get("said_not_covered") is not None]
        if flags:
            lines.append(f"| said_not_covered | {len(flags)} | {_fmt(sum(a == b for a, b in flags) / len(flags))} | | | |")
        lines.append("")
        lines += ["Per reply (judge / label): " + " · ".join(
            f"`{e['id']}` " + " ".join(f"{d[0].upper()}{graded[e['id']].get(d)}/{e[who].get(d) if e[who].get(d) is not None else '-'}" for d in DIMENSIONS) for e in have), ""]
    return "\n".join(lines)


async def validate(judge: Judge) -> str | None:
    """Grade the labelled replies as they are in labels.yaml (not fresh ones: the agent is not deterministic) and report agreement."""
    entries = load_labels()
    todo = [e for e in entries if any(labelled([e], who) for who in LABELLERS)]
    if not todo:
        return None
    graded = {}
    for e in todo:
        clauses = await clause_texts(list(e.get("retrieved") or []))
        g = await judge.grade(e["question"], e["lang"], clauses, e["reply"])
        if g.get("error"):
            print(f"  validate {e['id']}: {g['error']}", flush=True)
            continue
        graded[e["id"]] = g
        print(f"  validate {e['id']}: F{g['faithfulness']} R{g['relevance']} C{g['completeness']} not_covered={g['said_not_covered']}", flush=True)
    return agreement_report(entries, graded)


# ---------------------------------------------------------------- label stubs from a run
class _Dumper(yaml.SafeDumper):
    pass


def _str_presenter(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_Dumper.add_representer(str, _str_presenter)


def draft_labels(run_dir: Path, ids: list[str]) -> tuple[Path, Path]:
    """Label stubs (labels_draft.yaml) and a reading sheet with the clause texts (labels_context.md) for chosen answers of a run,
    written into the run's folder (the only path the eval container shares with the laptop); copy them to eval/generation/ by hand."""
    rows = {r["id"]: r for r in read_run(run_dir)}
    stubs, sheet = [], ["# Reading sheet for the generation labels", "",
                        "For each reply: the question, the clauses the agent retrieved (in the question's language) and the reply. Label in "
                        "labels.yaml with the rubric of `src/falcon/eval/generation.py` (RUBRIC): faithfulness, relevance, completeness 1-5, "
                        "said_not_covered true/false. Judge the reply against these clause texts only.", ""]
    for i in ids:
        r = rows[i]
        empty = {d: None for d in DIMENSIONS} | {"said_not_covered": None, "note": ""}
        stubs.append({"id": i, "lang": r["lang"], "answerable": r["answerable"], "question": r["question"], "retrieved": r["deterministic"].get("retrieved", []),
                      "reply": r["reply"].strip(), "hozayfa": dict(empty), "claude": dict(empty)})
        sheet += [f"## {i}", "", f"**Question ({r['lang']}):** {r['question']}", "", "**Retrieved clauses:**", ""]
        for c in r["clauses"]:
            sheet.append(f"- [{c['clause_id']}] {c.get('article') or ''}: {c.get('text_' + r['lang']) or c.get('text_en') or c.get('text_ar') or ''}")
        if not r["clauses"]:
            sheet.append("- (none retrieved)")
        sheet += ["", "**Reply:**", "", *("> " + line for line in r["reply"].strip().splitlines()), ""]
    draft = run_dir / "labels_draft.yaml"
    context = run_dir / "labels_context.md"
    draft.write_text(yaml.dump(stubs, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=130), encoding="utf-8")
    context.write_text("\n".join(sheet), encoding="utf-8")
    return draft, context


# ---------------------------------------------------------------- main
async def main_async(args) -> None:
    judge = Judge(args.judge)
    started = time.time()
    if args.draft_labels:
        draft, context = draft_labels(Path(args.draft_labels), [i.strip() for i in (args.ids or "").split(",") if i.strip()])
        print(f"written {draft} and {context}")
        return
    if args.validate:
        text = await validate(judge)
        if text is None:
            raise SystemExit("no labels filled in eval/generation/labels.yaml yet")
        out = RESULTS / f"generation-validate-{datetime.now():%Y%m%d-%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(text, encoding="utf-8")
        print("\n" + text + f"\n\nwritten to {out}")
        return
    if args.rejudge:
        rows = read_run(Path(args.rejudge))
        reset = f"as in {Path(args.rejudge).name} (graded again)"
        out = RESULTS / f"generation-{datetime.now():%Y%m%d-%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
    elif args.resume:
        out = Path(args.resume)
        previous = [r for r in read_run(out) if not r.get("error")]
        done = {r["id"] for r in previous}
        items = [i for i in load_questions(args.filter) if i["id"] not in done]
        reset = "no (resumed run)"
        print(f"resuming {out.name}: {len(previous)} answers kept, {len(items)} questions to ask · judge {judge.model}", flush=True)
        (out / "answers.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in previous), encoding="utf-8")
        rows = previous + await run_questions(items, False, out)
    else:
        items = load_questions(args.filter)
        if not items:
            raise SystemExit("no questions selected")
        reset = "no" if args.no_reset else "yes"
        out = RESULTS / f"generation-{datetime.now():%Y%m%d-%H%M%S}"
        out.mkdir(parents=True, exist_ok=True)
        print(f"{len(items)} questions · agent {settings.llm_provider}:{settings.llm_model} · judge {judge.model} · reset {reset} · writing to {out}", flush=True)
        rows = await run_questions(items, not args.no_reset, out)
    await judge_rows(rows, judge, force=bool(args.rejudge))
    validation = None if args.no_validate else await validate(judge)
    text = report(rows, judge, started, reset, validation)
    write_run(out, rows, text)
    print("\n" + text)
    print(f"\nwritten to {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--filter", default=None, help="substring of question ids (e.g. medical, _ar)")
    ap.add_argument("--no-reset", action="store_true", help="keep the demo data (another session is using the stack)")
    ap.add_argument("--judge", default=None, help="judge model id on OpenRouter (default: JUDGE_MODEL setting)")
    ap.add_argument("--resume", default=None, help="a results folder of an interrupted run: ask only the questions it has no answer for")
    ap.add_argument("--rejudge", default=None, help="a results folder: grade its answers again without asking the agent")
    ap.add_argument("--validate", action="store_true", help="grade the hand-labelled replies only and print the agreement")
    ap.add_argument("--no-validate", action="store_true", help="skip the label validation after a run")
    ap.add_argument("--draft-labels", default=None, help="a results folder: write label stubs and a reading sheet for --ids into it")
    ap.add_argument("--ids", default=None, help="comma-separated answer ids for --draft-labels")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
