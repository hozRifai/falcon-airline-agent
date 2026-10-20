"""The agent graph.

    prepare ──► guard ──► agent ──► (no tool calls) ──► END
                  │       ▲  │
                  └► (blocked: fixed refusal) ──► END
                 │  └──► write tool among the calls? ──► gate (interrupt: person confirms) ──► tools ──┐
                 │                             └──────────────── read-only calls ───────────► tools ──┤
                 └─────────────────────────────────────────────────────────────────────────────────────┘

- prepare: deterministic language detection on the last human message; the reply language is a fact, not a guess.
- guard: the input guardrail (guardrails.py, decision #29): personal data is removed from the message before anything else
  sees it, the message is classified (patterns, then a small model), and an out-of-scope request or a prompt injection is
  answered with a fixed refusal without the agent ever running; social engineering and abuse reach the agent with a note.
- agent: one model call with the persona's tools bound; the system prompt carries role, language, time, customer name.
  Every final draft (no tool calls) is reviewed against the conversation's tool trace (review.py: amounts, action claims,
  policy sources, plus the output guards below); a failing draft goes back to the model with the exact problem, up to
  REVIEW_MAX_ROUNDS times, then the customer gets a safe "could not resolve" message. The outcome is recorded in
  state["verification"] and returned to the UI and the evaluation harness.
- gate: LangGraph `interrupt()` before any tool in WRITE_TOOLS. The UI shows a button with the exact action; on
  resume with {approved: false} the tool calls are answered with a "declined" tool message, so the model
  cannot proceed as if the action happened.
- tools: executes the MCP tools and records every call (name, args, latency, ok, summary) into the turn log,
  which the API returns as "What I checked".
- memory: the whole state is checkpointed per session in Postgres (schema `agent`), so a conversation survives
  a restart and a confirmation can be resumed from another request.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from falcon.core.arabic import preferred_reply_language
from falcon.core.config import settings
from falcon.core.mcp_server import redact
from falcon.core.telemetry import tracer

from .guardrails import guard as guard_message
from .guardrails import refusal
from .personas import WRITE_TOOLS, tools_for_topic
from .prompts import system_prompt
from .review import HELD, HELD_POLICY, POLICY_HOLD_PROBLEMS, citation_support, cited_clause_ids, model_review, review_draft

log = logging.getLogger("falcon.runtime")

# Said when the tool-call cap cuts a turn off. It promises no handover: none has been created at that point.
CUT_OFF = {
    "en": "I could not finish this request: it needed more steps than I am allowed in one turn. Please tell me again what you would like, or ask for a human agent.",
    "ar": "لم أتمكن من إكمال هذا الطلب لأنه احتاج إلى خطوات أكثر مما يُسمح لي به في رد واحد. يرجى إخباري مرة أخرى بما تريد، أو طلب التحدث إلى موظف.",
}


# Said by the runtime, not the model, when two replies in a row could not be resolved and a handover case was logged (decision #43).
ESCALATED = {
    "en": ("I have not been able to resolve this for you twice in a row, so I have logged case #{case} for a human agent with a summary of our "
           "conversation; the team will reply to you by email. I can still help with your bookings in the meantime."),
    "ar": ("لم أتمكن من معالجة طلبك مرتين متتاليتين، لذلك سجّلت الحالة رقم {case} لموظف مع ملخص لمحادثتنا؛ سيرد عليك الفريق بالبريد الإلكتروني. "
           "يمكنني في هذه الأثناء مساعدتك في حجوزاتك."),
}
UNRESOLVED_STATUSES = {"held", "cut_off"}


# "Press the confirm button below" with no tool call: the button exists only when a write tool is called (decision #41 again: the
# full run of 19 Oct lost a rebooking this way after a correct quote, in Arabic). With the quote on the table, the runtime raises the card.
CARD_ANNOUNCE_RE = re.compile(r"confirm(?:ation)? button|press confirm|click confirm|زر التأكيد|اضغط\S* (?:على )?(?:زر )?(?:التأكيد|تأكيد)|سيظهر (?:لك )?زر", re.IGNORECASE)


def card_shortcut(state: AgentState, response: AIMessage) -> AIMessage | None:
    """A final draft that tells the customer to press the confirm button, with an allowed unconsumed quote from this turn: the matching
    write call instead, so the gate shows the card the draft promised."""
    if response.tool_calls or not CARD_ANNOUNCE_RE.search(_text_of(response)):
        return None
    quote = open_quote(list(state.get("messages", [])), this_turn=True)
    if not quote:
        return None
    tool = QUOTE_ACTION[quote["kind"]]
    args = {"quote_id": quote["quote_id"]} if tool == "create_booking" else {"pnr": quote.get("pnr"), "quote_id": quote["quote_id"]}
    log.warning("draft announced the confirm button without a tool call: raising the card for %s %s", tool, quote["quote_id"])
    return AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"card-{quote['quote_id']}"}], additional_kwargs=response.additional_kwargs,
                     usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)


# Output guards (FAILURE_DIARY #29-31). A chunk of 20-200 characters repeated back to back three times is a
# degenerate generation, not a list (list items differ in flight number or date). An announced check with no tool
# call, before any tool in the turn, leaves the customer waiting for something that will never come.
REPEATED = re.compile(r"(.{20,200}?)\1\1", re.DOTALL)
# DeepSeek's own scaffolding, seen verbatim in replies (FAILURE_DIARY #29, #47); never part of an answer to a customer
LEAK_RE = re.compile(r"Use the results below to formulate an answer[^.\n]*(?:needed)?\.?|unless additional information is needed\.?", re.IGNORECASE)
REPEATED_RUN = re.compile(r"(.{20,200}?)\1+", re.DOTALL)
ANNOUNCED = re.compile(r"\b(I'?ll|I will|let me|I am going to|I'm going to|let's|I'm) (now |go ahead and |start |just )?(check|look|verify|search|find|review|cancel|change|book|process|log|proceed|rebook|add|show|start)\b|"
                       r"\bsearching now\b|\bone moment\b|\bplease wait\b|\b(click|press|tap) (the |on the )?confirm(ation)? button\b|\bconfirm(ation)? button (will|below)\b|"
                       r"سأتحقق|سأقوم بالتحقق|سأبحث|أبحث عن|لنبحث|دعني أبحث|سأبدأ البحث|جاري البحث|سأقوم بالبحث|سأعرض|سأراجع|لحظة من فضلك|يرجى الانتظار|سأقوم ب|سألغي|سأحجز|سأغيّر|سأسجل|سأسجّل|زر التأكيد",
                       re.IGNORECASE)
RETRY_NUDGE = {
    "repeated": ("Your previous draft repeated the same text over and over and was discarded. Write the reply to the "
                 "person now, once, in the language of their last message, from the tool results above. Do not call "
                 "a tool unless you still need information you do not have."),
    "announced": ("Your previous draft said you would do or check something, or told the customer to press a confirmation button, but called "
                  "no tool, so nothing would happen and no button exists. Call the tool now, in this reply, instead of announcing it: the "
                  "confirmation button appears only when you call the action tool (cancel_booking, change_booking, create_booking, ...)."),
}
GARBLED = {
    "en": "Sorry, my reply came out garbled. Anything listed under \"What I checked\" did happen; ask me again and I will summarise it.",
    "ar": "عذرًا، خرج ردّي مشوشًا. ما يظهر تحت \"ما الذي تحققت منه\" قد تم بالفعل؛ اسألني مرة أخرى وسألخصه لك.",
}


def _reply_problem(response: AIMessage, state: "AgentState") -> str | None:
    text = _text_of(response)
    if REPEATED.search(text):
        return "repeated"
    if not response.tool_calls and not state.get("turn_tool_calls") and ANNOUNCED.search(text):
        return "announced"
    return None


def _strip_leak(response: AIMessage) -> AIMessage:
    """DeepSeek's scaffolding phrase is removed from a draft in place, tool calls kept: it is cosmetic, and retrying it four
    times once cost a booking (the model kept emitting it; the final run of 18 Oct)."""
    if isinstance(response.content, str) and LEAK_RE.search(response.content):
        response.content = LEAK_RE.sub("", response.content).strip()
    return response


def _strip_repeats(text: str, lang: str) -> str:
    left = REPEATED_RUN.sub("", LEAK_RE.sub("", text)).strip()
    return left if len(left) >= 20 else GARBLED.get(lang, GARBLED["en"])


def _add_usage(a: dict, b: dict) -> dict:
    return {k: (a.get(k) or 0) + (b.get(k) or 0) for k in ("input_tokens", "output_tokens", "total_tokens")}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    unresolved: int  # replies in a row the customer could not use (held or cut off); reset by any reply that went out (decision #43)
    lang: str
    role: str
    user_name: str
    preferred_lang: str
    turn_log: list[dict]
    turn_tool_calls: int
    pending: dict | None
    verification: dict | None
    guardrail: dict | None
    disclosed: list[str]  # kinds of travel-affecting facts disclosed so far in this thread (decision #34)


def _text_of(msg: AnyMessage) -> str:
    c = msg.content
    if isinstance(c, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
    return str(c)


def _summary(result: Any) -> str:
    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
    return text[:400]


def _tools_from(config: RunnableConfig) -> dict[str, Any]:
    return {t.name: t for t in config["configurable"].get("tools", [])}


async def prepare(state: AgentState, config: RunnableConfig) -> dict:
    last = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
    lang = preferred_reply_language(_text_of(last), state.get("lang", config["configurable"].get("default_lang", "en"))) if last else state.get("lang", "en")
    return {"lang": lang, "turn_log": [], "turn_tool_calls": 0, "pending": None, "verification": None}


def previous_reply(messages: list[AnyMessage]) -> str | None:
    """The assistant's last text reply before the customer's latest message: the classifier's context (decision #51)."""
    msgs = list(messages or [])
    last_human = max((i for i, m in enumerate(msgs) if isinstance(m, HumanMessage)), default=None)
    scope = msgs[:last_human] if last_human is not None else msgs
    for m in reversed(scope):
        if isinstance(m, AIMessage) and _text_of(m).strip():
            return _text_of(m)
    return None


async def guard(state: AgentState, config: RunnableConfig) -> dict:
    """Input guardrail (decision #29). The app's opening message is not a customer message and is not classified."""
    last = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
    text = _text_of(last) if last else ""
    if last is None or not text.strip() or text.startswith("[[opening]]"):
        return {"guardrail": None}
    lang = state.get("lang", "en")
    pre = config["configurable"].get("guard_precomputed")  # the API ran the classifier while it fetched the tools (decision #47)
    context = previous_reply(state.get("messages", []))
    if pre and pre.get("text") == text and pre.get("context") == context:
        clean, g = pre["clean"], pre["guard"]
    else:
        clean, g = await guard_message(text, lang, context)
    record = g.to_dict()
    prior = list(state.get("disclosed") or [])
    kinds = list(dict.fromkeys(d.kind for d in g.disclosures))
    # a kind disclosed earlier in this conversation: the reviewer then accepts a policy check anywhere in it, not only in this turn
    record["repeat"] = [k for k in kinds if k in prior]
    out: dict = {"guardrail": record, "disclosed": sorted(set(prior) | set(kinds))}
    if clean != text:
        # same id: the redacted text replaces the message in the checkpoint, so the model, the logs and the traces never see the data
        out["messages"] = [HumanMessage(content=clean, id=last.id)]
    if g.action == "block":
        log.warning("guardrail blocked a message: %s (%s)", g.label, g.reason)
        out["messages"] = out.get("messages", []) + [AIMessage(content=refusal(g.label, lang))]
        out["verification"] = {"status": "blocked", "rounds": [], "amounts": [], "evidence": 0, "checked": True, "reviewer": "guardrail"}
    return out


# A bare "yes" after a quote is not a judgement (decision #41): the runtime raises the confirmation card itself. DeepSeek, asked in
# Arabic, wrote "I will cancel now, press the confirm button" instead of calling the tool often enough that the Arabic customer saw
# the cut-off message and no card (FAILURE_DIARY #53).
# The vocabulary lives in falcon.core.confirmation so the voice channel uses the very same words (VOICE_PLAN rule 3).
from falcon.core.confirmation import CONFIRM_CORE, CONFIRM_FILLER, is_bare_confirmation  # noqa: E402,F401


QUOTE_ACTION = {"cancel": "cancel_booking", "change": "change_booking", "bag": "add_extra_bag", "book": "create_booking"}


OFFER_KEYS = ("kind", "pnr", "refund_method", "refund", "fee", "amount_due", "details")


def open_quote(messages: list[AnyMessage], this_turn: bool = False) -> dict | None:
    """The allowed quote the assistant presented in its last turn (or, with this_turn, since the customer's last message), when no
    write has consumed it yet; None otherwise. Two different offers in that turn (credit or card, two flights) are a question, not
    a quote: None, so no shortcut picks one (FAILURE_DIARY #65); the same offer quoted again counts once, and the latest is kept."""
    scope = messages if this_turn else messages[:-1]
    last_human = max((i for i, m in enumerate(scope) if isinstance(m, HumanMessage)), default=None)
    if last_human is None:
        return None
    quote = None
    offers: set[str] = set()
    for m in scope[last_human + 1:]:
        if isinstance(m, ToolMessage) and (m.name or "").startswith("quote_"):
            try:
                data = json.loads(_text_of(m))
            except Exception:
                continue
            if isinstance(data, dict) and data.get("allowed") and data.get("quote_id") and data.get("kind") in QUOTE_ACTION:
                quote = data
                offers.add(json.dumps({k: data.get(k) for k in OFFER_KEYS}, sort_keys=True, default=str))
    if not quote or len(offers) > 1:
        return None
    consumed = any(isinstance(m, ToolMessage) and (m.name or "") == QUOTE_ACTION[quote["kind"]] and quote["quote_id"] in _text_of(m) for m in messages)
    return None if consumed else quote


def confirmation_shortcut(state: AgentState) -> dict | None:
    """The write tool call for a bare confirmation of the quote just presented, so the card is raised without a model call."""
    msgs = state.get("messages", [])
    if not msgs or not isinstance(msgs[-1], HumanMessage) or not is_bare_confirmation(_text_of(msgs[-1])):
        return None
    quote = open_quote(msgs)
    if not quote:
        return None
    tool = QUOTE_ACTION[quote["kind"]]
    args = {"quote_id": quote["quote_id"]} if tool == "create_booking" else {"pnr": quote.get("pnr"), "quote_id": quote["quote_id"]}
    return {"name": tool, "args": args, "id": f"confirm-{quote['quote_id']}"}


def route_after_guard(state: AgentState) -> str:
    if (state.get("guardrail") or {}).get("action") == "block":
        return END
    return "confirm" if confirmation_shortcut(state) else "agent"


async def confirm(state: AgentState, config: RunnableConfig) -> dict:
    tc = confirmation_shortcut(state)
    log.info("confirmation shortcut: %s for %s", tc["name"], tc["args"].get("quote_id"))
    return {"messages": [AIMessage(content="", tool_calls=[tc])], "turn_tool_calls": state.get("turn_tool_calls", 0) + 1}


async def _review(response: AIMessage, state: AgentState) -> tuple[list[str], str, dict | None]:
    """All checks on one draft: output guards, then (for a final reply) the deterministic review and the optional model reviewer.
    Returns (problems, nudge for the retry, review record)."""
    guard = _reply_problem(response, state)
    if guard:
        return [guard], RETRY_NUDGE[guard], None
    if response.tool_calls:
        return [], "", None
    r = review_draft(response, state["messages"], lang=state.get("lang"), guardrail=state.get("guardrail"))
    record = {"problems": list(r.problems), "verdict": r.verdict.to_dict() if r.verdict else None}
    if r.ok and settings.review_citations and cited_clause_ids(_text_of(response)):
        r = await citation_support(response, state["messages"])
        record["problems"] = list(r.problems)
        if r.reviewer_note:
            record["reviewer_note"] = r.reviewer_note
    if r.ok and settings.review_llm:
        r = await model_review(response, state["messages"])
        record["problems"] = list(r.problems)
        record["reviewer_note"] = r.reviewer_note
    return list(r.problems), r.nudge, record


def turn_topic(state: AgentState) -> str | None:
    """The topic the guard's classifier named for this message, when routing is on and a real classifier named it (decision #46)."""
    g = state.get("guardrail") or {}
    if not settings.topic_routing or settings.llm_provider == "fake" or g.get("source") != "llm":
        return None
    topic = g.get("topic") or "general"
    return topic if topic != "general" else None


async def agent(state: AgentState, config: RunnableConfig) -> dict:
    llm = config["configurable"]["llm"]
    topic = turn_topic(state)
    tools = tools_for_topic(topic, list(_tools_from(config).values()))
    prompt = system_prompt(state["role"], lang=state["lang"], name=state["user_name"], preferred_lang=state.get("preferred_lang", "en"), topic=topic)
    model = llm.bind_tools(tools) if tools else llm
    note = (state.get("guardrail") or {}).get("note")  # what the guardrail flagged or removed in this turn's message
    messages = [SystemMessage(content=prompt + ("\n\n" + note if note else ""))] + list(state["messages"])
    lang = state.get("lang", "en")
    try:
        emit = get_stream_writer()  # "review" events for the UI: a draft sent back, a reply held (stream_mode "custom")
    except Exception:  # noqa: BLE001 - outside a streaming run (tests), there is nobody to tell
        emit = lambda _event: None  # noqa: E731
    started = time.perf_counter()
    response: AIMessage = await model.ainvoke(messages)
    response = _strip_leak(response)
    usage = dict(response.usage_metadata or {})
    rounds: list[dict] = []
    held = False
    last_record = None
    while True:
        problems, nudge, record = await _review(response, state)
        if record is not None:
            last_record = record
        if not problems:
            break
        rounds.append({"round": len(rounds) + 1, "problems": problems, **({"reviewer_note": record.get("reviewer_note")} if record and record.get("reviewer_note") else {})})
        emit({"type": "review", "round": len(rounds), "problems": problems, "final": len(rounds) > settings.review_max_rounds})
        if len(rounds) > settings.review_max_rounds:
            # Every round failed. A repeated-text draft can still be salvaged by stripping the repeats; anything else is held.
            if problems == ["repeated"]:
                response = AIMessage(content=_strip_repeats(_text_of(response), lang), id=response.id, usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)
            else:
                log.warning("reviewer held a reply after %d rounds: %s", len(rounds) - 1, [r["problems"] for r in rounds])
                with tracer("falcon.runtime").start_as_current_span("reviewer.hold", attributes={"falcon.rounds": len(rounds) - 1, "falcon.problems": ",".join(problems), "falcon.lang": lang}):
                    held_text = (HELD_POLICY if set(problems) & POLICY_HOLD_PROBLEMS else HELD).get(lang) or HELD["en"]
                    response = AIMessage(content=held_text, id=response.id, usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)
                held = True
            break
        log.warning("reviewer sent a draft back (round %d): %s", len(rounds), problems)
        try:
            # The draft is discarded, tool calls included (runaway drafts made needless reads); the model sees only the correction.
            with tracer("falcon.runtime").start_as_current_span("reviewer.round", attributes={"falcon.round": len(rounds), "falcon.problems": ",".join(problems), "falcon.nudge": nudge[:500]}):
                retried = _strip_leak(await model.ainvoke(messages + [SystemMessage(content=nudge)]))
        except Exception as exc:  # noqa: BLE001 - the retry is best effort; the customer still gets a safe reply
            log.warning("corrective retry failed (%s)", exc)
            if problems == ["repeated"]:
                response = AIMessage(content=_strip_repeats(_text_of(response), lang), id=response.id, usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)
            else:
                response = AIMessage(content=HELD.get(lang, HELD["en"]), id=response.id, usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)
                held = True
            break
        usage = _add_usage(usage, retried.usage_metadata or {})
        retried.usage_metadata = usage
        response = retried
    latency = int((time.perf_counter() - started) * 1000)
    response.additional_kwargs = {**(response.additional_kwargs or {}), "falcon_latency_ms": latency}
    if not held:
        response = card_shortcut(state, response) or response
    if response.tool_calls:
        # The same call repeated within one response (seen with DeepSeek) is one call, not a reason to cut the turn off.
        seen, unique = set(), []
        for tc in response.tool_calls:
            key = (tc["name"], json.dumps(tc["args"], sort_keys=True, default=str))
            if key not in seen:
                seen.add(key)
                unique.append(tc)
        if len(unique) < len(response.tool_calls):
            log.warning("dropped %d repeated tool calls", len(response.tool_calls) - len(unique))
            response = AIMessage(content=response.content, tool_calls=unique, id=response.id,
                                 additional_kwargs={k: v for k, v in response.additional_kwargs.items() if k != "tool_calls"},
                                 usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)
    calls = len(response.tool_calls or [])
    total = state.get("turn_tool_calls", 0) + calls
    if calls and total > settings.max_tool_calls_per_turn:
        # Safety valve: a looping model gets cut off instead of burning budget.
        log.warning("tool-call cap hit (%s): turn so far %s, next %s", settings.max_tool_calls_per_turn,
                    [e["tool"] for e in state.get("turn_log", [])], [tc["name"] for tc in response.tool_calls])
        response = AIMessage(content=CUT_OFF.get(lang, CUT_OFF["en"]), additional_kwargs=response.additional_kwargs)
        total = state.get("turn_tool_calls", 0)
        return {"messages": [response], "turn_tool_calls": total, "verification": {"status": "cut_off", "rounds": rounds}, "unresolved": state.get("unresolved", 0) + 1}
    out: dict = {"messages": [response], "turn_tool_calls": total}
    if not response.tool_calls:
        verdict = (last_record or {}).get("verdict") or {}
        out["verification"] = {
            "status": "held" if held else ("retried" if rounds else "verified"),
            "rounds": rounds, "amounts": verdict.get("amounts", []), "evidence": verdict.get("evidence", 0),
            "checked": verdict.get("checked", True), "reviewer": "rules+llm" if settings.review_llm else "rules",
        }
        out["unresolved"] = state.get("unresolved", 0) + 1 if held else 0
    return out


def needs_escalation(state: AgentState, config: RunnableConfig) -> bool:
    """Two unresolved replies in a row (decision #43), and the handover tool is available to this persona."""
    limit = settings.escalate_after_unresolved
    status = (state.get("verification") or {}).get("status")
    return bool(limit) and status in UNRESOLVED_STATUSES and state.get("unresolved", 0) >= limit and "handover_to_human" in _tools_from(config)


async def escalate(state: AgentState, config: RunnableConfig) -> dict:
    """The runtime logs the handover case itself and replaces the held reply with the case number: no model call, nothing to hold."""
    lang = state.get("lang", "en")
    held: AIMessage = state["messages"][-1]
    humans = [_text_of(m) for m in state["messages"] if isinstance(m, HumanMessage)][-3:]
    summary = "Automatic handover after two unresolved replies in a row. Recent customer messages: " + " | ".join(h[:300] for h in humans)
    already = next((e for e in state.get("turn_log", []) if e.get("tool") == "handover_to_human" and e.get("ok") and (e.get("result_json") or {}).get("request_id")), None)
    if already:
        # The model itself logged a case this turn and then could not say so acceptably (the full run of 19 Oct logged four cases for one
        # customer this way): reuse that case, never open a second one.
        data, ok, entry = already["result_json"], True, None
    else:
        tool = _tools_from(config)["handover_to_human"]
        started = time.perf_counter()
        try:
            raw = await tool.ainvoke({"summary": summary[:1000], "reason": "unresolved"})
            result = raw if isinstance(raw, str) else ("".join(b.get("text", "") for b in raw if isinstance(b, dict)) if isinstance(raw, list) else json.dumps(raw, default=str))
        except Exception as exc:  # noqa: BLE001 - the case system is down: the held reply stands, the customer is not promised a case
            result = f"Error: {exc}"
        latency = int((time.perf_counter() - started) * 1000)
        try:
            data = json.loads(result)
        except Exception:
            data = None
        ok = isinstance(data, dict) and bool(data.get("request_id"))
        entry = {"tool": "handover_to_human", "args": {"reason": "unresolved"}, "ok": ok, "latency_ms": latency, "summary": _summary(result), "result_json": data if isinstance(data, dict) else None,
                 "escalation": True}
    log.warning("escalated to a human after %d unresolved replies: %s", state.get("unresolved", 0), "case %s" % data.get("request_id") if ok else result[:200])
    with tracer("falcon.runtime").start_as_current_span("escalation", attributes={"falcon.ok": ok, "falcon.unresolved": state.get("unresolved", 0), "falcon.lang": lang}):
        pass
    turn_log = list(state.get("turn_log", [])) + ([entry] if entry else [])
    if not ok:
        return {"turn_log": turn_log}
    text = ESCALATED.get(lang, ESCALATED["en"]).format(case=data["request_id"])
    reply = AIMessage(content=text, id=held.id, usage_metadata=held.usage_metadata, response_metadata=held.response_metadata, additional_kwargs=held.additional_kwargs)
    verification = {**(state.get("verification") or {}), "status": "escalated", "case": data["request_id"]}
    return {"messages": [reply], "turn_log": turn_log, "verification": verification, "unresolved": 0}


def route_after_agent(state: AgentState, config: RunnableConfig) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if any(tc["name"] in WRITE_TOOLS for tc in last.tool_calls):
            return "gate"
        return "tools"
    return "escalate" if needs_escalation(state, config) else END


QUOTE_CARD_KEYS = ("kind", "pnr", "fee", "refund", "amount_due", "refund_method", "currency", "fare_family", "reason_en", "reason_ar", "quote_id",
                   "flights", "passengers", "details", "exception_if_approved", "fare_per_passenger", "lead_passenger", "payment")


def _describe_action(tc: dict, messages: list[AnyMessage]) -> dict:
    """Human-readable confirmation card: what will happen, to which flight, with the quoted amount when there is one.
    The quote may have been fetched in an earlier turn, so look through the conversation's tool results. An exception
    request has no quote id; the latest quote for that booking still tells the card which flight and what an approval would refund."""
    name, args = tc["name"], tc["args"]
    quote = None
    for m in reversed(messages):
        if not (isinstance(m, ToolMessage) and (m.name or "").startswith("quote_")):
            continue
        try:
            data = json.loads(_text_of(m))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if args.get("quote_id"):
            if data.get("quote_id") == args["quote_id"]:
                quote = {k: data.get(k) for k in QUOTE_CARD_KEYS}
                break
        elif args.get("pnr") and data.get("pnr") == args.get("pnr"):
            quote = {k: data.get(k) for k in QUOTE_CARD_KEYS}
            break
    return {"tool": name, "args": redact(args), "quote": quote}


async def gate(state: AgentState, config: RunnableConfig) -> dict:
    last: AIMessage = state["messages"][-1]
    writes = [tc for tc in last.tool_calls if tc["name"] in WRITE_TOOLS]
    payload = {"type": "confirm", "actions": [_describe_action(tc, state["messages"]) for tc in writes], "lang": state["lang"]}
    decision = interrupt(payload)  # pauses here; the API resumes with {"approved": bool, "note": str}
    with tracer("falcon.runtime").start_as_current_span("confirmation.decision", attributes={
            "falcon.approved": bool(decision and decision.get("approved")), "falcon.actions": ",".join(tc["name"] for tc in writes)}):
        pass
    if decision and decision.get("approved"):
        return {"pending": None}
    note = (decision or {}).get("note") or ""
    declined = [ToolMessage(content=json.dumps({"declined": True, "message": "The person did not confirm this action. Nothing was changed." + (f" Note: {note}" if note else "")}),
                            tool_call_id=tc["id"], name=tc["name"]) for tc in last.tool_calls]
    log = list(state.get("turn_log", [])) + [{"tool": tc["name"], "args": redact(tc["args"]), "ok": False, "latency_ms": 0, "summary": "declined by the person", "declined": True} for tc in last.tool_calls]
    return {"messages": declined, "turn_log": log, "pending": None}


def route_after_gate(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) else "agent"


async def tools_node(state: AgentState, config: RunnableConfig) -> dict:
    tools = _tools_from(config)
    last: AIMessage = state["messages"][-1]
    out_messages: list[ToolMessage] = []
    log = list(state.get("turn_log", []))
    for tc in last.tool_calls:
        tool = tools.get(tc["name"])
        started = time.perf_counter()
        if tool is None:
            result, ok = f"Error: tool {tc['name']} is not available to this user.", False
        else:
            try:
                raw = await tool.ainvoke(tc["args"])
                result = raw if isinstance(raw, str) else ("".join(b.get("text", "") for b in raw if isinstance(b, dict)) if isinstance(raw, list) else json.dumps(raw, default=str))
                ok = not result.startswith("Error")
            except Exception as exc:  # the tool layer already converts errors; this is the last net
                result, ok = f"Error: {exc}", False
        latency = int((time.perf_counter() - started) * 1000)
        try:
            result_json = json.loads(result)
        except Exception:
            result_json = None
        entry = {"tool": tc["name"], "args": redact(tc["args"]), "ok": ok, "latency_ms": latency, "summary": _summary(result), "result_json": result_json if isinstance(result_json, dict) else None}
        if tc["name"] in ("search_policy", "get_clause", "get_article", "check_travel_eligibility") and isinstance(result_json, dict):
            # clause ids returned to the model: the evaluation checks that cited ids come from here (grounding)
            ids = [r.get("clause_id") for r in result_json.get("results", [])] if "results" in result_json else [result_json.get("clause_id")] if "clause_id" in result_json else [c.get("clause_id") for c in result_json.get("clauses", [])]
            entry["clause_ids"] = [i for i in ids if i]
        elif isinstance(result_json, dict) and isinstance(result_json.get("clauses"), list) and all(isinstance(c, str) for c in result_json["clauses"]):
            entry["clause_ids"] = list(result_json["clauses"])  # a rules-engine result that names the clauses it rests on (decision #38) grounds a citation too
        log.append(entry)
        out_messages.append(ToolMessage(content=result, tool_call_id=tc["id"], name=tc["name"]))
    return {"messages": out_messages, "turn_log": log}


def build_graph(checkpointer=None):
    g = StateGraph(AgentState)
    g.add_node("prepare", prepare)
    g.add_node("guard", guard)
    g.add_node("confirm", confirm)
    g.add_node("agent", agent)
    g.add_node("gate", gate)
    g.add_node("escalate", escalate)
    g.add_node("tools", tools_node)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "guard")
    g.add_conditional_edges("guard", route_after_guard, {"agent": "agent", "confirm": "confirm", END: END})
    g.add_edge("confirm", "gate")
    g.add_conditional_edges("agent", route_after_agent, {"gate": "gate", "tools": "tools", "escalate": "escalate", END: END})
    g.add_edge("escalate", END)
    g.add_conditional_edges("gate", route_after_gate, {"tools": "tools", "agent": "agent"})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)
