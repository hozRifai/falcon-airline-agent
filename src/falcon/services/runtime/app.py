"""Agent runtime: the FastAPI app that owns identity, sessions, the LangGraph agent and the web UI.

Run:  uvicorn falcon.services.runtime.app:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, Any

import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from falcon.core.arabic import preferred_reply_language
from falcon.core.auth import UserContext, mint_session_token, verify_session_token
from falcon.core.clock import now
from falcon.core.config import REPO_ROOT, settings
from falcon.core.db import Database, connection_options
from falcon.core.mcp_client import client_for
from falcon.core.telemetry import current_ids, link_to, setup_tracing, tracer

from .graph import build_graph, previous_reply, text_of
from .guardrails import guard as guard_message
from .llm import make_chat_model, model_label
from .personas import CUSTOMER_TOOLS, SERVERS, SERVICE_ROLES, get_persona, list_personas

log = logging.getLogger("falcon.runtime")
WEB_DIST = REPO_ROOT / "web" / "dist"

db = Database(settings.dsn_agent, max_size=8)
state: dict[str, Any] = {"graph": None, "saver_pool": None, "llm": None}
_tool_cache: dict[str, tuple[float, list, list[str]]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.open()
    # The conversation checkpoints go through a pool, not one long-lived connection: a dropped connection (a database restart or
    # failover) is replaced on the next turn instead of failing every turn until the runtime is restarted.
    pool = AsyncConnectionPool(settings.dsn_agent, min_size=1, max_size=10, open=False, check=AsyncConnectionPool.check_connection,
                               kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row, "options": connection_options()})
    await pool.open()
    saver = AsyncPostgresSaver(pool)
    await saver.setup()  # creates the checkpoint tables in schema `agent`
    state["saver_pool"] = pool
    state["graph"] = build_graph(saver)
    state["llm"] = make_chat_model()
    log.info("runtime ready; model=%s", model_label())
    yield
    await pool.close()
    await db.close()


app = FastAPI(title="Falcon Airline agent runtime", lifespan=lifespan)
setup_tracing("falcon-runtime", app, langchain=True)  # a no-op without OTEL_EXPORTER_OTLP_ENDPOINT


# ---------------------------------------------------------------- identity
async def current_user(authorization: Annotated[str | None, Header()] = None) -> UserContext:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in first.")
    try:
        return verify_session_token(authorization.split(" ", 1)[1])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise HTTPException(401, "Session expired or invalid; sign in again.") from None


def _is_admin(request: Request) -> bool:
    """The evaluation harness and scripts/talk.py: full tool results in the done event, free-text trace tags."""
    return bool(settings.admin_token) and request.headers.get("x-admin-token") == settings.admin_token


async def check_budget() -> None:
    spent = await db.fetch_one("SELECT input_tokens + output_tokens AS t FROM agent.token_budget WHERE day = %s", (date.today(),))
    if spent and spent["t"] >= settings.daily_token_budget:
        raise HTTPException(429, "The demo's daily model budget is used up. Please try again tomorrow.")


# ---------------------------------------------------------------- the tools of a session
class NoSystemsReachable(RuntimeError):
    """None of the MCP servers answered tool discovery."""


TOOLS_TTL_S, DEGRADED_TTL_S = 600, 30  # a list missing a server is retried soon, so a recovered server is back within 30 s


async def tools_for(user: UserContext) -> tuple[list, list[str]]:
    """(tools, unavailable servers) for this session. Each MCP server is discovered on its own: one server down or refusing the
    token removes its tools, not everyone's; only when none answers is the turn refused."""
    cached = _tool_cache.get(user.session_id)
    if cached and time.time() - cached[0] < (DEGRADED_TTL_S if cached[2] else TOOLS_TTL_S):
        return cached[1], cached[2]
    found = await asyncio.gather(*(client_for(user, SERVICE_ROLES, [name]).get_tools() for name in SERVERS), return_exceptions=True)
    tools, unavailable = [], []
    for name, result in zip(SERVERS, found, strict=True):
        if isinstance(result, BaseException):
            log.warning("MCP server %s unavailable for session %s: %r", name, user.session_id, result)
            unavailable.append(name)
        else:
            tools += [t for t in result if t.name in CUSTOMER_TOOLS]
    if len(unavailable) == len(SERVERS):
        raise NoSystemsReachable(", ".join(unavailable))
    _tool_cache[user.session_id] = (time.time(), tools, unavailable)
    return tools, unavailable


def graph_config(user: UserContext, thread: str, tools: list, unavailable: list[str] | None = None) -> dict:
    return {"configurable": {"thread_id": f"{user.session_id}:{thread}", "tools": list(tools), "llm": state["llm"],
                             "default_lang": user.lang, "unavailable": list(unavailable or [])}}


# ---------------------------------------------------------------- personas and sign-in
@app.get("/api/personas")
async def personas():
    return {"personas": await list_personas(db), "frozen_now": now().isoformat(), "model": model_label(), "trace_ui_url": settings.trace_ui_url or None}


class SwitchBody(BaseModel):
    user_id: str
    reset: bool = True


_reset_lock = asyncio.Lock()


async def reset_demo_data() -> None:
    async with _reset_lock:
        proc = await asyncio.to_thread(subprocess.run, [sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], capture_output=True, text=True)
        if proc.returncode != 0:
            raise HTTPException(500, f"Reset failed: {(proc.stdout + proc.stderr)[-300:]}")
        _tool_cache.clear()


@app.post("/api/demo/switch")
async def demo_switch(body: SwitchBody):
    """The demo's sign-in: pick a customer, put the demo data back to the seed (so every customer starts from their story, whatever
    the previous one did), and start a fresh session as them. The audit log and the turn traces are kept."""
    persona = await get_persona(db, body.user_id)
    if not persona:
        raise HTTPException(404, "Unknown persona.")
    if body.reset:
        await reset_demo_data()
    user = UserContext(user_id=persona["user_id"], display_name=persona["display_name"], customer_id=persona["customer_id"], lang=persona["lang"], session_id=uuid.uuid4().hex[:12])
    return {"token": mint_session_token(user), "user": user.model_dump(), "persona": persona, "reset": body.reset}


# ---------------------------------------------------------------- data for the UI (through the same MCP seam as the agent)
def _parse(result: Any) -> Any:
    if isinstance(result, list):
        result = "".join(b.get("text", "") for b in result if isinstance(b, dict))
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return {"error": str(result)}


async def _call(user: UserContext, name: str, **kwargs) -> Any:
    tools = {t.name: t for t in (await tools_for(user))[0]}
    if name not in tools:
        raise HTTPException(403, f"{name} is not available to this user.")
    return _parse(await tools[name].ainvoke(kwargs))


@app.get("/api/bookings")
async def bookings(user: UserContext = Depends(current_user)):
    listing = await _call(user, "list_my_bookings")
    details = [await _call(user, "get_booking", pnr=b["pnr"]) for b in listing.get("bookings", [])[:5]]
    return {"bookings": details, "now": now().isoformat()}


@app.get("/api/search")
async def search(origin: str, destination: str, date_from: str, date_to: str | None = None, passengers: int = 1, exclude_weekdays: str | None = None, user: UserContext = Depends(current_user)):
    skip = [w for w in (exclude_weekdays or "").split(",") if w]
    return await _call(user, "search_flights", origin=origin, destination=destination, date_from=date_from, date_to=date_to, passengers=passengers, exclude_weekdays=skip)


# ---------------------------------------------------------------- chat
class ChatBody(BaseModel):
    message: str
    thread: str = "main"
    lang: str | None = None  # the app's language: the default when a message's script is ambiguous


class OpenBody(BaseModel):
    thread: str = "main"
    lang: str | None = None


class ConfirmBody(BaseModel):
    approved: bool
    note: str | None = None
    thread: str = "main"


MODEL_UNAVAILABLE = {
    "en": "Sorry, I could not reach my language model just now, so nothing was checked or changed. Please send your message again in a moment.",
    "ar": "عذرًا، تعذّر الوصول إلى نموذج اللغة الآن، لذلك لم يتم التحقق من أي شيء أو تغييره. يرجى إرسال رسالتك مرة أخرى بعد قليل.",
}
SYSTEMS_UNAVAILABLE = {
    "en": "Sorry, I cannot reach the booking systems right now, so nothing was checked or changed. Please try again in a moment.",
    "ar": "عذرًا، لا يمكنني الوصول إلى أنظمة الحجز الآن، لذلك لم يتم التحقق من أي شيء أو تغييره. يرجى المحاولة مرة أخرى بعد قليل.",
}
OPENING = {
    # Sent as the customer's (hidden) first message when a conversation opens, so the agent speaks first from the data:
    # greet by name, check the bookings and flight status, flag what needs attention, offer the options. The UI never shows it.
    "en": "[[opening]] Hello. I just opened the chat. Please check my bookings and the status of my flights, tell me if anything needs my attention, and ask how you can help.",
    "ar": "[[opening]] مرحباً. فتحت المحادثة الآن. من فضلك تحقق من حجوزاتي وحالة رحلاتي، وأخبرني إن كان هناك ما يستدعي انتباهي، واسألني كيف يمكنك المساعدة.",
}


def _sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False, default=str)}


def _ui_lang(user: UserContext, requested: str | None) -> UserContext:
    """The language the app is set to wins over the customer record's preference: Arabic chosen means everything in Arabic."""
    return user.model_copy(update={"lang": requested if requested in ("en", "ar") else user.lang})


def _public(turn_log: list[dict], verbose: bool) -> list[dict]:
    """The turn log as the UI sees it; the evaluation harness (verbose) also gets the full tool results, so the scorer can check amounts."""
    return [{k: v for k, v in e.items() if k != "result_json" or verbose} for e in turn_log]


async def _record_turn(sql: str, params: tuple) -> dict | None:
    """The turn log must never take the reply down with it: a database hiccup is logged and the customer still gets the done event."""
    try:
        return await db.fetch_one(sql, params)
    except Exception:
        log.exception("could not record the turn")
        return None


def _done(message: str, lang: str, turn_log: list[dict], started: float, trace_id: str | None, **extra) -> dict:
    return _sse("done", {"message": message, "pending": None, "lang": lang, "what_i_checked": turn_log, "usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0},
                         "latency_ms": int((time.perf_counter() - started) * 1000), "model_latency_ms": 0, "model": model_label(), "trace_id": trace_id, **extra})


async def _run_turn(user: UserContext, thread: str, graph_input: Any, verbose: bool, tags: str):
    """One customer turn as Server-Sent Events: token | tool | review | confirm | done. One root span (`chat.turn`) carries the session,
    the thread, the language and the reviewer's verdict; its trace id goes back to the UI and into agent.turns."""
    with tracer("falcon.runtime").start_as_current_span("chat.turn", attributes={
            "falcon.session_id": user.session_id, "falcon.user_id": user.user_id, "falcon.thread": thread,
            "falcon.resume": not isinstance(graph_input, dict), **({"falcon.tags": tags} if tags else {})}) as turn_span:
        async for event in _turn_events(user, thread, graph_input, verbose, turn_span):
            yield event


async def _turn_events(user: UserContext, thread: str, graph_input: Any, verbose: bool, turn_span):
    trace_id, span_id = current_ids()
    started = time.perf_counter()
    said_now = text_of(graph_input["messages"][-1]) if isinstance(graph_input, dict) and graph_input.get("messages") else ""
    lang_guess = preferred_reply_language(said_now, user.lang) if said_now else user.lang
    # The input guardrail (one classifier call, ~1.2 s) does not depend on the tool list, so it runs while the tools are
    # fetched; the guard node reuses the verdict when the text matches, and classifies itself otherwise.
    pre_task, context = None, None
    if said_now.strip() and not said_now.startswith("[[opening]]"):
        try:  # the classifier reads the assistant's previous reply: one checkpoint read, no model call
            snapshot = await state["graph"].aget_state({"configurable": {"thread_id": f"{user.session_id}:{thread}"}})
            context = previous_reply((snapshot.values or {}).get("messages", []) + list(graph_input["messages"]))
        except Exception as exc:  # noqa: BLE001 - without it the guard node classifies with the context itself
            log.warning("could not read the thread for the classifier's context: %s", exc)
        pre_task = asyncio.create_task(guard_message(said_now, lang_guess, context))
    try:
        tools, unavailable = await tools_for(user)
    except Exception:
        if pre_task:
            pre_task.cancel()
        log.exception("tool discovery failed for session %s", user.session_id)
        yield _done(SYSTEMS_UNAVAILABLE.get(lang_guess, SYSTEMS_UNAVAILABLE["en"]), lang_guess, [], started, trace_id, error="systems_unavailable")
        return
    config = graph_config(user, thread, tools, unavailable)
    if pre_task:
        try:
            clean, g = await pre_task
            config["configurable"]["guard_precomputed"] = {"text": said_now, "context": context, "clean": clean, "guard": g}
        except Exception as exc:  # noqa: BLE001 - the guard node classifies itself then
            log.warning("precomputed guardrail failed, the guard node will classify: %s", exc)
    graph = state["graph"]
    seen_tools = 0
    final_text, interrupted = "", None
    first_token_ms: int | None = None  # time to the first streamed token: the silence the customer sits through
    try:
        async for mode, chunk in graph.astream(graph_input, config, stream_mode=["updates", "messages", "custom"]):
            if mode == "custom":
                if isinstance(chunk, dict) and chunk.get("type") == "review":
                    yield _sse("review", chunk)  # the reviewer sent a draft back (or held the reply): the UI drops the streamed draft
            elif mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "agent" and isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None) and text_of(msg):
                    if first_token_ms is None:
                        first_token_ms = int((time.perf_counter() - started) * 1000)
                    yield _sse("token", {"text": text_of(msg)})
            else:
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupted = update[0].value if update else None
                        yield _sse("confirm", interrupted)
                        continue
                    if not isinstance(update, dict):
                        continue
                    if node in ("tools", "gate", "escalate") and update.get("turn_log"):
                        for entry in update["turn_log"][seen_tools:]:
                            yield _sse("tool", {k: v for k, v in entry.items() if k != "result_json"})
                        seen_tools = len(update["turn_log"])
                    msgs = update.get("messages") or []
                    if node == "escalate" and msgs:
                        final_text = text_of(msgs[-1])  # the runtime replaced the held reply with the case number
                    elif node in ("agent", "guard") and msgs and isinstance(msgs[-1], AIMessage) and not getattr(msgs[-1], "tool_calls", None):
                        # Text that accompanies tool calls is never shown: the reviewer only sees final drafts, and DeepSeek once put a
                        # whole invented price list in front of its search call, which the customer saw unreviewed.
                        text = text_of(msgs[-1]).strip()
                        if text:
                            final_text = f"{final_text}\n{text}".strip()
    except Exception:
        # The model provider failed (network, timeout after retries). Say so in the customer's language; nothing was changed.
        log.exception("turn failed for session %s", user.session_id)
        try:
            values = (await graph.aget_state(config)).values
        except Exception:  # noqa: BLE001 - if the checkpoint store is the thing that failed, still answer the customer
            values = {}
        lang = values.get("lang") or lang_guess
        turn_log = _public(values.get("turn_log", []), False)
        await _record_turn(
            "INSERT INTO agent.turns(session_id,user_id,thread,lang_in,lang_out,model,input_tokens,output_tokens,cached_tokens,latency_ms,tool_calls,finish,trace_id,span_id) VALUES (%s,%s,%s,%s,%s,%s,0,0,0,%s,%s,'error',%s,%s) RETURNING id",
            (user.session_id, user.user_id, thread, lang, lang, model_label(), int((time.perf_counter() - started) * 1000), json.dumps(turn_log, default=str), trace_id, span_id))
        yield _done(MODEL_UNAVAILABLE.get(lang, MODEL_UNAVAILABLE["en"]), lang, turn_log, started, trace_id, error="model_unavailable")
        return
    try:
        values = (await graph.aget_state(config)).values
    except Exception:
        log.exception("could not read the turn's final state")
        values = {}
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    model_latency = 0
    for m in reversed(values.get("messages", [])):  # the model calls of this turn: back to the customer's message
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage):
            u = m.usage_metadata or {}
            usage["input_tokens"] += u.get("input_tokens", 0) or 0
            usage["output_tokens"] += u.get("output_tokens", 0) or 0
            usage["cached_tokens"] += (u.get("input_token_details") or {}).get("cache_read", 0) or 0
            model_latency += (m.additional_kwargs or {}).get("falcon_latency_ms", 0)
    latency = int((time.perf_counter() - started) * 1000)
    lang = values.get("lang") or user.lang
    verification = values.get("verification") if not interrupted else None
    guardrail = values.get("guardrail")
    stored_log = _public(values.get("turn_log", []), False)
    turn_span.set_attributes({"falcon.lang": lang, "falcon.tool_calls": len(stored_log), "falcon.finish": "interrupt" if interrupted else "end",
                              "falcon.verification.status": (verification or {}).get("status", "n/a"), "falcon.verification.rounds": len((verification or {}).get("rounds", [])),
                              "falcon.guardrail.label": (guardrail or {}).get("label", "n/a"), "falcon.guardrail.action": (guardrail or {}).get("action", "n/a"),
                              "falcon.tokens.input": usage["input_tokens"], "falcon.tokens.output": usage["output_tokens"], "falcon.latency_ms": latency,
                              "falcon.first_token_ms": first_token_ms if first_token_ms is not None else -1})
    row = await _record_turn(
        "INSERT INTO agent.turns(session_id,user_id,thread,lang_in,lang_out,model,input_tokens,output_tokens,cached_tokens,latency_ms,first_token_ms,tool_calls,finish,verification,guardrail,trace_id,span_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (user.session_id, user.user_id, thread, lang, lang, model_label(), usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"], latency, first_token_ms, json.dumps(stored_log, default=str),
         "interrupt" if interrupted else "end", json.dumps(verification, default=str) if verification else None, json.dumps(guardrail, default=str) if guardrail else None, trace_id, span_id))
    await _record_turn("INSERT INTO agent.token_budget(day,input_tokens,output_tokens) VALUES (%s,%s,%s) ON CONFLICT (day) DO UPDATE SET input_tokens = agent.token_budget.input_tokens + EXCLUDED.input_tokens, output_tokens = agent.token_budget.output_tokens + EXCLUDED.output_tokens RETURNING day",
                       (date.today(), usage["input_tokens"], usage["output_tokens"]))
    yield _sse("done", {"message": final_text, "pending": interrupted, "lang": lang, "what_i_checked": _public(values.get("turn_log", []), verbose), "usage": usage, "latency_ms": latency,
                        "model_latency_ms": model_latency, "first_token_ms": first_token_ms, "model": model_label(), "verification": verification, "guardrail": guardrail,
                        "turn_id": row["id"] if row else None, "trace_id": trace_id})


def _stream(request: Request, user: UserContext, thread: str, graph_input: Any) -> EventSourceResponse:
    admin = _is_admin(request)
    tags = (request.headers.get("x-falcon-tags") or "")[:200] if admin else ""  # the harness sends `eval,task:<id>,run:<n>` for the trace UI
    return EventSourceResponse(_run_turn(user, thread, graph_input, verbose=admin, tags=tags))


@app.post("/api/chat")
async def chat(body: ChatBody, request: Request, user: UserContext = Depends(current_user)):
    await check_budget()
    text = body.message.strip()
    if not text or len(text) > 2000:
        raise HTTPException(400, "Message must be 1 to 2000 characters.")
    user = _ui_lang(user, body.lang)
    return _stream(request, user, body.thread, {"messages": [HumanMessage(content=text)], "user_name": user.display_name})


@app.post("/api/chat/open")
async def open_chat(body: OpenBody, request: Request, user: UserContext = Depends(current_user)):
    """The agent's proactive opening line for a freshly selected customer ('your flight tomorrow was cancelled; I can rebook you
    for free or refund 965 AED')."""
    await check_budget()
    user = _ui_lang(user, body.lang)
    return _stream(request, user, body.thread, {"messages": [HumanMessage(content=OPENING.get(user.lang, OPENING["en"]))], "user_name": user.display_name})


@app.post("/api/chat/confirm")
async def confirm(body: ConfirmBody, request: Request, user: UserContext = Depends(current_user)):
    return _stream(request, user, body.thread, Command(resume={"approved": body.approved, "note": body.note}))


@app.get("/api/chat/history")
async def history(thread: str = "main", user: UserContext = Depends(current_user)):
    snapshot = await state["graph"].aget_state(graph_config(user, thread, []))  # reading the history needs the thread, not the tools
    out = []
    for m in snapshot.values.get("messages", []):
        if isinstance(m, HumanMessage) and not text_of(m).startswith("[[opening]]"):
            out.append({"role": "user", "text": text_of(m)})
        elif isinstance(m, AIMessage) and not m.tool_calls:
            out.append({"role": "assistant", "text": text_of(m)})
    pending = next((task.interrupts[0].value for task in snapshot.tasks if task.interrupts), None)
    return {"messages": out, "pending": pending, "lang": snapshot.values.get("lang")}


# ---------------------------------------------------------------- feedback (thumbs on a reply)
class FeedbackBody(BaseModel):
    rating: str  # up | down
    thread: str = "main"
    turn_id: int | None = None  # the agent.turns row the reply came from (the done event carries it)
    message: str | None = None  # the reply as the customer saw it, kept with the rating so a review needs no join


@app.post("/api/feedback")
async def feedback(body: FeedbackBody, user: UserContext = Depends(current_user)):
    """Stored as data for the service owner; nothing in the demo acts on it automatically."""
    if body.rating not in ("up", "down"):
        raise HTTPException(400, "rating must be 'up' or 'down'.")
    turn = await db.fetch_one("SELECT trace_id, span_id FROM agent.turns WHERE id = %s AND session_id = %s", (body.turn_id, user.session_id)) if body.turn_id else None
    with tracer("falcon.runtime").start_as_current_span("customer.feedback", links=link_to((turn or {}).get("trace_id"), (turn or {}).get("span_id")),
                                                        attributes={"falcon.rating": body.rating, "falcon.session_id": user.session_id, "falcon.turn_id": body.turn_id or 0}):
        row = await db.fetch_one("INSERT INTO agent.feedback(session_id,user_id,thread,turn_id,rating,message) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                                 (user.session_id, user.user_id, body.thread, body.turn_id, body.rating, (body.message or "")[:4000] or None))
    return {"ok": True, "feedback_id": row["id"]}


# ---------------------------------------------------------------- health, readiness, metrics
@app.get("/api/health")
async def health():
    """Liveness: the process answers. Dependencies are /api/ready's business, so an outage never gets a healthy process restarted."""
    return {"ok": True, "now": now().isoformat(), "model": model_label(), "tracing": bool(settings.otel_exporter_otlp_endpoint)}


READY_TIMEOUT_S = 5.0
PROBE_USER = UserContext(user_id="readiness-probe", display_name="readiness probe", session_id="readiness")


async def _check_checkpoints() -> None:
    async with state["saver_pool"].connection() as conn:
        await conn.execute("SELECT 1")


async def _check_server(name: str) -> None:
    await client_for(PROBE_USER, SERVICE_ROLES, [name]).get_tools()


@app.get("/api/ready")
async def ready():
    """Readiness, for a load balancer or the platform's probe: the database, the conversation checkpoints and each MCP server (a real
    tool listing with a real service token). 200 when all answer, 503 naming what does not; error types only, never details."""
    checks = {"database": db.fetch_one("SELECT 1 AS ok"), "checkpoints": _check_checkpoints(), **{name: _check_server(name) for name in SERVERS}}
    results = await asyncio.gather(*(asyncio.wait_for(c, READY_TIMEOUT_S) for c in checks.values()), return_exceptions=True)
    report = {}
    for name, r in zip(checks, results, strict=True):
        if isinstance(r, BaseException):
            log.warning("readiness: %s is down: %r", name, r)
        report[name] = f"down ({type(r).__name__})" if isinstance(r, BaseException) else "ok"
    ok = all(v == "ok" for v in report.values())
    return JSONResponse({"ready": ok, "checks": report}, status_code=200 if ok else 503)


@app.get("/api/metrics/summary")
async def metrics_summary(x_admin_token: Annotated[str | None, Header()] = None):
    """Business metrics from the runtime's own tables: turns, tokens, latency, tool calls and failures, reviewer outcomes, thumbs."""
    if x_admin_token != settings.admin_token:
        raise HTTPException(403, "Admin token required.")
    turns = await db.fetch_one("SELECT count(*) AS turns, coalesce(sum(input_tokens),0) AS input_tokens, coalesce(sum(output_tokens),0) AS output_tokens, coalesce(avg(latency_ms),0)::int AS avg_latency_ms FROM agent.turns")
    tools = await db.fetch_all("SELECT tool, count(*) AS calls, sum(CASE WHEN ok THEN 0 ELSE 1 END) AS failures, avg(latency_ms)::int AS avg_ms FROM agent.audit_log GROUP BY tool ORDER BY calls DESC")
    gate = await db.fetch_all("SELECT verification->>'status' AS status, count(*) AS n FROM agent.turns WHERE verification IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")
    thumbs = await db.fetch_all("SELECT rating, count(*) AS n FROM agent.feedback GROUP BY rating")
    return {"turns": turns, "tools": [dict(t) for t in tools], "verification": {g["status"]: g["n"] for g in gate}, "feedback": {t["rating"]: t["n"] for t in thumbs}}


# ---------------------------------------------------------------- the web app (built with Vite), with the SPA fallback
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        candidate = WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        # the page must be revalidated on every load: it names the hashed bundles, and a cached copy kept showing the previous build
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:

    @app.get("/")
    async def root():
        return JSONResponse({"ok": True, "hint": "web app not built; run `npm run build` in web/", "api": "/api/health"})
