"""Agent runtime: FastAPI app that owns identity, sessions, the LangGraph agent, and the web UI.

Run:  uvicorn falcon.services.runtime.app:app --port 8000
"""

from __future__ import annotations

import asyncio
import base64  # noqa: F401 - kept for callers that post base64; the endpoint below takes multipart
import json
import logging
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.types import Command
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from falcon.core.arabic import preferred_reply_language
from falcon.core.auth import UserContext, mint_session_token, verify_session_token
from falcon.core.clock import now
from falcon.core.config import REPO_ROOT, settings
from falcon.core.db import Database
from falcon.core.mcp_client import client_for
from falcon.core.telemetry import current_ids, link_to, setup_tracing, tracer

from .guardrails import guard as guard_message
from .graph import build_graph, previous_reply
from .llm import make_chat_model, model_label
from .personas import API_TOOLS, ROLE_PROFILES, get_persona, list_personas

log = logging.getLogger("falcon.runtime")
WEB_DIST = REPO_ROOT / "web" / "dist"

db = Database(settings.dsn_agent, max_size=8)
state: dict[str, Any] = {"graph": None, "saver_conn": None}
_tool_cache: dict[str, tuple[float, list]] = {}
_ip_buckets: dict[str, list[float]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.open()
    conn = await AsyncConnection.connect(settings.dsn_agent, autocommit=True, prepare_threshold=0, row_factory=dict_row)
    saver = AsyncPostgresSaver(conn)
    await saver.setup()  # creates checkpoint tables in schema `agent` (search_path of agent_rw)
    state["saver_conn"] = conn
    state["graph"] = build_graph(saver)
    state["llm"] = make_chat_model()
    log.info("runtime ready; model=%s", model_label())
    yield
    await conn.close()
    await db.close()


app = FastAPI(title="Falcon Airline agent runtime", lifespan=lifespan)
setup_tracing("falcon-runtime", app, langchain=True)  # decision #27: no-op without OTEL_EXPORTER_OTLP_ENDPOINT / App Insights


# ---------------------------------------------------------------- auth helpers
async def current_user(authorization: Annotated[str | None, Header()] = None) -> UserContext:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Sign in first.")
    try:
        return verify_session_token(authorization.split(" ", 1)[1])
    except Exception:
        raise HTTPException(401, "Session expired or invalid; sign in again.")


def rate_limit(request: Request) -> None:
    if settings.admin_token and request.headers.get("x-admin-token") == settings.admin_token:
        return  # the evaluation harness and admin tooling are not the public
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0].strip()
    bucket = _ip_buckets.setdefault(ip, [])
    cutoff = time.time() - 60
    bucket[:] = [t for t in bucket if t > cutoff]
    if len(bucket) >= settings.rate_limit_per_minute:
        raise HTTPException(429, "Too many requests from this address; try again in a minute.")
    bucket.append(time.time())


async def check_budget(session_id: str) -> None:
    turns = await db.fetch_one("SELECT count(*) AS n FROM agent.turns WHERE session_id = %s", (session_id,))
    if turns and turns["n"] >= settings.max_turns_per_session:
        raise HTTPException(429, "This demo session has reached its turn limit. Sign in again to start a new one.")
    spent = await db.fetch_one("SELECT input_tokens + output_tokens AS t FROM agent.token_budget WHERE day = %s", (date.today(),))
    if spent and spent["t"] >= settings.daily_token_budget:
        raise HTTPException(429, "The demo's daily model budget is used up. Please try again tomorrow.")


async def tools_for(user: UserContext) -> list:
    cached = _tool_cache.get(user.session_id)
    if cached and time.time() - cached[0] < 600:
        return cached[1]
    profile = ROLE_PROFILES[user.role]
    client = client_for(user, profile["service_roles"], profile["servers"])
    tools = [t for t in await client.get_tools() if t.name in profile["tools"] or t.name in API_TOOLS]
    _tool_cache[user.session_id] = (time.time(), tools)
    return tools


def graph_config(user: UserContext, thread: str, tools: list) -> dict:
    return {"configurable": {"thread_id": f"{user.session_id}:{thread}", "tools": [t for t in tools if t.name not in API_TOOLS], "llm": state["llm"], "default_lang": user.lang}}


# ---------------------------------------------------------------- personas & sign-in
@app.get("/api/personas")
async def personas():
    rows = await list_personas(db)
    return {"personas": rows, "frozen_now": now().isoformat(), "model": model_label(), "trace_ui_url": settings.trace_ui_url or None,
            "requires_code": bool(settings.demo_access_code)}


class LoginBody(BaseModel):
    user_id: str
    access_code: str | None = None


@app.post("/api/auth/demo-login")
async def demo_login(body: LoginBody, request: Request):
    rate_limit(request)
    if settings.demo_access_code and body.access_code != settings.demo_access_code:
        raise HTTPException(403, "Wrong access code.")
    persona = await get_persona(db, body.user_id)
    if not persona:
        raise HTTPException(404, "Unknown persona.")
    user = UserContext(user_id=persona["user_id"], display_name=persona["display_name"], role=persona["role"], customer_id=persona["customer_id"], lang=persona["lang"], session_id=uuid.uuid4().hex[:12])
    return {"token": mint_session_token(user), "user": user.model_dump(), "persona": dict(persona)}


@app.get("/api/me")
async def me(user: UserContext = Depends(current_user)):
    return user.model_dump()


class SwitchBody(BaseModel):
    user_id: str
    reset: bool = True


_reset_lock = asyncio.Lock()


@app.post("/api/demo/switch")
async def demo_switch(body: SwitchBody, request: Request):
    """The demo's persona switcher: reset the demo data to the seed (so every customer starts from their story, whatever the
    previous one did), then sign in as that customer with a fresh session. Demo-only by nature: it wipes bookings, cases and
    conversation checkpoints; the audit log and turn traces are kept (the seed never truncates them)."""
    rate_limit(request)
    persona = await get_persona(db, body.user_id)
    if not persona:
        raise HTTPException(404, "Unknown persona.")
    if body.reset:
        async with _reset_lock:
            proc = await asyncio.to_thread(subprocess.run, [sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], capture_output=True, text=True)
            if proc.returncode != 0:
                raise HTTPException(500, f"Reset failed: {(proc.stdout + proc.stderr)[-300:]}")
            _tool_cache.clear()
    user = UserContext(user_id=persona["user_id"], display_name=persona["display_name"], role=persona["role"], customer_id=persona["customer_id"], lang=persona["lang"], session_id=uuid.uuid4().hex[:12])
    return {"token": mint_session_token(user), "user": user.model_dump(), "persona": dict(persona), "reset": body.reset}


# ---------------------------------------------------------------- data for the UI (through the same MCP seam as the agent)
def _parse(result: Any) -> Any:
    if isinstance(result, list):
        result = "".join(b.get("text", "") for b in result if isinstance(b, dict))
    try:
        return json.loads(result)
    except Exception:
        return {"error": str(result)}


async def _call(user: UserContext, name: str, **kwargs) -> Any:
    tools = {t.name: t for t in await tools_for(user)}
    if name not in tools:
        raise HTTPException(403, f"{name} is not available to this user.")
    return _parse(await tools[name].ainvoke(kwargs))


@app.get("/api/bookings")
async def bookings(user: UserContext = Depends(current_user)):
    listing = await _call(user, "list_my_bookings")
    details = []
    for b in listing.get("bookings", [])[:5]:
        details.append(await _call(user, "get_booking", pnr=b["pnr"]))
    return {"bookings": details, "now": now().isoformat()}


@app.get("/api/search")
async def search(origin: str, destination: str, date_from: str, date_to: str | None = None, passengers: int = 1, exclude_weekdays: str | None = None, user: UserContext = Depends(current_user)):
    skip = [w for w in (exclude_weekdays or "").split(",") if w]
    return await _call(user, "search_flights", origin=origin, destination=destination, date_from=date_from, date_to=date_to, passengers=passengers, exclude_weekdays=skip)


@app.get("/api/fare-rules")
async def fare_rules(user: UserContext = Depends(current_user)):
    return await _call(user, "get_fare_rules")


ATTACHMENT_TYPES = {"application/pdf", "image/jpeg", "image/png"}
ATTACHMENT_MAX_BYTES = 2 * 1024 * 1024


@app.post("/api/attachments")
async def upload_attachment(request: Request, file: UploadFile = File(...), user: UserContext = Depends(current_user)):
    """A document the customer attaches in the chat (decision #33): stored in the reservations system for the customer-care team.
    The agent learns its name and size through list_my_attachments, never its contents."""
    rate_limit(request)
    data = await file.read()
    if not data or len(data) > ATTACHMENT_MAX_BYTES:
        raise HTTPException(413, "Attach a PDF, JPEG or PNG of up to 2 MB.")
    ctype = (file.content_type or "").split(";")[0].strip().lower()
    if ctype not in ATTACHMENT_TYPES:
        raise HTTPException(415, "Attach a PDF, JPEG or PNG.")
    return await _call(user, "store_attachment", filename=(file.filename or "document")[-120:], content_type=ctype, content_base64=base64.b64encode(data).decode())


# ---------------------------------------------------------------- chat
class ChatBody(BaseModel):
    message: str
    thread: str = "main"
    lang: str | None = None  # the app's language (decision #40): the default when a message's script is ambiguous


class ConfirmBody(BaseModel):
    approved: bool
    note: str | None = None
    thread: str = "main"


MODEL_UNAVAILABLE = {
    "en": "Sorry, I could not reach my language model just now, so nothing was checked or changed. Please send your message again in a moment.",
    "ar": "عذرًا، تعذّر الوصول إلى نموذج اللغة الآن، لذلك لم يتم التحقق من أي شيء أو تغييره. يرجى إرسال رسالتك مرة أخرى بعد قليل.",
}


def _sse(event: str, data: Any) -> dict:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False, default=str)}


def _is_admin(request: Request) -> bool:
    return bool(settings.admin_token) and request.headers.get("x-admin-token") == settings.admin_token


def _tags(request: Request) -> str:
    """Free-text tags for the trace (the evaluation harness sends `eval,task:<id>,run:<n>`); admin callers only."""
    return (request.headers.get("x-falcon-tags") or "")[:200] if _is_admin(request) else ""


async def _run_and_stream(user: UserContext, thread: str, graph_input: Any, verbose: bool = False, tags: str = ""):
    """verbose (evaluation harness, admin token): the done event carries full tool results, so the scorer can check every
    amount in a reply against what the tools returned. The web app gets the lean log. One root span per turn (`chat.turn`)
    carries session, thread, persona, language and the reviewer's verdict; its trace id goes back to the UI and into agent.turns."""
    with tracer("falcon.runtime").start_as_current_span("chat.turn", attributes={
            "falcon.session_id": user.session_id, "falcon.user_id": user.user_id, "falcon.thread": thread, "falcon.persona": user.role,
            "falcon.resume": not isinstance(graph_input, dict), **({"falcon.tags": tags} if tags else {})}) as turn_span:
        async for event in _run_turn(user, thread, graph_input, verbose, turn_span):
            yield event


SYSTEMS_UNAVAILABLE = {
    "en": "Sorry, I cannot reach the booking systems right now, so nothing was checked or changed. Please try again in a moment.",
    "ar": "عذرًا، لا يمكنني الوصول إلى أنظمة الحجز الآن، لذلك لم يتم التحقق من أي شيء أو تغييره. يرجى المحاولة مرة أخرى بعد قليل.",
}


async def _record_turn(sql: str, params: tuple) -> dict | None:
    """The turn log must never take the reply down with it: a schema older than the code (reset.sh not run) or a database hiccup
    is logged and the customer still gets the done event (FAILURE_DIARY #37)."""
    try:
        return await db.fetch_one(sql, params)
    except Exception:  # noqa: BLE001
        log.exception("could not record the turn (is the database schema older than the code? run scripts/reset.sh)")
        return None


def _text(content) -> str:
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content if isinstance(b, dict))


async def _run_turn(user: UserContext, thread: str, graph_input: Any, verbose: bool, turn_span):
    trace_id, span_id = current_ids()
    started = time.perf_counter()
    # Decision #47: the input guardrail (one classifier call, ~1.2 s) does not depend on the tool list, so it runs while the tools are
    # fetched; the guard node reuses the verdict when the text matches, and classifies itself otherwise.
    pre_task = None
    said_now = ""
    context = None
    if isinstance(graph_input, dict) and graph_input.get("messages"):
        said_now = _text(graph_input["messages"][-1].content)
        if said_now.strip() and not said_now.startswith("[[opening]]"):
            try:  # decision #51: the classifier reads the assistant's previous reply; one checkpoint read, no model call
                snapshot = await state["graph"].aget_state({"configurable": {"thread_id": f"{user.session_id}:{thread}"}})
                context = previous_reply((snapshot.values or {}).get("messages", []) + list(graph_input["messages"]))
            except Exception as exc:  # noqa: BLE001 - without it the guard node classifies with the context itself
                log.warning("could not read the thread for the classifier's context: %s", exc)
            pre_task = asyncio.create_task(guard_message(said_now, preferred_reply_language(said_now, user.lang), context))
    try:
        tools = await tools_for(user)
    except Exception:  # noqa: BLE001 - an MCP server down must not leave the chat hanging without a final event
        if pre_task:
            pre_task.cancel()
        log.exception("tool discovery failed for session %s", user.session_id)
        said = graph_input["messages"][-1].content if isinstance(graph_input, dict) and graph_input.get("messages") else ""
        lang = preferred_reply_language(said, user.lang) if said else user.lang
        yield _sse("done", {"message": SYSTEMS_UNAVAILABLE.get(lang, SYSTEMS_UNAVAILABLE["en"]), "pending": None, "lang": lang, "what_i_checked": [],
                            "usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, "latency_ms": int((time.perf_counter() - started) * 1000), "model_latency_ms": 0,
                            "model": model_label(), "error": "systems_unavailable", "trace_id": trace_id})
        return
    config = graph_config(user, thread, tools)
    if pre_task:
        try:
            clean, g = await pre_task
            config["configurable"]["guard_precomputed"] = {"text": said_now, "context": context, "clean": clean, "guard": g}
        except Exception as exc:  # noqa: BLE001 - the guard node classifies itself then
            log.warning("precomputed guardrail failed, the guard node will classify: %s", exc)
    graph = state["graph"]
    seen_tools = 0
    final_text, interrupted = "", None
    first_token_ms: int | None = None  # time to first streamed token: the silence the customer sits through (what a voice channel would feel)
    try:
        async for mode, chunk in graph.astream(graph_input, config, stream_mode=["updates", "messages", "custom"]):
            if mode == "custom":
                if isinstance(chunk, dict) and chunk.get("type") == "review":
                    yield _sse("review", chunk)  # the reviewer sent a draft back (or held the reply): the UI drops the streamed draft
                continue
            if mode == "messages":
                msg, meta = chunk
                if meta.get("langgraph_node") == "agent" and isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                    text = msg.content if isinstance(msg.content, str) else "".join(b.get("text", "") for b in msg.content if isinstance(b, dict))
                    if text:
                        if first_token_ms is None:
                            first_token_ms = int((time.perf_counter() - started) * 1000)
                        yield _sse("token", {"text": text})
            else:
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        interrupted = update[0].value if update else None
                        yield _sse("confirm", interrupted)
                    elif node in ("tools", "gate") and isinstance(update, dict) and update.get("turn_log"):
                        for entry in update["turn_log"][seen_tools:]:
                            yield _sse("tool", {k: v for k, v in entry.items() if k != "result_json"})
                        seen_tools = len(update["turn_log"])
                    elif node == "escalate" and isinstance(update, dict) and update.get("messages"):
                        # decision #43: the runtime replaced the held reply with the case number; the held text must not reach the customer
                        c = update["messages"][-1].content
                        final_text = c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))
                        for entry in update.get("turn_log", [])[seen_tools:]:
                            yield _sse("tool", {k: v for k, v in entry.items() if k != "result_json"})
                        seen_tools = len(update.get("turn_log", []))
                    elif node in ("agent", "guard") and isinstance(update, dict):  # guard: the fixed refusal of a blocked message
                        msgs = update.get("messages") or []
                        if msgs and isinstance(msgs[-1], AIMessage) and not getattr(msgs[-1], "tool_calls", None):
                            # Text that accompanies tool calls is never shown: the reviewer only sees final drafts, and DeepSeek once put a
                            # whole invented price list in front of its search call, which the customer saw unreviewed (FAILURE_DIARY #47).
                            c = msgs[-1].content
                            text = c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))
                            if text.strip():
                                final_text = (final_text + "\n" + text).strip() if final_text else text
    except Exception:
        # The model provider failed (network, timeout after retries). Without this the stream just ended and the
        # chat hung with no reply (FAILURE_DIARY #26). Say so in the customer's language; nothing was changed.
        log.exception("turn failed for session %s", user.session_id)
        try:
            values = (await graph.aget_state(config)).values
        except Exception:  # noqa: BLE001 - if the checkpoint store is the thing that failed, still answer the customer
            values = {}
        said = graph_input["messages"][-1].content if isinstance(graph_input, dict) and graph_input.get("messages") else ""
        lang = values.get("lang") or (preferred_reply_language(said, user.lang) if said else user.lang)
        turn_log = [{k: v for k, v in e.items() if k != "result_json"} for e in values.get("turn_log", [])]
        latency = int((time.perf_counter() - started) * 1000)
        await _record_turn(
            "INSERT INTO agent.turns(session_id,user_id,thread,lang_in,lang_out,model,input_tokens,output_tokens,cached_tokens,latency_ms,tool_calls,finish,trace_id,span_id) VALUES (%s,%s,%s,%s,%s,%s,0,0,0,%s,%s,'error',%s,%s) RETURNING id",
            (user.session_id, user.user_id, thread, lang, lang, model_label(), latency, json.dumps(turn_log, default=str), trace_id, span_id))
        yield _sse("done", {"message": MODEL_UNAVAILABLE.get(lang, MODEL_UNAVAILABLE["en"]), "pending": None, "lang": lang, "what_i_checked": turn_log,
                            "usage": {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, "latency_ms": latency, "model_latency_ms": 0,
                            "model": model_label(), "error": "model_unavailable", "trace_id": trace_id})
        return
    try:
        snapshot = await graph.aget_state(config)
        values = snapshot.values
    except Exception:  # noqa: BLE001 - the checkpoint store failed after the reply was produced
        log.exception("could not read the turn's final state")
        values = {}
    turn_log = [{k: v for k, v in e.items() if k != "result_json" or verbose} for e in values.get("turn_log", [])]
    stored_log = [{k: v for k, v in e.items() if k != "result_json"} for e in values.get("turn_log", [])]
    usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    model_latency = 0
    for m in reversed(values.get("messages", [])):
        if isinstance(m, HumanMessage):
            break
        if isinstance(m, AIMessage):
            u = m.usage_metadata or {}
            usage["input_tokens"] += u.get("input_tokens", 0) or 0
            usage["output_tokens"] += u.get("output_tokens", 0) or 0
            usage["cached_tokens"] += (u.get("input_token_details") or {}).get("cache_read", 0) or 0
            model_latency += (m.additional_kwargs or {}).get("falcon_latency_ms", 0)
    latency = int((time.perf_counter() - started) * 1000)
    verification = values.get("verification") if not interrupted else None
    guardrail = values.get("guardrail")
    turn_span.set_attributes({"falcon.lang": values.get("lang") or "", "falcon.tool_calls": len(stored_log), "falcon.finish": "interrupt" if interrupted else "end",
                              "falcon.verification.status": (verification or {}).get("status", "n/a"), "falcon.verification.rounds": len((verification or {}).get("rounds", [])),
                              "falcon.guardrail.label": (guardrail or {}).get("label", "n/a"), "falcon.guardrail.action": (guardrail or {}).get("action", "n/a"),
                              "falcon.tokens.input": usage["input_tokens"], "falcon.tokens.output": usage["output_tokens"], "falcon.latency_ms": latency,
                              "falcon.first_token_ms": first_token_ms if first_token_ms is not None else -1})
    row = await _record_turn(
        "INSERT INTO agent.turns(session_id,user_id,thread,lang_in,lang_out,model,input_tokens,output_tokens,cached_tokens,latency_ms,first_token_ms,tool_calls,finish,verification,guardrail,trace_id,span_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
        (user.session_id, user.user_id, thread, values.get("lang"), values.get("lang"), model_label(), usage["input_tokens"], usage["output_tokens"], usage["cached_tokens"], latency, first_token_ms, json.dumps(stored_log, default=str),
         "interrupt" if interrupted else "end", json.dumps(verification, default=str) if verification else None, json.dumps(guardrail, default=str) if guardrail else None, trace_id, span_id))
    await _record_turn("INSERT INTO agent.token_budget(day,input_tokens,output_tokens) VALUES (%s,%s,%s) ON CONFLICT (day) DO UPDATE SET input_tokens = agent.token_budget.input_tokens + EXCLUDED.input_tokens, output_tokens = agent.token_budget.output_tokens + EXCLUDED.output_tokens RETURNING day",
                       (date.today(), usage["input_tokens"], usage["output_tokens"]))
    yield _sse("done", {"message": final_text, "pending": interrupted, "lang": values.get("lang") or user.lang, "what_i_checked": turn_log, "usage": usage, "latency_ms": latency, "model_latency_ms": model_latency,
                        "first_token_ms": first_token_ms,
                        "model": model_label(), "verification": verification, "guardrail": guardrail, "turn_id": row["id"] if row else None, "trace_id": trace_id})


@app.post("/api/chat")
async def chat(body: ChatBody, request: Request, user: UserContext = Depends(current_user)):
    rate_limit(request)
    await check_budget(user.session_id)
    text = body.message.strip()
    if not text or len(text) > 2000:
        raise HTTPException(400, "Message must be 1 to 2000 characters.")
    user = _ui_lang(user, body.lang)
    initial = {"messages": [HumanMessage(content=text)], "role": user.role, "user_name": user.display_name, "preferred_lang": user.lang}
    return EventSourceResponse(_run_and_stream(user, body.thread, initial, verbose=_is_admin(request), tags=_tags(request)))


OPENING = {
    # Sent as the customer's (hidden) first message when a conversation opens, so the agent speaks first from the data:
    # greet by name, check the bookings and flight status, flag what needs attention, offer the options. The UI never shows it.
    "en": "[[opening]] Hello. I just opened the chat. Please check my bookings and the status of my flights, tell me if anything needs my attention, and ask how you can help.",
    "ar": "[[opening]] مرحباً. فتحت المحادثة الآن. من فضلك تحقق من حجوزاتي وحالة رحلاتي، وأخبرني إن كان هناك ما يستدعي انتباهي، واسألني كيف يمكنك المساعدة.",
}


class OpenBody(BaseModel):
    thread: str = "main"
    lang: str | None = None  # the app's language (decision #40): the opening is in it, whatever the customer record says


def _ui_lang(user: UserContext, requested: str | None) -> UserContext:
    """The language the app is set to wins over the customer record's preference: Arabic chosen means everything in Arabic."""
    lang = requested if requested in ("en", "ar") else user.lang
    return user.model_copy(update={"lang": lang})


@app.post("/api/chat/open")
async def open_chat(body: OpenBody, request: Request, user: UserContext = Depends(current_user)):
    """The agent's proactive opening line for a freshly selected customer (the disruption-recovery moment: 'your flight
    tomorrow was cancelled; I can rebook you for free or refund 965 AED')."""
    rate_limit(request)
    await check_budget(user.session_id)
    user = _ui_lang(user, body.lang)
    text = OPENING.get(user.lang, OPENING["en"])
    initial = {"messages": [HumanMessage(content=text)], "role": user.role, "user_name": user.display_name, "preferred_lang": user.lang}
    return EventSourceResponse(_run_and_stream(user, body.thread, initial, verbose=_is_admin(request), tags=_tags(request)))


@app.post("/api/chat/confirm")
async def confirm(body: ConfirmBody, request: Request, user: UserContext = Depends(current_user)):
    rate_limit(request)
    return EventSourceResponse(_run_and_stream(user, body.thread, Command(resume={"approved": body.approved, "note": body.note}), verbose=_is_admin(request), tags=_tags(request)))


@app.get("/api/chat/history")
async def history(thread: str = "main", user: UserContext = Depends(current_user)):
    tools = await tools_for(user)
    snapshot = await state["graph"].aget_state(graph_config(user, thread, tools))
    out = []
    for m in snapshot.values.get("messages", []):
        if isinstance(m, HumanMessage):
            if str(m.content).startswith("[[opening]]"):
                continue  # the hidden opener (see /api/chat/open)
            out.append({"role": "user", "text": m.content})
        elif isinstance(m, AIMessage) and not m.tool_calls:
            out.append({"role": "assistant", "text": m.content if isinstance(m.content, str) else ""})
    pending = None
    for task in snapshot.tasks:
        if task.interrupts:
            pending = task.interrupts[0].value
    return {"messages": out, "pending": pending, "lang": snapshot.values.get("lang")}


# ---------------------------------------------------------------- feedback (thumbs on a reply)
class FeedbackBody(BaseModel):
    rating: str                 # "up" | "down"
    thread: str = "main"
    turn_id: int | None = None  # the agent.turns row the reply came from (the done event carries it)
    comment: str | None = None
    message: str | None = None  # the reply text as the customer saw it (kept with the rating so a review needs no join)


@app.post("/api/feedback")
async def feedback(body: FeedbackBody, request: Request, user: UserContext = Depends(current_user)):
    """Thumbs up/down on an assistant reply. Stored as data for the service owner (rate over time, which replies fail);
    nothing in the demo acts on it automatically. A review of flagged conversations is the natural next step."""
    rate_limit(request)
    if body.rating not in ("up", "down"):
        raise HTTPException(400, "rating must be 'up' or 'down'.")
    turn = await db.fetch_one("SELECT trace_id, span_id FROM agent.turns WHERE id = %s AND session_id = %s", (body.turn_id, user.session_id)) if body.turn_id else None
    with tracer("falcon.runtime").start_as_current_span("customer.feedback", links=link_to((turn or {}).get("trace_id"), (turn or {}).get("span_id")),
                                                        attributes={"falcon.rating": body.rating, "falcon.session_id": user.session_id, "falcon.turn_id": body.turn_id or 0,
                                                                    "falcon.rated_trace_id": (turn or {}).get("trace_id") or ""}):
        row = await db.fetch_one(
            "INSERT INTO agent.feedback(session_id,user_id,thread,turn_id,rating,comment,message) VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (user.session_id, user.user_id, body.thread, body.turn_id, body.rating, (body.comment or "")[:1000] or None, (body.message or "")[:4000] or None))
    return {"ok": True, "feedback_id": row["id"]}


# ---------------------------------------------------------------- admin & health
@app.post("/api/admin/reset")
async def admin_reset(x_admin_token: Annotated[str | None, Header()] = None):
    if x_admin_token != settings.admin_token:
        raise HTTPException(403, "Admin token required.")
    proc = await asyncio.to_thread(subprocess.run, [sys.executable, str(REPO_ROOT / "db" / "apply.py"), "--seed-only"], capture_output=True, text=True)
    _tool_cache.clear()
    return {"ok": proc.returncode == 0, "output": (proc.stdout + proc.stderr)[-500:]}


@app.get("/api/health")
async def health():
    return {"ok": True, "now": now().isoformat(), "model": model_label(), "auth_mode": settings.auth_mode, "tracing": bool(settings.otel_exporter_otlp_endpoint)}


@app.get("/api/metrics/summary")
async def metrics_summary(x_admin_token: Annotated[str | None, Header()] = None):
    if x_admin_token != settings.admin_token:
        raise HTTPException(403, "Admin token required.")
    turns = await db.fetch_one("SELECT count(*) AS turns, coalesce(sum(input_tokens),0) AS input_tokens, coalesce(sum(output_tokens),0) AS output_tokens, coalesce(avg(latency_ms),0)::int AS avg_latency_ms FROM agent.turns")
    tools = await db.fetch_all("SELECT tool, count(*) AS calls, sum(CASE WHEN ok THEN 0 ELSE 1 END) AS failures, avg(latency_ms)::int AS avg_ms FROM agent.audit_log GROUP BY tool ORDER BY calls DESC")
    gate = await db.fetch_all("SELECT verification->>'status' AS status, count(*) AS n FROM agent.turns WHERE verification IS NOT NULL GROUP BY 1 ORDER BY 2 DESC")
    thumbs = await db.fetch_all("SELECT rating, count(*) AS n FROM agent.feedback GROUP BY rating")
    return {"turns": turns, "tools": [dict(t) for t in tools], "verification": {g["status"]: g["n"] for g in gate}, "feedback": {t["rating"]: t["n"] for t in thumbs}}


# ---------------------------------------------------------------- web app (built with Vite) with SPA fallback
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    async def spa(path: str):
        candidate = WEB_DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        # the page must be revalidated on every load: it names the hashed bundles, and a cached copy kept showing the previous
        # build after a rebuild (FAILURE_DIARY #71); the bundles themselves are content-hashed and may be cached
        return FileResponse(WEB_DIST / "index.html", headers={"Cache-Control": "no-cache"})
else:

    @app.get("/")
    async def root():
        return JSONResponse({"ok": True, "hint": "web app not built; run `npm run build` in web/", "api": "/api/health"})
