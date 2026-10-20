"""One voice call against the Falcon runtime: the four rules of docs/VOICE_PLAN.md §2, without Pipecat (unit-testable).

The runtime is the brain: every utterance goes to the same /api/chat (or /api/chat/confirm) the app uses, with the customer's
session token, so the guard, the tools, the reviewer, the gate and the escalation are untouched. This module only decides what is
spoken:

1. only the reviewed reply, the "done" event's message; the token stream is never spoken (speech cannot be taken back);
2. progress is timed, not triggered by the agent (decision #52): when the reply has not come PROGRESS_AFTER seconds after the turn
   was sent, one fixed phrase in the call's language; if it is still not there STILL_AFTER seconds later, one second fixed phrase;
   never more than the two, none for a fast reply, and no model writes them;
3. a "done" with "pending" (the confirmation card) is spoken from the card's fields, and the next utterance is matched by the chat's
   bare-confirmation vocabulary: yes -> confirm approved=true, no -> approved=false, anything else -> the card once more, then declined;
4. everything spoken passes through the deterministic normaliser (speech.normalise).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Protocol

import httpx

from falcon.core.confirmation import is_bare_confirmation, is_bare_decline
from falcon.services.voice.speech import normalise

log = logging.getLogger("falcon.voice")

PROGRESS = {"en": "Give me a minute to search for the right answer for you.", "ar": "امنحني دقيقة لأبحث لك عن الإجابة الصحيحة."}
STILL = {"en": "I'm still checking, thank you for waiting.", "ar": "ما زلت أتحقق، شكراً على صبرك."}
# Decision #52: on the first tool event the phrase came 7 to 10 s after the customer stopped and then 15 s of silence followed, so it
# sounded like a first answer that said nothing (FAILURE_DIARY #69). Timed from the moment the turn is sent, as voice platforms do.
PROGRESS_AFTER = 1.2
STILL_AFTER = 10.0
UNAVAILABLE = {"en": "Sorry, I could not reach our systems just now, so nothing was checked or changed. Please say that again in a moment.",
               "ar": "عذرًا، تعذّر الوصول إلى أنظمتنا الآن، لذلك لم يتم التحقق من أي شيء أو تغييره. يرجى إعادة طلبك بعد قليل."}
NOT_HEARD = {"en": "I did not hear a yes or a no.", "ar": "لم أسمع نعم أو لا."}
ASK = {"en": "Say yes to confirm, or no to leave it as it is.", "ar": "قل نعم للتأكيد، أو لا لتركه كما هو."}
NO_ANSWER_NOTE = "voice call: no clear yes or no after the card was read twice"
LABELS = {"en": {"create_booking": "Book this flight", "cancel_booking": "Cancel booking", "change_booking": "Change flight",
                 "add_extra_bag": "Add baggage", "request_exception": "Log an exception case for the customer-care team"},
          "ar": {"create_booking": "حجز هذه الرحلة", "cancel_booking": "إلغاء الحجز", "change_booking": "تغيير الرحلة",
                 "add_extra_bag": "إضافة أمتعة", "request_exception": "تسجيل حالة استثناء لفريق خدمة العملاء"}}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
          "ar": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]}

Speak = Callable[[str, str], Awaitable[None]]  # (text already normalised, kind: progress | reply | card | error)
Observe = Callable[[str, dict], Awaitable[None]]  # (kind: user | done | spoken, payload): the transcript mirror (phase 2)


class Runtime(Protocol):
    def chat(self, message: str) -> AsyncIterator[tuple[str, dict]]: ...
    def confirm(self, approved: bool, note: str | None = None) -> AsyncIterator[tuple[str, dict]]: ...


class RuntimeClient:
    """The runtime's SSE endpoints with one customer's session token and one thread: (event, data) pairs."""

    def __init__(self, base_url: str, token: str, thread: str, lang: str, http: httpx.AsyncClient | None = None, headers: dict | None = None):
        self.base_url, self.token, self.thread, self.lang = base_url.rstrip("/"), token, thread, lang
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0))
        self.headers = headers or {}  # the evaluation adds its admin token and trace tags; a call sends none

    async def _stream(self, path: str, body: dict) -> AsyncIterator[tuple[str, dict]]:
        headers = {"Authorization": f"Bearer {self.token}", "Accept": "text/event-stream", **self.headers}
        async with self.http.stream("POST", self.base_url + path, json=body, headers=headers) as r:
            r.raise_for_status()
            event, data = "message", []
            async for line in r.aiter_lines():
                line = line.rstrip("\r")
                if not line:
                    if data:
                        yield event, json.loads("\n".join(data))
                    event, data = "message", []
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].lstrip())
            if data:
                yield event, json.loads("\n".join(data))

    def chat(self, message: str) -> AsyncIterator[tuple[str, dict]]:
        return self._stream("/api/chat", {"message": message, "thread": self.thread, "lang": self.lang})

    def confirm(self, approved: bool, note: str | None = None) -> AsyncIterator[tuple[str, dict]]:
        return self._stream("/api/chat/confirm", {"approved": approved, "note": note, "thread": self.thread})


def _when(departure: Any, lang: str) -> str:
    """The quote's departure ("2026-10-21 19:40 (Gulf time)" or ISO) as "21 October 19:40"; the zone note is dropped (every time
    the agent says is Gulf time)."""
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})[T ](\d{2}:\d{2})", str(departure or ""))
    if not m:
        return str(departure or "")
    return f"{int(m.group(3))} {MONTHS[lang][int(m.group(2)) - 1]} {m.group(4)}"


def card_text(pending: dict, lang: str) -> str:
    """The confirmation card as one spoken paragraph, from the same fields the app's card shows (web/src/Chat.tsx ConfirmCard)."""
    ar = lang == "ar"
    parts = ["للتأكيد:" if ar else "To confirm:"]
    for a in pending.get("actions") or []:
        q, args = a.get("quote") or {}, a.get("args") or {}
        cur = q.get("currency") or "AED"
        head = LABELS[lang].get(a.get("tool"), a.get("tool") or "")
        parts.append(f"{head} {args['pnr']}." if args.get("pnr") else f"{head}.")
        for f in q.get("flights") or []:
            route = f"{f.get('origin')} {'إلى' if ar else 'to'} {f.get('destination')}"
            parts.append(f"{'الرحلة' if ar else 'Flight'} {f.get('flight_number')}، {route}، {_when(f.get('departure'), lang)}." if ar
                         else f"Flight {f.get('flight_number')}, {route}, {_when(f.get('departure'), lang)}.")
        if a.get("tool") == "request_exception" and q.get("exception_if_approved"):
            e = q["exception_if_approved"]
            parts.append(f"في حال موافقة الفريق: {e.get('refund')} {cur}، والرسوم {e.get('fee')} {cur}." if ar
                         else f"If the team approves: {e.get('refund')} {cur} back, fee {e.get('fee')} {cur}.")
        elif q and q.get("kind") == "cancel":
            if float(q.get("refund") or 0) > 0:
                how = ("كرصيد سفر" if q.get("refund_method") == "credit" else "إلى البطاقة") if ar else ("as travel credit" if q.get("refund_method") == "credit" else "to your card")
                parts.append(f"المبلغ المسترد {q.get('refund')} {cur} {how}، والرسوم {q.get('fee')} {cur}." if ar
                             else f"You get back {q.get('refund')} {cur} {how}, fee {q.get('fee')} {cur}.")
            else:
                parts.append("لا استرداد لهذه الدرجة." if ar else "No refund on this fare.")
        elif q:
            fee = f"، والرسوم {q.get('fee')} {cur}" if ar else f", fee {q.get('fee')} {cur}"
            parts.append((f"المبلغ المستحق {q.get('amount_due')} {cur}" if ar else f"Amount due {q.get('amount_due')} {cur}")
                         + (fee if float(q.get("fee") or 0) > 0 else "") + ".")
        if args.get("reason"):
            parts.append(f"{'السبب' if ar else 'Reason'}: {args['reason']}.")
    parts.append(ASK[lang])
    return " ".join(parts)


class CallSession:
    """One call: utterances in, spoken text out. One runtime turn at a time: whatever the customer says while a turn is running is
    buffered, and all of it goes to the agent as one message when that turn ends (FAILURE_DIARY #66: a sentence cut into fragments
    at every pause was answered fragment by fragment)."""

    def __init__(self, runtime: Runtime, lang: str, speak: Speak, observe: Observe | None = None,
                 progress_after: float = PROGRESS_AFTER, still_after: float = STILL_AFTER):
        self.runtime, self.lang, self.speak, self.observe = runtime, ("ar" if lang == "ar" else "en"), speak, observe
        self.progress_after, self.still_after = progress_after, still_after
        self._sent_at = 0.0
        self._speaking = asyncio.Lock()  # one line at a time: a reply never interleaves with a progress phrase being pushed
        self.pending: dict | None = None
        self.card_repeats = 0
        self._buffer: list[str] = []
        self._busy = False

    async def _say(self, text: str, kind: str) -> None:
        spoken = normalise(text, self.lang)
        if spoken:
            async with self._speaking:
                # announced before it is spoken: the app paces the reply's text by the spoken words that follow (word by word)
                if self.observe:  # after_s: seconds since the turn was sent (the caller's wait, before TTS start)
                    await self.observe("spoken", {"kind": kind, "text": text, "spoken": spoken,
                                                  "after_s": round(time.monotonic() - self._sent_at, 2) if self._sent_at else None})
                await self.speak(spoken, kind)

    async def handle(self, utterance: str) -> None:
        text = (utterance or "").strip()
        if not text:  # an empty turn (noise, a cough) is never sent to the agent and never answered
            return
        self._buffer.append(text)
        if self._busy:  # the running turn's loop below picks it up, merged with anything else said meanwhile
            return
        self._busy = True
        try:
            while self._buffer:
                merged = " ".join(self._buffer)
                self._buffer.clear()
                await self._turn(merged)
        finally:
            self._busy = False

    async def _turn(self, text: str) -> None:
        if self.observe:
            await self.observe("user", {"text": text, "pending": bool(self.pending)})
        if self.pending is not None:
            if is_bare_confirmation(text):
                events = self.runtime.confirm(True)
            elif is_bare_decline(text):
                events = self.runtime.confirm(False)
            elif self.card_repeats == 0:
                self.card_repeats = 1
                await self._say(f"{NOT_HEARD[self.lang]} {card_text(self.pending, self.lang)}", "card")
                return
            else:
                events = self.runtime.confirm(False, NO_ANSWER_NOTE)
            self.pending, self.card_repeats = None, 0
        else:
            events = self.runtime.chat(text)
        await self._run(events)

    async def _progress(self) -> None:
        """The two timed phrases of rule 2; cancelled the moment the reply arrives. A phrase already started is finished (shield)."""
        await asyncio.sleep(self.progress_after)
        await asyncio.shield(self._say(PROGRESS[self.lang], "progress"))
        await asyncio.sleep(self.still_after)
        await asyncio.shield(self._say(STILL[self.lang], "progress"))

    async def _run(self, events: AsyncIterator[tuple[str, dict]]) -> None:
        self._sent_at = time.monotonic()
        progress = asyncio.create_task(self._progress())
        try:
            async for event, data in events:
                if event == "done":
                    progress.cancel()
                    if self.observe:
                        await self.observe("done", data)
                    await self._say(data.get("message") or "", "reply")
                    if data.get("pending"):
                        self.pending, self.card_repeats = data["pending"], 0
                        await self._say(card_text(self.pending, self.lang), "card")
                # token / review / confirm events are deliberately not spoken (rule 1); the card comes from done.pending
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            progress.cancel()
            log.warning("voice turn failed: %s", exc)
            await self._say(UNAVAILABLE[self.lang], "error")
        finally:
            progress.cancel()
