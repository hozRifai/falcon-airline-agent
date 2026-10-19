"""A scripted stand-in for the language model.

Two uses:
1. Tests: the graph, the confirmation gate, identity injection and the evaluation harness are exercised
   without any API key or cost. `ScriptedChatModel` replays a fixed list of responses.
2. Demo without a key: `RuleBasedFakeModel` follows simple keyword rules through the real tools, so the
   web app, the "What I checked" panel and the confirm button can be seen working end to end. It is
   obviously not an LLM: it does not understand, it pattern-matches. Switch LLM_PROVIDER / add a key
   for the real thing.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from falcon.core.calendar import WEEKDAY_AR, WEEKDAY_EN, periods
from falcon.core.clock import now

PNR_RE = re.compile(r"\b(?!FA\d)(?=[A-Z0-9]{6}\b)(?=[A-Z]*\d)[A-Z0-9]{6}\b")  # six characters with at least one digit ("CANCEL" is not a PNR, nor is flight FA2921)
FLIGHT_RE = re.compile(r"\b(FA\d{1,4})\b", re.IGNORECASE)
DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")
ROUTE_RE = re.compile(r"\bfrom\s+([a-z][a-z ]*?)\s+to\s+([a-z][a-z]*)", re.IGNORECASE)
ROUTE_AR_RE = re.compile(r"من\s+(\S+)\s+(?:إلى|الى|ل)\s*(\S+)")
# phrase → key in falcon.core.calendar.periods (the real model reads the same calendar from the prompt)
PERIOD_PHRASES = [("first week of next month", "first_week_next_month"), ("first week of november", "first_week_next_month"),
                  ("الأسبوع الأول من الشهر القادم", "first_week_next_month"), ("next weekend", "next_weekend"), ("نهاية الأسبوع القادم", "next_weekend"),
                  ("next week", "next_week"), ("الأسبوع القادم", "next_week"), ("الاسبوع القادم", "next_week"), ("الأسبوع المقبل", "next_week"),
                  ("this weekend", "this_weekend"), ("نهاية الأسبوع", "this_weekend"), ("next month", "next_month"), ("الشهر القادم", "next_month"),
                  ("this week", "this_week"), ("هذا الأسبوع", "this_week"), ("tomorrow", "tomorrow"), ("غدا", "tomorrow"), ("غدًا", "tomorrow")]


def _tool_call(name: str, args: dict) -> dict:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}", "type": "tool_call"}


def _content_json(msg: BaseMessage) -> Any:
    content = msg.content
    if isinstance(content, list):
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    try:
        return json.loads(content)
    except Exception:
        return None


class ScriptedChatModel(BaseChatModel):
    """Replays responses in order. Each item is an AIMessage or a dict {content, tool_calls}."""

    script: list[Any] = []
    tools: list[Any] = []
    index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        clone = self.model_copy()
        clone.tools = list(tools)
        clone.script = self.script  # share the cursor
        return clone

    def _next(self) -> AIMessage:
        if self.index >= len(self.script):
            return AIMessage(content="(script exhausted)")
        item = self.script[self.index]
        self.script[self.index] = item  # keep
        type(self).index = self.index + 1  # advance shared cursor
        if isinstance(item, AIMessage):
            return item
        return AIMessage(content=item.get("content", ""), tool_calls=[_tool_call(tc["name"], tc.get("args", {})) for tc in item.get("tool_calls", [])])

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next())])


class RuleBasedFakeModel(BaseChatModel):
    """Keyword rules over the real tools. Enough to show every flow in the demo without a model key."""

    tools: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "rule-based-fake"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
        clone = self.model_copy()
        clone.tools = list(tools)
        return clone

    # ------------------------------------------------------------ helpers
    def _has(self, name: str) -> bool:
        return any(getattr(t, "name", None) == name for t in self.tools)

    @staticmethod
    def _meta(messages: list[BaseMessage]) -> tuple[str, str]:
        role, lang = "customer", "en"
        for m in messages:
            if isinstance(m, SystemMessage):
                r = re.search(r"ROLE:\s*(\w+)", m.content if isinstance(m.content, str) else "")
                l = re.search(r"REPLY_LANGUAGE:\s*(\w+)", m.content if isinstance(m.content, str) else "")
                role = r.group(1) if r else role
                lang = l.group(1) if l else lang
        return role, lang

    @staticmethod
    def _name(messages: list[BaseMessage]) -> str:
        for m in messages:
            if isinstance(m, SystemMessage):
                r = re.search(r"Signed-in customer:\s*([^(\n]+)", m.content if isinstance(m.content, str) else "")
                if r:
                    return r.group(1).strip()
        return ""

    @staticmethod
    def _last_human(messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    @staticmethod
    def _disclosures(messages: list[BaseMessage]) -> tuple[dict[str, tuple[int | None, str | None]], str]:
        """The guardrail note of this turn (guardrails.describe_disclosure, decision #34): kind -> (figure, unit), and the first question
        the note asks search_policy. The real model reads the same note; this is the scripted model's version of noticing."""
        labels = (("a pregnancy", "pregnancy"), ("a newborn or infant", "infant"), ("a child travelling alone", "child_alone"),
                  ("a medical condition, illness or recent surgery", "medical"), ("a special-assistance need", "assistance"), ("an animal", "animal"),
                  ("a restricted or dangerous item", "dangerous_item"), ("a travel-document problem", "travel_documents"), ("something the Conditions may restrict", "other"))
        for m in messages:
            text = m.content if isinstance(m, SystemMessage) and isinstance(m.content, str) else ""
            if "message discloses" not in text:
                continue
            found: dict[str, tuple[int | None, str | None]] = {}
            for label, kind in labels:
                if label not in text:
                    continue
                r = re.search(re.escape(label) + r", stated as (\d+) (weeks|months|days|years)", text)
                found[kind] = (int(r.group(1)), r.group(2)) if r else (None, None)
            q = re.search(r"search_policy for: (.+?)\. Then tell", text, re.S)
            return found, (q.group(1).split("; ")[0] if q else "")
        return {}, ""

    @staticmethod
    def _tool_results(messages: list[BaseMessage]) -> list[tuple[str, Any]]:
        out = []
        for m in messages:
            if isinstance(m, ToolMessage):
                out.append((m.name or "", _content_json(m) if _content_json(m) is not None else m.content))
        return out

    @staticmethod
    def _text(lang: str, en: str, ar: str) -> str:
        return ar if lang == "ar" else en

    # ------------------------------------------------------------ main
    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any) -> ChatResult:
        role, lang = self._meta(messages)
        human = self._last_human(messages)
        low = human.lower()
        results = self._tool_results(messages)
        last_tool, last_res = (results[-1] if results and isinstance(messages[-1], ToolMessage) else (None, None))
        T = lambda en, ar: self._text(lang, en, ar)  # noqa: E731

        def ai(content: str = "", calls: list[dict] | None = None) -> ChatResult:
            msg = AIMessage(content=content, tool_calls=[_tool_call(c["name"], c["args"]) for c in (calls or [])])
            return ChatResult(generations=[ChatGeneration(message=msg)])

        opening = human.startswith("[[opening]]")  # the app opened the conversation: greet from the bookings (see /api/chat/open)

        # ---- continuing after a tool result
        if last_tool:
            if isinstance(last_res, str) and last_res.lower().startswith(("error", "forbidden")):
                return ai(T(f"I could not do that: {last_res}", f"لم أتمكن من ذلك: {last_res}"))
            if opening and last_tool == "list_my_bookings":
                bookings = (last_res or {}).get("bookings", []) if isinstance(last_res, dict) else []
                first = self._name(messages).split()[0] if self._name(messages) else ""
                active = [b for b in bookings if b["status"] == "confirmed"]
                lines = [T(f"Hello {first}." if first else "Hello.", f"مرحباً {first}." if first else "مرحباً.")]
                if not active:
                    lines.append(T("You have no active bookings. How can I help?", "لا توجد لديك حجوزات فعّالة. كيف يمكنني المساعدة؟"))
                    return ai(" ".join(lines))
                for b in active[:3]:
                    flights = ", ".join(b.get("flights", []))
                    lines.append(T(f"Booking {b['pnr']}: {flights}.", f"الحجز {b['pnr']}: {flights}."))
                disrupted = [b for b in active if b.get("needs_attention")]
                if disrupted:
                    lines.append(T("One of your flights has been cancelled by the airline: you are entitled to a full refund by any method, or a free rebooking on the same route. Which would you prefer?",
                                   "إحدى رحلاتك ألغتها شركة الطيران: يحق لك استرداد كامل المبلغ بأي طريقة، أو إعادة حجز مجانية على المسار نفسه. ماذا تفضّل؟"))
                else:
                    lines.append(T("How can I help you today?", "كيف يمكنني مساعدتك اليوم؟"))
                return ai(" ".join(lines))
            if last_tool == "list_my_bookings":
                bookings = (last_res or {}).get("bookings", []) if isinstance(last_res, dict) else []
                active = [b for b in bookings if b["status"] == "confirmed"] or bookings
                if not active:
                    return ai(T("You have no bookings on this account.", "لا توجد حجوزات على حسابك."))
                b = active[0]
                if any(k in low for k in ("email", "e-mail", "mail me", "بريد", "إيميل", "ايميل")) and self._has("send_itinerary"):
                    return ai("", [{"name": "send_itinerary", "args": {"pnr": b["pnr"]}}])
                if any(k in low for k in ("cancel", "إلغاء", "الغي", "ألغ", "الغ", "إلغ")):
                    method = "cash" if any(k in low for k in ("cash", "card", "original", "بطاقة", "نقد")) else "credit"
                    return ai("", [{"name": "quote_cancellation", "args": {"pnr": b["pnr"], "refund_method": method}}])
                if any(k in low for k in ("status", "delay", "late", "on time", "حالة", "تأخير", "متأخر", "موعد")):
                    fl = b.get("flights", [""])[0].split()
                    date = DATE_RE.search(" ".join(b.get("flights", [])))
                    return ai("", [{"name": "get_flight_status", "args": {"flight_number": fl[0] if fl else "", "date": date.group(1) if date else ""}}])
                lines = [f"- {x['pnr']}: {x['route']} on {x['first_departure']} ({x['fare_family']}, {x['status']})" for x in bookings]
                return ai(T("Here are your bookings:\n" + "\n".join(lines), "هذه حجوزاتك:\n" + "\n".join(lines)))
            if last_tool == "quote_cancellation" and isinstance(last_res, dict):
                q = last_res
                if q.get("allowed"):
                    return ai(T(f"Cancellation is possible for booking {q['pnr']} ({q['fare_family']} fare). Fee {q['fee']} AED, refund {q['refund']} AED as {'travel credit' if q['refund_method']=='credit' else 'a refund to your card'}. Shall I go ahead? Reply 'yes' to confirm.",
                                f"يمكن إلغاء الحجز {q['pnr']} (درجة {q['fare_family']}). الرسوم {q['fee']} درهم، والمبلغ المسترد {q['refund']} درهم {'كرصيد سفر' if q['refund_method']=='credit' else 'إلى بطاقتك'}. هل أتابع؟ أجب بـ'نعم' للتأكيد."))
                extra = ""
                if q.get("exception_if_approved"):
                    extra = T(f" I can log an exception case for our customer-care team; if approved the refund would be {q['exception_if_approved']['refund']} AED. Say 'request exception' and tell me your reason.",
                              f" يمكنني تسجيل حالة استثناء لفريق خدمة العملاء؛ إذا تمت الموافقة سيكون المبلغ المسترد {q['exception_if_approved']['refund']} درهم. قل 'طلب استثناء' واذكر سببك.")
                return ai(T(f"I'm sorry, this cancellation is not allowed: {q['reason_en']}{extra}", f"عذراً، لا يمكن إلغاء هذا الحجز: {q['reason_ar']}{extra}"))
            if last_tool == "cancel_booking" and isinstance(last_res, dict):
                return ai(T(f"Done. Booking {last_res.get('pnr')} is cancelled; {last_res.get('refund_amount')} AED will be refunded as {last_res.get('refund_method')} (reference {last_res.get('reference')}).",
                            f"تم. الحجز {last_res.get('pnr')} ملغى؛ سيتم استرداد {last_res.get('refund_amount')} درهم ({last_res.get('refund_method')}) — المرجع {last_res.get('reference')}."))
            if last_tool == "list_my_attachments" and isinstance(last_res, dict):
                atts = last_res.get("attachments") or []
                pnr = next((r.get("pnr") for _, r in reversed(results) if isinstance(r, dict) and r.get("pnr")), None) or (PNR_RE.search(human.upper()).group(0) if PNR_RE.search(human.upper()) else None)
                if atts and pnr and self._has("request_exception"):
                    humans = " ".join((getattr(m, "content", "") or "") for m in messages if getattr(m, "type", "") == "human").lower()
                    kind = "bereavement" if any(k in humans for k in ("passed away", "died", "death", "توفي", "وفاة")) else "late_cancellation"
                    return ai("", [{"name": "request_exception", "args": {"pnr": pnr, "kind": kind, "reason": human[:300], "refund_method": "credit",
                                                                          "evidence_attachment_ids": [a["id"] for a in atts]}}])
                return ai(T("An exception case for a medical reason needs supporting evidence. Please attach your doctor's note or document with the paperclip in the chat, and I will log the case with it.",
                            "تسجيل حالة استثناء لسبب طبي يحتاج إلى مستند داعم. يرجى إرفاق التقرير الطبي أو المستند بزر المشبك في المحادثة، وسأسجّل الحالة معه."))
            if last_tool == "request_exception" and isinstance(last_res, dict):
                return ai(T(f"I have logged case #{last_res.get('request_id')} for our customer-care team with your reason and the proposed refund. You will receive an email once it has been reviewed; nothing changes on the booking until then.",
                            f"سجّلت الحالة رقم #{last_res.get('request_id')} لفريق خدمة العملاء مع سببك والمبلغ المقترح. ستصلك رسالة بالبريد الإلكتروني بعد مراجعتها؛ لن يتغير شيء في الحجز حتى ذلك الحين."))
            if last_tool == "get_flight_status" and isinstance(last_res, dict):
                f = (last_res.get("flights") or [{}])[0]
                return ai(T(f"Flight {f.get('flight_number')} on {f.get('date')} is {f.get('status')}" + (f", delayed {f.get('delay_minutes')} minutes ({f.get('reason')})." if f.get("delay_minutes") else "."),
                            f"الرحلة {f.get('flight_number')} بتاريخ {f.get('date')} حالتها {f.get('status')}" + (f"، متأخرة {f.get('delay_minutes')} دقيقة ({f.get('reason')})." if f.get("delay_minutes") else ".")))
            if last_tool == "quote_booking" and isinstance(last_res, dict):
                q = last_res
                if q.get("allowed"):
                    f = (q.get("flights") or [{}])[0]
                    return ai(T(f"Flight {f.get('flight_number')} on {f.get('departure')}, {q['fare_family']} fare: {q.get('fare_per_passenger')} AED per passenger, total {q['amount_due']} AED charged to the card on file. Shall I book it? Reply 'yes' to confirm.",
                                f"الرحلة {f.get('flight_number')} يوم {f.get('departure')}، درجة {q['fare_family']}: {q.get('fare_per_passenger')} درهم للراكب، الإجمالي {q['amount_due']} درهم يُحصَّل من البطاقة المسجلة. هل أحجزها؟ أجب بـ'نعم' للتأكيد."))
                return ai(T(f"I cannot book this flight: {q['reason_en']}", f"لا يمكنني حجز هذه الرحلة: {q['reason_ar']}"))
            if last_tool == "send_itinerary" and isinstance(last_res, dict):
                return ai(T(f"Done. The itinerary for {last_res.get('pnr')} has been sent to {last_res.get('to')} (reference {last_res.get('reference')}).",
                            f"تم. أُرسل مسار الرحلة للحجز {last_res.get('pnr')} إلى {last_res.get('to')} (المرجع {last_res.get('reference')})."))
            if last_tool == "create_booking" and isinstance(last_res, dict) and self._has("send_itinerary") and any(
                    k in (getattr(m, "content", "") or "").lower() for m in messages if getattr(m, "type", "") == "human" for k in ("email", "e-mail", "mail me", "بريد", "إيميل", "ايميل")):
                return ai("", [{"name": "send_itinerary", "args": {"pnr": last_res.get("pnr")}}])  # the customer asked for the ticket by email in the same breath
            if last_tool == "create_booking" and isinstance(last_res, dict):
                return ai(T(f"Done. Your booking {last_res.get('pnr')} is confirmed: {last_res.get('flight')}; {last_res.get('amount_charged')} AED charged to the card on file.",
                            f"تم. حجزك {last_res.get('pnr')} مؤكد: {last_res.get('flight')}؛ حُصِّل {last_res.get('amount_charged')} درهم من البطاقة المسجلة."))
            if last_tool == "search_flights" and isinstance(last_res, dict):
                opts = last_res.get("options_cheapest_first") or []
                if not opts:
                    return ai(T(f"I found no Falcon flights for {last_res.get('dates')}.", f"لم أجد رحلات لفالكون في الفترة {last_res.get('dates')}."))
                o = opts[0]
                p = o["price_per_passenger_aed"]
                d = DATE_RE.search(o.get("departure_local", ""))
                wd = __import__("datetime").date.fromisoformat(d.group(1)).weekday() if d else None
                return ai(T(f"The cheapest option between {last_res.get('dates')} is flight {o['flight_number']} on {WEEKDAY_EN[wd] + ' ' if wd is not None else ''}{o.get('departure_local')} "
                            f"({o['origin']} to {o['destination']}): Special {p['SPECIAL']} AED per passenger (Saver {p['SAVER']}, Flex {p['FLEX']}). I searched the whole period, {last_res.get('days_searched')} days.",
                            f"أرخص خيار في الفترة {last_res.get('dates')} هو الرحلة {o['flight_number']} يوم {WEEKDAY_AR[wd] + ' ' if wd is not None else ''}{o.get('departure_local')} "
                            f"({o['origin']} إلى {o['destination']}): الدرجة الخاصة {p['SPECIAL']} درهم للمسافر (الموفرة {p['SAVER']}، المرنة {p['FLEX']}). بحثت في الفترة كاملة، {last_res.get('days_searched')} أيام."))
            if last_tool == "check_travel_eligibility" and isinstance(last_res, dict):
                lines = [((f.get("rule_ar") if lang == "ar" else f.get("rule_en")) or "") for f in last_res.get("findings", [])]
                if last_res.get("status") == "ok":
                    lines.append(T("Nothing in the Conditions prevents the trip as described.", "لا شيء في شروط النقل يمنع الرحلة كما وصفتها."))
                else:
                    lines.append(T("If this means you cannot travel as booked, I can quote a cancellation under your fare rules, or log an exception case for the customer-care team with your medical document attached. Which would you prefer?",
                                   "إن كان هذا يعني أنك لن تتمكني من السفر كما هو محجوز، يمكنني تسعير الإلغاء وفق قواعد درجتك، أو تسجيل حالة استثناء لفريق خدمة العملاء مع إرفاق المستند الطبي. ماذا تفضّلين؟"))
                return ai("\n".join(lines))
            if last_tool == "search_policy" and isinstance(last_res, dict):
                hits = last_res.get("results", [])[:2]
                if not hits:
                    return ai(T("I could not find that in the Conditions of Carriage. I can hand you to a colleague.", "لم أجد ذلك في شروط النقل. يمكنني تحويلك إلى زميل."))
                parts = [f"[{h['clause_id']}] " + ((h.get('text_ar') if lang == 'ar' and h.get('text_ar') else h.get('text_en')) or "")[:300] for h in hits]
                return ai(T("According to the Conditions of Carriage:\n" + "\n".join(parts), "وفقاً لشروط النقل:\n" + "\n".join(parts)))
            return ai(T("Done.", "تم."))

        # ---- first step for a new user message
        if opening and self._has("list_my_bookings"):
            return ai("", [{"name": "list_my_bookings", "args": {}}])
        disclosed, check_question = self._disclosures(messages)  # decision #34: the Conditions are checked before anything else in the turn
        if disclosed and self._has("check_travel_eligibility"):
            args: dict[str, Any] = {}
            if "pregnancy" in disclosed and disclosed["pregnancy"][0] is not None:
                value, unit = disclosed["pregnancy"]
                args["pregnancy_weeks" if unit == "weeks" else "pregnancy_months"] = value
                if re.search(r"\btwins?\b|توأم|توام", human, re.I):
                    args["multiple_pregnancy"] = True
            if "infant" in disclosed and disclosed["infant"][0] is not None:
                value, unit = disclosed["infant"]
                args["infant_age_days"] = value * {"days": 1, "weeks": 7, "months": 30}.get(unit or "days", 1)
            if "child_alone" in disclosed and disclosed["child_alone"][0] is not None:
                args["child_age_years"], args["travelling_alone"] = disclosed["child_alone"][0], True
            if args:
                return ai("", [{"name": "check_travel_eligibility", "args": args}])
            if check_question and self._has("search_policy"):
                return ai("", [{"name": "search_policy", "args": {"query": check_question, "top_k": 4}}])
        pnr_m = PNR_RE.search(human.upper())
        pnr_txt = pnr_m.group(0) if pnr_m else None
        quote_id = next((r.get("quote_id") for _, r in reversed(results) if isinstance(r, dict) and r.get("quote_id") and r.get("allowed")), None)
        # customer
        if any(k in low for k in ("yes", "confirm", "go ahead", "نعم", "أكد", "موافق", "تابع")) and quote_id:
            quoted = next((r for _, r in reversed(results) if isinstance(r, dict) and r.get("quote_id") == quote_id), {})
            if quoted.get("kind") == "book":
                return ai("", [{"name": "create_booking", "args": {"quote_id": quote_id}}])
            return ai("", [{"name": "cancel_booking", "args": {"pnr": quoted.get("pnr"), "quote_id": quote_id}}])
        last_search = next((r for _, r in reversed(results) if isinstance(r, dict) and r.get("options_cheapest_first")), None)
        if last_search and self._has("quote_booking") and any(k in low for k in ("book it", "book that", "book this", "book the", "go ahead and book", "احجزها", "احجز لي", "احجز هذه", "احجز الرحلة")):
            opt = last_search["options_cheapest_first"][0]
            fare = "FLEX" if any(k in low for k in ("flex", "المرنة")) else "SAVER" if any(k in low for k in ("saver", "الموفرة")) else "SPECIAL"
            return ai("", [{"name": "quote_booking", "args": {"flight_instance_id": opt["flight_instance_id"], "fare_family": fare, "passengers": 1}}])
        if any(k in low for k in ("email", "e-mail", "mail me", "بريد", "إيميل", "ايميل")) and self._has("send_itinerary") and not any(k in low for k in ("book it", "احجز")):
            pnr = pnr_txt or next((r.get("pnr") for _, r in reversed(results) if isinstance(r, dict) and r.get("pnr") and r.get("status") == "confirmed"), None)
            if pnr:
                return ai("", [{"name": "send_itinerary", "args": {"pnr": pnr}}])
            return ai("", [{"name": "list_my_bookings", "args": {}}])
        if any(k in low for k in ("exception", "استثناء", "supervisor", "مشرف", "attached", "attachment", "أرفقت", "مرفق", "passed away", "died", "death", "توفي", "وفاة")):
            pnr = pnr_txt if pnr_m else next((r.get("pnr") for _, r in reversed(results) if isinstance(r, dict) and r.get("pnr")), None)
            if pnr and self._has("list_my_attachments"):
                return ai("", [{"name": "list_my_attachments", "args": {}}])  # evidence first (decision #33); the result decides what happens next
            if pnr:
                return ai("", [{"name": "request_exception", "args": {"pnr": pnr, "kind": "late_cancellation", "reason": human[:300], "refund_method": "credit"}}])
        if any(k in low for k in ("cancel", "إلغاء", "الغي", "ألغ", "الغ", "إلغ", "refund", "استرداد")):
            if pnr_m and self._has("quote_cancellation"):
                method = "cash" if any(k in low for k in ("cash", "card", "original", "بطاقة", "نقد")) else "credit"
                return ai("", [{"name": "quote_cancellation", "args": {"pnr": pnr_txt, "refund_method": method}}])
            return ai("", [{"name": "list_my_bookings", "args": {}}])
        if any(k in low for k in ("status", "delay", "late", "on time", "حالة", "تأخير", "متأخر")):
            return ai("", [{"name": "list_my_bookings", "args": {}}])
        route = ROUTE_RE.search(human) or ROUTE_AR_RE.search(human)
        if route and self._has("search_flights"):
            key = next((k for phrase, k in PERIOD_PHRASES if phrase in low), None)
            dates = DATE_RE.findall(human)
            if key:
                period = next(p for p in periods(now().date()) if p.key == key)
                start, end = str(period.start), str(period.end)
            elif dates:
                start, end = dates[0], dates[-1]
            else:
                return ai(T("Which dates would you like to travel? A day or a period (for example next week) works.", "في أي تاريخ تريد السفر؟ يمكنك ذكر يوم أو فترة (مثل الأسبوع القادم)."))
            return ai("", [{"name": "search_flights", "args": {"origin": route.group(1).strip(), "destination": route.group(2).strip(), "date_from": start, "date_to": end, "passengers": 1}}])
        if any(k in low for k in ("booking", "trip", "flight", "حجز", "حجوزات", "رحل")) and not any(k in low for k in ("policy", "allowed", "baggage", "bag", "أمتعة", "شروط")):
            return ai("", [{"name": "list_my_bookings", "args": {}}])
        if any(k in low for k in ("baggage", "bag", "luggage", "policy", "condition", "compensation", "denied", "infant", "أمتعة", "حقيبة", "شروط", "تعويض", "رضيع", "سياسة")):
            return ai("", [{"name": "search_policy", "args": {"query": human, "top_k": 4}}])
        return ai(T("Hello! I'm Falcon's assistant (scripted demo model, no API key configured). I can show your bookings, quote a cancellation, check a flight status, or answer policy questions.",
                    "مرحباً! أنا مساعد فالكون (نموذج تجريبي بدون مفتاح). يمكنني عرض حجوزاتك، أو تسعير الإلغاء، أو التحقق من حالة الرحلة، أو الإجابة عن أسئلة الشروط."))
