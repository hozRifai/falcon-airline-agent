"""Faithfulness checks shared by the runtime's verification gate and the evaluation scorer.

One definition, two uses: the gate in `runtime/verify.py` blocks a reply before the customer sees it, the
scorer in `eval/metrics.py` reports the same checks over recorded conversations. Both are deterministic and check a
reply against the trace of the conversation, never against the model's opinion:

- amounts     every amount of money in the reply must appear in a tool result of this conversation or in the customer's
              own words. Money has one source in this system (the rules engine, through the quote tools) and one canonical
              form (a number), so exact matching is the right test. Arithmetic by the model is itself a defect: the prompt
              says "repeat its numbers exactly".
- claims      "your booking has been cancelled / changed / refunded" needs a successful write in this conversation, or a
              booking state from the reservations system that says so (a cancellation decided elsewhere, a status check).

Deliberately not checked here: clause citations, dates, deadlines and policy meaning. Those are paraphrase problems
(the model may legitimately say "21 October" for 2026-10-21, or cite the parent of the clause it read), so they belong
to the advisory model review, not to a blocking regex.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .arabic import DIGITS

# Tools that change the world for a customer; a successful result from one of them is evidence for a "done" claim.
WRITE_TOOLS = {"create_booking", "cancel_booking", "change_booking", "add_extra_bag", "request_exception"}
# Side effects that need no confirmation button but are still the only evidence for "I have sent/done it".
EVIDENCE_TOOLS = WRITE_TOOLS | {"send_itinerary", "handover_to_human"}  # a logged case is the evidence for "I have logged a case"
# Tools whose results describe the booking's state in the reservations system; a cancelled status there is also evidence.
BOOKING_STATE_TOOLS = {"get_booking", "list_my_bookings", "get_request_status", "create_booking", "cancel_booking", "change_booking"}
ACTION_EVIDENCE_RE = re.compile(r'"status":\s*"(cancelled|changed|approved|completed)"|"already_done":\s*true', re.IGNORECASE)

# money in a reply: "1,291 AED", "AED 100", "100.00 per passenger", "١٬٢٩١ درهم"; plain integers without a currency
# mark or decimals are not counted (hours, clause numbers, dates have their own shapes)
MONEY_RE = re.compile(r"(?:AED|د\.إ|درهم(?:اً|ا)?)\s*\**\s*(\d[\d,٬ ]*(?:\.\d{1,2})?)|(\d[\d,٬ ]*(?:\.\d{1,2})?)\s*\**\s*(?:AED|د\.إ|درهم(?:اً|ا)?)|\b(\d{1,3}(?:[,٬]\d{3})+(?:\.\d{2})?|\d+\.\d{2})\b")
# clause references ("clause 8.10", "Article 10.2.1", "[7.1]", "البند 8.10") are not money; they are removed before amounts are read
CLAUSE_REF_RE = re.compile(r"(?:\b(?:clauses?|articles?|sections?|paragraphs?)\s*(?:no\.?\s*)?|البند\s*|البنود\s*|المادة\s*|المواد\s*|الفقرة\s*|الفقرات\s*|\[)\s*\d{1,2}(?:\.\d{1,2}){1,3}(?:\s*(?:,|and|or|و|&|/)\s*\d{1,2}(?:\.\d{1,2}){1,3})*\]?",
                           re.IGNORECASE)
# A claim that the agent itself completed an action on the booking. Statements about the flight ("your flight has been
# cancelled by the airline") are facts from tools, not claims, and "will be refunded" / "سيتم استرداد" are promises about a
# pending step, so the patterns are anchored on booking/reservation/refund nouns and on past-tense verbs with a word start.
DONE_CLAIM_RE = re.compile(
    r"\b(booking|reservation)\b[^.\n]{0,40}?\b(has been|have been|was|were|is now)\b( successfully)? (cancelled|canceled|changed|rebooked|moved|updated)\b|"
    r"\b(booking|reservation)\b[^.\n]{0,40}?\b(has been|have been|is now)\b( successfully)? (booked|created|made)\b|"
    r"\b(ticket|e-?ticket|itinerary|confirmation|receipt|email|e-?mail|note|request|case)\b[^.\n]{0,40}?\b(has been|have been|was|were|is now)\b( successfully)? (sent|emailed|e-mailed|forwarded|resent|delivered|submitted|logged|escalated|registered)\b|"
    r"\b(I|we)('ve| have) (cancelled|canceled|changed|rebooked|refunded|processed|submitted|moved|booked|reserved|sent|emailed|e-mailed|forwarded|notified|escalated|registered|applied|charged|logged|opened|raised|filed|created)\b|"
    r"\b(I|we)('ve| have) (successfully |already |now )?(logged|opened|raised|filed|created|booked|cancelled|canceled|changed|sent|emailed)\b|"
    r"(?<![؀-ۿ])تم (إرسال|ارسال) |(?<![؀-ۿ])أرسلت (لك |إليك )?|"
    r"(?<![؀-ۿ])تم تسجيل (الحالة|حالة|طلب|الطلب|حالتك|طلبك)|(?<![؀-ۿ])(?:لقد )?قمت[يِ]? (?:بالفعل )?ب(?:إلغاء|الغاء|تغيير|تعديل|حجز|تسجيل|إرسال|ارسال)|(?<![؀-ۿ])(?:لقد )?سج[ّ]?لت (?:لك |الحالة|حالة|طلب)|"
    r"(?<![؀-ۿ])(?:لقد )?(?:ألغيت|الغيت|غي[ّ]?رت|حجزت|أرسلت) (?:لك |حجزك|الحجز|الرحلة|التذكرة|مسار)|"
    r"(?<![؀-ۿ])تم (حجز|إنشاء حجز|إنشاء الحجز) |"
    r"\brefund (has been|was) (issued|processed|sent|credited)\b|"
    r"(?<![؀-ۿ])تم (إلغاء|الغاء) (حجزك|الحجز|حجز)|(?<![؀-ۿ])تم (تغيير|تعديل) (حجزك|الحجز)|"
    r"(?<![؀-ۿ])تم (استرداد|إصدار استرداد|رد) (المبلغ|مبلغ)|(?<![؀-ۿ])أُلغي حجزك|(?<![؀-ۿ])ألغيت (حجزك|الحجز)", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A case, request or reference number the reply gives: it must appear in a tool result ("رقم الحالة هو EX12345" was invented, 19 Oct)
# "booking reference M8T2LD" is a PNR the customer gave, not a case number: only case/request/ticket words count, and a number the
# customer themselves wrote is allowed (the re-run of 19 Oct held a correct "I couldn't find M8T2LD" four times for this)
CASE_REF_RE = re.compile(r"(?:\b(?:case|request|ticket)\s*(?:number|no\.?|id)?\s*#?\s*|رقم (?:الحالة|الطلب|التذكرة)[^A-Za-z0-9٠-٩]{0,12})([A-Z]{0,4}[-]?\d{1,10})\b", re.IGNORECASE)


def norm_amount(s: str) -> str:
    s = s.translate(DIGITS).replace(",", "").replace("٬", "").replace(" ", "")
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s


def amounts_in(text: str) -> set[str]:
    """Amounts of money in a reply, normalised to two decimals; clause references are not amounts."""
    out = set()
    for m in MONEY_RE.finditer(CLAUSE_REF_RE.sub(" ", text or "")):
        raw = next(g for g in m.groups() if g)
        out.add(norm_amount(raw))
    return out


def numbers_in(blob: str) -> set[str]:
    """Every number in a tool result (any field), normalised to two decimals, so 1291, "1291.00" and 1,291 agree."""
    return {norm_amount(n) for n in NUMBER_RE.findall((blob or "").translate(DIGITS))}


def is_success(tool: str, result_text: str) -> bool:
    """A tool result that is not an error and not a declined confirmation."""
    t = (result_text or "").lstrip()
    if t.startswith("Error"):
        return False
    try:
        data = json.loads(t)
    except (json.JSONDecodeError, TypeError):
        return True
    return not (isinstance(data, dict) and data.get("declined"))


def action_evidence(tool_results: list[tuple[str, str]]) -> bool:
    """Something in this conversation justifies a 'done' claim: a successful write, or a booking state that says so."""
    for tool, text in tool_results:
        if tool in EVIDENCE_TOOLS and is_success(tool, text):
            return True
        if tool in BOOKING_STATE_TOOLS and is_success(tool, text) and ACTION_EVIDENCE_RE.search(text or ""):
            return True
    return False


@dataclass
class Verdict:
    amounts: list[str] = field(default_factory=list)              # amounts found in the reply
    unsupported_amounts: list[str] = field(default_factory=list)  # not in any tool result nor in the customer's words
    false_claim: bool = False                                     # says an action was done; nothing in the trace did it
    evidence: int = 0                                             # tool results consulted
    checked: bool = True                                          # False when there were no tool results to check against

    @property
    def ok(self) -> bool:
        return not self.unsupported_amounts and not self.false_claim

    def problems(self) -> list[str]:
        out = []
        if self.unsupported_amounts:
            out.append("unsupported_amounts")
        if self.false_claim:
            out.append("false_claim")
        return out

    def nudge(self) -> str:
        """The corrective instruction for one retry: precise about what was wrong, silent about everything else."""
        parts = []
        if self.unsupported_amounts:
            amts = ", ".join(self.unsupported_amounts)
            parts.append(f"Your previous draft was not sent: it mentions the amount(s) {amts}, which do not appear in any tool result or in the "
                         "customer's own message. State only figures returned by tools, exactly as returned (no totals, differences or conversions "
                         "computed by you). If you need a figure you do not have, call the tool now instead of estimating.")
        if self.false_claim:
            parts.append("Your previous draft was not sent: it says an action on the booking was completed (cancelled, changed, refunded), but no "
                         "such action has succeeded in this conversation. Do not claim it. If the customer wants it, call the action tool; "
                         "they confirm with a button before anything changes.")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {"amounts": self.amounts, "unsupported_amounts": self.unsupported_amounts, "false_claim": self.false_claim,
                "evidence": self.evidence, "checked": self.checked}


def check_reply(text: str, tool_results: list[tuple[str, str]], customer_texts: list[str]) -> Verdict:
    """Check one reply against the conversation's tool results (name, result text) and the customer's messages."""
    v = Verdict(evidence=len(tool_results))
    said = amounts_in(text)
    v.amounts = sorted(said)
    if said:
        allowed: set[str] = set()
        for _, result_text in tool_results:
            allowed |= numbers_in(result_text)
        for c in customer_texts:
            allowed |= amounts_in(c) | numbers_in(c)
        v.unsupported_amounts = sorted(a for a in said if a not in allowed)
    if DONE_CLAIM_RE.search(text or "") and not action_evidence(tool_results):
        v.false_claim = True
    refs = {m.group(1).translate(DIGITS).upper() for m in CASE_REF_RE.finditer((text or "").translate(DIGITS))}
    if refs:
        seen = (" ".join(r for _, r in tool_results) + " " + " ".join(customer_texts)).translate(DIGITS).upper()
        if any(ref not in seen for ref in refs):
            v.false_claim = True  # a number the systems never issued
    return v
