"""Personal data in chat text: find it, remove it, name what was removed.

Two uses, one definition (decision #29): the guard node redacts a customer's message before the model, the logs or a
trace see it ("my card is 4111 1111 1111 1111" becomes "[payment card ending 1111, removed]"), and the reviewer refuses a
draft reply that would send any of these back out. Deterministic on purpose: a card number has a checksum, an Emirates ID
and an IBAN have a fixed shape, so a regex plus Luhn is both faster and more honest than asking a model whether a number
looks sensitive. Passport numbers are not matched: their shape (a letter and seven digits) collides with nothing here but
has no checksum, and a false positive would hide a booking reference from the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# 13-19 digits with optional spaces or dashes, validated by Luhn below (so dates, amounts and PNRs never match)
CARD_RE = re.compile(r"(?<![\dA-Za-z])(?:\d[ -]?){12,18}\d(?![\dA-Za-z])")
# UAE IBAN: AE + 2 check digits + 19 digits, with optional grouping spaces; other countries: 2 letters, 2 digits, 11-30 alphanumerics
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{4}){2,7}(?:[ -]?[A-Z0-9]{1,4})?\b")
EMIRATES_ID_RE = re.compile(r"\b784[ -]?\d{4}[ -]?\d{7}[ -]?\d\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
# UAE mobile numbers (+971 5x xxx xxxx, 05x xxx xxxx) and international numbers with a + prefix and 10-14 digits
PHONE_RE = re.compile(r"(?<![\w+])(?:(?:\+|00)971[ -]?|0)5\d(?:[ -]?\d){7}(?![\w])|(?<![\w])\+\d{1,3}(?:[ -]?\d){9,13}(?![\w])")

LABELS = {"card": "payment card", "iban": "bank account (IBAN)", "emirates_id": "Emirates ID", "email": "email address", "phone": "phone number"}


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


@dataclass
class Finding:
    kind: str
    text: str          # what was matched, as written
    replacement: str   # what it became


@dataclass
class Redaction:
    text: str
    findings: list[Finding] = field(default_factory=list)

    @property
    def kinds(self) -> list[str]:
        seen: list[str] = []
        for f in self.findings:
            if f.kind not in seen:
                seen.append(f.kind)
        return seen

    @property
    def changed(self) -> bool:
        return bool(self.findings)


def _card_matches(text: str) -> list[tuple[int, int, str]]:
    out = []
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            out.append((m.start(), m.end(), digits))
    return out


def find(text: str) -> list[Finding]:
    """Every piece of personal data in the text, in order of appearance (cards first so a card is never read as a phone)."""
    norm = text.translate(ARABIC_DIGITS)  # one-to-one, so positions in `norm` are positions in `text`
    spans: list[tuple[int, int, Finding]] = []

    def taken(s: int, e: int) -> bool:
        return any(not (e <= a or s >= b) for a, b, _ in spans)

    for s, e, digits in _card_matches(norm):
        spans.append((s, e, Finding("card", text[s:e], f"[payment card ending {digits[-4:]}, removed]")))
    for s, e, kind, rep in (
        *[(m.start(), m.end(), "iban", "[IBAN removed]") for m in IBAN_RE.finditer(norm) if norm[m.start():m.start() + 2].isalpha()],
        *[(m.start(), m.end(), "emirates_id", "[Emirates ID removed]") for m in EMIRATES_ID_RE.finditer(norm)],
        *[(m.start(), m.end(), "email", "[email address removed]") for m in EMAIL_RE.finditer(norm)],
        *[(m.start(), m.end(), "phone", "[phone number removed]") for m in PHONE_RE.finditer(norm)],
    ):
        if not taken(s, e):
            spans.append((s, e, Finding(kind, text[s:e], rep)))
    spans.sort(key=lambda t: t[0])
    return [f for _, _, f in spans]


def redact(text: str) -> Redaction:
    """The text with every piece of personal data replaced by a placeholder that says what was removed."""
    norm = text.translate(ARABIC_DIGITS)
    spans: list[tuple[int, int, Finding]] = []
    for s, e, digits in _card_matches(norm):
        spans.append((s, e, Finding("card", text[s:e], f"[payment card ending {digits[-4:]}, removed]")))
    for pattern, kind, rep in ((IBAN_RE, "iban", "[IBAN removed]"), (EMIRATES_ID_RE, "emirates_id", "[Emirates ID removed]"),
                               (EMAIL_RE, "email", "[email address removed]"), (PHONE_RE, "phone", "[phone number removed]")):
        for m in pattern.finditer(norm):
            if kind == "iban" and not norm[m.start():m.start() + 2].isalpha():
                continue
            if any(not (m.end() <= a or m.start() >= b) for a, b, _ in spans):
                continue
            spans.append((m.start(), m.end(), Finding(kind, text[m.start():m.end()], rep)))
    spans.sort(key=lambda t: t[0])
    out, last, findings = [], 0, []
    for s, e, f in spans:
        out.append(text[last:s])
        out.append(f.replacement)
        findings.append(f)
        last = e
    out.append(text[last:])
    return Redaction("".join(out), findings)
