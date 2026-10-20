"""The bare yes / bare no vocabulary (decision #41), shared by the chat runtime and the voice channel.

A message that only says yes (EN / AR), with at most filler words around it, is a confirmation with nothing new to decide: the
chat raises the confirmation card itself, and the voice channel presses the confirm button. A bare no declines the spoken card.
Anything else is not a decision. Pure module (no LangGraph), so the voice service can import it.
"""

from __future__ import annotations

import re

CONFIRM_CORE = {"yes", "yeah", "yep", "ok", "okay", "sure", "confirm", "confirmed", "proceed", "agreed",
                "نعم", "أجل", "اجل", "موافق", "موافقة", "أوافق", "اوافق", "تمام", "أكد", "تأكيد", "تابع", "نفذ", "حسنا", "طيب", "ماشي"}
CONFIRM_FILLER = {"please", "go", "ahead", "do", "it", "that", "with", "the", "and", "sure", "fine", "thanks", "thank", "you", "cancel", "cancellation", "change", "booking", "book", "bag",
                  "من", "فضلك", "لو", "سمحت", "الإلغاء", "الالغاء", "بالإلغاء", "بالالغاء", "التغيير", "بالتغيير", "الحجز", "بالحجز", "المتابعة", "ذلك", "هذا", "شكرا", "شكرًا", "اعمل", "افعل", "أكمل", "اكمل", "كمل"}
# The voice card's "no" (VOICE_PLAN rule 3). Filler is narrower than for yes: "cancel" is a filler after yes ("yes, cancel it") but
# must not turn "no, cancel" into anything, so a decline allows only politeness and "it/that/this" around the no.
DECLINE_CORE = {"no", "nope", "nah", "don", "dont", "stop", "leave", "decline", "declined",
                "لا", "كلا", "لأ", "توقف", "اتركه", "اترك", "ارفض", "أرفض", "رافض"}
DECLINE_FILLER = {"please", "thanks", "thank", "you", "it", "that", "this", "not", "now", "do", "t", "leave", "as", "is", "for", "no",
                  "شكرا", "شكرًا", "من", "فضلك", "لو", "سمحت", "الآن", "الان", "هذا", "ذلك", "مش", "أريد", "اريد", "تلغي", "تلغيه", "شيء", "شي", "كما", "هو"}
_WORD_RE = re.compile(r"\w+")
_TASHKEEL_RE = re.compile("[ً-ْـ]")  # diacritics and tatweel: "حسنًا" is "حسنا"


def _words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(_TASHKEEL_RE.sub("", text or ""))]


def is_bare_confirmation(text: str) -> bool:
    """True for a message that only says yes (EN / AR), with at most filler words around it: nothing new to decide."""
    words = _words(text)
    return bool(words) and any(w in CONFIRM_CORE for w in words) and all(w in CONFIRM_CORE or w in CONFIRM_FILLER for w in words)


def is_bare_decline(text: str) -> bool:
    """True for a message that only says no (EN / AR), with at most politeness around it. Never true together with a yes."""
    words = _words(text)
    if not words or any(w in CONFIRM_CORE for w in words):
        return False
    return any(w in DECLINE_CORE for w in words) and all(w in DECLINE_CORE or w in DECLINE_FILLER for w in words)
