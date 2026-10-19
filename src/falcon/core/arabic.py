"""Arabic text helpers: normalisation (airport names, scorer), script detection, digit folding.

Why normalise: Arabic keyword search fails without it. "الإلغاء" and "الالغاء" are the same
word with different alef forms; "ة"/"ه" and "ى"/"ي" are written inconsistently; diacritics
and tatweel (ـ) are decoration. BM25 / Postgres full-text search sees different strings
unless we fold them first. The same folding is applied to documents and to queries.
"""

from __future__ import annotations

import re
import unicodedata

_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_ARABIC_LETTERS = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]")
_LATIN_LETTERS = re.compile(r"[A-Za-z]")

_ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
_EASTERN_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {ord(c): str(i) for i, c in enumerate(_ARABIC_INDIC)}
_DIGIT_MAP.update({ord(c): str(i) for i, c in enumerate(_EASTERN_ARABIC_INDIC)})


def fold_digits(text: str) -> str:
    """٢٠٢٦ -> 2026 so dates and PNRs typed in Arabic-Indic digits still match."""
    return text.translate(_DIGIT_MAP)


def normalize_arabic(text: str) -> str:
    """Canonical form for indexing and querying. Lossy on purpose."""
    text = unicodedata.normalize("NFKC", text)  # presentation forms (ﻻ, ﺔ ...) -> base letters
    text = fold_digits(text)
    text = _DIACRITICS.sub("", text)
    text = text.replace(_TATWEEL, "")
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    text = text.replace("ک", "ك").replace("ی", "ي").replace("ھ", "ه")  # Persian/Urdu letter forms that OCR engines emit for Arabic
    text = text.replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_script(text: str) -> str:
    """Return 'ar', 'en' or 'mixed' from the letters actually used (deterministic, no model)."""
    ar = len(_ARABIC_LETTERS.findall(text))
    la = len(_LATIN_LETTERS.findall(text))
    if ar == 0 and la == 0:
        return "unknown"
    if ar == 0:
        return "en"
    if la == 0:
        return "ar"
    share = ar / (ar + la)
    if share >= 0.7:
        return "ar"
    if share <= 0.3:
        return "en"
    return "mixed"


def preferred_reply_language(text: str, previous: str = "en") -> str:
    """The rule the agent is held to: answer in the language of the LAST message.
    Mixed messages (Arabic sentence with a Latin PNR or airport code) count as Arabic if
    Arabic letters dominate; otherwise keep the previous language."""
    script = detect_script(text)
    if script in ("ar", "en"):
        return script
    if script == "mixed":
        ar = len(_ARABIC_LETTERS.findall(text))
        la = len(_LATIN_LETTERS.findall(text))
        return "ar" if ar >= la else previous
    return previous


# What stays Latin in an Arabic reply by design: airport codes, PNRs, flight numbers, AED, dates, times, clause ids.
_CODE_TOKEN = re.compile(r"\b(?:[A-Z]{2,3}|FA\d{1,4}|[A-Z0-9]{6}|AED|Gulf time|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\d{1,2}(?:\.\d{1,2}){1,3})\b")


def reply_script(text: str) -> str:
    """The script a reply is written in ('ar', 'en', 'mixed', 'unknown'), ignoring the tokens that are Latin in either language.
    The runtime reviewer holds a draft to it (FAILURE_DIARY #43) and the evaluation scorer reports it; one definition for both."""
    return detect_script(_CODE_TOKEN.sub(" ", text or ""))
