"""Speech normaliser (docs/VOICE_PLAN.md rule 4): the reviewed reply, rewritten so a TTS voice says codes and amounts right.

Deterministic, no model: booking codes and flight numbers are spelled character by character ("K3P7QZ" -> "K three P seven Q Z"),
clause ids are read digit by digit with "point" ("7.4.4" -> "seven point four point four"), amounts with a currency become words
("AED 1,291.00" -> "one thousand two hundred ninety one dirhams"), ISO dates become "20 October 2026", markdown and list markers go.
Arabic gets the same treatment with Arabic number words and Arabic names for the Latin letters, because the Arabic voice reads
the Arabic reply and the codes in it stay Latin.
"""

from __future__ import annotations

import re
import unicodedata
from decimal import Decimal

from num2words import num2words

DIGITS = {"en": ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
          "ar": ["صفر", "واحد", "اثنان", "ثلاثة", "أربعة", "خمسة", "ستة", "سبعة", "ثمانية", "تسعة"]}
LETTERS_AR = {"A": "إيه", "B": "بي", "C": "سي", "D": "دي", "E": "إي", "F": "إف", "G": "جي", "H": "إتش", "I": "آي", "J": "جيه",
              "K": "كيه", "L": "إل", "M": "إم", "N": "إن", "O": "أو", "P": "بي", "Q": "كيو", "R": "آر", "S": "إس", "T": "تي",
              "U": "يو", "V": "في", "W": "دبليو", "X": "إكس", "Y": "واي", "Z": "زد"}
POINT = {"en": "point", "ar": "نقطة"}
TO = {"en": " to ", "ar": " إلى "}
AND = {"en": " and ", "ar": " و"}
MONTHS = {"en": ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
          "ar": ["يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر"]}
# Every currency the seed uses is AED; the word for it in each language, and its minor unit.
CURRENCY = {"en": ("dirham", "dirhams", "fils"), "ar": ("درهم", "درهم", "فلس")}

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_AMOUNT_BEFORE = re.compile(rf"\bAED\s?({_NUM})")
_AMOUNT_AFTER = re.compile(rf"({_NUM})\s?(?:AED\b|دراهم|درهماً|درهما|درهم|dirhams?\b)")
_CLAUSE = re.compile(r"\b(clause|article|section|condition|المادة|مادة|البند|بند|الفقرة|فقرة)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)
_DOTTED = re.compile(r"(?<![\d.])\d+(?:\.\d+){2,}(?!\d)(?!\.\d)")  # 7.4.4 without a keyword (an amount never has two dots)
_CODE = re.compile(r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9]+(?:-[A-Z0-9]+)*\b")  # K3P7QZ, FA929, CR-K3P7QZ-6ACE33
_CAPS = re.compile(r"\b[A-Z]{2,4}\b")  # airport and other letter codes: DXB, CAI
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_ROUTE_DASH = re.compile(r"(?<=[A-Z])\s?[–-]\s?(?=[A-Z]{3}\b)")  # DXB–KWI
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.MULTILINE)
_HEADER = re.compile(r"^\s*#+\s*", re.MULTILINE)
_EMPHASIS = re.compile(r"(\*\*|__|\*|`)")
_SENTENCE_END = tuple(".!?:؟،;")


def number_words(n: int, lang: str) -> str:
    """1291 -> "one thousand two hundred ninety one" / "ألف ومئتان وواحد وتسعون"."""
    if lang == "ar":
        return re.sub(r"\bو\s+", "و", num2words(n, lang="ar")).strip()
    return re.sub(r"\s+", " ", num2words(n, lang="en").replace(",", "").replace("-", " ").replace(" and ", " ")).strip()


def spell(code: str, lang: str) -> str:
    """Character by character, a pause at each hyphen: "CR-K3P7QZ" -> "C R, K three P seven Q Z"."""
    parts = []
    for chunk in code.split("-"):
        chars = [DIGITS[lang][int(c)] if c.isdigit() else (LETTERS_AR.get(c, c) if lang == "ar" else c) for c in chunk]
        parts.append(" ".join(chars))
    return "، ".join(parts) if lang == "ar" else ", ".join(parts)


def amount_words(raw: str, lang: str) -> str:
    value = Decimal(raw.replace(",", ""))
    whole, minor = int(value), int((value - int(value)) * 100)
    one, many, sub = CURRENCY[lang]
    text = f"{number_words(whole, lang)} {one if whole == 1 else many}"
    if minor:
        text += f"{AND[lang]}{number_words(minor, lang)} {sub}"
    return text


def _dotted(digits: str, lang: str) -> str:
    return f" {POINT[lang]} ".join(" ".join(DIGITS[lang][int(c)] for c in part) for part in digits.split("."))


def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = _LINK.sub(r"\1", text)
    text = _HEADER.sub("", text)
    text = _BULLET.sub("", text)
    text = _EMPHASIS.sub("", text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # each line is said as its own sentence: a list item without a full stop still gets a pause
    return " ".join(ln if ln.endswith(_SENTENCE_END) else ln + "." for ln in lines)


def normalise(text: str, lang: str) -> str:
    """The reply as the voice should say it. lang is the call's language ("ar" or "en")."""
    lang = "ar" if lang == "ar" else "en"
    text = _strip_markdown(text or "")
    text = _AMOUNT_BEFORE.sub(lambda m: amount_words(m.group(1), lang), text)
    text = _AMOUNT_AFTER.sub(lambda m: amount_words(m.group(1), lang), text)
    text = _CLAUSE.sub(lambda m: f"{m.group(1)} {_dotted(m.group(2), lang)}", text)
    text = _DOTTED.sub(lambda m: _dotted(m.group(0), lang), text)
    text = _ROUTE_DASH.sub(TO[lang], text)
    text = _CODE.sub(lambda m: spell(m.group(0), lang), text)
    text = _ISO_DATE.sub(lambda m: f"{int(m.group(3))} {MONTHS[lang][int(m.group(2)) - 1]} {m.group(1)}", text)
    text = _CAPS.sub(lambda m: spell(m.group(0), lang), text)
    text = text.replace("→", TO[lang]).replace("—", ", ").replace("–", ", ")
    text = "".join(c for c in text if unicodedata.category(c) not in ("So", "Cs"))  # emoji and pictographs (✈)
    text = re.sub(r"\s+([,.،])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    return re.sub(r"\s+", " ", text).strip()
