"""Word and character error rate for comparing speech-to-text services on the same call (decision #55).

Both sides are normalised the same way before counting, so a service is not charged for spelling style: punctuation and Arabic
diacritics go, alef / yaa / taa marbuta forms are unified, digits become words ("23" and "ثلاثة وعشرون" compare equal), and a code
spelled out character by character ("K 3 P 7 Q Z") is joined back into one token ("k3p7qz").
"""

from __future__ import annotations

import re

from num2words import num2words

_TASHKEEL = re.compile("[ً-ْٰـ]")
_PUNCT = re.compile(r"[^\w\s]|_")
_DIGITS = re.compile(r"\d+")
_SINGLE = re.compile(r"^[a-z0-9]$")
_CODE = re.compile(r"^(?=.*\d)[a-z0-9]{2,8}$")  # a code built so far: "k3p7q", "fa"
# what common WER normalisers do for English (Whisper's EnglishTextNormalizer among them): a contraction and its expansion are the
# same words, and "zed" is the letter Z; applied to both sides, so no service gains or loses by its spelling style (21 Oct)
_EN_SAME = {"i'd": "i would", "i'm": "i am", "it's": "it is", "don't": "do not", "can't": "cannot", "i'll": "i will", "zed": "z"}
_CODE_SPLIT = re.compile(r"\b([a-z]{1,3}) (\d{2,4})\b")  # "fa 929" -> "fa929": a flight number written with a space


def normalise(text: str, lang: str) -> list[str]:
    t = _TASHKEEL.sub("", (text or "").lower()).replace("’", "'")
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ة", "ه")
    if lang != "ar":
        t = " ".join(_EN_SAME.get(w, w) for w in t.split())
    t = _PUNCT.sub(" ", t)
    if lang != "ar":
        t = " ".join(_EN_SAME.get(w, w) for w in t.split())  # "zed." is "zed" only once the full stop is gone
    t = _CODE_SPLIT.sub(r"\1\2", t)
    words: list[str] = []
    run = False  # the last token is a run of single characters ("k 3 p 7 q z" -> "k3p7qz")
    for w in t.split():
        single = bool(_SINGLE.match(w))
        if single and (run or (words and _CODE.match(words[-1]))):  # "k3p7q z" -> "k3p7qz"
            words[-1] += w
        else:
            words.append(w)
        run = single
    out: list[str] = []
    for w in words:
        if _DIGITS.fullmatch(w):
            spoken = num2words(int(w), lang="ar" if lang == "ar" else "en").replace(",", " ").replace("-", " ")
            spoken = spoken.replace("أ", "ا").replace("إ", "ا").replace("ة", "ه")
            out += re.sub(r"\bو\s+", "و", spoken).split() if lang == "ar" else spoken.replace(" and ", " ").split()
        else:
            out.append(w)
    return out


def _distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_rates(reference: str, hypothesis: str, lang: str) -> tuple[float, float]:
    """(WER, CER) of the hypothesis against the reference, after normalisation."""
    ref, hyp = normalise(reference, lang), normalise(hypothesis, lang)
    if not ref:
        return (0.0, 0.0) if not hyp else (1.0, 1.0)
    rc, hc = "".join(ref), "".join(hyp)
    return _distance(ref, hyp) / len(ref), _distance(list(rc), list(hc)) / max(1, len(rc))
