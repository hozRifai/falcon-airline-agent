"""Parse the Conditions of Carriage PDFs (English and Arabic) into clause rows with shared clause ids.

Source: Emirates' public "Conditions of Carriage for Passengers and Baggage" (published 28 December 2024), used
with credit for demonstration. Both PDFs are generated from Word, so the text layer is intact: article headings
("ARTICLE 10 — REFUNDS" / "المادة10" + "-" + title), clause numbers on their own line or at the start of a
paragraph ("10.1.1 We will refund ..."), and Article 1 as unnumbered definitions in the same order in both
languages. The clause id ("10.2.1") is the key that links the two languages.

One defect in the Arabic text layer, repaired here: ligature glyphs come out with their letters reversed. The
lam-alef ligature "لا" is one glyph in the font; PyMuPDF expands it to two characters and, when it reverses the
right-to-left run into logical order, reverses those two as well ("الرحلات" comes out as "الرحالت", "لا" as "ال";
1,188 occurrences, plus 27 of the four-letter "ريال" ligature in "مونتريال"). The letters of an expanded ligature
are detectable: all but the last have a zero-width bounding box. `pdf_text_ar` reads character geometry and puts
every such run back in order, which needs no dictionary; `repair_arabic_text` then fixes the handful of cases the
geometry misses (heading fonts, where the ligature letters do get a width) and the definite article detached from
its word by a justification gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from falcon.core.arabic import fold_digits, normalize_arabic

NUM_LINE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?$")
NUM_INLINE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,3})\.?\s+(\S.*)$")
ART_EN = re.compile(r"^ARTICLE\s+(\d+)\s*[—–-]\s*(.*)$")
ART_AR = re.compile(r"^المادة\s*(\d{1,2})$")
DASH = re.compile(r"^[—–-]\s*(.*)$")
DEF_EN = re.compile(r"^“([^”]+)”(?:,?\s*(?:and\s+)?“[^”]+”)*\s+(?:means|mean|has|have|are|is)\b")  # a quoted word at a line start is otherwise a continuation
DEF_AR_VERB = re.compile(r"^(تعني|يعني|يُقصد|يقصد|هي|هو)\b")
QUOTE = re.compile(r'["“”]')

# Definitions are linked by position; these anchors re-synchronise the two lists when one language merges or
# skips a term (the Arabic version folds "Itinerary/Receipt" into the previous entry).
ANCHORS_AR_TO_EN = {"اياتا": "IATA", "ايكاو": "ICAO", "حقوق السحب": "SDR", "وارسو": "Warsaw Convention", "مونتريال": "Montreal Convention", "الموقع الشبكي": "Website", "الظروف القاهره": "Force Majeure", "التعرفه": "Tariff", "التذكره": "Ticket"}


@dataclass
class Clause:
    clause_id: str
    article_no: int
    article_title: str
    text: str


AR_LETTER = r"[ء-ي]"


def repair_arabic_text(text: str) -> str:
    """Second pass after `pdf_text_ar`, for what character geometry cannot see.

    - 'ا' + hamza-alef + 'ل' is never valid Arabic: it is the reversed ligature 'لأ' / 'لإ' / 'لآ'; same for
      'اال' = 'الا'. A safety net: on the source PDF the geometric repair leaves none of these.
    - A standalone 'ال' or 'وال' followed by a word is the definite article separated from its word by a
      justification gap that the extractor turned into a space ('بشروط ال نقل' -> 'بشروط النقل'). After the
      geometric repair a real 'لا' (not) never appears as 'ال', so the join is safe. A stray '.' in the gap is the
      sentence's final period displaced by the right-to-left extraction; it is dropped.
    """
    text = re.sub(r"ا([أإآ])ل", r"ال\1", text)
    text = re.sub(r"اال", "الا", text)
    text = re.sub(rf"(?<!{AR_LETTER})(و?ال) +\.? *(?={AR_LETTER})", r"\1", text)
    text = re.sub(rf"\)\(({AR_LETTER}ـ?)\)?", r"(\1)", text)  # list markers come out mirrored: ')(أ' for '(أ)'
    return text


LEAD_PUNCT = re.compile(r"^([.،:؛)]+)(.*)$")


def _trailing_punct(line: str) -> str:
    """Right-to-left extraction emits a line's final punctuation first ('.التعرفة المطبقة'); put it back at the end,
    in reading order ('.)' was ').'). Quotes are left alone: a leading quote is usually the body's opening quote."""
    m = LEAD_PUNCT.match(line)
    if m and re.search(AR_LETTER, m.group(2)):
        return m.group(2).strip() + m.group(1)[::-1]
    return line


def pdf_text(path: Path) -> str:
    import pymupdf

    doc = pymupdf.open(str(path))
    return "\n".join(page.get_text("text") for page in doc)


def _inside(a: dict, b: dict, eps: float = 0.05) -> bool:
    """Character a sits horizontally inside character b's box: both came from one glyph."""
    return bool(a["c"].strip() and b["c"].strip()) and a["bbox"][0] >= b["bbox"][0] - eps and a["bbox"][2] <= b["bbox"][2] + eps


def _reorder_ligatures(chars: list[dict]) -> list[dict]:
    """Letters of an expanded ligature glyph come out in reverse order, each boxed inside the next (the last one
    carries the glyph's width; in body text the others have zero width, in heading fonts a partial one). Real
    neighbouring glyphs never overlap this way, so reversing every such run restores reading order. On the
    source PDF this touches exactly 1,198 lam-alef ligatures and 27 'ريال' ligatures, and nothing else."""
    out, i = [], 0
    while i < len(chars):
        j = i
        while j + 1 < len(chars) and _inside(chars[j], chars[j + 1]):
            j += 1
        out.extend(reversed(chars[i : j + 1]))
        i = j + 1
    return out


AR_LETTER_RE = re.compile(AR_LETTER)
SENTENCE_PUNCT = set(".،؛:")


def _drop_fake_spaces(chars: list[dict]) -> list[dict]:
    """A justification gap inside a word becomes a space in the text layer ("رح لة", "أشك ال"). Geometry tells the two
    apart: a real space leaves 2.4-4 pt between the neighbouring glyphs (at 11 pt), a word broken by a gap leaves
    none (-0.2 to 0.7 pt measured over all 13,000 spaces of the source PDF; heading fonts overlap by about -1 pt and
    are left alone). Reading order is right to left, so the previous glyph is the one on the right."""
    out = []
    for i, c in enumerate(chars):
        if c["c"] == " " and 0 < i < len(chars) - 1:
            prev, nxt = chars[i - 1], chars[i + 1]
            if AR_LETTER_RE.match(prev["c"]) and AR_LETTER_RE.match(nxt["c"]) and -0.5 < prev["bbox"][0] - nxt["bbox"][2] < 1.0:
                continue
        out.append(c)
    return out


def _place_punctuation(chars: list[dict]) -> list[dict]:
    """The extractor drops a sentence's final period somewhere inside the line ("است.عمال") or at its start; its box
    says where it was drawn. A punctuation mark with later glyphs to its right (logically before it, in RTL) is moved
    to just before the first later glyph to its left, or to the end of the line. Digits' dots ("10.2.2") are left alone."""
    out = list(chars)
    i = 0
    while i < len(out):
        c = out[i]
        if c["c"] in SENTENCE_PUNCT:
            prev_c = out[i - 1]["c"] if i > 0 else ""
            next_c = out[i + 1]["c"] if i + 1 < len(out) else ""
            if not (prev_c.isdigit() or next_c.isdigit()):
                later = [j for j in range(i + 1, len(out)) if out[j]["c"].strip()]
                if any(out[j]["bbox"][0] > c["bbox"][2] + 0.5 for j in later):
                    dest = next((j for j in later if out[j]["bbox"][2] <= c["bbox"][0] + 0.5), None)
                    out.pop(i)
                    if dest is None:
                        end = len(out)
                        while end > i and not out[end - 1]["c"].strip():
                            end -= 1
                        out.insert(end, c)
                    else:
                        out.insert(dest - 1, c)
                    continue
        i += 1
    return out


def pdf_text_ar(path: Path) -> str:
    """Arabic text layer with reversed ligatures, broken words and displaced punctuation repaired from character
    geometry (see module docstring). Line and block structure is the same as `pdf_text`."""
    import pymupdf

    doc = pymupdf.open(str(path))
    lines: list[str] = []
    for page in doc:
        for block in page.get_text("rawdict")["blocks"]:
            for line in block.get("lines", []):
                chars = _reorder_ligatures([ch for span in line["spans"] for ch in span["chars"]])
                if line["spans"] and line["spans"][0]["size"] < 13:  # body text; heading glyph boxes overlap and are left alone
                    chars = _place_punctuation(_drop_fake_spaces(chars))
                lines.append("".join(ch["c"] for ch in chars))
            lines.append("")
    return repair_arabic_text("\n".join(lines))


def _key(clause_id: str) -> list[int]:
    return [int(x) for x in clause_id.split(".")]


def _fix_number_typo(num: str, article: int) -> str:
    """The Arabic source numbers the animals section "88.10" (a doubled digit in the document itself). A leading
    component that is the current article's number typed twice is read as that article."""
    first, _, rest = num.partition(".")
    if len(first) == 2 and first == str(article) * 2 and rest:
        return f"{article}.{rest}"
    return num


def _lines(raw: str) -> list[str]:
    out = []
    for l in raw.splitlines():
        l = fold_digits(l).replace(" ", " ").strip()
        if not l or re.fullmatch(r"\d{1,3}", l) or re.fullmatch(r"[.…\s]+", l):
            continue
        out.append(l)
    return out


def _arabic_headings(lines: list[str]) -> dict[int, tuple[int, str, int]]:
    """Body headings: 'المادة4' then '—' then the title (or '- title' on one line). Returns {line index: (n, title, next index)}."""
    heads = {}
    for i, l in enumerate(lines):
        m = ART_AR.match(l)
        if not m:
            continue
        n, title, k = int(m.group(1)), "", i + 1
        if k < len(lines) and DASH.match(lines[k]):
            rest = DASH.match(lines[k]).group(1).strip()
            k += 1
            if rest:
                title = rest
            elif k < len(lines) and not NUM_LINE.match(lines[k]):
                title, k = lines[k], k + 1
        heads[i] = (n, title.strip(" ."), k)
    return heads


def _parse(raw: str, lang: str) -> tuple[list[str], list[Clause]]:
    """Return (article-1 lines, numbered clauses)."""
    lines = _lines(raw)
    heads = _arabic_headings(lines) if lang == "ar" else {}
    clauses: list[Clause] = []
    article, title, cur, in_toc = 0, "", None, True
    art1: list[str] = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if lang == "en":
            m = ART_EN.match(l)
            if m and "...." not in l:
                n = int(m.group(1))
                if not (in_toc and n != 1):
                    in_toc = False
                    if cur:
                        clauses.append(cur)
                        cur = None
                    title = m.group(2).strip(" .")
                    i += 1
                    if i < len(lines) and re.fullmatch(r"[A-Z][A-Z ,&-]+", lines[i]):  # heading wrapped onto a second line
                        title, i = f"{title} {lines[i].strip(' .')}", i + 1
                    article, title = n, re.sub(r"\s+,", ",", title).title()
                    continue
        elif i in heads:
            n, t, k = heads[i]
            if not (in_toc and n != 1):
                in_toc = False
                if cur:
                    clauses.append(cur)
                    cur = None
                article, title = n, t
            i = k
            continue
        if in_toc:
            i += 1
            continue
        if article == 1:
            art1.append(l)
            i += 1
            continue
        nm = NUM_LINE.match(l) or NUM_INLINE.match(l)
        num = _fix_number_typo(nm.group(1), article) if nm else None
        if nm and int(num.split(".")[0]) in (article, article + 1) and (cur is None or _key(num) > _key(cur.clause_id)):
            # numbers only move forward: a lone "7.1" after 7.1.3 is a cross-reference split from its "Article"
            if cur:
                clauses.append(cur)
            article = int(num.split(".")[0])
            cur = Clause(num, article, title, nm.group(2).strip() if nm.re is NUM_INLINE else "")
            i += 1
            continue
        if lang == "ar":
            l = _trailing_punct(l)
        if cur is None:
            cur = Clause(f"{article}.0", article, title, l)
        else:
            cur.text = (cur.text + " " + l).strip()
        i += 1
    if cur:
        clauses.append(cur)
    clauses = [c for c in clauses if c.text.strip()]
    for c in clauses:
        c.text = re.sub(r"\s+", " ", c.text).strip()
    return art1, clauses


def _definitions_en(art1: list[str]) -> list[tuple[str, str]]:
    """[(term, full text)] in document order."""
    defs: list[list[str]] = []
    for l in art1:
        m = DEF_EN.match(l)
        if m:
            defs.append([m.group(1), l])
        elif defs:
            defs[-1][1] += " " + l
    return [(t, re.sub(r"\s+", " ", x)) for t, x in defs]


def _definitions_ar(art1: list[str]) -> list[tuple[str, str]]:
    defs: list[list[str]] = []
    pending: str | None = None
    for raw_line in art1:
        l = raw_line.lstrip(".:،؛ ")  # RTL extraction puts the sentence's final period at the start of the line
        if DEF_AR_VERB.match(l):
            if pending is not None:
                defs.append([pending, pending + " " + l])
                pending = None
            elif defs:
                defs[-1][1] += " " + l
        elif QUOTE.match(raw_line) and len(l) < 90 and "." not in l and not DEF_AR_VERB.search(l):  # term lines START with a quote
            if pending:
                defs.append([pending, pending])
            pending = l.strip('"“” ')
        elif pending is not None:
            pending += " " + l
        elif defs:
            defs[-1][1] += " " + l
    if pending:
        defs.append([pending, pending])
    return [(t, re.sub(r"\s+", " ", x)) for t, x in defs]


def _link_definitions(en_defs, ar_defs) -> list[tuple[str, str | None, str, str | None]]:
    """Walk both lists in order; anchors re-synchronise. Returns [(en_term, ar_term, en_text, ar_text)]."""
    en_index = {t.lower(): i for i, (t, _) in enumerate(en_defs)}
    out = []
    j = 0
    for i, (en_term, en_text) in enumerate(en_defs):
        ar_term = ar_text = None
        if j < len(ar_defs):
            # does the next Arabic term anchor to a later English term? then this English term has no Arabic counterpart
            nxt = normalize_arabic(ar_defs[j][0])
            anchored = next((en for ar_key, en in ANCHORS_AR_TO_EN.items() if nxt.startswith(ar_key)), None)
            if anchored and en_index.get(anchored.lower(), i) > i:
                out.append((en_term, None, en_text, None))
                continue
            ar_term, ar_text = ar_defs[j]
            j += 1
        out.append((en_term, ar_term, en_text, ar_text))
    return out


def parse_both(en_pdf: Path, ar_pdf: Path) -> tuple[list[Clause], list[Clause], list[tuple[str, str]]]:
    """Parse both PDFs. Returns (english clauses, arabic clauses, glossary pairs). Definitions get ids 1.1.n."""
    en_art1, en_clauses = _parse(pdf_text(en_pdf), "en")
    ar_art1, ar_clauses = _parse(pdf_text_ar(ar_pdf), "ar")
    linked = _link_definitions(_definitions_en(en_art1), _definitions_ar(ar_art1))
    en_defs, ar_defs, glossary = [], [], []
    for n, (en_term, ar_term, en_text, ar_text) in enumerate(linked, start=1):
        en_defs.append(Clause(f"1.1.{n}", 1, "Definitions", en_text))
        if ar_text:
            ar_defs.append(Clause(f"1.1.{n}", 1, "التعاريف", ar_text))
            term_ar = re.split(r"\s*[()]?\s*(?:تعني|يعني)\b", ar_term.split('"')[0].split("،")[0])[0]
            term_ar = re.sub(r"[A-Za-z$]+", "", term_ar).strip(" .:()")
            term_ar = re.sub(r"\s+و$", "", term_ar)  # two terms on one line: '"كوبون المسافر" و"إيصال المسافر"'
            glossary.append((en_term, term_ar))
    en_titles = {c.article_no: c.article_title for c in en_clauses}
    for c in ar_clauses:  # English titles are the canonical article names in the index
        c.article_title = en_titles.get(c.article_no, c.article_title)
    return en_defs + en_clauses, ar_defs + ar_clauses, glossary
