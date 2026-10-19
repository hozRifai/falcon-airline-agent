"""The Conditions of Carriage parser: both PDFs, linked by clause id, Arabic ligatures repaired."""

import collections
import re
from pathlib import Path

import pytest

from falcon.core.config import REPO_ROOT
from falcon.policy import parse_coc

DATA = REPO_ROOT / "data" / "policy"


@pytest.fixture(scope="module")
def parsed():
    if not (DATA / "coc_en.pdf").exists():
        pytest.skip("policy PDFs not present")
    return parse_coc.parse_both(DATA / "coc_en.pdf", DATA / "coc_ar.pdf")


def test_clauses_link_across_languages(parsed):
    en, ar, _ = parsed
    en_ids = {c.clause_id for c in en}
    ar_ids = {c.clause_id for c in ar}
    assert len(en) >= 340 and len(ar) >= 340
    assert len(en_ids & ar_ids) >= 350
    assert en_ids == ar_ids, (sorted(en_ids ^ ar_ids))  # 353 of 353 at the time of writing; "88.10" in the source is read as 8.10
    for clauses in (en, ar):
        dup = [k for k, v in collections.Counter(c.clause_id for c in clauses).items() if v > 1]
        assert not dup, dup
    assert len({c.article_no for c in en}) == 18


def test_definitions_aligned(parsed):
    en, ar, glossary = parsed
    assert sum(c.article_no == 1 for c in en) == 37
    assert sum(c.article_no == 1 for c in ar) == 37
    terms = dict(glossary)
    assert terms["Ticket"] == "التذكرة"
    assert terms["Montreal Convention"] == "اتفاقية مونتريال"  # a four-letter ligature, reversed in the raw text layer
    assert terms["Passenger Coupon"] == "كوبون المسافر"


def test_arabic_text_has_no_reversed_ligatures(parsed):
    _, ar, _ = parsed
    text = " ".join(c.text for c in ar)
    tokens = collections.Counter(re.findall(r"[؀-ۿ]+", text))
    assert tokens["ال"] == 0 and tokens["وال"] == 0 and tokens["إال"] == 0  # 'لا', 'ولا', 'إلا' reversed
    assert not re.search(r"ا[أإآ]ل|اال", text)
    assert "الرحالت" not in text and "مونتلاير" not in text
    assert "الرحلات" in text and "لا يجوز" in text
    assert not re.search(r"\)\([ء-ي]", text), "list markers must read (أ), not )(أ"


def ar_text_of(clauses, clause_id):
    return next(c.text for c in clauses if c.clause_id == clause_id)


def test_article_titles_are_canonical(parsed):
    en, ar, _ = parsed
    titles = {c.article_no: c.article_title for c in en}
    assert titles[9] == "Schedules, Delays, Cancellation Of Flights And Denied Boarding"
    assert titles[10] == "Refunds"
    assert all(c.article_title == titles[c.article_no] for c in ar if c.article_no != 1)


def test_trailing_punctuation_rule():
    assert parse_coc._trailing_punct(".التعرفة المطبقة") == "التعرفة المطبقة."
    assert parse_coc._trailing_punct(".)وإلا ستنطبق") == "وإلا ستنطبق)."
    assert parse_coc._trailing_punct('"الطاقم') == '"الطاقم'
    assert parse_coc.repair_arabic_text("بشروط ال نقل") == "بشروط النقل"
    assert parse_coc.repair_arabic_text("لا يجوز") == "لا يجوز"
    assert parse_coc.repair_arabic_text("يلي: )(أ ألغينا") == "يلي: (أ) ألغينا"


def test_words_and_periods_are_whole(parsed):
    _, ar, _ = parsed
    text = " ".join(c.text for c in ar)
    assert not re.search(r"[ء-ي]\.[ء-ي]", text), "a displaced period inside a word (است.عمال)"
    for broken, whole in (("رح لة", "رحلة"), ("أشك ال", "أشكال"), ("والالت زام", "والالتزام"), ("الأورا ق", "الأوراق")):
        assert broken not in text and whole in text, broken  # justification gaps inside words, repaired from glyph geometry
    assert ar_text_of(ar, "8.10").startswith("الحيوانات")  # numbered "88.10" in the source document
