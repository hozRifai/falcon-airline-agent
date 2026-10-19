"""The generation evaluation's pure parts: the question set, the deterministic checks, the judge's JSON parsing, the agreement maths.
No model is called here; the judge itself is measured by `python -m falcon.eval.generation --validate` against the hand labels."""

import json
import math

import pytest

from falcon.eval import generation as g

SEARCH = {"query": "newborn", "available": True, "confidence": 0.71, "rewritten": False,
          "results": [{"clause_id": "7.4.5", "article": "7 – Refusal and limitation of carriage", "score": 0.03, "similarity": 0.71,
                       "text_en": "Carriage of newly born infants will generally not be permitted until 7 Days after delivery.", "text_ar": "لا يُسمح عادةً بنقل الأطفال حديثي الولادة قبل مرور 7 أيام على الولادة."},
                      {"clause_id": "7.4.1", "article": "7 – Refusal and limitation of carriage", "score": 0.02, "similarity": 0.6, "text_en": "Pregnant passengers ...", "text_ar": None}]}
TURN = [{"tool": "list_my_bookings", "args": {}, "ok": True, "summary": "...", "result_json": {"bookings": []}},
        {"tool": "search_policy", "args": {"query": "newborn"}, "ok": True, "summary": "2 clauses", "result_json": SEARCH}]
ITEM = {"id": "newborn_en", "topic": "newborn", "lang": "en", "question": "How soon after giving birth can my baby fly?", "relevant": {"7.4.5": 2}, "answerable": True, "source": "gold"}


def test_question_set_covers_gold_and_extra_questions_in_both_languages():
    items = g.load_questions()
    ids = {i["id"] for i in items}
    assert len(items) == 48 and sum(i["lang"] == "en" for i in items) == 24 and sum(i["lang"] == "ar" for i in items) == 24
    assert sum(not i["answerable"] for i in items) == 4
    assert {"medical_emergency_cancellation_en", "medical_emergency_cancellation_ar", "infant_child_baggage_en", "infant_child_baggage_ar"} <= ids
    assert g.load_questions("medical") and all("medical" in i["id"] for i in g.load_questions("medical"))


def test_policy_trace_collects_clause_texts_and_calls():
    tools, clauses, calls = g.policy_trace(TURN)
    assert [n for n, _ in tools] == ["list_my_bookings", "search_policy"]
    assert [c["clause_id"] for c in clauses] == ["7.4.5", "7.4.1"] and clauses[0]["text_ar"].startswith("لا")
    assert calls == [{"tool": "search_policy", "args": {"query": "newborn"}, "ok": True, "latency_ms": None, "confidence": 0.71, "rewritten": False, "available": True}]


def test_deterministic_checks_mirror_the_reviewer():
    tools, clauses, _ = g.policy_trace(TURN)
    good = g.deterministic("Under clause 7.4.5, newborns are generally not carried until 7 days after delivery.", tools, clauses, ITEM)
    assert good["cites_clause"] and good["cited"] == ["7.4.5"] and good["answering_retrieved"] and good["answering_cited"]
    assert not good["unretrieved_clause"] and not good["unsupported_quantities"] and not good["unsupported_regulation"] and not good["uncited_policy_answer"]
    bad = g.deterministic("Clause 8.1.2 allows 10 kg for infants, as EU Regulation 261/2004 requires.", tools, clauses, ITEM)
    assert bad["unretrieved_clause"] and bad["unsupported_quantities"] and bad["unsupported_regulation"] and not bad["answering_cited"]
    uncited = g.deterministic("The Conditions of Carriage say newborns wait a week.", tools, clauses, ITEM)
    assert uncited["uncited_policy_answer"] and not uncited["cites_clause"]
    honest = g.deterministic("The Conditions of Carriage do not cover student discounts; a human agent can help.", tools, clauses, {**ITEM, "relevant": {}, "answerable": False})
    assert honest["not_covered_pattern"] and honest["answering_retrieved"] is None and not honest["uncited_policy_answer"]
    nothing = g.deterministic("Hello!", [], [], ITEM)
    assert not nothing["policy_tool_called"] and nothing["answering_retrieved"] is False


def test_judge_json_is_parsed_tolerantly_and_scores_clamped():
    raw = '```json\n{"faithfulness": 5, "relevance": "4", "completeness": 7, "said_not_covered": "false", "reason": "ok"}\n```'
    out = g.normalise_scores(g.parse_judge_json(raw))
    assert out == {"faithfulness": 5, "relevance": 4, "completeness": 5, "said_not_covered": False, "reason": "ok"}
    prose = 'Here is my grading: {"faithfulness": 2, "relevance": 3, "completeness": 1, "said_not_covered": true, "reason": "invented 8.1.2"} done.'
    assert g.parse_judge_json(prose)["said_not_covered"] is True
    assert g.parse_judge_json("not json at all") == {}
    missing = g.normalise_scores({"faithfulness": 3})
    assert missing["relevance"] is None and missing["said_not_covered"] is None
    glitch = g.normalise_scores({"faith游戏副本ness": 1, "relevance": 5, "completeness": 1, "said_not_covered": False, "reason": "r"})  # Qwen, 18 Oct, 1 of 48
    assert glitch["faithfulness"] == 1 and glitch["relevance"] == 5
    assert g.normalise_scores({"score": 4})["faithfulness"] is None  # no near miss, no guess


def test_agreement_exact_within_one_and_bias():
    a = g.agreement([(5, 5), (4, 5), (2, 4), (3, 3)])
    assert a["n"] == 4 and a["exact"] == 0.5 and a["within_1"] == 0.75 and math.isclose(a["mean_abs"], 0.75) and math.isclose(a["bias"], -0.75)
    assert g.agreement([])["n"] == 0 and g.agreement([])["exact"] is None


def test_honest_needs_the_not_covered_flag_and_nothing_invented():
    assert g.honest({"judge": {"said_not_covered": True, "faithfulness": 5}}) is True
    assert g.honest({"judge": {"said_not_covered": True, "faithfulness": 2}}) is False
    assert g.honest({"judge": {"said_not_covered": False, "faithfulness": 5}}) is False
    assert g.honest({"judge": {"error": "x"}}) is None


def test_brief_uses_the_question_language_and_falls_back_to_english():
    clauses = [{"clause_id": "7.4.5", "article": "7", "text_en": "English text", "text_ar": "نص عربي"}, {"clause_id": "7.4.1", "article": "7", "text_en": "Only English", "text_ar": None}]
    brief = g.render_brief("س؟", "ar", clauses, "رد")
    assert "[7.4.5] 7: نص عربي" in brief and "[7.4.1] 7: Only English" in brief and brief.endswith("REPLY:\nرد")
    assert "(none: the assistant retrieved no clause" in g.render_brief("q", "en", [], "")


def test_labels_file_is_ready_for_hand_labelling():
    if not g.LABELS.exists():
        pytest.skip("eval/generation/labels.yaml is written after the first run")
    entries = g.load_labels()
    assert len(entries) == 10 and {e["lang"] for e in entries} == {"en", "ar"}
    ids = {e["id"] for e in entries}
    assert any(i.startswith("medical_emergency_cancellation") for i in ids) and any(i.startswith("infant_child_baggage") for i in ids)
    assert sum(not e["answerable"] for e in entries) >= 2
    for e in entries:
        assert e["reply"].strip() and isinstance(e["retrieved"], list) and e["question"]
        assert all(e["hozayfa"][d] is None for d in g.DIMENSIONS)                       # Hozayfa's labels are still to be filled
        assert all(isinstance(e["claude"][d], int) and 1 <= e["claude"][d] <= 5 for d in g.DIMENSIONS)  # the builder's labels are in
        assert isinstance(e["claude"]["said_not_covered"], bool)
    assert len(g.labelled(entries, "claude")) == 10 and g.labelled(entries, "hozayfa") == []
    json.dumps(entries, ensure_ascii=False)


def test_rows_are_appended_as_they_land_and_graded_rows_are_kept_on_resume(tmp_path):
    import asyncio

    g.append_row(tmp_path, {"id": "a_en", "reply": "x"})
    g.append_row(tmp_path, {"id": "b_ar", "reply": "ر", "judge": {"faithfulness": 5, "relevance": 5, "completeness": 5, "said_not_covered": False, "reason": "ok"}})
    g.append_row(None, {"id": "ignored"})  # no folder: nothing written
    rows = g.read_run(tmp_path)
    assert [r["id"] for r in rows] == ["a_en", "b_ar"] and rows[1]["reply"] == "ر"

    class StubJudge:
        calls = 0

        async def grade(self, question, lang, clauses, reply):
            self.calls += 1
            return {"faithfulness": 3, "relevance": 3, "completeness": 3, "said_not_covered": False, "reason": "stub"}

    rows = [{"id": "a_en", "question": "q", "lang": "en", "clauses": [], "reply": "x"},
            {"id": "b_ar", "question": "q", "lang": "ar", "clauses": [], "reply": "ر", "judge": {"faithfulness": 5, "relevance": 5, "completeness": 5, "said_not_covered": False, "reason": "ok"}},
            {"id": "c_en", "question": "q", "lang": "en", "clauses": [], "reply": "", "error": "stack down"},
            {"id": "d_en", "question": "q", "lang": "en", "clauses": [], "reply": "y", "judge": {"error": "timeout"}}]
    stub = StubJudge()
    asyncio.run(g.judge_rows(rows, stub))
    assert stub.calls == 2 and rows[0]["judge"]["faithfulness"] == 3 and rows[1]["judge"]["faithfulness"] == 5 and "judge" not in rows[2] and rows[3]["judge"]["faithfulness"] == 3
    asyncio.run(g.judge_rows(rows, stub, force=True))
    assert stub.calls == 5 and rows[1]["judge"]["faithfulness"] == 3
