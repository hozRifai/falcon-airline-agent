import math

from falcon.eval.retrieval import load_gold, score_ranking


def test_perfect_and_empty_rankings():
    rel = {"6.1": 2, "6.2": 1}
    best = score_ranking(["6.1", "6.2", "x"], rel)
    assert best["recall@1"] == 0.5 and best["recall@3"] == 1.0 and best["hit@1"] == 1.0 and best["mrr"] == 1.0
    assert math.isclose(best["ndcg@10"], 1.0)
    none = score_ranking(["a", "b"], rel)
    assert none["recall@10"] == 0 and none["mrr"] == 0 and none["ndcg@10"] == 0 and none["first_primary_rank"] is None


def test_rank_matters():
    rel = {"7.4.5": 2}
    third = score_ranking(["a", "b", "7.4.5"], rel)
    assert third["hit@1"] == 0 and third["hit@3"] == 1 and math.isclose(third["mrr"], 1 / 3)
    assert math.isclose(third["ndcg@10"], 1 / math.log2(4))


def test_gold_set_shape():
    items = load_gold()
    answerable = [i for i in items if i["answerable"]]
    assert len(answerable) == 40 and {i["lang"] for i in answerable} == {"en", "ar"}
    assert all(any(g == 2 for g in i["relevant"].values()) for i in answerable)


def test_aliases_append_the_contract_words_only_when_they_match():
    from falcon.policy.aliases import expand
    q, added = expand("I missed my outbound flight, is my return still valid?")
    assert added and "onward reservations" in q and q.startswith("I missed my outbound flight")
    q2, added2 = expand("ما هي السياسة عندما يكون إلغاء الحجز بسبب حالة طبية طارئة؟")
    assert "Force Majeure" in q2 and len(added2) == 1
    assert expand("Can I bring my cat?") == ("Can I bring my cat?", [])
