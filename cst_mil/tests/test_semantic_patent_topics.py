import json

import numpy as np
import pytest

from cst_mil.research_followup.semantic_patent_followup.review import (
    export_review_packets,
    machine_quality_audit,
    merge_assisted_reviews,
)
from cst_mil.research_followup.semantic_patent_followup.topics import (
    VARIANTS,
    interpret_topics,
    phrase_candidates,
)


def rows(texts):
    return [
        {"post_id": "p", "comment_id": f"m{i}", "raw_text": text, "order": i, "parent_id": None}
        for i, text in enumerate(texts)
    ]


def interpret(messages, *, vectors=None, scores=None, variant="full"):
    n = len(messages)
    return interpret_topics(
        messages,
        np.tile([1.0, 0.0], (n, 1)) if vectors is None else vectors,
        np.full((n, 3), 0.1) if scores is None else scores,
        category_thresholds={"attack": 0.5, "hate": 0.5, "spam_scam": 0.5},
        variant=variant,
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("不能让外国专家到吗？", "不能让外国专家到吗？"),
        ("这样的事会不会发生到学校？", "这样的事会不会发生到学校？"),
        ("没有任何一种逃避能得到赞赏", "没有任何一种逃避能得到赞赏"),
        ("Do not bully classmates?", "do not bully classmates?"),
        ("não ofenda colegas!", "não ofenda colegas!"),
        ("na\u0303o ofenda colegas?", "não ofenda colegas?"),
    ],
)
def test_protected_phrase_is_complete_original_clause(raw, expected):
    candidates = phrase_candidates(raw)
    assert set(candidates) == {expected}
    start, end = candidates[expected]
    assert raw[start:end] == raw


def test_polarity_ablation_really_removes_scope_protection():
    assert "外国专家" not in phrase_candidates("不能让外国专家到吗？")
    assert "外国专家" in phrase_candidates("不能让外国专家到吗？", protect_polarity=False)
    assert "bully classmates" in phrase_candidates(
        "Do not bully classmates?", protect_polarity=False
    )


def test_long_protected_clause_abstains_without_partial_assertion():
    assert not phrase_candidates("不要" + "围攻同学" * 30 + "？")
    assert "认真训练" in phrase_candidates("不要辱骂队员，认真训练")


def test_words_complete_no_generic_titles_and_supported_unicode():
    candidates = phrase_candidates("下次比赛还去看吗")
    assert "比" not in candidates and "还" not in candidates and "看" not in candidates
    # A question remains a question even without an explicit question mark.
    result = interpret(rows(["não ofenda colegas", "não ofenda colegas"]))
    assert machine_quality_audit(rows(["não ofenda colegas", "não ofenda colegas"]), result)[
        "passed"
    ]


@pytest.mark.parametrize("text", [" 没错。 ", "yes!", "sim", "（同意）"])
def test_whole_response_stoplist_only(text):
    assert phrase_candidates(text) == {}
    assert phrase_candidates("同意这个训练方案")
    assert phrase_candidates("支持同学不要骂人")
    assert phrase_candidates("yes improved training")


def test_short_reply_links_only_direct_semantic_parent_without_score_copy():
    messages = rows(["校园比赛需要认真训练", "比赛表现应该通过训练改善", "没错", "同意"])
    messages[2]["parent_id"] = "m0"
    messages[3]["parent_id"] = "m2"
    scores = np.array([[0.9, 0.1, 0.2], [0.8, 0.1, 0.2], [0.03, 0.04, 0.01], [0.05, 0.02, 0.01]])
    result = interpret(messages, scores=scores)
    topic = result["topics"][0]
    assert topic["context_linked_message_ids"] == ["m2"]
    assert topic["name_statistics_message_ids"] == ["m0", "m1"]
    assert result["message_scores"][2]["risk_score"] == 0.04
    assert result["message_scores"][2]["category_scores"] == scores[2].tolist()
    assert result["message_scores"][2]["topic_ids"] == [0]
    assert result["message_scores"][2]["topic_strengths"] == [0.0]
    assert not result["message_scores"][3]["assigned"]  # no cascading
    assert all(
        support["message_id"] != "m2"
        for supports in topic["keyword_supports"].values()
        for support in supports
    )
    assert machine_quality_audit(messages, result)["passed"]
    ablated = interpret(messages, scores=scores, variant="no_reply_link")
    assert ablated["coverage"] == 0.5 and result["coverage"] == 0.75
    assert ablated["topics"][0]["keywords"] == topic["keywords"]


@pytest.mark.parametrize("kind", ["missing", "self", "different_post", "cycle"])
def test_invalid_parent_never_creates_context_membership(kind):
    messages = rows(["校园比赛认真训练", "校园比赛专注训练", "没错", "同意"])
    if kind == "missing":
        messages[2]["parent_id"] = "absent"
    elif kind == "self":
        messages[2]["parent_id"] = "m2"
    elif kind == "different_post":
        messages[2]["parent_id"] = "m0"
        messages[2]["post_id"] = "other"
    else:
        messages[2]["parent_id"] = "m0"
        messages[0]["parent_id"] = "m2"
    result = interpret(messages)
    assert not result["message_scores"][2]["assigned"]
    assert result["message_scores"][2]["rejected_reply_reason"]


def test_no_parent_no_topic_and_long_unnameable_message_not_inherited():
    messages = rows(["校园比赛认真训练", "校园比赛专注训练", "没错", "啊 " * 15])
    messages[3]["parent_id"] = "m0"
    result = interpret(messages)
    assert not result["message_scores"][2]["assigned"]
    assert not result["message_scores"][3]["assigned"]


def test_counter_distance_ablation_changes_grouping_without_risk_override():
    messages = rows(
        [
            "你这个废物赶紧滚出队伍",
            "这种废物别参加比赛",
            "不要骂队员，停止人身攻击",
            "不要围攻同学，停止辱骂",
        ]
    )
    scores = np.array([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.1, 0.1]])
    full = interpret(messages, scores=scores)
    ablated = interpret(messages, scores=scores, variant="no_counter_distance")
    assert len(full["topics"]) == 2 and len(ablated["topics"]) == 1
    assert full["interpretation_provenance"]["counter_pairs_constrained"] == 4
    for a, b in zip(full["message_scores"], ablated["message_scores"], strict=True):
        assert a["category_scores"] == b["category_scores"]


def test_each_variant_preserves_risk_and_original_support():
    messages = rows(["Should we stop bullying?", "We should stop bullying?", "yes"])
    messages[2]["parent_id"] = "m0"
    scores = np.array([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    for variant in VARIANTS:
        result = interpret(messages, variant=variant, scores=scores)
        assert [item["category_scores"] for item in result["message_scores"]] == scores.tolist()
        assert machine_quality_audit(messages, result)["passed"]
        assert all(len(item["topic_ids"]) <= 1 for item in result["message_scores"])


def test_empty_singleton_zero_vectors_and_bad_inputs():
    assert interpret([])["coverage"] == 0
    assert interpret(rows(["你好同学"]))["topics"] == []
    assert interpret(rows(["训练比赛", "训练比赛"]), vectors=np.zeros((2, 2)))["topics"] == []
    with pytest.raises(ValueError):
        interpret(rows(["训练比赛"]), scores=np.array([[2.0, 0.1, 0.1]]))
    with pytest.raises(ValueError):
        interpret(rows(["训练比赛"]), variant="made_up")
    duplicated = rows(["训练比赛", "训练比赛"])
    duplicated[1]["comment_id"] = "m0"
    with pytest.raises(ValueError):
        interpret(duplicated)


def test_machine_audit_catches_tampered_span_and_missing_member():
    messages = rows(["训练比赛", "训练比赛"])
    result = interpret(messages)
    assert machine_quality_audit(messages, result)["passed"]
    first = next(iter(result["topics"][0]["keyword_supports"].values()))[0]
    first["start"] += 1
    result["topics"][0]["message_ids"].append("absent")
    audited = machine_quality_audit(messages, result)
    assert not audited["passed"]
    assert {item["kind"] for item in audited["violations"]} >= {
        "invalid_original_span",
        "invalid_topic_ids",
    }


def test_two_review_packets_hide_variants_and_labels_and_preserve_differences(tmp_path):
    messages = rows(["校园比赛认真训练", "校园比赛专注训练", "没错"])
    messages[2]["parent_id"] = "m0"
    cases = [
        {
            "case_id": "c1",
            "messages": messages,
            "labels": [1, 0, 1],
            "variants": {variant: interpret(messages, variant=variant) for variant in VARIANTS},
        }
    ]
    manifest = export_review_packets(cases, tmp_path)
    assert manifest["case_count"] == 1
    mapping = json.loads(
        (tmp_path / "audit_only_identity_mapping.json").read_text(encoding="utf-8")
    )
    reviewers = []
    for reviewer in (1, 2):
        payload = json.loads(
            (tmp_path / f"reviewer_{reviewer}" / "case_001.json").read_text(encoding="utf-8")
        )
        encoded = json.dumps(payload)
        assert "labels" not in encoded and "interpretation_provenance" not in encoded
        assert all(variant not in encoded for variant in VARIANTS)
        reviewers.append(
            [
                {
                    "case_id": "c1",
                    "anonymous_configuration": alias,
                    "name_faithfulness": reviewer - 1,
                    "group_consistency": 2,
                    "coverage_appropriateness": 1,
                    "stance_separation": None,
                    "message_ids": ["m0"],
                    "reason": "有原文依据",
                }
                for alias in mapping[f"reviewer_{reviewer}"].values()
            ]
        )
    merged = merge_assisted_reviews(*reviewers, mapping)
    assert len(merged["disagreements"]) == len(VARIANTS)
    assert merged["semantic_accuracy"] == "not_estimated_without_human_gold"
    with pytest.raises(ValueError):
        merge_assisted_reviews(reviewers[0][:-1], reviewers[1], mapping)
