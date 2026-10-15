import pytest

from cst_mil.research_followup.round2.execution import evaluate, fitting_data


def test_fitting_data_excludes_test_and_rejects_overlap():
    sessions = [
        dict(post_id="a", split="train", label=0),
        dict(post_id="b", split="dev", label=1),
        dict(post_id="c", split="test2023", label=1),
    ]
    messages = [dict(post_id=x["post_id"]) for x in sessions]
    fit, local = fitting_data(sessions, messages)
    assert {s["post_id"] for s in fit} == {"a", "b"}
    assert {m["post_id"] for m in local} == {"a", "b"}
    with pytest.raises(ValueError, match="Duplicate"):
        fitting_data(sessions + [dict(post_id="a", split="test", label=0)], messages)


def test_evaluation_distinguishes_hit_from_full_message_recall():
    rows = [
        dict(post_id="a", label=1, risk_probability=0.9, evidence=[{"comment_id": "1"}]),
        dict(post_id="b", label=0, risk_probability=0.1, evidence=[]),
    ]
    grouped = {
        "a": [dict(comment_id="1", comment_label=1), dict(comment_id="2", comment_label=1)],
        "b": [dict(comment_id="3", comment_label=0)],
    }
    metrics = evaluate(rows, 0.5, grouped)
    assert metrics["top3_hit_rate"] == 1
    assert metrics["dangerous_comment_recall"] == 0.5
    assert metrics["ap"] == metrics["auc"] == 1
