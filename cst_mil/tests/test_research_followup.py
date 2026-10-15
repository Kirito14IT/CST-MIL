import numpy as np
import pytest

from cst_mil.research_followup.ablation import crossfit, make_features
from cst_mil.research_followup.audit import evidence_details, paired_bootstrap, ranking_at_k


def test_ranking_ties_are_stable_and_positive_denominator_is_explicit():
    rows = [
        dict(post_id="b", label=1, risk_probability=0.5),
        dict(post_id="a", label=0, risk_probability=0.5),
        dict(post_id="c", label=1, risk_probability=0.1),
    ]
    out = ranking_at_k(rows, [1, 2, 30])
    assert out[0]["precision"] == 0
    assert out[1]["recall"] == 0.5
    assert out[2]["reviewed"] == 3


def test_evidence_uses_rank_before_label_filter_and_short_sessions():
    messages = [
        dict(post_id="a", comment_id="x", comment_label=None),
        dict(post_id="a", comment_id="y", comment_label=1),
        dict(post_id="a", comment_id="z", comment_label=0),
    ]
    predictions = [
        dict(
            post_id="a",
            message_scores=[
                dict(comment_id="y", rank=2, risk_probability=0.2),
                dict(comment_id="x", rank=1, risk_probability=0.8),
                dict(comment_id="z", rank=3, risk_probability=0.1),
            ],
        )
    ]
    out = evidence_details(predictions, messages, k=1)
    assert out["hit_count"] == 0
    assert out["n_sessions_with_positive_comments"] == 1
    assert evidence_details(predictions, messages, k=3)["hit_count"] == 1


def test_paired_bootstrap_identical_scores_zero_and_deterministic():
    y = np.array([0, 0, 1, 1])
    scores = {"a": np.array([0.1, 0.3, 0.4, 0.9]), "cst_mil_r0": np.array([0.1, 0.3, 0.4, 0.9])}
    out = paired_bootstrap(y, scores, repeats=20)
    for row in out:
        assert row["delta"] == row["ci_low"] == row["ci_high"] == 0
    assert out == paired_bootstrap(y, scores, repeats=20)


def test_no_topics_removes_all_topic_derived_structure():
    msgs = [dict(comment_id="a", parent_id=None), dict(comment_id="b", parent_id="a")]
    risks = np.array([0.2, 0.8])
    a = make_features(msgs, np.eye(2), risks, structure=True, use_topics=False)
    b = make_features(
        msgs, np.array([[0.5, 0.5], [0.5, 0.5]]), risks, structure=True, use_topics=False
    )
    np.testing.assert_array_equal(a, b)
    assert len(a) == 12


def test_crossfit_keeps_complete_bags_disjoint():
    bags = np.repeat(np.arange(6), 2)
    x = np.arange(24).reshape(12, 2) / 24
    y = np.array([0, 1, 0, 1, 0, 1])
    scores, classifier, history = crossfit(x, bags, y, head="lr", config={"seed": 42})
    assert np.isfinite(scores).all()
    assert classifier is not None
    held = []
    for fold in history:
        assert not set(fold["train_bags"]) & set(fold["held_bags"])
        held.extend(fold["held_bags"])
    assert sorted(held) == list(range(6))


def test_crossfit_rejects_unrepresented_bags():
    with pytest.raises(ValueError):
        crossfit(np.eye(3), np.array([0, 0, 1]), np.array([0, 1, 0]), head="lr", config={})
