"""The BERTopic exception permits numeric drift, never changed decisions."""

from copy import deepcopy
from importlib import import_module

import pytest


def check(predictions, reference):
    module = import_module("cst_mil.research_followup.round2.timing_revision")
    return module.validate_parity(predictions, reference)


def rows():
    return [
        dict(
            post_id="a",
            risk_probability=0.7,
            prediction=1,
            evidence=[{"comment_id": "a1"}, {"comment_id": "a2"}],
            topics=[{"topic_id": 3}, {"topic_id": 1}],
        ),
        dict(
            post_id="b",
            risk_probability=0.2,
            prediction=0,
            evidence=[{"comment_id": "b1"}],
            topics=[{"topic_id": 2}],
        ),
    ]


def test_small_drift_passes_when_all_decisions_are_unchanged():
    reference = rows()
    predictions = deepcopy(reference)
    predictions[0]["risk_probability"] += 3e-6
    result = check(predictions, reference)
    assert result["accepted"] is True
    assert result["max_saved_prediction_difference"] == pytest.approx(3e-6)


def test_probability_drift_above_authorized_limit_is_rejected():
    reference = rows()
    predictions = deepcopy(reference)
    predictions[0]["risk_probability"] += 1.01e-5
    with pytest.raises(ValueError, match="probability"):
        check(predictions, reference)


def test_ranking_change_is_rejected_even_within_numeric_tolerance():
    reference = rows()
    reference[1]["risk_probability"] = 0.700001
    predictions = deepcopy(reference)
    predictions[0]["risk_probability"] = 0.700002
    with pytest.raises(ValueError, match="ranking"):
        check(predictions, reference)


def test_breaking_a_tie_is_rejected_even_if_sorted_ids_do_not_change():
    reference = rows()
    reference[1]["risk_probability"] = 0.7
    predictions = deepcopy(reference)
    predictions[0]["risk_probability"] += 1e-6
    with pytest.raises(ValueError, match="ranking"):
        check(predictions, reference)


def test_classification_change_is_rejected_within_numeric_tolerance():
    reference = rows()
    predictions = deepcopy(reference)
    predictions[0]["risk_probability"] += 1e-6
    predictions[0]["prediction"] = 0
    with pytest.raises(ValueError, match="classification"):
        check(predictions, reference)


def test_reordering_top_three_is_rejected_with_unchanged_probabilities():
    reference = rows()
    predictions = deepcopy(reference)
    predictions[0]["evidence"].reverse()
    with pytest.raises(ValueError, match="Top-3"):
        check(predictions, reference)


def test_topic_id_change_is_rejected_with_unchanged_probabilities():
    reference = rows()
    predictions = deepcopy(reference)
    predictions[0]["topics"][0]["topic_id"] = 4
    with pytest.raises(ValueError, match="topic"):
        check(predictions, reference)


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_probability_never_passes_parity(probability):
    reference = rows()
    predictions = deepcopy(reference)
    predictions[0]["risk_probability"] = probability
    with pytest.raises(ValueError, match="probability"):
        check(predictions, reference)


def test_missing_session_cannot_be_accepted_as_a_partial_timing_pass():
    reference = rows()
    with pytest.raises(ValueError, match="session"):
        check(reference[:1], reference)
