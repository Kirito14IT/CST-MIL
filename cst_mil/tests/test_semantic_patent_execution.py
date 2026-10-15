from copy import deepcopy

import pytest

from cst_mil.research_followup.semantic_patent_followup.common import MAIN_TASKS
from cst_mil.research_followup.semantic_patent_followup.execution import choose_candidate


def values(gain_a=0.0, gain_b=0.0):
    baseline = {
        t: {m: 0.8 for m in ("AP", "roc_auc", "macro_f1", "precision", "recall")}
        for t in MAIN_TASKS
    }
    result = {v: deepcopy(baseline) for v in ("full", "context_a", "context_b")}
    result["context_a"]["sccd.attack"]["AP"] += gain_a
    result["context_b"]["sccd.attack"]["AP"] += gain_b
    return result


def test_no_gain_falls_back():
    assert choose_candidate(values())["selected_architecture"] == "full"


def test_exact_boundary_accepted_and_tie_simpler():
    assert choose_candidate(values(0.01, 0.01))["selected_architecture"] == "context_a"


def test_larger_eligible_gain_selected():
    assert choose_candidate(values(0.01, 0.02))["selected_architecture"] == "context_b"


@pytest.mark.parametrize(
    "task,metric",
    [
        ("conda.attack", "AP"),
        ("scamspot.spam_scam", "AP"),
        ("sccd.conversation", "AP"),
        ("sccd.conversation", "roc_auc"),
        ("sccd.attack", "macro_f1"),
    ],
)
def test_any_guardrail_failure_rejects(task, metric):
    data = values(0.03)
    data["context_a"][task][metric] -= 0.011
    assert choose_candidate(data)["selected_architecture"] == "full"


def test_guardrail_exact_minus_point_01_allowed():
    data = values(0.01)
    data["context_a"]["hatebr.attack"]["AP"] -= 0.01
    result = choose_candidate(data)
    assert result["selected_architecture"] == "context_a"
    assert result["TEST_used"] is False
