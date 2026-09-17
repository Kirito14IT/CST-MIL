from __future__ import annotations

import numpy as np
import pytest

from cst_mil.bench import dataset
from cst_mil.bench.common import build_windows, normalise_rows, read_json, write_json
from cst_mil.bench.dataset import load_suite
from cst_mil.bench.evaluation import (
    evidence_metrics,
    replacement_gate,
    stopping_gate,
    topic_quality,
)
from cst_mil.bench.runner import run_lock
from cst_mil.bench.verification import _equal


def test_nested_cohorts_and_smoke_boundary():
    _, pilot, pm = load_suite("pilot300")
    _, full, fm = load_suite("full677")
    _, smoke, sm = load_suite("smoke5")
    full_ids = {s["post_id"]: s for s in full}
    assert (len(pilot), len(pm), len(full), len(fm)) == (300, 17312, 677, 38872)
    assert all(full_ids[s["post_id"]]["split"] == s["split"] for s in pilot)
    assert sum(s["fresh_test"] for s in full) == 76
    assert len(smoke) == 5 and len(sm) > 5
    assert all(full_ids[s["post_id"]]["split"] == "train" for s in smoke)
    assert not any(m.get("comment_label") is not None for m in fm if m["split"] != "test")


def test_windows_cover_tail_and_keep_duplicate_evidence():
    s = {"post_id": "x"}
    m = [
        {"post_id": "x", "comment_id": str(i), "order": i, "model_text": "repeat"}
        for i in range(12)
    ]
    windows = build_windows(s, m)
    assert set(v for w in windows for v in w["message_ids"]) == set(map(str, range(12)))
    assert windows[-1]["end"] == 11
    assert windows[0]["text"] == "repeat" and len(windows[0]["message_ids"]) == 5


def test_empty_and_invalid_topics():
    np.testing.assert_array_equal(normalise_rows([[0, 0], [1, 1]]), [[0, 0], [0.5, 0.5]])
    with pytest.raises(ValueError):
        normalise_rows([[float("nan")]])
    quality = topic_quality([], ["文本"], [{"coverage": 0}])
    assert quality["npmi"] == -1 and quality["topic_diversity"] == 0


def test_evidence_respects_returned_tie_ranking():
    rows = [{"comment_id": str(i), "risk_probability": 0.5, "rank": 5 - i} for i in range(5)]
    result = evidence_metrics(
        [{"post_id": "p", "message_scores": rows}],
        [{"post_id": "p", "comment_id": str(i), "comment_label": int(i == 4)} for i in range(5)],
    )
    assert result["hit_rate_at_3"] == 1


def test_predeclared_gates():
    base = {
        "risk": {"macro_f1": 0.8},
        "topics": {"npmi": 0.1, "topic_diversity": 0.7, "coverage": 0.8},
        "functional_checks_passed": True,
    }
    assert stopping_gate(base, base)
    assert not replacement_gate(base, base)
    newer = {**base, "risk": {"macro_f1": 0.82}}
    assert replacement_gate(newer, base)
    bad = {**newer, "topics": {**base["topics"], "npmi": -0.1}}
    assert not replacement_gate(bad, base)
    assert not replacement_gate({**newer, "functional_checks_passed": False}, base)


def test_coverage_counts_messages_not_sessions():
    result = topic_quality(
        [],
        [],
        [
            {"coverage": 1, "coverage_details": {"assigned_messages": 1, "total_messages": 1}},
            {"coverage": 0, "coverage_details": {"assigned_messages": 0, "total_messages": 100}},
        ],
    )
    assert result["coverage"] == pytest.approx(1 / 101)
    assert result["coverage_macro_session"] == 0.5


def test_manifest_metadata_tamper_is_rejected(monkeypatch):
    original = dataset.read_json

    def changed(path):
        data = original(path)
        if path.name == "manifest.json":
            data["order"] = list(reversed(data["order"]))
        return data

    monkeypatch.setattr(dataset, "read_json", changed)
    with pytest.raises(ValueError, match="manifest/content fingerprint"):
        load_suite("smoke5")


def test_lock_release_on_error_and_atomic_json(tmp_path):
    with pytest.raises(RuntimeError):
        with run_lock(tmp_path / "run"):
            assert (tmp_path / "run/.lock").exists()
            raise RuntimeError("controlled interruption")
    assert not (tmp_path / "run/.lock").exists()
    with run_lock(tmp_path / "run"):
        with pytest.raises(RuntimeError, match="already active"):
            with run_lock(tmp_path / "run"):
                pass
    write_json(tmp_path / "value.json", {"中文": [1, 2, 3]})
    assert read_json(tmp_path / "value.json") == {"中文": [1, 2, 3]}


def test_resume_comparison_checks_shape_not_only_probability():
    _equal({"topics": [{"id": 1}], "p": 0.3}, {"topics": [{"id": 1}], "p": 0.30000001})
    with pytest.raises(AssertionError):
        _equal({"topics": [1]}, {"topics": [1, 2]})
