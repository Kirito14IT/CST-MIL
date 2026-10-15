from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from cst_mil.research_followup.full_joint_comparison.analysis import (
    METRICS, METHODS, TaskPrediction, _cost_table, _jsonl, _weighted_metrics, bootstrap_paired, cluster_components,
    independent_metrics, summarize, top3_metrics,
)


def test_weighted_bootstrap_metrics_match_sklearn_including_ties() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([.4, .4, .7, .8, .4, .1])
    weights = np.array([[1, 1, 2, 3, 1, 1], [0, 2, 1, 1, 0, 2]])
    actual = _weighted_metrics(y, p, .4, weights)
    for i, w in enumerate(weights):
        prediction = p >= .4
        expected = [average_precision_score(y, p, sample_weight=w),
                    roc_auc_score(y, p, sample_weight=w),
                    f1_score(y, prediction, average="macro", labels=[0, 1], sample_weight=w, zero_division=0),
                    precision_score(y, prediction, sample_weight=w, zero_division=0),
                    recall_score(y, prediction, sample_weight=w, zero_division=0)]
        assert actual[i] == pytest.approx(expected, abs=1e-12)


def test_unknown_messages_connect_true_groups_by_duplicate_components() -> None:
    rows = [{"record_id": "a", "source": "conda", "eval_group": "match1", "duplicate_component": "dup1"},
            {"record_id": "b", "source": "conda", "eval_group": "match2", "duplicate_component": "dup1", "observed_mask": [False] * 3},
            {"record_id": "c", "source": "conda", "eval_group": "match2", "duplicate_component": "dup2"},
            {"record_id": "d", "source": "conda", "eval_group": "match3", "duplicate_component": "dup3"}]
    groups = cluster_components(rows)
    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["c"] != groups["d"]
    with pytest.raises(ValueError, match="genuine eval_group"):
        cluster_components([{"record_id": "bad", "source": "hatebr"}])


def test_paired_bootstrap_averages_metrics_not_scores_and_logs_degeneracy() -> None:
    labels = np.array([0, 0, 1, 1])
    ids, groups = list("abcd"), list("abcd")
    first = TaskPrediction(ids, labels, np.array([.05, .8, .7, .6]), .5, groups)
    second = TaskPrediction(ids, labels, np.array([.8, .05, .7, .6]), .5, groups)
    good = TaskPrediction(ids, labels, np.array([.1, .2, .8, .9]), .5, groups)
    distribution, meta = bootstrap_paired({"a": [first, second], "b": [good, good]}, count=100, seed=3)
    repeated, again = bootstrap_paired({"a": [first, second], "b": [good, good]}, count=100, seed=3)
    assert np.array_equal(distribution["a"], repeated["a"], equal_nan=True)
    assert meta == again
    assert meta["degenerate"] > 0
    # Whenever every item is sampled once, AP is the per-seed mean .5833,
    # whereas averaging the scores first would produce AP=1.
    expected = .5 * (independent_metrics(labels, first.scores, .5)["AP"] + independent_metrics(labels, second.scores, .5)["AP"])
    assert np.any(np.isclose(distribution["a"][:, 0], expected))
    assert expected < average_precision_score(labels, .5 * (first.scores + second.scores))
    mismatch = TaskPrediction(ids[::-1], labels, first.scores, .5, groups)
    with pytest.raises(ValueError, match="cohorts/groups"):
        bootstrap_paired({"a": [first], "bad": [mismatch]}, count=10)


def test_bootstrap_reports_cluster_concentration_and_rejects_single_cluster_ci() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([.1, .9, .2, .8])
    ids = list("abcd")
    pred = TaskPrediction(ids, y, p, .5, ["large", "large", "large", "small"])
    _, meta = bootstrap_paired({"a": [pred]}, count=20, seed=2)
    assert meta["n_clusters"] == 2
    assert meta["largest_cluster_n"] == 3
    assert meta["largest_cluster_fraction"] == .75
    assert meta["size_concentration_effective_cluster_count"] == pytest.approx(1 / (.75 ** 2 + .25 ** 2))
    singleton = TaskPrediction(ids, y, p, .5, ["one"] * 4)
    draws, metadata = bootstrap_paired({"a": [singleton], "b": [singleton]}, count=20)
    assert np.isnan(draws["a"]).all()
    assert metadata["status"] == "insufficient_clusters"
    assert metadata["attempted"] == 0
    assert metadata["interval_available"] is False


def test_top3_denominator_is_positive_gold_sessions_and_fallback_is_not_evidence() -> None:
    rows = []
    for sid, labels in (("a", [1, 1, 1, 1]), ("b", [0]), ("c", [1])):
        for i, label in enumerate(labels):
            rows.append({"record_id": f"{sid}{i}", "session_id": sid, "source": "sccd",
                         "labels": [label, 0, 0], "observed_mask": [True, False, False],
                         "scores": {"attack": 1 - i / 10}})
    rows.append({"record_id": "fallback", "session_id": "a", "source": "sccd",
                 "is_fallback": True, "labels": [0, 0, 0], "observed_mask": [False] * 3,
                 "scores": {"attack": 1.0}})
    actual = top3_metrics(rows)
    assert actual["positive_gold_sessions"] == 2
    assert actual["hit_at_3"] == 1
    assert actual["dangerous_comment_recall_at_3"] == pytest.approx(4 / 5)
    assert actual["selected_comments_all_sessions"] == 5


def test_top3_score_ties_preserve_original_message_order_before_ids() -> None:
    rows = [{"record_id": rid, "message_id": rid, "session_id": "s", "source": "sccd",
             "order": order, "labels": [label, 0, 0], "observed_mask": [True, False, False],
             "scores": {"attack": .5}}
            for rid, order, label in (("z-first", 0, 1), ("b-second", 1, 0),
                                      ("c-third", 2, 0), ("a-last", 3, 0))]
    result = top3_metrics(rows[::-1])
    assert result["dangerous_selected_comments"] == 1
    assert result["hit_at_3"] == 1
    assert "original order" in result["ranking"]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _toy_run(root: Path, method: str, seed: int, *, flip_test: bool) -> None:
    run = root / "runs" / method / f"seed_{seed}"
    run.mkdir(parents=True)
    thresholds = {"attack": .5, "hate": .5, "spam_scam": .5,
                  "conversation": .5, "conversation_common16": .5}
    (run / "run_manifest.json").write_text(json.dumps({"method": method, "seed": seed, "thresholds": thresholds}), encoding="utf-8")
    for split in ("dev", "test"):
        rows = []
        for source in ("sccd", "conda", "hatebr", "toxicn", "scamspot"):
            for i in range(4):
                label = i % 2
                score = .8 if label else .2
                if split == "test" and flip_test:
                    score = 1 - score
                sid = f"{split}-s{i // 2}"
                masks = [source != "scamspot", source == "toxicn", source == "scamspot"]
                rows.append({"record_id": f"{source}-{split}-{i}", "message_id": str(i),
                             "session_id": sid, "source": source, "split": split,
                             "labels": [label] * 3, "observed_mask": masks,
                             "scores": dict.fromkeys(("attack", "hate", "spam_scam"), score),
                             "eval_group": f"{source}-{split}-{i}",
                             "duplicate_component": f"{source}-{split}-{i}",
                             "text_length": 20 if i < 2 else 40})
        sessions = [{"session_id": f"{split}-s{i}", "label": i,
                     "score": .8 if i else .2, "common16_score": .7 if i else .3,
                     "threshold": .5, "common16_threshold": .5} for i in range(2)]
        _write_jsonl(run / f"predictions_message_{split}.jsonl", rows)
        _write_jsonl(run / f"predictions_session_{split}.jsonl", sessions)


def test_full_summary_uses_validation_selection_and_excludes_unknown_masks(tmp_path: Path) -> None:
    for seed in (42, 43):
        _toy_run(tmp_path, "lda", seed, flip_test=True)
        _toy_run(tmp_path, "cst_v1_1", seed, flip_test=False)
    result = summarize(tmp_path, bootstrap=30, methods=("lda", "cst_v1_1"), seeds=(42, 43))
    assert result["selected_baselines"]["toxicn.hate"]["method"] == "lda"
    assert result["status"] == "complete"
    assert len(result["summary"]) == 2 * 8 * 2
    assert all(r["n"] == 4 for r in result["summary"] if r["task"] == "toxicn.hate")
    assert (tmp_path / "analysis" / "length_strata.csv").exists()
    assert (tmp_path / "analysis" / "paired_contrasts.csv").exists()
    assert (tmp_path / "analysis" / "costs.csv").exists()
    summary = (tmp_path / "analysis" / "analysis_summary.md").read_text(encoding="utf-8")
    assert "全部4次训练结果" in summary
    assert "../runs/lda/seed_42/predictions_message_test.jsonl" in summary
    assert "N/A" in summary
    assert not any(r["task"] == "pooled" for r in result["summary"])


def test_missing_run_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        summarize(tmp_path, bootstrap=10, methods=("lda",), seeds=(42,))


def test_analysis_jsonl_preserves_unicode_separators_in_strings(tmp_path: Path) -> None:
    rows = [{"record_id": "a", "text": "中文\u2028English\u2029Português\u0085"},
            {"record_id": "b", "text": "العربية\u0085日本語\u2029ไทย\u2028"}]
    path = tmp_path / "unicode.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8-sig")
    with path.open(encoding="utf-8-sig") as handle:
        assert len(list(handle)) == 2
    assert _jsonl(path) == rows


def test_cost_csv_preserves_three_measurements_and_unknowns() -> None:
    rows = _cost_table([{"method": "lda", "warm_total_seconds": [1., 2., 3.],
                         "GPU_required": False, "model_bytes": 0,
                         "comment_throughput_per_second_by_source": {"conda": 2.5}}],
                       ["lda", "cst_v1_1"])
    assert rows[0]["warm_total_seconds_mean"] == 2
    assert rows[0]["warm_total_seconds_sd"] == 1
    assert rows[0]["model_bytes"] == 0
    assert rows[0]["GPU_required"] is False
    assert rows[1]["model_bytes"] == "N/A"
    assert rows[1]["measurement_status"] == "unavailable"
    assert rows[0]["dependency_sizing_scope"] == "N/A"
    assert rows[0]["cold_start_definition"] == "N/A"
    assert "deserialization only" in rows[0]["model_load_seconds_definition"]
    assert "lazy encoder initialization" in rows[0]["first_inference_seconds_definition"]
    assert "not the entire test set" in rows[0]["first_inference_seconds_definition"]
    assert "not a separate full cold-start end-to-end rerun" in rows[0]["training_seconds_cold_definition"]
    assert "must not be summed as campaign elapsed time" in rows[0]["training_seconds_cold_definition"]


def test_figure_exports_real_analysis_and_all_required_qa(tmp_path: Path) -> None:
    from cst_mil.research_followup.full_joint_comparison.figures import plot
    skill = Path.home() / ".codex/skills/nature-figure"
    if not skill.exists():
        pytest.skip("Local figure QA skill unavailable")
    for method in METHODS:
        for seed in (42, 43, 44):
            _toy_run(tmp_path, method, seed, flip_test=method == "lda")
        cost = {"training_seconds_cold": 10, "training_seconds_cached": 8,
                "warm_total_seconds": [1, 1.1, .9], "peak_tree_rss_bytes": 10 * 1024 ** 2,
                "model_bytes": 1024 ** 2, "encoder_asset_bytes": 2 * 1024 ** 2,
                "runtime_dependency_bytes": 3 * 1024 ** 2, "GPU_required": False}
        (tmp_path / "runs" / method / "seed_42/cost.json").write_text(json.dumps(cost), encoding="utf-8")
    summarize(tmp_path, bootstrap=30)
    outputs = plot(tmp_path)
    assert len(outputs) == 6
    for path in outputs:
        assert Path(path).exists()
    qa = tmp_path / "figures/qa"
    assert (qa / "performance.alignment.json").exists()
    assert (qa / "costs.collisions.json").exists()
