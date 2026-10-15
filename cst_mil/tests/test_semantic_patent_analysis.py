"""Small standalone analysis cohorts, with a deliberate non-TEST group bridge."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from cst_mil.research_followup.semantic_patent_followup import analysis


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_groups(rows):
    parents = {r["base_eval_group"]: r["base_eval_group"] for r in rows}

    def root(node):
        while node != parents[node]:
            node = parents[node]
        return node

    first = {}
    for row in rows:
        prior = first.setdefault(row["duplicate_component"], row["base_eval_group"])
        a, b = root(prior), root(row["base_eval_group"])
        parents[b] = a
    members = {}
    for node in parents:
        members.setdefault(root(node), set()).add(node)
    names = {node: "eval:" + hashlib.sha256("|".join(sorted(values)).encode()).hexdigest()
             for node, values in members.items()}
    for row in rows:
        row["eval_group"] = names[root(row["base_eval_group"])]


def make_campaign(path: Path):
    base = path / "upstream"
    output = path / "followup"
    records, sessions = [], []
    for split in ("train", "dev", "test"):
        for source in ("sccd", "conda", "hatebr", "toxicn", "scamspot"):
            for i in range(4):
                sid = f"{split}-session-{i // 2}" if source == "sccd" else None
                rid = f"{source}-{split}-{i}"
                labels, mask = [i % 2, 1 - i % 2, i % 2], [True, source == "toxicn", False]
                if source == "scamspot":
                    mask = [False, False, True]
                group = "sccd:session:" + sid if source == "sccd" else f"{source}:real-group:{split}-{i}"
                records.append({"record_id": rid, "message_id": rid, "session_id": sid,
                                "source": source, "split": split, "parent_id": None,
                                "order": i % 2, "text": "完整\u2028正文 " + rid,
                                "labels": labels, "observed_mask": mask,
                                "base_eval_group": group, "duplicate_component": "dup:" + rid,
                                "fit_eligible": True})
            if source == "sccd":
                sessions += [{"session_id": f"{split}-session-{i}", "split": split, "label": i}
                             for i in range(2)]
    by_id = {r["record_id"]: r for r in records}
    # Both TEST sessions connect through two rows in one TRAIN session. A graph
    # built from TEST alone would incorrectly report two independent clusters.
    by_id["sccd-train-0"]["duplicate_component"] = "bridge-left"
    by_id["sccd-test-0"]["duplicate_component"] = "bridge-left"
    by_id["sccd-train-1"]["duplicate_component"] = "bridge-right"
    by_id["sccd-test-2"]["duplicate_component"] = "bridge-right"
    freeze_groups(records)
    jsonl(base / "prepared/records.jsonl", records)
    jsonl(base / "prepared/sessions.jsonl", sessions)
    save(base / "prepared/manifest.json", {"artifacts": {
        name: sha(base / "prepared" / name) for name in ("records.jsonl", "sessions.jsonl")}})
    protocol = {"base_output": str(base), "input_hashes": {}, "selection": {
        "minimum_primary_gain": .01, "maximum_other_AP_drop": .01,
        "maximum_SCCD_session_AP_AUC_drop": .01, "maximum_SCCD_attack_macro_F1_drop": .01,
        "tie_order": ["context_a", "context_b"], "tie_tolerance": 1e-12}}
    save(output / "protocol.json", protocol)
    save(output / "selection_freeze.json", {"selected_architecture": "full"})
    for variant in (*analysis.ARCHITECTURES, *analysis.ABLATIONS):
        for seed in analysis.SEEDS:
            folder = output / "runs" / variant / f"seed_{seed}"
            all_predictions = {}
            all_sessions = {}
            for split in ("dev", "test"):
                pred = []
                for row in records:
                    if row["split"] != split:
                        continue
                    scores = [.9 if label else .1 for label in row["labels"]]
                    if variant == "no_char":
                        scores = [1 - value for value in scores]
                    pred.append({**row, "scores": scores, "text_length": len(row["text"])})
                all_predictions[split] = pred
                all_sessions[split] = [{**session,
                    "score": .9 if session["label"] else .1,
                    "common16_score": .8 if session["label"] else .2}
                    for session in sessions if session["split"] == split]
            dev = all_predictions["dev"]
            thresholds = {}
            for j, category in enumerate(analysis.frozen.CATEGORIES):
                observed = [r for r in dev if r["observed_mask"][j]]
                thresholds[category] = analysis.select_threshold_independent(
                    [r["labels"][j] for r in observed], [r["scores"][j] for r in observed])
            thresholds.update(conversation=.9, conversation_common16=.8)
            manifest = {"seed": seed, "config": {"architecture_base": "full" if variant in analysis.ABLATIONS else variant,
                                                 "ablation": variant if variant in analysis.ABLATIONS else None},
                        "thresholds": thresholds, "conversation_threshold": .9, "common16_threshold": .8}
            save(folder / "manifest.json", manifest)
            save(folder / "complete.json", {"status": "complete", "artifacts": {}})
            save(folder / "test_complete.json", {"status": "complete", "artifacts": {}})
            for split in ("dev", "test"):
                for row in all_sessions[split]:
                    row.update(threshold=.9, common16_threshold=.8)
                jsonl(folder / f"predictions_message_{split}.jsonl", all_predictions[split])
                jsonl(folder / f"predictions_session_{split}.jsonl", all_sessions[split])
    return output, records, sessions


def test_independent_threshold_matches_brute_force_including_lower_tie():
    rng = np.random.default_rng(4)
    for n in (2, 5, 30):
        labels = rng.integers(0, 2, size=n)
        scores = rng.choice([0., .2, .5, .8, 1.], size=n)
        candidates = np.unique(np.r_[0., scores, 1.])
        results = [f1_score(labels, scores >= c, average="macro", zero_division=0) for c in candidates]
        expected = candidates[np.argmax(results)]
        assert analysis.select_threshold_independent(labels, scores) == expected


def test_full_analysis_uses_non_test_bridges_and_keeps_all_effects(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    summary = analysis.summarize(output, count=12)
    assert summary["status"] == "completed"
    assert len(summary["metrics"]) == 9 * 2 * 8
    assert summary["n_ablation_effects"] == 6 * 8 * 5
    assert summary["selection_audit"]["selected_architecture"] == "full"
    audit = summary["bootstrap"]["sccd.conversation"]
    assert audit["n_clusters"] == 1
    assert audit["status"] == "insufficient_clusters"
    assert audit["test_only_groups_reconstructed"] is False
    assert audit["interval_available"] is False
    assert (output / "analysis/bootstrap/test__conda.attack.npz").exists()
    assert (output / "analysis/per_seed_ablation_effects.csv").exists()
    gold, _, _ = analysis.load_frozen_gold(output)
    assert len(gold) == 60
    assert summary["effect_conclusions"]["interval_unavailable"] >= 2 * 6 * 5


def test_independent_metric_numbers(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    gold, sessions, _ = analysis.load_frozen_gold(output)
    folder = output / "runs/full/seed_42"
    tasks, rows = analysis.extract_tasks(
        analysis._jsonl(folder / "predictions_message_test.jsonl"),
        analysis._jsonl(folder / "predictions_session_test.jsonl"),
        analysis._json(folder / "manifest.json"), gold, sessions, "test")
    task = tasks["conda.attack"]
    result = analysis.frozen.independent_metrics(task.labels, task.scores, task.threshold)
    assert result["AP"] == average_precision_score(task.labels, task.scores) == 1.
    assert result["roc_auc"] == roc_auc_score(task.labels, task.scores) == 1.
    assert result["macro_f1"] == f1_score(task.labels, task.scores >= task.threshold, average="macro") == 1.
    top3 = analysis.frozen.top3_metrics(rows)
    assert top3["positive_gold_sessions"] == 2
    assert top3["hit_sessions"] == 2
    assert top3["dangerous_comment_recall_at_3"] == 1


@pytest.mark.parametrize("field,value", [
    ("labels", [1, 1, 1]), ("observed_mask", [False, False, False]),
    ("eval_group", "fake:group"), ("parent_id", "fake"), ("text_length", 0),
])
def test_prediction_cannot_change_gold_metadata(field, value, tmp_path):
    output, _, _ = make_campaign(tmp_path)
    gold, sessions, _ = analysis.load_frozen_gold(output)
    folder = output / "runs/full/seed_42"
    messages = analysis._jsonl(folder / "predictions_message_test.jsonl")
    messages[0][field] = value
    with pytest.raises(ValueError):
        analysis.extract_tasks(messages, analysis._jsonl(folder / "predictions_session_test.jsonl"),
                               analysis._json(folder / "manifest.json"), gold, sessions, "test")


def test_duplicate_or_missing_prediction_rejected(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    gold, sessions, _ = analysis.load_frozen_gold(output)
    folder = output / "runs/full/seed_42"
    messages = analysis._jsonl(folder / "predictions_message_test.jsonl")
    with pytest.raises(ValueError, match="Incomplete/duplicate"):
        analysis.extract_tasks(messages[:-1] + [messages[0]],
                               analysis._jsonl(folder / "predictions_session_test.jsonl"),
                               analysis._json(folder / "manifest.json"), gold, sessions, "test")


def test_frozen_data_tampering_rejected(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    base = Path(analysis._json(output / "protocol.json")["base_output"])
    with (base / "prepared/records.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="fingerprint changed"):
        analysis.load_frozen_gold(output)


def test_wrong_saved_dev_threshold_rejected(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    folder = output / "runs/full/seed_42"
    manifest = analysis._json(folder / "manifest.json")
    manifest["thresholds"]["attack"] = .5
    with pytest.raises(ValueError, match="independent DEV selection"):
        analysis.verify_dev_thresholds(analysis._jsonl(folder / "predictions_message_dev.jsonl"),
                                       analysis._jsonl(folder / "predictions_session_dev.jsonl"), manifest)


def test_seed_mismatch_prevents_reporting(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    path = output / "runs/full/seed_42/manifest.json"
    manifest = analysis._json(path)
    manifest["seed"] = 99
    save(path, manifest)
    with pytest.raises(ValueError, match="architecture/ablation/seed"):
        analysis.summarize(output, count=2)


def test_ablation_on_wrong_selected_architecture_rejected(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    path = output / "runs/no_char/seed_42/manifest.json"
    manifest = analysis._json(path)
    manifest["config"]["architecture_base"] = "context_b"
    save(path, manifest)
    with pytest.raises(ValueError, match="architecture/ablation/seed"):
        analysis.summarize(output, count=2)


def test_frozen_selection_is_independently_checked(tmp_path):
    output, _, _ = make_campaign(tmp_path)
    summary = analysis.summarize(output, count=2)
    with pytest.raises(ValueError, match="independent DEV-rule"):
        analysis.selection_audit(summary["metrics"], "context_a",
                                 analysis._json(output / "protocol.json")["selection"])


def test_effect_conclusions_do_not_hide_negative_or_zero():
    assert analysis._interval_conclusion(.01, .02) == "supports_positive_contribution"
    assert analysis._interval_conclusion(-.02, -.01) == "ablation_higher_on_this_metric"
    assert analysis._interval_conclusion(-.01, .01) == "insufficient_evidence_of_difference"
    assert analysis._interval_conclusion(None, None) == "interval_unavailable"


def test_score_averaging_is_not_used_in_seed_metric_aggregation():
    # Mean of per-seed AP is 5/6; AP of the averaged scores is 1.0.
    y = np.asarray([0, 1])
    arrays = [np.asarray([.9, .8]), np.asarray([.1, .8]), np.asarray([.2, .9])]
    rows = [{"split": "test", "variant": "full", "seed": seed, "task": "toy",
             **analysis.frozen.independent_metrics(y, scores, .5)}
            for seed, scores in zip(analysis.SEEDS, arrays, strict=True)]
    result = analysis.aggregate_metrics(rows)[0]
    assert result["AP_mean"] == pytest.approx((.5 + 1 + 1) / 3)
    assert result["AP_mean"] != average_precision_score(y, np.mean(arrays, axis=0))


def test_sealed_artifact_hash_must_match(tmp_path):
    artifact = tmp_path / "prediction.jsonl"
    artifact.write_text("old", encoding="utf-8")
    seal = tmp_path / "complete.json"
    save(seal, {"status": "complete", "artifacts": {str(artifact): "wrong"}})
    with pytest.raises(ValueError, match="Sealed artifact changed"):
        analysis._verify_seal(seal)
