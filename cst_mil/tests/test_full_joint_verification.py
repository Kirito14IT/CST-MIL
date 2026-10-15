"""Light independent-audit fixtures; no model fitting or formal campaign audit."""

import copy

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cst_mil.research_followup.full_joint_comparison import verification as audit


def toy_cohort(n=512):
    records, predictions, bags, sessions = [], [], [], []
    sources = ("sccd", "conda", "toxicn", "scamspot", "hatebr")
    thresholds = {
        "attack": 0.5,
        "hate": 0.4,
        "spam_scam": 0.6,
        "conversation": 0.7,
        "conversation_common16": 0.65,
    }
    for index in range(n):
        source, rid = sources[index % 5], f"row-{index:04d}"
        labels = [index % 2, (index // 2) % 2, index % 2]
        observed = (
            [True, True, False]
            if source == "toxicn"
            else [False, False, True]
            if source == "scamspot"
            else [True, False, False]
        )
        text = "x" * (30 if index % 2 else 31)
        row = {
            "record_id": rid,
            "source": source,
            "split": "test",
            "session_id": rid,
            "message_id": rid + ":m",
            "labels": labels,
            "observed_mask": observed,
            "duplicate_component": rid,
            "eval_group": "group:" + rid,
            "parent_id": None,
            "order": 0,
            "is_fallback": False,
            "text": text,
        }
        records.append(row)
        prediction = {k: v for k, v in row.items() if k != "text"}
        prediction.update(
            text_length=len(text),
            scores={
                name: (0.8 if label else 0.2)
                for name, label in zip(audit.CATEGORIES, labels, strict=True)
            },
        )
        predictions.append(prediction)
        if source == "sccd":
            sessions.append({"session_id": rid, "split": "test", "label": labels[0]})
            bags.append(
                {
                    "session_id": rid,
                    "split": "test",
                    "label": labels[0],
                    "score": 0.9 if labels[0] else 0.1,
                    "common16_score": 0.8 if labels[0] else 0.2,
                    "threshold": 0.7,
                    "common16_threshold": 0.65,
                }
            )
    return records, sessions, predictions, bags, thresholds


def test_512_row_cohort_labels_masks_thresholds_and_metrics():
    records, sessions, predictions, bags, thresholds = toy_cohort()
    proof = audit.verify_prediction_cohort(records, sessions, predictions, bags, thresholds, "test")
    assert proof["messages"] == 512 and proof["metadata_and_masks_exact"]
    metrics, strata = audit.task_metrics(predictions, bags, thresholds)
    assert len(metrics) == 8 and len(strata) == 12
    for value in metrics.values():
        assert value["AP"] == value["roc_auc"] == value["macro_f1"] == 1
    for task in metrics:
        if task.startswith("sccd.conversation"):
            continue
        assert (
            sum(strata[task, name]["n"] for name in ("le_30_chars", "gt_30_chars"))
            == metrics[task]["n"]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", [1, 1, 1]),
        ("observed_mask", [False, False, False]),
        ("text_length", 7),
        ("eval_group", "invented"),
    ],
)
def test_prediction_cohort_rejects_mismatched_gold_or_metadata(field, value):
    records, sessions, predictions, bags, thresholds = toy_cohort()
    predictions[0][field] = value
    with pytest.raises(ValueError):
        audit.verify_prediction_cohort(records, sessions, predictions, bags, thresholds, "test")


def test_prediction_cohort_rejects_missing_ids_and_changed_session_threshold():
    records, sessions, predictions, bags, thresholds = toy_cohort()
    with pytest.raises(ValueError, match="IDs"):
        audit.verify_prediction_cohort(
            records, sessions, predictions[:-1], bags, thresholds, "test"
        )
    bags[0]["threshold"] = 0.5
    with pytest.raises(ValueError, match="threshold"):
        audit.verify_prediction_cohort(records, sessions, predictions, bags, thresholds, "test")


def test_independent_metrics_match_direct_sklearn_with_tied_scores():
    rng = np.random.default_rng(42)
    labels, scores = rng.integers(0, 2, 512), rng.integers(0, 10, 512) / 10
    result = audit.independent_metrics(labels, scores, 0.4)
    assert result["AP"] == average_precision_score(labels, scores)
    assert result["roc_auc"] == roc_auc_score(labels, scores)
    assert result["macro_f1"] == f1_score(labels, scores >= 0.4, labels=[0, 1], average="macro")


def test_empty_and_single_class_metrics_remain_explicit():
    assert audit.independent_metrics([], [], 0.5)["AP"] is None
    negative = audit.independent_metrics([0, 0], [0.2, 0.6], 0.5)
    assert negative["AP"] is negative["roc_auc"] is None
    positive = audit.independent_metrics([1, 1], [0.2, 0.6], 0.5)
    assert positive["AP"] == 1 and positive["roc_auc"] is None


def test_top3_handles_ties_small_sessions_unknown_and_fallback():
    _, _, predictions, _, _ = toy_cohort(16)
    rows = [r for r in predictions if r["source"] == "sccd"]
    for row in rows:
        row["session_id"] = "same-session"
        row["scores"]["attack"] = 0.5
        row["labels"][0] = 0
    rows[0]["is_fallback"] = True
    rows[0]["labels"][0] = 1
    rows[1]["observed_mask"][0] = False
    rows[1]["labels"][0] = 1
    rows[3]["labels"][0] = 1
    result = audit.independent_top3(rows)
    assert result["dangerous_gold_comments"] == 1
    assert result["selected_comments_all_sessions"] == 3
    assert result["hit_sessions"] == result["dangerous_selected_comments"] == 1
    assert audit.independent_top3(rows[:1])["selected_comments_all_sessions"] == 0


def test_clusters_merge_true_groups_through_unknown_duplicate_bridge():
    _, _, rows, _, _ = toy_cohort(16)
    sccd = [r for r in rows if r["source"] == "sccd"]
    sccd[1]["duplicate_component"] = sccd[0]["duplicate_component"]
    sccd[1]["observed_mask"] = [False, False, False]
    result = audit.cluster_counts(sccd)
    assert result["sccd.conversation"] == 3
    assert result["sccd.attack"] == 3


def test_final_heads_require_identical_ids_and_correct_source_class_weights():
    records, _, _, _, _ = toy_cohort()
    for row in records:
        row.update(split="train", fit_eligible=True)
    allowed = {r["record_id"] for r in records}
    heads = []
    for j, category in enumerate(audit.CATEGORIES):
        chosen = [r for r in records if r["observed_mask"][j]]
        counts = {}
        for row in chosen:
            key = row["source"], row["labels"][j]
            counts[key] = counts.get(key, 0) + 1
        raw = np.array([1 / (2 * counts[r["source"], r["labels"][j]]) for r in chosen])
        heads.append(
            {
                "category": category,
                "seed": 42,
                "n_observed": len(chosen),
                "observed_record_ids": [r["record_id"] for r in chosen],
                "sample_weight": (raw * (len(raw) / raw.sum())).tolist(),
            }
        )
    signature = audit._head_signature(heads, records, allowed, 42)
    assert len(signature) == 64
    bad = copy.deepcopy(heads)
    bad[0]["sample_weight"][0] += 0.01
    with pytest.raises(ValueError, match="weights"):
        audit._head_signature(bad, records, allowed, 42)
    bad = copy.deepcopy(heads)
    bad[0]["observed_record_ids"].pop()
    with pytest.raises(ValueError, match="IDs"):
        audit._head_signature(bad, records, allowed, 42)


def test_partial_audit_missing_experiment_is_incomplete_and_never_passes(tmp_path):
    result = audit.audit(tmp_path, partial=True)
    assert result["status"] == "incomplete" and result["completed_runs_audited"] == 0
    assert (tmp_path / "completion_audit.json").is_file()


def fit_fixture(seed=42):
    records, sessions, _, _, _ = toy_cohort()
    for row in records:
        row.update(split="train", fit_eligible=True)
    for row in sessions:
        row["split"] = "train"
    records[1]["duplicate_component"] = records[0]["duplicate_component"]

    def heads(allowed):
        result = []
        for j, name in enumerate(audit.CATEGORIES):
            rows = [r for r in records if r["record_id"] in allowed and r["observed_mask"][j]]
            counts = {}
            for row in rows:
                key = row["source"], row["labels"][j]
                counts[key] = counts.get(key, 0) + 1
            classes = {
                source: {label for s, label in counts if s == source} for source, _ in counts
            }
            raw = np.array(
                [
                    1 / (len(classes[r["source"]]) * counts[r["source"], r["labels"][j]])
                    for r in rows
                ]
            )
            result.append(
                {
                    "category": name,
                    "seed": seed,
                    "n_observed": len(rows),
                    "observed_record_ids": [r["record_id"] for r in rows],
                    "sample_weight": (raw * (len(raw) / raw.sum())).tolist(),
                }
            )
        return result

    all_train = {r["record_id"] for r in records}
    details = {
        "parameters": {"method": "nmf", "seed": seed, "threads": 4, "folds": 3},
        "fit_audit": {
            "method": "nmf",
            "seed": seed,
            "threads": 4,
            "test_records_used_for_fit": 0,
            "representation_fit_record_ids": list(all_train),
            "pca_seed": None,
            "backend": {"seed": seed},
            "folds": [],
            "head_audits": {"final": heads(all_train)},
        },
    }
    labels = np.array([s["label"] for s in sessions])
    for fold, (_, held_indices) in enumerate(
        StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels)
    ):
        held = {sessions[i]["session_id"] for i in held_indices}
        held_rows = [r for r in records if r["source"] == "sccd" and r["session_id"] in held]
        components = {r["duplicate_component"] for r in held_rows}
        allowed = {
            r["record_id"]
            for r in records
            if r["record_id"] in all_train
            and not (r["source"] == "sccd" and r["session_id"] in held)
            and r["duplicate_component"] not in components
        }
        details["fit_audit"]["folds"].append(
            {
                "fold": fold,
                "seed": seed,
                "held_session_ids": list(held),
                "held_record_ids": [r["record_id"] for r in held_rows],
                "held_duplicate_components": list(components),
                "allowed_record_ids": list(allowed),
                "n_allowed": len(allowed),
            }
        )
        details["fit_audit"]["head_audits"][f"fold_{fold}"] = heads(allowed)
    return details, records, sessions


def test_512_row_fit_protocol_three_folds_and_cross_source_duplicate_exclusion():
    details, records, sessions = fit_fixture()
    result = audit.verify_fit_details(details, records, sessions, "nmf", 42, 4)
    assert result["train_records"] == 512 and result["fold_coverage_exact"]
    assert len(result["oof_signatures"]) == 3
    bad = copy.deepcopy(details)
    first = bad["fit_audit"]["folds"][0]
    first["allowed_record_ids"].append(first["held_record_ids"][0])
    first["n_allowed"] += 1
    with pytest.raises(ValueError, match="OOF allowed"):
        audit.verify_fit_details(bad, records, sessions, "nmf", 42, 4)
    bad = copy.deepcopy(details)
    bad["fit_audit"]["backend"]["seed"] = 43
    with pytest.raises(ValueError, match="Topic backend"):
        audit.verify_fit_details(bad, records, sessions, "nmf", 42, 4)


@pytest.mark.parametrize(
    "labels,scores,threshold,macro",
    [
        ([0, 1], [0.1, 0.9], 0.9, 1.0),
        ([0, 1], [0.0, 1.0], 1.0, 1.0),
        ([0, 1], [0.5, 0.5], 0.0, 1 / 3),
        ([0, 1], [1.0, 1.0], 0.0, 1 / 3),
        ([0, 0], [0.0, 0.0], 1.0, 1.0),
        ([1, 1], [1.0, 1.0], 0.0, 1.0),
        ([0, 1], [0.9, 0.1], 0.0, 1 / 3),
    ],
)
def test_threshold_independent_scan_boundaries_ties_and_absent_classes(
    labels, scores, threshold, macro
):
    proof = audit.independent_dev_threshold(labels, scores)
    assert proof["threshold"] == threshold
    assert proof["macro_f1"] == pytest.approx(macro, abs=1e-15)
    assert proof["tie_tolerance"] == 1e-12


def test_threshold_scan_matches_exhaustive_sklearn_without_training_selector():
    rng = np.random.default_rng(43)
    labels, scores = rng.integers(0, 2, 512), rng.integers(0, 25, 512) / 24
    result = audit.independent_dev_threshold(labels, scores)
    best, selected = -1.0, None
    for threshold in sorted({0.0, 1.0, *scores.tolist()}):
        macro = f1_score(labels, scores >= threshold, average="macro", zero_division=0)
        if macro > best + 1e-12:
            best, selected = macro, threshold
    assert result["threshold"] == selected and result["macro_f1"] == best


def test_threshold_verification_pools_sources_masks_unknowns_and_checks_five_values():
    _, _, messages, bags, _ = toy_cohort()
    for row in messages + bags:
        row["split"] = "dev"
    thresholds = {name: 0.8 for name in audit.CATEGORIES}
    thresholds.update(conversation=0.9, conversation_common16=0.8)
    result = audit.verify_dev_thresholds(messages, bags, thresholds)
    assert result["status"] == "passed" and len(result["thresholds"]) == 5
    for name in audit.CATEGORIES:
        assert result["thresholds"][name]["n_observed"] == sum(
            audit.category(row, "observed_mask", name) for row in messages
        )
    wrong = {**thresholds, "attack": 0.5}
    with pytest.raises(ValueError, match="selected by dev"):
        audit.verify_dev_thresholds(messages, bags, wrong)
    messages[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation predictions only"):
        audit.verify_dev_thresholds(messages, bags, thresholds)


def test_empty_validation_threshold_branch_is_explicitly_unverifiable():
    with pytest.raises(audit.IncompleteAudit, match="No observed validation"):
        audit.independent_dev_threshold([], [])
