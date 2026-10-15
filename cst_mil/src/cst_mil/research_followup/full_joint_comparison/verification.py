"""Independent completion audit; never refits models or modifies experiment evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .common import digest, identity, read, read_jsonl, save, verify_hashes

CATEGORIES = ("attack", "hate", "spam_scam")
MESSAGE_TASKS = (
    ("sccd", "attack"),
    ("conda", "attack"),
    ("hatebr", "attack"),
    ("toxicn", "attack"),
    ("toxicn", "hate"),
    ("scamspot", "spam_scam"),
)
METRICS = ("AP", "roc_auc", "macro_f1", "precision", "recall")
TOLERANCE = 1e-10


class IncompleteAudit(Exception):
    """Expected evidence has not yet been produced by the sequential campaign."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def category(row: dict, field: str, name: str):
    value = row[field]
    return value[name] if isinstance(value, dict) else value[CATEGORIES.index(name)]


def independent_metrics(labels, scores, threshold: float) -> dict:
    """A fresh sklearn computation, independent of training and analysis modules."""
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    require(y.ndim == p.ndim == 1 and len(y) == len(p), "Metric vector shape mismatch")
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid probability")
    require(set(y.tolist()) <= {0, 1}, "Invalid binary labels")
    n, positive = len(y), int(y.sum())
    result = {"n": n, "positive": positive, "threshold": float(threshold)}
    if not n:
        return {**result, **dict.fromkeys(METRICS)}
    predicted = p >= threshold
    return {
        **result,
        "AP": float(average_precision_score(y, p)) if positive else None,
        "roc_auc": float(roc_auc_score(y, p)) if 0 < positive < n else None,
        "macro_f1": float(f1_score(y, predicted, labels=[0, 1], average="macro", zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
    }


def independent_dev_threshold(labels, scores) -> dict:
    """Scan sorted confusion counts using the frozen lower-tie selector semantics.

    This implementation deliberately uses a Python pointer over sorted pairs,
    rather than the training selector's vectorized cumulative/searchsorted path.
    Scores equal to a candidate remain positive (the decision rule is >=).
    A class with an F1 denominator of zero is absent from the macro average.
    """
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    require(y.ndim == p.ndim == 1 and len(y) == len(p), "Threshold vector shape mismatch")
    if not len(y):
        raise IncompleteAudit(
            "No observed validation labels; training rejects this selector branch"
        )
    require(set(y.tolist()) <= {0, 1}, "Threshold labels are not binary")
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)), "Invalid validation scores")
    ordered = sorted(zip(p.tolist(), y.tolist(), strict=True))
    candidates = sorted({0.0, 1.0, *p.tolist()})
    tp, fp, tn, fn = int(y.sum()), len(y) - int(y.sum()), 0, 0
    cursor, best_score, selected = 0, -1.0, None
    for threshold in candidates:
        while cursor < len(ordered) and ordered[cursor][0] < threshold:
            if ordered[cursor][1] == 1:
                tp, fn = tp - 1, fn + 1
            else:
                fp, tn = fp - 1, tn + 1
            cursor += 1
        class_scores = []
        for correct, denominator in ((tn, 2 * tn + fp + fn), (tp, 2 * tp + fp + fn)):
            if denominator:
                class_scores.append(2 * correct / denominator)
        macro = sum(class_scores) / len(class_scores)
        if macro > best_score + 1e-12:
            best_score = macro
            selected = {
                "threshold": float(threshold),
                "macro_f1": macro,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
    return {
        **selected,
        "n_observed": len(y),
        "positive": int(y.sum()),
        "candidate_count": len(candidates),
        "tie_tolerance": 1e-12,
        "selector": "ascending candidates {0, unique(scores), 1}; improve > 1e-12; lower tie",
        "absent_class_rule": "omit class when 2*correct+FP+FN is zero",
    }


def verify_dev_thresholds(messages: list[dict], bags: list[dict], thresholds: dict) -> dict:
    """Prove each of five frozen thresholds came from all observed dev gold."""
    require(
        all(row["split"] == "dev" for row in messages + bags),
        "Threshold verification accepts validation predictions only",
    )
    evidence = {}
    for name in CATEGORIES:
        observed = [row for row in messages if category(row, "observed_mask", name)]
        evidence[name] = independent_dev_threshold(
            [category(row, "labels", name) for row in observed],
            [category(row, "scores", name) for row in observed],
        )
    for name, score in (("conversation", "score"), ("conversation_common16", "common16_score")):
        evidence[name] = independent_dev_threshold(
            [row["label"] for row in bags], [row[score] for row in bags]
        )
    for name, proof in evidence.items():
        difference = abs(proof["threshold"] - float(thresholds[name]))
        require(
            difference <= TOLERANCE, f"Frozen {name} threshold was not selected by dev macro-F1"
        )
        proof.update(frozen_threshold=float(thresholds[name]), threshold_abs_difference=difference)
    return {
        "status": "passed",
        "all_sources_pooled_with_observed_masks": True,
        "independent_of_training_selector": True,
        "thresholds": evidence,
        "threshold_comparison_tolerance": TOLERANCE,
    }


def verify_prediction_cohort(
    records: list[dict],
    sessions: list[dict],
    messages: list[dict],
    bags: list[dict],
    thresholds: dict,
    split: str,
) -> dict:
    expected = {r["record_id"]: r for r in records if r["split"] == split}
    actual = {r["record_id"]: r for r in messages}
    require(len(actual) == len(messages) and set(actual) == set(expected), "Prediction IDs differ")
    fields = (
        "source",
        "split",
        "session_id",
        "message_id",
        "labels",
        "observed_mask",
        "duplicate_component",
        "eval_group",
        "parent_id",
        "order",
        "is_fallback",
    )
    for rid, original in expected.items():
        row = actual[rid]
        for key in fields:
            require(row.get(key) == original.get(key), f"Prediction {rid} changed {key}")
        require(row["text_length"] == len(original["text"]), f"Text length changed: {rid}")
        for name in CATEGORIES:
            score = float(category(row, "scores", name))
            require(np.isfinite(score) and 0 <= score <= 1, f"Invalid score: {rid}/{name}")
    expected_bags = {s["session_id"]: s for s in sessions if s["split"] == split}
    actual_bags = {s["session_id"]: s for s in bags}
    require(
        len(actual_bags) == len(bags) and set(actual_bags) == set(expected_bags),
        "Conversation prediction IDs differ",
    )
    for sid, original in expected_bags.items():
        row = actual_bags[sid]
        require(row["split"] == split and row["label"] == original["label"], "Bag gold changed")
        for score, threshold, key in (
            ("score", "threshold", "conversation"),
            ("common16_score", "common16_threshold", "conversation_common16"),
        ):
            require(float(row[threshold]) == float(thresholds[key]), "Frozen bag threshold changed")
            require(np.isfinite(row[score]) and 0 <= row[score] <= 1, "Invalid bag score")
    return {"messages": len(messages), "sessions": len(bags), "metadata_and_masks_exact": True}


def _head_signature(heads: list[dict], records: list[dict], allowed: set[str], seed: int) -> str:
    by_id = {r["record_id"]: r for r in records}
    signatures = []
    require([h["category"] for h in heads] == list(CATEGORIES), "Head categories changed")
    for j, head in enumerate(heads):
        ids, weights = head["observed_record_ids"], np.asarray(head["sample_weight"], dtype=float)
        expected = {rid for rid in allowed if by_id[rid]["observed_mask"][j]}
        require(len(ids) == len(set(ids)) and set(ids) == expected, "Observed head IDs differ")
        require(head["seed"] == seed and len(weights) == len(ids), "Head seed/weights mismatch")
        require(np.isfinite(weights).all() and np.all(weights > 0), "Invalid training weights")
        if ids:
            counts = Counter((by_id[rid]["source"], by_id[rid]["labels"][j]) for rid in ids)
            classes = {
                source: {label for s, label in counts if s == source} for source, _ in counts
            }
            raw = np.array(
                [
                    1
                    / (
                        len(classes[by_id[rid]["source"]])
                        * counts[by_id[rid]["source"], by_id[rid]["labels"][j]]
                    )
                    for rid in ids
                ]
            )
            expected_weights = raw * (len(raw) / raw.sum())
            require(
                np.max(np.abs(weights - expected_weights)) <= TOLERANCE,
                "Source/class weights differ from independent formula",
            )
        require(head["n_observed"] == len(ids), "Observed count mismatch")
        signatures.append({"category": head["category"], "ids": ids, "weights": weights.tolist()})
    return identity(signatures)


def verify_fit_details(
    details: dict, records: list[dict], sessions: list[dict], method: str, seed: int, threads: int
) -> dict:
    audit, parameters = details["fit_audit"], details["parameters"]
    require(
        parameters["method"] == method and parameters["seed"] == seed,
        "Method/seed not propagated to parameters",
    )
    require(
        parameters["threads"] == threads and parameters["folds"] == 3, "CPU/fold protocol changed"
    )
    require(
        audit["seed"] == seed and audit["method"] == method and audit["threads"] == threads,
        "Method/seed not propagated to fit audit",
    )
    require(audit["test_records_used_for_fit"] == 0, "Test records used for fitting")
    train = {
        r["record_id"] for r in records if r["split"] == "train" and r.get("fit_eligible", True)
    }
    fit_ids = audit["representation_fit_record_ids"]
    require(
        len(fit_ids) == len(set(fit_ids)) and set(fit_ids) == train,
        "Representation fit is not exactly eligible train records",
    )
    final_signature = _head_signature(audit["head_audits"]["final"], records, train, seed)
    train_sessions = [s for s in sessions if s["split"] == "train"]
    labels = np.array([s["label"] for s in train_sessions])
    expected_folds = list(
        StratifiedKFold(3, shuffle=True, random_state=seed).split(np.zeros(len(labels)), labels)
    )
    require(len(audit["folds"]) == 3, "Expected three OOF folds")
    coverage, fold_signatures = [], []
    for fold, (_, held_indices) in enumerate(expected_folds):
        held = {train_sessions[i]["session_id"] for i in held_indices}
        evidence = audit["folds"][fold]
        require(evidence["fold"] == fold and evidence["seed"] == seed, "OOF seed/fold changed")
        require(set(evidence["held_session_ids"]) == held, "Held sessions do not match frozen seed")
        held_records = [r for r in records if r["source"] == "sccd" and r["session_id"] in held]
        held_ids = {r["record_id"] for r in held_records}
        components = {r["duplicate_component"] for r in held_records}
        allowed = {
            r["record_id"]
            for r in records
            if r["record_id"] in train
            and not (r["source"] == "sccd" and r["session_id"] in held)
            and r["duplicate_component"] not in components
        }
        require(set(evidence["held_record_ids"]) == held_ids, "Held fold omitted SCCD comments")
        require(
            set(evidence["held_duplicate_components"]) == components,
            "Held duplicate catalog differs",
        )
        require(
            set(evidence["allowed_record_ids"]) == allowed
            and evidence["n_allowed"] == len(allowed),
            "OOF allowed rows leak held comments or duplicate components",
        )
        fold_signatures.append(
            _head_signature(audit["head_audits"][f"fold_{fold}"], records, allowed, seed)
        )
        coverage.extend(evidence["held_session_ids"])
    require(
        Counter(coverage) == Counter(s["session_id"] for s in train_sessions),
        "OOF folds do not cover every train conversation exactly once",
    )
    expected_pca = seed if method in {"cst_v1_1", "minilm_lr"} else None
    require(audit["pca_seed"] == expected_pca, "PCA random seed changed")
    if audit.get("backend"):
        require(audit["backend"]["seed"] == seed, "Topic backend random seed changed")
    return {
        "training_ids_signature": identity(fit_ids),
        "final_heads_signature": final_signature,
        "oof_signatures": fold_signatures,
        "train_records": len(train),
        "train_sessions": len(train_sessions),
        "fold_coverage_exact": True,
    }


def task_metrics(messages: list[dict], bags: list[dict], thresholds: dict) -> tuple[dict, dict]:
    metrics, strata = {}, {}
    for source, name in MESSAGE_TASKS:
        rows = [r for r in messages if r["source"] == source and category(r, "observed_mask", name)]
        task = source + "." + name
        metrics[task] = independent_metrics(
            [category(r, "labels", name) for r in rows],
            [category(r, "scores", name) for r in rows],
            thresholds[name],
        )
        for stratum in ("le_30_chars", "gt_30_chars"):
            chosen = [r for r in rows if (r["text_length"] <= 30) == (stratum == "le_30_chars")]
            strata[task, stratum] = independent_metrics(
                [category(r, "labels", name) for r in chosen],
                [category(r, "scores", name) for r in chosen],
                thresholds[name],
            )
    for task, score in (("conversation", "score"), ("conversation_common16", "common16_score")):
        metrics["sccd." + task] = independent_metrics(
            [r["label"] for r in bags], [r[score] for r in bags], thresholds[task]
        )
    return metrics, strata


def independent_top3(messages: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in messages:
        if row["source"] == "sccd":
            groups[row["session_id"]].append(row)
    positive = hit = dangerous = found = selected_total = 0
    for rows in groups.values():
        real = [r for r in rows if not r["is_fallback"]]
        gold = {
            r["record_id"]
            for r in real
            if category(r, "observed_mask", "attack") and category(r, "labels", "attack") == 1
        }
        ranked = sorted(
            real,
            key=lambda r: (
                -float(category(r, "scores", "attack")),
                r["order"],
                r["message_id"],
                r["record_id"],
            ),
        )[:3]
        chosen = {r["record_id"] for r in ranked}
        positive += bool(gold)
        hit += bool(gold & chosen)
        dangerous += len(gold)
        found += len(gold & chosen)
        selected_total += len(chosen)
    return {
        "n_sessions": len(groups),
        "positive_gold_sessions": positive,
        "hit_sessions": hit,
        "hit_at_3": hit / positive if positive else None,
        "dangerous_gold_comments": dangerous,
        "dangerous_selected_comments": found,
        "dangerous_comment_recall_at_3": found / dangerous if dangerous else None,
        "selected_comments_all_sessions": selected_total,
    }


def cluster_counts(messages: list[dict]) -> dict:
    parent, nodes = {}, {}

    def find(node):
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for row in messages:
        group = row["session_id"] if row["source"] == "sccd" else row["eval_group"]
        node = f"group:{row['source']}:{group}"
        nodes[row["record_id"]] = node
        other = "duplicate:" + row["duplicate_component"]
        left, right = find(node), find(other)
        parent[right] = left
    result = {}
    for source, name in MESSAGE_TASKS:
        result[source + "." + name] = len(
            {
                find(nodes[r["record_id"]])
                for r in messages
                if r["source"] == source and category(r, "observed_mask", name)
            }
        )
    result["sccd.conversation"] = len(
        {find(nodes[r["record_id"]]) for r in messages if r["source"] == "sccd"}
    )
    result["sccd.conversation_common16"] = result["sccd.conversation"]
    return result


def _csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _compare(actual: dict, expected: dict, fields) -> float:
    maximum = 0.0
    for field in fields:
        observed, target = actual[field], expected[field]
        if target is None:
            require(observed in (None, ""), f"Expected missing {field}, got {observed}")
        else:
            difference = abs(float(observed) - float(target))
            require(
                np.isfinite(difference) and difference <= TOLERANCE, f"Metric mismatch: {field}"
            )
            maximum = max(maximum, difference)
    return maximum


def audit(output: Path, *, partial: bool = False) -> dict:
    """Write requirement-by-requirement proof; missing evidence is never a pass."""
    output = Path(output).resolve()
    checks, run_evidence, expected_metrics, expected_strata, expected_top3 = [], [], {}, {}, {}
    fit_signatures, reference_test, state_hashes = {}, None, {}

    def check(name, function):
        try:
            proof = function()
        except (FileNotFoundError, IncompleteAudit) as error:
            checks.append({"requirement": name, "status": "incomplete", "reason": str(error)})
        except Exception as error:
            checks.append(
                {
                    "requirement": name,
                    "status": "failed",
                    "reason": str(error),
                    "exception": type(error).__name__,
                }
            )
        else:
            checks.append({"requirement": name, "status": "passed", "proof": proof})

    prerequisite = [
        output / "experiment.json",
        output / "prepared/records.jsonl",
        output / "prepared/sessions.jsonl",
    ]
    missing = [str(path) for path in prerequisite if not path.is_file()]
    if missing:
        result = {
            "schema_version": 1,
            "status": "incomplete",
            "partial_requested": partial,
            "completed_runs_audited": 0,
            "required_runs": 21,
            "requirement_counts": {"incomplete": 1},
            "requirements": [
                {
                    "requirement": "campaign_prerequisites",
                    "status": "incomplete",
                    "missing": missing,
                }
            ],
            "verification_code_sha256": digest(Path(__file__)),
            "no_model_refitting_or_prediction_reexecution": True,
        }
        save(output / "completion_audit.json", result)
        return result
    experiment = read(prerequisite[0])
    records, sessions = read_jsonl(prerequisite[1]), read_jsonl(prerequisite[2])

    def prepared_check():
        from .data import load_prepared

        rows, bags, manifest = load_prepared(output)
        verify_hashes(experiment["code_hashes"])
        verify_hashes(experiment["encoder"]["asset_hashes"])
        require(
            digest(output / "prepared/manifest.json") == experiment["dataset_sha256"],
            "Dataset freeze changed",
        )
        require(
            manifest["fingerprint"] == experiment["dataset_fingerprint"], "Dataset identity changed"
        )
        require(
            Counter(s["split"] for s in bags) == {"train": 406, "dev": 135, "test": 136},
            "SCCD split changed",
        )
        return {"records": len(rows), "sessions": len(bags), "input_and_code_hashes_verified": True}

    check("frozen_data_code_encoder_and_406_135_136_split", prepared_check)
    methods, seeds = experiment["methods"], experiment["seeds"]
    require(
        len(methods) == 7 and seeds == [42, 43, 44], "Expected seven methods and three fixed seeds"
    )
    for method in methods:
        for seed in seeds:
            folder = output / "runs" / method / f"seed_{seed}"

            def run_check(folder=folder, method=method, seed=seed):
                nonlocal reference_test
                run_id = identity(
                    {"experiment": identity(experiment), "method": method, "seed": seed}
                )
                manifest = read(folder / "run_manifest.json")
                require(
                    manifest["identity"] == run_id
                    and manifest["method"] == method
                    and manifest["seed"] == seed,
                    "Run identity changed",
                )
                for stage in ("train", "test"):
                    state_path = folder / f"{stage}_state.json"
                    seal = read(state_path)
                    require(seal["identity"] == run_id, "Run seal identity changed")
                    if seal["status"] != "complete":
                        raise IncompleteAudit(f"{method}/{seed}/{stage} is not sealed complete")
                    required_artifacts = (
                        [
                            "model.joblib",
                            "run_manifest.json",
                            "fit_details.json",
                            "metrics.json",
                            "predictions_message_dev.jsonl",
                            "predictions_session_dev.jsonl",
                        ]
                        if stage == "train"
                        else [
                            "test_freeze.json",
                            "predictions_message_test.jsonl",
                            "predictions_session_test.jsonl",
                        ]
                    )
                    require(
                        {str((folder / name).resolve()) for name in required_artifacts}
                        <= set(seal["artifacts"]),
                        "Run seal omitted required model/prediction evidence",
                    )
                    verify_hashes(seal["artifacts"])
                    state_hashes[str(state_path)] = digest(state_path)
                require(
                    digest(folder / "model.joblib") == manifest["model_sha256"],
                    "Model seal changed",
                )
                test_freeze = read(folder / "test_freeze.json")
                require(
                    test_freeze["identity"] == run_id
                    and test_freeze["model_sha256"] == manifest["model_sha256"]
                    and test_freeze["thresholds"] == manifest["thresholds"],
                    "Test threshold/model freeze changed",
                )
                proof = verify_fit_details(
                    read(folder / "fit_details.json"),
                    records,
                    sessions,
                    method,
                    seed,
                    experiment["threads"],
                )
                signature = {
                    k: proof[k]
                    for k in ("training_ids_signature", "final_heads_signature", "oof_signatures")
                }
                if seed in fit_signatures:
                    require(
                        signature == fit_signatures[seed],
                        "Methods used unequal training IDs, OOF folds, or weights",
                    )
                else:
                    fit_signatures[seed] = signature
                for split in ("dev", "test"):
                    messages = read_jsonl(folder / f"predictions_message_{split}.jsonl")
                    bags = read_jsonl(folder / f"predictions_session_{split}.jsonl")
                    verify_prediction_cohort(
                        records, sessions, messages, bags, manifest["thresholds"], split
                    )
                    if split == "dev":
                        proof["validation_threshold_selection"] = verify_dev_thresholds(
                            messages, bags, manifest["thresholds"]
                        )
                    metrics, strata = task_metrics(messages, bags, manifest["thresholds"])
                    expected_metrics.update(
                        {(method, seed, split, task): value for task, value in metrics.items()}
                    )
                    expected_strata.update(
                        {
                            (method, seed, split, task, name): value
                            for (task, name), value in strata.items()
                        }
                    )
                    expected_top3[method, seed, split] = independent_top3(messages)
                    if split == "test" and reference_test is None:
                        reference_test = messages
                run_evidence.append({"method": method, "seed": seed})
                return {
                    "method": method,
                    "seed": seed,
                    **proof,
                    "predictions_and_thresholds_exact": True,
                }

            check(f"run_{method}_{seed}_identity_supervision_oof_predictions", run_check)

        def cost_check(method=method):
            folder = output / "runs" / method / "seed_42"
            seal, cost = read(folder / "cost_state.json"), read(folder / "cost.json")
            manifest = read(folder / "run_manifest.json")
            expected = identity(
                {
                    "run": manifest["identity"],
                    "model": manifest["model_sha256"],
                    "cost_code": digest(Path(__file__).with_name("cost.py")),
                    "repetitions": 3,
                    "threads": experiment["threads"],
                    "batch": "SCCD-session/other64",
                    "test_seal_sha256": digest(folder / "test_state.json"),
                    "score_tolerance": 1e-4,
                }
            )
            require(cost["identity"] == seal["identity"] == expected, "Cost identity changed")
            if seal["status"] != "complete" or cost["status"] != "complete":
                raise IncompleteAudit(f"{method} cost is not complete")
            require(
                str((folder / "cost.json").resolve()) in seal["artifacts"],
                "Cost seal omitted cost evidence",
            )
            verify_hashes(seal["artifacts"])
            require(
                cost["repetitions"] == 3 and len(cost["warm_total_seconds"]) == 3,
                "Cost repetitions changed",
            )
            require(
                cost["seed"] == 42
                and cost["method"] == method
                and cost["paid_API_calls"] == 0
                and cost["GPU_required"] is False,
                "Cost execution policy changed",
            )
            require(
                cost["resources"]["peak_affinity_width"] <= experiment["threads"],
                "CPU budget exceeded",
            )
            require(
                max(
                    cost["prediction_consistency"]["max_message_abs_diff"],
                    cost["prediction_consistency"]["max_session_abs_diff"],
                )
                <= 1e-4,
                "Cost prediction drift",
            )
            for repetition in (1, 2, 3):
                rows = [r for r in cost["rows"] if r["repetition"] == repetition]
                counts = Counter()
                for row in rows:
                    counts[row["source"]] += row["n_records"]
                require(
                    counts == Counter(r["source"] for r in records if r["split"] == "test"),
                    "Cost skipped test data",
                )
                require(
                    abs(
                        sum(r["seconds"] for r in rows) - cost["warm_total_seconds"][repetition - 1]
                    )
                    < 1e-8,
                    "Cost total differs from measured batches",
                )
            return {"seed": 42, "repetitions": 3, "all_test_records_measured": True}

        check(f"cost_{method}_identity_full_test_three_repetitions", cost_check)

    def metrics_check():
        if len(run_evidence) != 21:
            raise IncompleteAudit("All 21 runs required for final metric verification")
        maximum = 0.0
        for filename, expected, extra in (
            ("per_seed_metrics.csv", expected_metrics, ()),
            ("length_strata.csv", expected_strata, ("stratum",)),
        ):
            rows = _csv(output / "analysis" / filename)
            actual = {
                (r["method"], int(r["seed"]), r["split"], r["task"], *(r[k] for k in extra)): r
                for r in rows
            }
            require(
                len(actual) == len(rows) and set(actual) == set(expected),
                f"Metric CSV cohort changed: {filename}",
            )
            for key, target in expected.items():
                maximum = max(
                    maximum, _compare(actual[key], target, ("n", "positive", "threshold", *METRICS))
                )
        rows = _csv(output / "analysis/top3.csv")
        actual = {(r["method"], int(r["seed"]), r["split"]): r for r in rows}
        require(
            set(actual) == set(expected_top3) and len(actual) == len(rows),
            "Top3 result coverage changed",
        )
        for key, target in expected_top3.items():
            maximum = max(maximum, _compare(actual[key], target, target.keys()))
        return {
            "max_abs_difference": maximum,
            "tolerance": TOLERANCE,
            "per_seed_tasks": len(expected_metrics),
            "length_strata": len(expected_strata),
            "top3_rows": len(expected_top3),
            "independent_sklearn": True,
        }

    check("independent_metrics_length_strata_and_top3", metrics_check)

    def bootstrap_check():
        analysis = read(output / "analysis/analysis.json")
        require(
            analysis["status"] == "complete"
            and analysis["methods"] == methods
            and analysis["seeds"] == seeds,
            "Analysis run set changed",
        )
        if analysis["missing_costs"]:
            raise IncompleteAudit("Analysis is missing costs")
        require(reference_test is not None, "No complete test predictions")
        counts = cluster_counts(reference_test)
        require(set(counts) == set(analysis["bootstrap"]), "Bootstrap task set changed")
        for task, n_clusters in counts.items():
            meta = analysis["bootstrap"][task]
            require(
                meta["requested"] == 2000 and meta["n_clusters"] == n_clusters,
                "Bootstrap count/clusters changed",
            )
            require(
                meta["seed_aggregation"]
                == "mean of per-seed metrics, never metric of averaged scores",
                "Bootstrap used a seed ensemble",
            )
            require(
                meta["attempted"] == (2000 if n_clusters >= 2 else 0),
                "Invalid bootstrap attempt count",
            )
            with np.load(
                output / "analysis" / f"bootstrap_{task.replace('.', '_')}.npz", allow_pickle=False
            ) as stored:
                require(set(stored.files) == set(methods), "Bootstrap methods changed")
                for method in methods:
                    require(stored[method].shape == (2000, 5), "Bootstrap array shape changed")
                valid = int(np.isfinite(stored[methods[0]][:, 0]).sum())
                require(valid == meta["valid"], "Valid bootstrap replicate count changed")
                require(
                    meta["valid"] + meta["degenerate"] == meta["attempted"],
                    "Bootstrap attempt accounting changed",
                )
        for summary in analysis["summary"]:
            values = [
                expected_metrics[summary["method"], seed, summary["split"], summary["task"]]
                for seed in seeds
            ]
            require(summary["n_seeds"] == 3, "Summary seed count changed")
            for metric in METRICS:
                observed = [v[metric] for v in values if v[metric] is not None]
                expected = {
                    "mean": float(np.mean(observed)) if observed else None,
                    "sd": float(np.std(observed, ddof=1))
                    if len(observed) > 1
                    else (0.0 if observed else None),
                }
                _compare(
                    {"mean": summary[metric + "_mean"], "sd": summary[metric + "_sd"]},
                    expected,
                    expected.keys(),
                )
        for task, selected in analysis["selected_baselines"].items():
            baselines = methods[:5]

            def validation_ap(method, task=task):
                values = [expected_metrics[method, seed, "dev", task]["AP"] for seed in seeds]
                return float(np.mean(values)) if all(v is not None for v in values) else -1.0

            winner = max(
                baselines, key=lambda method: (validation_ap(method), -baselines.index(method))
            )
            require(
                selected["method"] == winner, "Topic baseline was not selected by validation AP"
            )
            require(
                abs(selected["validation_mean_AP"] - validation_ap(winner)) <= TOLERANCE,
                "Selected baseline validation score changed",
            )
        verification = read(output / "analysis/verification.json")
        require(
            verification["no_seed_score_ensemble"] is True and verification["run_count"] == 21,
            "Analysis ensemble declaration changed",
        )
        return {
            "requested_per_task": 2000,
            "independent_cluster_counts": counts,
            "three_seed_metric_mean_verified": True,
        }

    check("bootstrap_dependency_clusters_and_no_seed_ensemble", bootstrap_check)

    def figures_check():
        manifest = read(output / "figures/figure_manifest.json")
        require(
            manifest["source_sha256"] == digest(output / "analysis/analysis.json"),
            "Figure data changed",
        )
        require(
            manifest["figure_source_sha256"] == digest(Path(__file__).with_name("figures.py")),
            "Figure code changed",
        )
        for name in ("performance", "costs"):
            for suffix in ("png", "pdf", "svg"):
                path = output / "figures" / f"{name}.{suffix}"
                if not path.is_file():
                    raise IncompleteAudit(f"Missing figure: {path}")
                require(path.stat().st_size > 100, f"Empty figure: {path}")
            qa = output / "figures/qa"
            results = read(qa / f"{name}.qa.json")
            require(
                {r["audit"] for r in results} == {"source", "font", "collisions"}
                and all(r["returncode"] == 0 for r in results),
                "Figure QA did not pass",
            )
            for kind in ("alignment", "collisions"):
                report = read(qa / f"{name}.{kind}.json")
                require(
                    report["verdict"] == "PASS"
                    and report["summary"]["fail"] == 0
                    and report["summary"]["warn"] == 0,
                    f"Figure {name} {kind} still has unresolved findings",
                )
        return {
            "figures": 2,
            "formats": ["PNG", "PDF", "SVG"],
            "source_font_alignment_collision_QA": True,
            "human_visual_inspection": "separate user-facing visual review required",
        }

    check("figure_exports_and_machine_qa", figures_check)

    def preservation_check():
        before, after = (
            read(output / "preservation_before.json"),
            read(output / "preservation_after.json"),
        )
        require(
            after["unchanged"] is True and after["checked_files"] == len(before),
            "Preservation verification incomplete",
        )
        verify_hashes(before)
        deployment = [
            path
            for path in before
            if path.replace("\\", "/").endswith("/app/config/deployment.json")
        ]
        require(len(deployment) == 1, "Default deployment is missing from preservation evidence")
        return {
            "old_files_verified": len(before),
            "default_deployment_sha256": before[deployment[0]],
        }

    check("old_evidence_and_default_deployment_unchanged", preservation_check)
    statuses = Counter(row["status"] for row in checks)
    status = (
        "failed" if statuses["failed"] else "incomplete" if statuses["incomplete"] else "passed"
    )
    if partial and len(run_evidence) < 21 and status != "failed":
        status = "incomplete"
    result = {
        "schema_version": 1,
        "status": status,
        "partial_requested": partial,
        "completed_runs_audited": len(run_evidence),
        "required_runs": 21,
        "requirement_counts": dict(statuses),
        "requirements": checks,
        "verification_code_sha256": digest(Path(__file__)),
        "run_seal_sha256": state_hashes,
        "no_model_refitting_or_prediction_reexecution": True,
    }
    save(output / "completion_audit.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.output, partial=args.partial)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "completed_runs_audited", "requirement_counts")}
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
