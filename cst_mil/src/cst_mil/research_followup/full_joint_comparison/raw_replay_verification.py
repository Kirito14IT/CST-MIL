"""Independent audit of a raw-inference evaluation revision of frozen weights.

Training is proved exclusively by original train seals. Evaluation-view manifests
and revision dev/test seals are deliberately never accepted as training seals.
No model is loaded, fitted, or used to regenerate predictions here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from . import verification as independent
from .common import digest, identity, read, read_jsonl, save, verify_hashes

SEMANTIC = frozenset({"bertopic", "cst_v1_1", "minilm_lr"})
TRADITIONAL = frozenset({"lda", "btm", "gsdmm", "nmf"})
GLOBAL_GROUP_PROTOCOL = "frozen-global-eval-group-closure-v1"
IncompleteAudit = independent.IncompleteAudit
require = independent.require


def verify_frozen_global_groups(records: list[dict]) -> dict:
    """Independently verify frozen IDs by bipartite graph traversal, without scores.

    Every retained train/dev/test record participates, including fully unknown
    labels. This does not redefine the already sealed prepared dependency IDs.
    The analysis implementation uses union-find; this audit uses graph traversal.
    """
    base_components, component_bases = defaultdict(set), defaultdict(set)
    groups_by_id, source_sets, seen = {}, defaultdict(set), set()
    for row in records:
        rid, source = row["record_id"], row["source"]
        require(rid not in seen, "Global dependency audit found duplicate record IDs")
        seen.add(rid)
        base, component = row["base_eval_group"], row["duplicate_component"]
        require(
            base.startswith(source + ":") and row["eval_group"],
            "Global dependency group lost genuine source identity",
        )
        if source == "sccd":
            require(
                base == "sccd:session:" + str(row["session_id"]),
                "Global dependency group does not identify the genuine SCCD session",
            )
        base_components[base].add(component)
        component_bases[component].add(base)
        groups_by_id[rid] = row["eval_group"]
        source_sets[row["eval_group"]].add(source)
    pending, expected, visited_components = set(base_components), {}, set()
    count = 0
    while pending:
        frontier, members = [pending.pop()], set()
        while frontier:
            base = frontier.pop()
            if base in members:
                continue
            members.add(base)
            pending.discard(base)
            for component in base_components[base]:
                if component in visited_components:
                    continue
                visited_components.add(component)
                frontier.extend(component_bases[component] - members)
        frozen_id = "eval:" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()
        expected.update({base: frozen_id for base in members})
        count += 1
    for row in records:
        require(
            groups_by_id[row["record_id"]] == expected[row["base_eval_group"]],
            "Frozen eval_group differs from the complete all-source train/dev/test closure",
        )
    return {
        "protocol": GLOBAL_GROUP_PROTOCOL,
        "verified": True,
        "records": len(records),
        "base_groups": len(base_components),
        "global_groups": count,
        "cross_source_global_groups": sum(len(sources) > 1 for sources in source_sets.values()),
        "unknown_records_included": sum(not any(row["observed_mask"]) for row in records),
        "scope": "all retained prepared train/dev/test rows before any score filtering",
        "global_ids_redefined": False,
        "algorithm": "independent bipartite base-group/duplicate-component graph traversal",
    }


def frozen_global_cluster_profiles(messages: list[dict]) -> dict:
    """Describe observed task subsets by authoritative cross-source frozen IDs."""
    duplicates, sessions, ids = {}, {}, set()
    for row in messages:
        require(row["record_id"] not in ids, "Prediction dependency audit found duplicate IDs")
        ids.add(row["record_id"])
        group = row["eval_group"]
        require(bool(group), "Prediction lacks its frozen global eval_group")
        previous = duplicates.setdefault(row["duplicate_component"], group)
        require(previous == group, "Cross-source duplicate dependency was split")
        if row["source"] == "sccd":
            previous = sessions.setdefault(row["session_id"], group)
            require(previous == group, "Genuine SCCD session dependency was split")
    task_counts = {
        source + "." + name: Counter(
            row["eval_group"]
            for row in messages
            if row["source"] == source and independent.category(row, "observed_mask", name)
        )
        for source, name in independent.MESSAGE_TASKS
    }
    task_counts["sccd.conversation"] = task_counts["sccd.conversation_common16"] = Counter(
        sessions.values()
    )
    result = {}
    for task, counts in task_counts.items():
        total, largest = sum(counts.values()), max(counts.values(), default=0)
        concentration = sum((size / total) ** 2 for size in counts.values()) if total else None
        result[task] = {
            "n_clusters": len(counts),
            "largest_cluster_n": largest,
            "largest_cluster_fraction": largest / total if total else None,
            "size_concentration_effective_cluster_count": 1 / concentration
            if concentration
            else None,
        }
    return result


def frozen_global_cluster_counts(messages: list[dict]) -> dict:
    """Count tasks without source-prefix splitting or score-conditioned grouping."""
    return {
        task: row["n_clusters"] for task, row in frozen_global_cluster_profiles(messages).items()
    }


def replay_source_metadata(revision: dict, *, module_dir: Path | None = None) -> dict:
    """Resolve the sole execution source sealed by this evaluation revision."""
    directory = Path(module_dir or Path(__file__).parent).resolve()
    candidates = [
        Path(path).resolve()
        for path in revision["code_hashes"]
        if re.fullmatch(r"raw_replay(?:_v[0-9]+)?\.py", Path(path).name)
    ]
    require(len(candidates) == 1, "Revision must bind exactly one raw replay execution source")
    path = candidates[0]
    require(path.parent == directory, "Raw replay execution source is outside the audited module")
    sha = digest(path)
    require(
        revision["code_hashes"][str(path)] == sha, "Raw replay execution source fingerprint changed"
    )
    name = "raw_replay_v1" if path.stem == "raw_replay" else path.stem
    return {
        "execution_source_path": str(path),
        "execution_source_sha256": sha,
        "evaluation_revision": name,
    }


def verify_evaluation_view(
    view: dict, original: dict, base_folder: Path, revision_identity: str
) -> dict:
    """Only evaluation thresholds may differ from the frozen training manifest."""
    require(view["manifest_role"] == "evaluation_view", "Revision manifest claims a training role")
    require(view["inherited_training"] is True, "Revision falsely claims newly trained weights")
    require(
        view["base_run_identity"] == original["identity"], "Inherited training identity changed"
    )
    require(view["identity"] == original["identity"], "Evaluation view replaced training identity")
    require(
        view["base_run_manifest_sha256"] == digest(base_folder / "run_manifest.json"),
        "Original run manifest provenance changed",
    )
    require(
        view["base_train_state_sha256"] == digest(base_folder / "train_state.json"),
        "Original training seal provenance changed",
    )
    require(
        Path(view["base_model_path"]).resolve() == (base_folder / "model.joblib").resolve(),
        "Revision changed the underlying model path",
    )
    require(
        view["model_sha256"] == original["model_sha256"] == digest(base_folder / "model.joblib"),
        "Revision changed the frozen weights",
    )
    require(
        view["evaluation_revision_identity"] == revision_identity, "Evaluation revision changed"
    )
    for key, value in original.items():
        if key not in {"thresholds", "identity"}:
            require(view.get(key) == value, f"Frozen training metadata changed: {key}")
    if view["method"] in TRADITIONAL:
        require(
            view["thresholds"] == original["thresholds"], "Inherited traditional thresholds changed"
        )
    require(
        set(view["thresholds"])
        == {"attack", "hate", "spam_scam", "conversation", "conversation_common16"},
        "Expected exactly five evaluation thresholds",
    )
    return {
        "weights_unchanged": True,
        "base_train_identity": original["identity"],
        "evaluation_role_explicit": True,
    }


def verify_revision_stage(
    folder: Path, revision_identity: str, method: str, seed: int, split: str
) -> dict:
    view_path = folder / "run_manifest.json"
    state_path = folder / f"{split}_state.json"
    seal = read(state_path)
    expected = identity(
        {
            "revision": revision_identity,
            "method": method,
            "seed": seed,
            "stage": split,
            "run_manifest_sha256": digest(view_path),
        }
    )
    require(seal["identity"] == expected, "Raw evaluation stage identity changed")
    if seal["status"] != "complete":
        raise IncompleteAudit(f"{method}/{seed}/{split} is not complete")
    files = (
        [folder / "cost.json", folder / "cost_provenance.json"]
        if split == "cost"
        else [folder / f"predictions_{unit}_{split}.jsonl" for unit in ("message", "session")]
    )
    require(
        {str(path.resolve()) for path in files} <= set(seal["artifacts"]),
        "Revision seal omitted complete predictions",
    )
    verify_hashes(seal["artifacts"])
    return seal


def independent_revision_threshold(labels, scores) -> dict:
    """Independent vectorized scan; raw executor uses the separate pointer scan."""
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    require(y.ndim == p.ndim == 1 and len(y) == len(p), "Threshold vector shape changed")
    if not len(y):
        raise IncompleteAudit("No observed DEV labels for independent threshold verification")
    require(
        set(y.tolist()) <= {0, 1} and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)),
        "Invalid independent threshold labels/scores",
    )
    order = np.argsort(p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    cumulative_positive = np.r_[0, np.cumsum(sorted_y)]
    candidates = np.unique(np.r_[0.0, p, 1.0])
    below = np.searchsorted(sorted_p, candidates, side="left")
    fn, tn = cumulative_positive[below], below - cumulative_positive[below]
    tp, fp = int(y.sum()) - fn, len(y) - int(y.sum()) - tn
    numerators = np.column_stack((2 * tn, 2 * tp))
    denominators = np.column_stack((2 * tn + fp + fn, 2 * tp + fp + fn))
    ratios = np.divide(
        numerators,
        denominators,
        out=np.zeros_like(numerators, dtype=float),
        where=denominators != 0,
    )
    macro = ratios.sum(axis=1) / np.count_nonzero(denominators, axis=1)
    winner, best = 0, -1.0
    for index, value in enumerate(macro):
        if value > best + 1e-12:
            winner, best = index, float(value)
    return {
        "threshold": float(candidates[winner]),
        "macro_f1": best,
        "n_observed": len(y),
        "positive": int(y.sum()),
        "candidate_count": len(candidates),
        "tie_tolerance": 1e-12,
    }


def verify_revision_thresholds(messages: list[dict], bags: list[dict], thresholds: dict) -> dict:
    require(
        all(r["split"] == "dev" for r in messages + bags), "Only DEV may select revision thresholds"
    )
    result = {}
    for name in independent.CATEGORIES:
        rows = [r for r in messages if independent.category(r, "observed_mask", name)]
        result[name] = independent_revision_threshold(
            [independent.category(r, "labels", name) for r in rows],
            [independent.category(r, "scores", name) for r in rows],
        )
    for name, score in (("conversation", "score"), ("conversation_common16", "common16_score")):
        result[name] = independent_revision_threshold(
            [r["label"] for r in bags], [r[score] for r in bags]
        )
    for name, proof in result.items():
        require(
            abs(proof["threshold"] - thresholds[name]) <= independent.TOLERANCE,
            f"Revision {name} threshold is not the DEV optimum",
        )
    return {
        "independent_vectorized_scan": True,
        "worker_pointer_scan_not_called": True,
        "thresholds": result,
    }


def verify_base_training(
    base: Path, experiment: dict, records: list[dict], sessions: list[dict], method: str, seed: int
) -> tuple[dict, dict]:
    folder = base / "runs" / method / f"seed_{seed}"
    original, seal = read(folder / "run_manifest.json"), read(folder / "train_state.json")
    expected = identity({"experiment": identity(experiment), "method": method, "seed": seed})
    require(original["identity"] == seal["identity"] == expected, "Base training identity changed")
    if seal["status"] != "complete":
        raise IncompleteAudit("Original training is not sealed complete")
    files = (
        "model.joblib",
        "run_manifest.json",
        "fit_details.json",
        "metrics.json",
        "predictions_message_dev.jsonl",
        "predictions_session_dev.jsonl",
    )
    require(
        {str((folder / name).resolve()) for name in files} <= set(seal["artifacts"]),
        "Base training seal omitted evidence",
    )
    verify_hashes(seal["artifacts"])
    require(
        digest(folder / "model.joblib") == original["model_sha256"], "Base model weights changed"
    )
    proof = independent.verify_fit_details(
        read(folder / "fit_details.json"), records, sessions, method, seed, experiment["threads"]
    )
    return original, proof


def verify_cost_coverage(cost: dict, records: list[dict], method: str, threads: int) -> dict:
    require(
        cost["status"] == "complete" and cost["method"] == method and cost["seed"] == 42,
        "Cost method/seed/status changed",
    )
    require(
        cost["repetitions"] == 3 and len(cost["warm_total_seconds"]) == 3,
        "Cost repetitions changed",
    )
    require(
        cost["GPU_required"] is False and cost["paid_API_calls"] == 0,
        "Cost used remote/GPU services",
    )
    require(cost["resources"]["peak_affinity_width"] <= threads, "Cost exceeded CPU budget")
    require(
        max(
            cost["prediction_consistency"]["max_message_abs_diff"],
            cost["prediction_consistency"]["max_session_abs_diff"],
        )
        <= 1e-4,
        "Cost raw predictions differ from their proper reference",
    )
    consistency = cost["prediction_consistency"]
    require(
        consistency["head_threshold_disagreements_all_repetitions"] == 0,
        "Cost head classifications changed across raw repetitions",
    )
    if method in SEMANTIC:
        require(
            consistency["session_threshold_disagreements_all_repetitions"]
            == {"main": 0, "common16": 0},
            "Cost conversation classifications changed across raw repetitions",
        )
    else:
        require(
            consistency["max_message_abs_diff"] == 0.0
            and consistency["max_session_abs_diff"] == 0.0,
            "Inherited traditional cost must have exact score consistency",
        )
    expected_counts = Counter(row["source"] for row in records if row["split"] == "test")
    for repetition in (1, 2, 3):
        batches = [row for row in cost["rows"] if row["repetition"] == repetition]
        observed = Counter()
        for row in batches:
            observed[row["source"]] += row["n_records"]
        require(observed == expected_counts, "Cost skipped or duplicated test records")
        require(
            abs(sum(row["seconds"] for row in batches) - cost["warm_total_seconds"][repetition - 1])
            < 1e-8,
            "Cost totals differ from measured batches",
        )
    return {
        "full_test_coverage": True,
        "warm_repetitions": 3,
        "seed": 42,
        "prediction_max_tolerance": 1e-4,
    }


def verify_no_consistency_retry(folder: Path, revision_identity: str) -> dict:
    for path in folder.rglob("cost_invalid.json"):
        invalid = read(path)
        require(
            not (
                invalid.get("status") == "invalid_prediction_consistency"
                and invalid.get("evaluation_revision_identity") == revision_identity
            ),
            f"Same-revision invalid consistency evidence forbids retry/acceptance: {path}",
        )
    return {"same_revision_invalid_consistency_retry": False}


def verify_metric_tables(
    output: Path, expected_metrics: dict, expected_strata: dict, expected_top3: dict
) -> dict:
    maximum = 0.0
    for filename, expected, extra in (
        ("per_seed_metrics.csv", expected_metrics, ()),
        ("length_strata.csv", expected_strata, ("stratum",)),
    ):
        rows = independent._csv(output / "analysis" / filename)
        actual = {
            (r["method"], int(r["seed"]), r["split"], r["task"], *(r[k] for k in extra)): r
            for r in rows
        }
        require(
            len(actual) == len(rows) and set(actual) == set(expected),
            "Revision metric coverage changed",
        )
        for key, target in expected.items():
            maximum = max(
                maximum,
                independent._compare(
                    actual[key], target, ("n", "positive", "threshold", *independent.METRICS)
                ),
            )
    rows = independent._csv(output / "analysis/top3.csv")
    actual = {(r["method"], int(r["seed"]), r["split"]): r for r in rows}
    require(
        len(actual) == len(rows) and set(actual) == set(expected_top3),
        "Revision Top3 coverage changed",
    )
    for key, target in expected_top3.items():
        maximum = max(maximum, independent._compare(actual[key], target, target.keys()))
    return {
        "max_abs_difference": maximum,
        "tolerance": independent.TOLERANCE,
        "per_seed_task_count": len(expected_metrics),
        "length_stratum_count": len(expected_strata),
        "top3_count": len(expected_top3),
        "independent_sklearn": True,
    }


def expected_selected_baselines(
    expected_metrics: dict, methods: list[str], seeds: list[int]
) -> dict:
    tasks = sorted({key[3] for key in expected_metrics if key[2] == "dev"})
    selected = {}
    baselines = methods[:5]
    for task in tasks:

        def average(method, task=task):
            values = [expected_metrics[method, seed, "dev", task]["AP"] for seed in seeds]
            return float(np.mean(values)) if all(v is not None for v in values) else -1.0

        winner = max(baselines, key=lambda method: (average(method), -baselines.index(method)))
        selected[task] = {"method": winner, "validation_mean_AP": average(winner)}
    return selected


def verify_dependency_input_hashes(output: Path, dependence: dict, declaration: dict) -> dict:
    """Read platform-native relative keys while still requiring unique sealed hashes."""
    declared = declaration["prediction_and_manifest_hashes"]
    proof = {}
    for name, sha in (
        ("prepared/records.jsonl", dependence["prepared_records_sha256"]),
        ("prepared/manifest.json", dependence["prepared_manifest_sha256"]),
    ):
        keys = [key for key in declared if key.replace("\\", "/") == name]
        require(len(keys) == 1, f"Expected one dependency input fingerprint key: {name}")
        require(
            sha == digest(output / name) == declared[keys[0]],
            "Analysis dependency metadata is not bound to the frozen prepared records",
        )
        proof[name] = sha
    return proof


def verify_bootstrap_and_summary(
    output: Path,
    expected_metrics: dict,
    reference_test: list[dict],
    methods: list[str],
    seeds: list[int],
    frozen_baselines: dict,
) -> dict:
    analysis = read(output / "analysis/analysis.json")
    require(
        analysis["status"] == "complete"
        and analysis["methods"] == methods
        and analysis["seeds"] == seeds,
        "Revision analysis method/seed set changed",
    )
    if analysis["missing_costs"]:
        raise IncompleteAudit("Revision analysis has missing costs")
    dependence = analysis["dependence_protocol"]
    require(
        dependence["protocol"] == GLOBAL_GROUP_PROTOCOL
        and dependence["verified"] is True
        and dependence["global_ids_redefined"] is False
        and dependence["posthoc_correction"] is True,
        "Analysis did not use the declared frozen global dependency correction",
    )
    profiles = frozen_global_cluster_profiles(reference_test)
    counts = {task: row["n_clusters"] for task, row in profiles.items()}
    require(set(counts) == set(analysis["bootstrap"]), "Revision bootstrap tasks changed")
    contrasts = {
        (row["task"], row["metric"], row["left"], row["right"]): row
        for row in analysis["contrasts"]
    }
    expected_contrast_keys = {
        (task, metric, left, right)
        for task in counts
        for metric in independent.METRICS
        for left, right in combinations(methods, 2)
    }
    require(
        len(contrasts) == len(analysis["contrasts"]) and set(contrasts) == expected_contrast_keys,
        "Revision contrast coverage changed",
    )
    for task, clusters in counts.items():
        metadata = analysis["bootstrap"][task]
        require(
            metadata["requested"] == 2000 and metadata["n_clusters"] == clusters,
            "Revision bootstrap count/clusters changed",
        )
        independent._compare(metadata, profiles[task], profiles[task].keys())
        require(
            metadata["attempted"] == (2000 if clusters >= 2 else 0),
            "Revision bootstrap attempts changed",
        )
        require(
            metadata["seed_aggregation"]
            == "mean of per-seed metrics, never metric of averaged scores",
            "Revision bootstrap uses seed ensemble scores",
        )
        with np.load(
            output / "analysis" / f"bootstrap_{task.replace('.', '_')}.npz", allow_pickle=False
        ) as arrays:
            require(set(arrays.files) == set(methods), "Revision bootstrap methods changed")
            require(
                all(arrays[method].shape == (2000, 5) for method in methods),
                "Revision bootstrap shape changed",
            )
            valid = int(np.isfinite(arrays[methods[0]][:, 0]).sum())
            require(
                valid == metadata["valid"]
                and metadata["valid"] + metadata["degenerate"] == metadata["attempted"],
                "Revision bootstrap accounting changed",
            )
            for left, right in combinations(methods, 2):
                differences = arrays[left] - arrays[right]
                for j, metric in enumerate(independent.METRICS):
                    samples = differences[:, j][np.isfinite(differences[:, j])]
                    lo, hi = (
                        np.quantile(samples, [0.025, 0.975]).tolist()
                        if len(samples)
                        else (None, None)
                    )
                    observed = float(
                        np.mean(
                            [expected_metrics[left, seed, "test", task][metric] for seed in seeds]
                        )
                    ) - float(
                        np.mean(
                            [expected_metrics[right, seed, "test", task][metric] for seed in seeds]
                        )
                    )
                    target = {
                        "difference": observed,
                        "ci_low": lo,
                        "ci_high": hi,
                        "valid_bootstrap": len(samples),
                        "attempted_bootstrap": metadata["attempted"],
                    }
                    independent._compare(
                        contrasts[task, metric, left, right], target, target.keys()
                    )
    for summary in analysis["summary"]:
        require(summary["n_seeds"] == 3, "Revision summary must contain three seeds")
        for metric in independent.METRICS:
            values = [
                expected_metrics[summary["method"], seed, summary["split"], summary["task"]][metric]
                for seed in seeds
            ]
            values = [v for v in values if v is not None]
            target = {
                "mean": float(np.mean(values)) if values else None,
                "sd": float(np.std(values, ddof=1))
                if len(values) > 1
                else (0.0 if values else None),
            }
            independent._compare(
                {"mean": summary[metric + "_mean"], "sd": summary[metric + "_sd"]},
                target,
                target.keys(),
            )
    expected = expected_selected_baselines(expected_metrics, methods, seeds)
    for source in (analysis["selected_baselines"], frozen_baselines):
        require(set(source) == set(expected), "Validation baseline tasks changed")
        for task, target in expected.items():
            require(
                source[task]["method"] == target["method"],
                "Baseline was not selected by revision validation AP",
            )
            independent._compare(source[task], target, ("validation_mean_AP",))
    declaration = read(output / "analysis/verification.json")
    require(
        declaration["no_seed_score_ensemble"] is True and declaration["run_count"] == 21,
        "Revision seed aggregation declaration changed",
    )
    require(
        declaration["dependence_protocol"]["protocol"] == GLOBAL_GROUP_PROTOCOL
        and declaration["dependence_protocol"]["verified"] is True
        and declaration["dependence_protocol"]["global_ids_redefined"] is False,
        "Analysis verification lost the frozen global dependency proof",
    )
    dependency_input_hashes = verify_dependency_input_hashes(output, dependence, declaration)
    return {
        "requested_bootstrap_per_task": 2000,
        "dependency_cluster_counts": counts,
        "dependence_protocol": GLOBAL_GROUP_PROTOCOL,
        "global_ids_redefined": False,
        "frozen_dependency_input_hashes": dependency_input_hashes,
        "three_seed_metric_mean_verified": True,
        "validation_baselines_match_pretest_freeze": True,
    }


def verify_figures(output: Path) -> dict:
    manifest = read(output / "figures/figure_manifest.json")
    require(
        manifest["source_sha256"] == digest(output / "analysis/analysis.json"), "Figure data drift"
    )
    require(
        manifest["figure_source_sha256"] == digest(Path(__file__).with_name("figures.py")),
        "Figure code drift",
    )
    for name in ("performance", "costs"):
        for suffix in ("png", "pdf", "svg"):
            path = output / "figures" / f"{name}.{suffix}"
            if not path.is_file():
                raise IncompleteAudit(f"Missing revision figure: {path}")
            require(path.stat().st_size > 100, "Empty revision figure")
        qa = output / "figures/qa"
        results = read(qa / f"{name}.qa.json")
        require(
            {r["audit"] for r in results} == {"source", "font", "collisions"}
            and all(r["returncode"] == 0 for r in results),
            "Figure QA return codes failed",
        )
        for kind in ("alignment", "collisions"):
            report = read(qa / f"{name}.{kind}.json")
            require(
                report["verdict"] == "PASS"
                and report["summary"]["fail"] == report["summary"]["warn"] == 0,
                "Figure QA has unresolved findings",
            )
    return {"figures": 2, "formats": ["PNG", "PDF", "SVG"], "machine_qa_passed": True}


def verify_raw_dev_preparation_costs(output: Path, methods: list[str]) -> dict:
    rows = independent._csv(output / "analysis/costs.csv")
    csv_costs = {row["method"]: row for row in rows}
    analysis = read(output / "analysis/analysis.json")
    json_costs = {row["method"]: row for row in analysis["costs"]}
    declaration = read(output / "analysis/verification.json")["prediction_and_manifest_hashes"]
    require(
        len(csv_costs) == len(rows) and set(csv_costs) == set(json_costs) == set(methods),
        "DEV preparation cost table coverage changed",
    )
    verified = {}
    for method in methods:
        path = output / "runs" / method / "seed_42/dev_provenance.json"
        provenance = read(path)
        startup = provenance.get("startup") or {}
        components = [
            float(provenance.get("replay_seconds", 0)),
            float(startup.get("model_load_seconds", 0)),
            float(startup.get("startup_warmup_seconds", 0)),
        ]
        require(
            np.isfinite(components).all() and min(components) >= 0, "Invalid DEV preparation times"
        )
        expected = sum(components)
        if method in TRADITIONAL:
            require(
                expected == 0.0 and provenance["kind"] == "inherited_sealed_dev",
                "Traditional DEV was re-executed while reporting zero preparation",
            )
        for source in (csv_costs[method], json_costs[method]):
            independent._compare(
                source,
                {"raw_dev_inference_preparation_seconds": expected},
                ("raw_dev_inference_preparation_seconds",),
            )
            definition = source["raw_dev_inference_preparation_definition"]
            require(
                "excludes threshold scanning and audit" in definition
                and "not included" in definition,
                "DEV preparation cost definition hides exclusions or double-counting",
            )
        require(
            declaration[str(path.relative_to(output))] == digest(path),
            "DEV preparation provenance hash changed",
        )
        verified[method] = expected
    return {
        "raw_dev_inference_preparation_seconds": verified,
        "definition": (
            "raw DEV prediction + loading + TRAIN warm-up; excludes threshold scan/audit"
        ),
        "original_training_fields_unchanged": True,
    }


def verify_warmup(startup: dict, records: list[dict], protocol: dict) -> dict:
    rows = sorted(
        (r for r in records if r["source"] == "conda" and r["split"] == "train"),
        key=lambda r: r["record_id"],
    )[:64]
    expected = [r["record_id"] for r in rows]
    require(
        len(expected) == 64
        and startup["warmup_record_ids"] == expected == protocol["warmup"]["record_ids"],
        "Raw stage warm-up did not use the first 64 TRAIN CONDA IDs",
    )
    require(
        startup["warmup_split"] == "train"
        and startup["warmup_source"] == "conda"
        and startup["warmup_repetitions"] == 1
        and startup["warmup_predictions_discarded"] is True,
        "Warm-up source/repetition/discard policy changed",
    )
    binding = identity(
        [{"record_id": r["record_id"], "text_sha256": identity(r["text"])} for r in rows]
    )
    require(binding == protocol["warmup"]["text_binding"], "Warm-up text binding changed")
    return {"warmup_count": 64, "train_only": True, "label_independent_id_selection": True}


def audit(output: Path, *, base: Path | None = None, partial: bool = False) -> dict:
    output = Path(output).resolve()
    checks, completed = [], []
    metrics, strata, top3, signatures = {}, {}, {}, {}
    reference_test = None
    replay_source = {}

    def check(name, action):
        try:
            proof = action()
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

    def finish():
        statuses = Counter(row["status"] for row in checks)
        status = (
            "failed" if statuses["failed"] else "incomplete" if statuses["incomplete"] else "passed"
        )
        result = {
            "schema_version": 1,
            "status": status,
            "partial_requested": partial,
            "evaluation_revision": output.name,
            "execution_source": replay_source,
            "completed_runs_audited": len(completed),
            "required_runs": 21,
            "requirement_counts": dict(statuses),
            "requirements": checks,
            "verification_code_sha256": digest(Path(__file__)),
            "independent_threshold_code": (
                "local cumulative-count scan; worker pointer scan not invoked"
            ),
            "training_seals_policy": (
                "original base train seals only; revision manifests are evaluation views"
            ),
            "no_model_loading_fitting_or_prediction_reexecution": True,
        }
        save(output / "completion_audit.json", result)
        return result

    manifest_path = output / "revision_manifest.json"
    if not manifest_path.is_file():
        checks.append(
            {
                "requirement": "raw_revision_exists",
                "status": "incomplete",
                "missing": str(manifest_path),
            }
        )
        return finish()
    revision = read(manifest_path)
    base = Path(base or revision["base_output"]).resolve()
    experiment = read(base / "experiment.json")
    protocol = read(output / "revision_protocol.json")
    records, sessions = (
        read_jsonl(base / "prepared/records.jsonl"),
        read_jsonl(base / "prepared/sessions.jsonl"),
    )
    methods, seeds = experiment["methods"], experiment["seeds"]

    def revision_identity_check():
        replay_source.update(replay_source_metadata(revision))
        require(Path(revision["base_output"]).resolve() == base, "Wrong base campaign")
        require(
            revision["identity"]
            == identity({k: v for k, v in revision.items() if k != "identity"}),
            "Revision identity changed",
        )
        require(
            revision["base_experiment_sha256"] == digest(base / "experiment.json")
            and revision["base_experiment_identity"] == identity(experiment),
            "Base experiment changed",
        )
        require(
            revision["protocol_sha256"] == digest(output / "revision_protocol.json"),
            "Revision protocol changed",
        )
        require(
            revision["model_configurations_changed"] is False
            and revision["model_files_modified"] is False,
            "Revision claims model changes",
        )
        require(
            len(methods) == 7
            and seeds == [42, 43, 44]
            and experiment["threads"] == protocol["threads"] == 4,
            "Revision changed method/seed/CPU policy",
        )
        verify_hashes(revision["code_hashes"])
        verify_hashes(experiment["code_hashes"])
        verify_hashes(experiment["encoder"]["asset_hashes"])
        require(
            digest(output / "experiment.json") == digest(base / "experiment.json"),
            "Revision changed experiment snapshot",
        )
        original_prepared = read(base / "prepared/manifest.json")
        require(
            digest(base / "prepared/manifest.json") == experiment["dataset_sha256"]
            and original_prepared["fingerprint"] == experiment["dataset_fingerprint"],
            "Base prepared cohort no longer matches the training experiment",
        )
        verify_hashes(original_prepared["configuration"]["input_hashes"])
        require(
            revision["prepared_manifest_sha256"]
            == digest(base / "prepared/manifest.json")
            == digest(output / "prepared/manifest.json"),
            "Revision changed prepared cohort",
        )
        for name, sha in original_prepared["artifacts"].items():
            require(
                digest(base / "prepared" / name) == digest(output / "prepared" / name) == sha,
                f"Revision prepared artifact changed: {name}",
            )
        require(
            Counter(s["split"] for s in sessions) == {"train": 406, "dev": 135, "test": 136},
            "SCCD split changed",
        )
        training = {(row["method"], row["seed"]): row for row in revision["base_training"]}
        require(
            len(training) == 21 and set(training) == {(m, s) for m in methods for s in seeds},
            "Revision training inventory changed",
        )
        require(
            set(protocol["semantic_methods"]) == SEMANTIC
            and set(protocol["traditional_methods"]) == TRADITIONAL
            and protocol["cost_policy"]["seed"] == 42
            and protocol["cost_policy"]["passes"] == 3
            and protocol["cost_policy"]["score_tolerance"] == 1e-4,
            "Revision method roles or cost acceptance policy changed",
        )
        require(
            protocol["test_counts"]
            == {
                "messages": sum(r["split"] == "test" for r in records),
                "sessions": sum(s["split"] == "test" for s in sessions),
            },
            "Revision test count changed",
        )
        return {
            "base_experiment_and_data_unchanged": True,
            "original_model_inventory": 21,
            "threads": 4,
            "frozen_global_dependency_proof": verify_frozen_global_groups(records),
        }

    check("revision_base_data_code_encoder_identity", revision_identity_check)

    for method in methods:
        for seed in seeds:
            folder, original_folder = (
                output / "runs" / method / f"seed_{seed}",
                base / "runs" / method / f"seed_{seed}",
            )

            def run_check(method=method, seed=seed, folder=folder, original_folder=original_folder):
                nonlocal reference_test
                original, proof = verify_base_training(
                    base, experiment, records, sessions, method, seed
                )
                view = read(folder / "run_manifest.json")
                proof.update(
                    verify_evaluation_view(view, original, original_folder, revision["identity"])
                )
                inherited = next(
                    row
                    for row in revision["base_training"]
                    if row["method"] == method and row["seed"] == seed
                )
                for field in (
                    "base_run_identity",
                    "base_model_path",
                    "model_sha256",
                    "base_run_manifest_sha256",
                    "base_train_state_sha256",
                ):
                    require(
                        inherited[field] == view[field],
                        "Revision training inventory/view provenance differs",
                    )
                require(
                    not (folder / "train_state.json").exists(),
                    "Evaluation revision forged a new training seal",
                )
                signature = {
                    key: proof[key]
                    for key in ("training_ids_signature", "final_heads_signature", "oof_signatures")
                }
                if seed in signatures:
                    require(
                        signatures[seed] == signature,
                        "Methods used unequal original supervision/OOF/weights",
                    )
                else:
                    signatures[seed] = signature
                for split in ("dev", "test"):
                    seal = verify_revision_stage(folder, revision["identity"], method, seed, split)
                    required = [
                        folder / "run_manifest.json",
                        folder / f"{split}_provenance.json",
                        folder
                        / ("validation_freeze.json" if split == "dev" else "test_freeze.json"),
                    ]
                    require(
                        {str(p.resolve()) for p in required} <= set(seal["artifacts"]),
                        "Stage omitted provenance/freeze",
                    )
                    provenance = read(folder / f"{split}_provenance.json")
                    messages = read_jsonl(folder / f"predictions_message_{split}.jsonl")
                    bags = read_jsonl(folder / f"predictions_session_{split}.jsonl")
                    independent.verify_prediction_cohort(
                        records, sessions, messages, bags, view["thresholds"], split
                    )
                    if method in SEMANTIC:
                        require(
                            provenance["kind"] == f"semantic_raw_{split}"
                            and provenance["embeddings_argument"] is None,
                            "Semantic revision did not use raw predictions",
                        )
                        require(
                            provenance["base_model_sha256"] == original["model_sha256"]
                            and provenance["original_model_file_modified"] is False,
                            "Raw provenance changed weights",
                        )
                        verify_warmup(provenance["startup"], records, protocol)
                        require(
                            seal["resources"]["peak_affinity_width"] <= 4,
                            "Raw replay exceeded CPU budget",
                        )
                    else:
                        require(
                            provenance["kind"] == f"inherited_sealed_{split}"
                            and provenance["no_prediction_reexecution"] is True,
                            "Traditional prediction inheritance is not explicit",
                        )
                        source_seal = original_folder / (
                            "train_state.json" if split == "dev" else "test_state.json"
                        )
                        source_state = read(source_seal)
                        require(
                            source_state["status"] == "complete"
                            and source_state["identity"] == original["identity"],
                            "Original prediction seal invalid",
                        )
                        verify_hashes(source_state["artifacts"])
                        require(
                            provenance[f"base_{'train' if split == 'dev' else 'test'}_state_sha256"]
                            == digest(source_seal),
                            "Inherited source seal changed",
                        )
                        verify_hashes(provenance["source_artifacts"])
                        for unit in ("message", "session"):
                            name = f"predictions_{unit}_{split}.jsonl"
                            require(
                                digest(folder / name) == digest(original_folder / name),
                                "Inherited prediction bytes changed",
                            )
                    if split == "dev":
                        threshold_proof = verify_revision_thresholds(
                            messages, bags, view["thresholds"]
                        )
                        proof["independent_validation_thresholds"] = threshold_proof
                        freeze = read(folder / "validation_freeze.json")
                        require(
                            freeze["only_dev_used_for_threshold_selection"] is True
                            and freeze["thresholds"] == view["thresholds"]
                            and freeze["model_sha256"] == original["model_sha256"]
                            and freeze["revision_identity"] == revision["identity"],
                            "DEV threshold freeze invalid",
                        )
                        verify_hashes(freeze["dev_prediction_hashes"])
                        for name, target in threshold_proof["thresholds"].items():
                            independent._compare(
                                freeze["threshold_selection"]["thresholds"][name],
                                target,
                                (
                                    "threshold",
                                    "macro_f1",
                                    "n_observed",
                                    "positive",
                                    "candidate_count",
                                ),
                            )
                    else:
                        global_freeze = read(output / "validation_reference_freeze.json")
                        freeze = read(folder / "test_freeze.json")
                        sha = digest(output / "validation_reference_freeze.json")
                        require(
                            freeze["validation_reference_freeze_sha256"]
                            == seal["validation_reference_freeze_sha256"]
                            == sha
                            and freeze["validation_reference_identity"]
                            == global_freeze["identity"],
                            "Test not bound to pretest DEV reference freeze",
                        )
                        require(
                            freeze["thresholds"] == view["thresholds"]
                            and freeze["model_sha256"] == original["model_sha256"]
                            and freeze["run_manifest_sha256"]
                            == digest(folder / "run_manifest.json"),
                            "Test view thresholds or weights changed",
                        )
                        require(
                            freeze["created_at_utc"] >= global_freeze["created_at_utc"],
                            "Revised test predates validation freeze",
                        )
                    values, length_values = independent.task_metrics(
                        messages, bags, view["thresholds"]
                    )
                    metrics.update(
                        {(method, seed, split, task): value for task, value in values.items()}
                    )
                    strata.update(
                        {
                            (method, seed, split, task, stratum): value
                            for (task, stratum), value in length_values.items()
                        }
                    )
                    top3[method, seed, split] = independent.independent_top3(messages)
                    if split == "test" and reference_test is None:
                        reference_test = messages
                completed.append((method, seed))
                return {
                    "method": method,
                    "seed": seed,
                    **proof,
                    "prediction_mode": "raw" if method in SEMANTIC else "inherited_sealed",
                }

            check(f"run_{method}_{seed}_original_training_and_revision_evaluation", run_check)

        def cost_check(method=method):
            folder, original_folder = (
                output / "runs" / method / "seed_42",
                base / "runs" / method / "seed_42",
            )
            verify_no_consistency_retry(folder, revision["identity"])
            view, cost = read(folder / "run_manifest.json"), read(folder / "cost.json")
            seal = verify_revision_stage(folder, revision["identity"], method, 42, "cost")
            provenance = read(folder / "cost_provenance.json")
            proof = verify_cost_coverage(cost, records, method, 4)
            if method in TRADITIONAL:
                require(
                    digest(folder / "cost.json")
                    == digest(original_folder / "cost.json")
                    == provenance["source_cost_sha256"],
                    "Inherited cost bytes changed",
                )
                original_seal = read(original_folder / "cost_state.json")
                require(
                    original_seal["status"] == "complete"
                    and original_seal["identity"] == cost["identity"],
                    "Original cost seal invalid",
                )
                verify_hashes(original_seal["artifacts"])
                require(
                    provenance["kind"] == "inherited_valid_cost"
                    and provenance["no_cost_remeasurement"] is True
                    and provenance["source_cost_state_sha256"]
                    == digest(original_folder / "cost_state.json")
                    and provenance["source_cost_identity"]
                    == seal["inherited_cost_identity"]
                    == cost["identity"],
                    "Inherited cost provenance invalid",
                )
                proof["cold_start_comparability"] = (
                    "legacy warm-up definition; compare resident three-pass costs only"
                )
            else:
                expected = identity(
                    {
                        "revision": revision["identity"],
                        "method": method,
                        "seed": 42,
                        "model_sha256": view["model_sha256"],
                        "run_manifest_sha256": digest(folder / "run_manifest.json"),
                        "test_state_sha256": digest(folder / "test_state.json"),
                        "raw_replay_code_sha256": replay_source_metadata(revision)[
                            "execution_source_sha256"
                        ],
                        "passes": 3,
                        "score_tolerance": 1e-4,
                    }
                )
                require(
                    cost["identity"]
                    == seal["cost_identity"]
                    == expected
                    == provenance["cost_identity"],
                    "Raw cost identity invalid",
                )
                require(
                    cost["evaluation_revision_identity"] == revision["identity"]
                    and cost["test_state_sha256"]
                    == provenance["revision_test_state_sha256"]
                    == digest(folder / "test_state.json")
                    and cost["run_manifest_sha256"] == digest(folder / "run_manifest.json"),
                    "Raw cost uses wrong test/view reference",
                )
                require(
                    provenance["kind"] == "semantic_raw_remeasured_cost"
                    and provenance["same_training_warmup_as_dev_and_test"] is True
                    and provenance["threshold_override_only"] is True
                    and provenance["base_model_sha256"] == view["model_sha256"],
                    "Raw cost provenance invalid",
                )
                verify_warmup(cost, records, protocol)
                require(
                    cost["prediction_consistency"]["thresholds_from_revision_manifest"] is True,
                    "Cost classified using old package thresholds",
                )
                for row in cost["pass_counts"]:
                    require(
                        row["messages"] == protocol["test_counts"]["messages"]
                        and row["sessions"] == protocol["test_counts"]["sessions"],
                        "Raw cost incomplete prediction pass",
                    )
                require(
                    [r["repetition"] for r in cost["pass_counts"]] == [1, 2, 3],
                    "Raw cost pass accounting changed",
                )
                proof["raw_test_binding_and_train_warmup"] = True
            return proof

        check(f"cost_{method}_inherited_or_raw_reference_and_complete_repetitions", cost_check)

    def validation_freeze_check():
        freeze = read(output / "validation_reference_freeze.json")
        require(
            freeze["identity"] == identity({k: v for k, v in freeze.items() if k != "identity"})
            and freeze["revision_identity"] == revision["identity"],
            "Global DEV freeze identity changed",
        )
        require(
            freeze["test_used_for_decisions"] is False and freeze["model_files_modified"] is False,
            "Global reference freeze used test or changed weights",
        )
        expected_paths = {
            str((output / "runs" / method / f"seed_{seed}" / name).resolve())
            for method in methods
            for seed in seeds
            for name in (
                "dev_state.json",
                "run_manifest.json",
                "validation_freeze.json",
                "predictions_message_dev.jsonl",
                "predictions_session_dev.jsonl",
            )
        }
        require(
            set(freeze["all_dev_artifacts"]) == expected_paths,
            "Global freeze did not bind all 21 DEV runs",
        )
        verify_hashes(freeze["all_dev_artifacts"])
        if len(completed) != 21:
            raise IncompleteAudit(
                "All 21 revised run proofs required for final reference verification"
            )
        selected = expected_selected_baselines(metrics, methods, seeds)
        require(
            set(selected) == set(freeze["selected_baselines"]) == set(freeze["references"]),
            "Global baseline task set changed",
        )
        for task, target in selected.items():
            require(
                freeze["selected_baselines"][task]["method"] == target["method"],
                "Global baseline not chosen by new DEV",
            )
            independent._compare(
                freeze["selected_baselines"][task], target, ("validation_mean_AP",)
            )
        for method in methods:
            for seed in seeds:
                for task, value in freeze["all_21_run_validation_metrics"][method][
                    str(seed)
                ].items():
                    independent._compare(
                        value,
                        metrics[method, seed, "dev", task],
                        ("n", "positive", "threshold", *independent.METRICS),
                    )
        return {
            "all_dev_bindings": len(expected_paths),
            "baseline_tasks": len(selected),
            "new_DEV_only": True,
        }

    check("global_new_dev_reference_freeze_before_revised_test", validation_freeze_check)

    def metrics_check():
        if len(completed) != 21:
            raise IncompleteAudit("All 21 revised runs are required for independent metrics")
        return verify_metric_tables(output, metrics, strata, top3)

    check("independent_sklearn_metrics_length_strata_top3", metrics_check)

    def bootstrap_check():
        if len(completed) != 21 or reference_test is None:
            raise IncompleteAudit(
                "All revised test predictions required for bootstrap verification"
            )
        return verify_bootstrap_and_summary(
            output,
            metrics,
            reference_test,
            methods,
            seeds,
            read(output / "validation_reference_freeze.json")["selected_baselines"],
        )

    check("revision_bootstrap_no_seed_ensemble_and_pretest_baseline_selection", bootstrap_check)
    check("revision_figure_exports_and_QA", lambda: verify_figures(output))
    check(
        "raw_dev_preparation_cost_separately_reported",
        lambda: verify_raw_dev_preparation_costs(output, methods),
    )

    def preservation_check():
        before, after = (
            read(base / "preservation_before.json"),
            read(base / "preservation_after.json"),
        )
        require(
            len(before) == 19196
            and after["checked_files"] == len(before)
            and after["unchanged"] is True,
            "Original 19196-file preservation audit incomplete",
        )
        verify_hashes(before)
        deployment = [
            p for p in before if p.replace("\\", "/").endswith("/app/config/deployment.json")
        ]
        require(len(deployment) == 1, "Default deployment missing from preservation evidence")
        return {
            "original_files_verified": len(before),
            "default_deployment_sha256": before[deployment[0]],
            "original_models_and_predictions_unchanged": True,
        }

    check("original_19196_files_and_default_deployment_preserved", preservation_check)
    return finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Manifest-bound raw replay evaluation revision directory",
    )
    parser.add_argument("--base", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    result = audit(args.output, base=args.base, partial=args.partial)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "completed_runs_audited", "requirement_counts")}
        )
    )
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
