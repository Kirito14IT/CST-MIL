"""Gold-ID-linked, frozen-global-cluster analysis of the semantic follow-up.

Saved training metrics are never used. Numerical resampling reuses the audited
independent evaluator from the previous campaign, with its source hash recorded.
Three seeds remain three models: metrics are averaged only after evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cst_mil.research_followup.full_joint_comparison import analysis as frozen

SEEDS = (42, 43, 44)
ARCHITECTURES = ("full", "context_a", "context_b")
ABLATIONS = ("no_char", "no_message_semantics", "no_session_semantics",
             "no_session_reply", "no_session_risk", "no_source_balance")
TASKS = (*frozen.MAIN_TASKS, "sccd.conversation_common16")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict]:
    # U+2028, U+2029 and NEL inside multilingual text are not JSONL delimiters.
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def _write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_threshold_independent(labels: Any, scores: Any) -> float:
    """Independently reconstruct the exact DEV macro-F1/lower-threshold rule."""
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if not len(y) or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Observed DEV labels and scores must have equal nonzero length")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid DEV scores")
    order = np.argsort(p, kind="stable")
    cumulative_positive = np.r_[0, np.cumsum(y[order])]
    candidates = np.unique(np.r_[0., p, 1.])
    below = np.searchsorted(p[order], candidates, side="left")
    fn, tn = cumulative_positive[below], below - cumulative_positive[below]
    tp, fp = int(y.sum()) - fn, len(y) - int(y.sum()) - tn
    denominators = np.column_stack([2 * tn + fp + fn, 2 * tp + fp + fn])
    numerators = np.column_stack([2 * tn, 2 * tp])
    f1 = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float),
                   where=denominators > 0)
    macro = f1.sum(axis=1) / (denominators > 0).sum(axis=1)
    threshold, best = float(candidates[0]), -1.
    for candidate, value in zip(candidates, macro, strict=True):
        if value > best + 1e-12:
            threshold, best = float(candidate), float(value)
    return threshold


def load_frozen_gold(output: Path) -> tuple[dict[str, dict], dict[str, dict], dict]:
    protocol = _json(output / "protocol.json")
    base = Path(protocol["base_output"])
    manifest_path = base / "prepared/manifest.json"
    manifest = _json(manifest_path)
    for name in ("records.jsonl", "sessions.jsonl"):
        path = base / "prepared" / name
        if _sha(path) != manifest["artifacts"][name]:
            raise ValueError(f"Frozen prepared {name} fingerprint changed")
        saved = protocol.get("input_hashes", {}).get(str(path))
        if saved is not None and saved != _sha(path):
            raise ValueError(f"Protocol-bound prepared {name} fingerprint changed")
    rows = _jsonl(base / "prepared/records.jsonl")
    group_audit = frozen.validate_frozen_global_groups(rows)
    records = {str(r["record_id"]): r for r in rows}
    session_rows = _jsonl(base / "prepared/sessions.jsonl")
    sessions = {str(r["session_id"]): r for r in session_rows}
    if len(sessions) != len(session_rows):
        raise ValueError("Frozen session IDs are not unique")
    by_session = defaultdict(list)
    for row in rows:
        if row["source"] == "sccd":
            by_session[str(row["session_id"])].append(row)
    for session in sessions.values():
        messages = by_session[str(session["session_id"])]
        if not messages or len({r["eval_group"] for r in messages}) != 1:
            raise ValueError("A frozen SCCD session lacks one coherent global group")
    group_audit.update(prepared_records_sha256=_sha(base / "prepared/records.jsonl"),
                       prepared_sessions_sha256=_sha(base / "prepared/sessions.jsonl"),
                       prepared_manifest_sha256=_sha(manifest_path))
    return records, sessions, group_audit


def extract_tasks(messages: list[dict], session_rows: list[dict], manifest: dict,
                  gold: dict[str, dict], session_gold: dict[str, dict], split: str
                  ) -> tuple[dict[str, frozen.TaskPrediction], list[dict]]:
    """Check the full cohort before applying any observed-label filter."""
    ids = [str(r["record_id"]) for r in messages]
    expected = {rid for rid, row in gold.items() if row["split"] == split}
    if len(set(ids)) != len(ids) or set(ids) != expected:
        raise ValueError(f"Incomplete/duplicate {split} message prediction IDs")
    normalized = []
    for prediction in messages:
        original = gold[str(prediction["record_id"])]
        for key in ("message_id", "source", "split", "session_id", "parent_id", "order",
                    "labels", "observed_mask", "duplicate_component", "eval_group"):
            if prediction.get(key) != original.get(key):
                raise ValueError(f"Prediction changed frozen gold/group field {key}")
        if prediction.get("text_length") != len(str(original["text"])):
            raise ValueError("Prediction text length differs from complete frozen text")
        score = np.asarray(prediction["scores"], dtype=float)
        if score.shape != (3,) or not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError("Prediction needs three finite probabilities in [0, 1]")
        normalized.append({**original, "scores": score.tolist(),
                           "is_fallback": not original.get("fit_eligible", True)})
    rows_by_id = {str(r["record_id"]): r for r in normalized}
    tasks = {}
    thresholds = manifest["thresholds"]
    for source, category in frozen.MESSAGE_TASKS:
        j = frozen.CATEGORIES.index(category)
        selected = sorted([r for r in normalized if r["source"] == source and r["observed_mask"][j]],
                          key=lambda row: str(row["record_id"]))
        if not selected:
            raise ValueError(f"No observed gold for {source}.{category}")
        ids = [str(r["record_id"]) for r in selected]
        tasks[f"{source}.{category}"] = frozen.TaskPrediction(
            ids, np.asarray([gold[rid]["labels"][j] for rid in ids], dtype=int),
            np.asarray([rows_by_id[rid]["scores"][j] for rid in ids], dtype=float),
            float(thresholds[category]), [str(gold[rid]["eval_group"]) for rid in ids],
            np.asarray([len(str(gold[rid]["text"])) for rid in ids]))
    ids = [str(r["session_id"]) for r in session_rows]
    expected_sessions = {sid for sid, r in session_gold.items() if r["split"] == split}
    if len(set(ids)) != len(ids) or set(ids) != expected_sessions:
        raise ValueError("Session prediction IDs differ from frozen session cohort")
    session_groups = {str(r["session_id"]): str(r["eval_group"])
                      for r in normalized if r["source"] == "sccd"}
    selected_sessions = sorted(session_rows, key=lambda r: str(r["session_id"]))
    for prediction in selected_sessions:
        original = session_gold[str(prediction["session_id"])]
        if prediction["label"] != original["label"] or prediction["split"] != original["split"]:
            raise ValueError("Conversation gold/split changed")
    ids = [str(r["session_id"]) for r in selected_sessions]
    for category, score_key, row_threshold, manifest_key in (
        ("conversation", "score", "threshold", "conversation_threshold"),
        ("conversation_common16", "common16_score", "common16_threshold", "common16_threshold"),
    ):
        threshold = float(manifest.get(manifest_key, thresholds[category]))
        if category in thresholds and float(thresholds[category]) != threshold:
            raise ValueError("Manifest conversation thresholds conflict")
        if any(float(r[row_threshold]) != threshold for r in selected_sessions):
            raise ValueError("Session prediction threshold is not frozen DEV threshold")
        tasks[f"sccd.{category}"] = frozen.TaskPrediction(
            ids, np.asarray([session_gold[sid]["label"] for sid in ids], dtype=int),
            np.asarray([r[score_key] for r in selected_sessions], dtype=float), threshold,
            [session_groups[sid] for sid in ids])
    for current in tasks.values():
        # Also validates probabilities and thresholds, including degenerate gold.
        frozen.independent_metrics(current.labels, current.scores, current.threshold)
    return tasks, normalized


def verify_dev_thresholds(messages: list[dict], sessions: list[dict], manifest: dict) -> dict:
    result = {}
    for j, category in enumerate(frozen.CATEGORIES):
        observed = [r for r in messages if r["observed_mask"][j]]
        result[category] = select_threshold_independent(
            [r["labels"][j] for r in observed], [r["scores"][j] for r in observed])
    for category, field in (("conversation", "score"), ("conversation_common16", "common16_score")):
        result[category] = select_threshold_independent([r["label"] for r in sessions],
                                                       [r[field] for r in sessions])
    for key, threshold in result.items():
        if abs(float(manifest["thresholds"][key]) - threshold) > 1e-12:
            raise ValueError(f"Saved {key} threshold differs from independent DEV selection")
    return result


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    result = []
    for split, variant, task in sorted({(r["split"], r["variant"], r["task"]) for r in rows}):
        selected = [r for r in rows if (r["split"], r["variant"], r["task"]) == (split, variant, task)]
        if sorted(r["seed"] for r in selected) != list(SEEDS):
            raise ValueError("Every reported comparison requires seeds 42/43/44")
        if len({(r["n"], r["positive"]) for r in selected}) != 1:
            raise ValueError("Gold cohort differs across seeds")
        first = selected[0]
        aggregate = {"split": split, "variant": variant, "task": task, "n_seeds": 3,
                     "n": first["n"], "positive": first["positive"], "prevalence": first["prevalence"]}
        for metric in frozen.METRICS:
            values = [r[metric] for r in selected if r[metric] is not None]
            aggregate[metric + "_mean"] = float(np.mean(values)) if values else None
            aggregate[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result.append(aggregate)
    return result


def selection_audit(aggregates: list[dict], selected_architecture: str, rule: dict) -> dict:
    lookup = {(r["variant"], r["task"]): r for r in aggregates if r["split"] == "dev"}
    checks = []
    for architecture in ("context_a", "context_b"):
        gain = lookup[architecture, "sccd.attack"]["AP_mean"] - lookup["full", "sccd.attack"]["AP_mean"]
        criteria = {"primary_AP_gain": gain >= float(rule["minimum_primary_gain"]) - 1e-12}
        changes = {"sccd.attack.AP": gain}
        for task in frozen.MAIN_TASKS:
            if task == "sccd.attack":
                continue
            metrics = ("AP", "roc_auc") if task == "sccd.conversation" else ("AP",)
            limit = float(rule["maximum_SCCD_session_AP_AUC_drop"] if task == "sccd.conversation"
                          else rule["maximum_other_AP_drop"])
            for metric in metrics:
                change = lookup[architecture, task][metric + "_mean"] - lookup["full", task][metric + "_mean"]
                changes[task + "." + metric] = change
                criteria[task + "." + metric] = change >= -limit - 1e-12
        change = lookup[architecture, "sccd.attack"]["macro_f1_mean"] - lookup["full", "sccd.attack"]["macro_f1_mean"]
        changes["sccd.attack.macro_f1"] = change
        criteria["sccd.attack.macro_f1"] = change >= -float(rule["maximum_SCCD_attack_macro_F1_drop"]) - 1e-12
        checks.append({"architecture": architecture, "primary_gain": gain, "changes": changes,
                       "criteria": criteria, "eligible": all(criteria.values())})
    eligible = [r for r in checks if r["eligible"]]
    winner = "full"
    if eligible:
        best = -float("inf")
        for architecture in rule.get("tie_order", ["context_a", "context_b"]):
            match = next((r for r in eligible if r["architecture"] == architecture), None)
            if match is not None and match["primary_gain"] > best + float(rule.get("tie_tolerance", 1e-12)):
                winner, best = architecture, match["primary_gain"]
    if winner != selected_architecture:
        raise ValueError("Frozen architecture choice differs from independent DEV-rule calculation")
    return {"selected_architecture": winner, "verified": True, "candidates": checks,
            "test_used_for_selection": False}


def _interval_conclusion(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "interval_unavailable"
    if lower > 0:
        return "supports_positive_contribution"
    if upper < 0:
        return "ablation_higher_on_this_metric"
    return "insufficient_evidence_of_difference"


def _verify_seal(path: Path) -> None:
    seal = _json(path)
    if seal.get("status") not in {"complete", "completed", "passed"}:
        raise ValueError(f"Run seal is not accepted: {path}")
    for artifact, expected in seal.get("artifacts", {}).items():
        target = Path(artifact)
        if not target.is_absolute():
            target = path.parent / target
        if not target.is_file() or _sha(target) != expected:
            raise ValueError(f"Sealed artifact changed: {target}")


def summarize(output: str | Path, count: int = 2000) -> dict:
    """Write means/SD, every seed, paired ablation CIs and gold message Top-3."""
    output = Path(output)
    if count < 1:
        raise ValueError("Positive bootstrap count required")
    protocol = _json(output / "protocol.json")
    choice = _json(output / "selection_freeze.json")
    selected = choice["selected_architecture"]
    if selected not in ARCHITECTURES:
        raise ValueError("Unknown frozen selected architecture")
    gold, session_gold, group_audit = load_frozen_gold(output)
    identities = {str(output / name): _sha(output / name)
                  for name in ("protocol.json", "selection_freeze.json")}
    datasets, rows, top3, threshold_audits = {}, [], [], {}
    for variant in (*ARCHITECTURES, *ABLATIONS):
        for seed in SEEDS:
            folder = output / "runs" / variant / f"seed_{seed}"
            for name in ("manifest.json", "complete.json"):
                path = folder / name
                if not path.exists():
                    raise FileNotFoundError(f"Missing sealed run {path}")
                identities[str(path)] = _sha(path)
                if name == "complete.json":
                    _verify_seal(path)
            manifest = _json(folder / "manifest.json")
            expected_config = {"architecture_base": selected if variant in ABLATIONS else variant,
                               "ablation": variant if variant in ABLATIONS else None}
            if manifest["seed"] != seed or manifest["config"] != expected_config:
                raise ValueError("Run architecture/ablation/seed does not match fixed comparison")
            for split in ("dev", "test"):
                paths = [folder / f"predictions_{unit}_{split}.jsonl" for unit in ("message", "session")]
                if split == "test" and not all(p.exists() for p in paths):
                    if variant == selected or variant in ABLATIONS:
                        raise FileNotFoundError(f"Missing final TEST predictions for {variant}/{seed}")
                    continue
                if split == "test":
                    seal = folder / "test_complete.json"
                    if not seal.exists():
                        raise FileNotFoundError(f"Unsealed TEST predictions: {folder}")
                    identities[str(seal)] = _sha(seal)
                    _verify_seal(seal)
                for path in paths:
                    identities[str(path)] = _sha(path)
                messages, sessions = _jsonl(paths[0]), _jsonl(paths[1])
                tasks, normalized = extract_tasks(messages, sessions, manifest, gold, session_gold, split)
                datasets[split, variant, seed] = tasks
                if split == "dev":
                    threshold_audits[f"{variant}/{seed}"] = verify_dev_thresholds(messages, sessions, manifest)
                for task, prediction in tasks.items():
                    metrics = frozen.independent_metrics(prediction.labels, prediction.scores, prediction.threshold)
                    rows.append({"split": split, "variant": variant, "seed": seed, "task": task, **metrics})
                top3.append({"split": split, "variant": variant, "seed": seed, **frozen.top3_metrics(normalized)})
    metrics = aggregate_metrics(rows)
    choice_audit = selection_audit(metrics, selected, protocol["selection"])
    identity = {"protocol": "semantic-patent-independent-analysis-v1", "count": count,
                "analysis_source_sha256": _sha(Path(__file__)),
                "audited_numeric_source_sha256": _sha(Path(frozen.__file__)),
                "input_sha256": identities, "global_group_audit": group_audit}
    target = output / "analysis"
    target.mkdir(parents=True, exist_ok=True)
    identity_path = target / "identity.json"
    if identity_path.exists() and _json(identity_path) != identity:
        raise ValueError("Analysis inputs/code changed; preserve old results and use an explicit revision")
    _write(identity_path, identity)
    _csv(target / "per_seed_metrics.csv", rows)
    _csv(target / "metrics.csv", metrics)
    _csv(target / "top3.csv", top3)
    _write(target / "threshold_audit.json", threshold_audits)
    _write(target / "selection_audit.json", choice_audit)
    effects, seed_differences, bootstrap_audits = [], [], {}
    boot_dir = target / "bootstrap"
    boot_dir.mkdir(exist_ok=True)
    for task_index, task in enumerate(TASKS):
        predictions = {variant: [datasets["test", variant, seed][task] for seed in SEEDS]
                       for variant in (selected, *ABLATIONS)}
        sampled, audit = frozen.bootstrap_paired(predictions, count=count, seed=20261015 + task_index)
        audit.update(global_group_protocol=group_audit["protocol"],
                     grouping_scope="frozen full TRAIN/DEV/TEST closure, before mask filtering",
                     test_only_groups_reconstructed=False)
        bootstrap_audits[task] = audit
        np.savez_compressed(boot_dir / f"test__{task}.npz", **sampled)
        for ablation in ABLATIONS:
            for j, metric in enumerate(frozen.METRICS):
                selected_scores, ablated_scores = [], []
                for seed in SEEDS:
                    selected_prediction = datasets["test", selected, seed][task]
                    ablated_prediction = datasets["test", ablation, seed][task]
                    first = frozen.independent_metrics(selected_prediction.labels, selected_prediction.scores,
                                                       selected_prediction.threshold)[metric]
                    second = frozen.independent_metrics(ablated_prediction.labels, ablated_prediction.scores,
                                                        ablated_prediction.threshold)[metric]
                    selected_scores.append(first)
                    ablated_scores.append(second)
                    seed_differences.append({"split": "test", "task": task, "metric": metric,
                                             "seed": seed, "selected_architecture": selected,
                                             "ablation": ablation, "selected": first, "ablated": second,
                                             "difference": first - second if first is not None and second is not None else None})
                differences = sampled[selected][:, j] - sampled[ablation][:, j]
                valid = differences[np.isfinite(differences)]
                lower, upper = (map(float, np.quantile(valid, [.025, .975])) if len(valid) else (None, None))
                first_mean = float(np.mean(selected_scores)) if all(x is not None for x in selected_scores) else None
                second_mean = float(np.mean(ablated_scores)) if all(x is not None for x in ablated_scores) else None
                effects.append({"split": "test", "task": task, "metric": metric,
                                "selected_architecture": selected, "ablation": ablation,
                                "selected_mean": first_mean, "ablation_mean": second_mean,
                                "difference": first_mean - second_mean if first_mean is not None and second_mean is not None else None,
                                "lower95": lower, "upper95": upper, "n_clusters": audit["n_clusters"],
                                "requested_replicates": count, "valid_replicates": int(len(valid)),
                                "interval_type": "paired cluster percentile 95%, exploratory, unadjusted",
                                "conclusion": _interval_conclusion(lower, upper)})
    _csv(target / "ablation_effects.csv", effects)
    _csv(target / "per_seed_ablation_effects.csv", seed_differences)
    _write(boot_dir / "audit.json", bootstrap_audits)
    summary = {
        "status": "completed", "selected_architecture": selected, "seeds": list(SEEDS),
        "n_seed_split_task_metrics": len(rows), "n_ablation_effects": len(effects),
        "bootstrap_count": count, "selection_audit": choice_audit,
        "global_groups": group_audit, "bootstrap": bootstrap_audits,
        "metrics": metrics, "effect_conclusions": dict(Counter(r["conclusion"] for r in effects)),
        "notes": ["Independent metrics computed from saved scores and frozen gold linked by actual record/message IDs.",
                  "Three-seed means/SD are separate-model results, not an ensemble.",
                  "Intervals resample frozen global dependency IDs, including non-TEST bridges; no TEST-only reconstruction.",
                  "Intervals are exploratory percentile intervals, without multiplicity adjustment or causal proof.",
                  "Negative effects and intervals crossing zero are retained; absence of significance is not equivalence.",
                  "common16 is a compatibility field name; matching ablations may contain 4, 10, or 16 common features.",
                  "Top-3 uses observed attack gold by ID, excludes fallback, and reports dangerous-message recall separately.",
                  "Previously examined TEST data are supplementary verification, not first unseen evaluation.",
                  "No topic-accuracy or hallucination-rate claim follows from classification scores."],
        "artifacts": {p.relative_to(target).as_posix(): _sha(p) for p in target.rglob("*")
                      if p.is_file() and p.name != "summary.json" and not p.name.endswith(".tmp")},
    }
    _write(target / "summary.json", summary)
    return summary
