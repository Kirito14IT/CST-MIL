"""Independent, cluster-paired analysis of the full joint comparison.

The evaluation unit is never changed to a seed-ensemble prediction.  Each
replicate computes a metric within each seed, and only then averages seeds.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METHODS = ("lda", "btm", "gsdmm", "nmf", "bertopic", "cst_v1_1", "minilm_lr")
TOPIC_BASELINES = METHODS[:5]
SEEDS = (42, 43, 44)
CATEGORIES = ("attack", "hate", "spam_scam")
MESSAGE_TASKS = (("sccd", "attack"), ("conda", "attack"),
                 ("hatebr", "attack"), ("toxicn", "attack"),
                 ("toxicn", "hate"), ("scamspot", "spam_scam"))
METRICS = ("AP", "roc_auc", "macro_f1", "precision", "recall")
MAIN_TASKS = tuple(f"{s}.{c}" for s, c in MESSAGE_TASKS) + ("sccd.conversation",)
GLOBAL_GROUP_PROTOCOL = "frozen-global-eval-group-closure-v1"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    # Only physical LF/CRLF boundaries delimit JSONL. Unicode line-separator
    # characters may legally occur inside an unescaped multilingual string.
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _category(row: Mapping[str, Any], field: str, category: str,
              default: Any = None) -> Any:
    value = row.get(field)
    if value is None and field == "observed_mask":
        value = row.get("masks")
    if value is None and field == "scores":
        value = row.get("scores3")
    if isinstance(value, Mapping):
        return value.get(category, default)
    if isinstance(value, (list, tuple)):
        return value[CATEGORIES.index(category)]
    return default


def _source(row: Mapping[str, Any]) -> str:
    return str(row.get("source", "")).lower().replace("-", "").replace("_", "")


def _record_id(row: Mapping[str, Any]) -> str:
    if row.get("record_id") is not None:
        return str(row["record_id"])
    if row.get("message_id") is None:
        raise ValueError("Prediction requires record_id or message_id")
    return f"{_source(row)}:{row.get('session_id', '')}:{row['message_id']}"


def independent_metrics(labels: Sequence[int], scores: Sequence[float],
                        threshold: float) -> dict[str, Any]:
    """Recompute prediction metrics without trusting saved training metrics."""
    y, p = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Binary labels and scores must be equally sized vectors")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Scores must be finite values in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    n, positive = len(y), int(y.sum())
    result = {"n": n, "positive": positive,
              "prevalence": positive / n if n else None, "threshold": float(threshold)}
    if not n:
        return {**result, "status": "no_observed_labels", **dict.fromkeys(METRICS),
                "tn": 0, "fp": 0, "fn": 0, "tp": 0}
    prediction = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel().tolist()
    both = 0 < positive < n
    return {**result, "status": "both_classes" if both else "single_class",
            "AP": float(average_precision_score(y, p)) if positive else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self.parent.setdefault(value, value)
        current = value
        while current != self.parent[current]:
            current = self.parent[current]
        while value != current:
            value, self.parent[value] = self.parent[value], current
        return current

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def validate_frozen_global_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Verify, never replace, the already frozen all-split dependency closure."""
    uf = _UnionFind()
    component_first: dict[str, str] = {}
    by_group: dict[str, set[str]] = {}
    by_id: set[str] = set()
    for row in rows:
        rid = _record_id(row)
        if rid in by_id:
            raise ValueError("Frozen global-group record IDs are not unique")
        by_id.add(rid)
        group, frozen = row["base_eval_group"], row["eval_group"]
        if not group.startswith(_source(row) + ":") or not frozen:
            raise ValueError("Frozen global group lost its genuine source-group identity")
        if _source(row) == "sccd" and group != "sccd:session:" + str(row["session_id"]):
            raise ValueError("Frozen global group does not identify the genuine SCCD session")
        uf.find(group)
        component = row["duplicate_component"]
        first = component_first.setdefault(component, group)
        uf.union(first, group)
        by_group.setdefault(group, set()).add(frozen)
    members: dict[str, set[str]] = {}
    for group in by_group:
        members.setdefault(uf.find(group), set()).add(group)
    expected = {
        root: "eval:" + hashlib.sha256("|".join(sorted(groups)).encode("utf-8")).hexdigest()
        for root, groups in members.items()
    }
    sources: dict[str, set[str]] = {}
    for row in rows:
        root = uf.find(row["base_eval_group"])
        if row["eval_group"] != expected[root]:
            raise ValueError("Frozen eval_group differs from the complete all-source closure")
        sources.setdefault(row["eval_group"], set()).add(_source(row))
    return {
        "protocol": GLOBAL_GROUP_PROTOCOL, "verified": True,
        "records": len(rows), "base_groups": len(by_group), "global_groups": len(members),
        "cross_source_global_groups": sum(len(value) > 1 for value in sources.values()),
        "unknown_records_included": sum(not any(row.get("observed_mask", ())) for row in rows),
        "scope": "all retained prepared train/dev/test rows before any score filtering",
        "global_ids_redefined": False,
    }


def _prepared_global_groups(root: Path) -> tuple[dict[str, dict], dict[str, Any]]:
    manifest_path, path = root / "prepared/manifest.json", root / "prepared/records.jsonl"
    if not manifest_path.exists():
        if (root / "experiment.json").exists() or (root / "revision_manifest.json").exists():
            raise FileNotFoundError("Registered experiments require frozen prepared global groups")
        return {}, {"protocol": GLOBAL_GROUP_PROTOCOL, "verified": False,
                    "scope": "unregistered unit-test fixture; not a frozen scientific cohort"}
    manifest = _json(manifest_path)
    actual = _sha(path)
    if actual != manifest["artifacts"]["records.jsonl"]:
        raise ValueError("Frozen prepared record fingerprint changed")
    keys = ("record_id", "source", "split", "session_id", "base_eval_group",
            "eval_group", "duplicate_component", "observed_mask")
    with path.open(encoding="utf-8-sig") as handle:
        rows = [{key: row[key] for key in keys}
                for line in handle if line.strip() if (row := json.loads(line))]
    audit = validate_frozen_global_groups(rows)
    audit.update(prepared_records_sha256=actual, prepared_manifest_sha256=_sha(manifest_path))
    return {_record_id(row): row for row in rows}, audit


def _verify_prediction_global_groups(messages: Sequence[Mapping[str, Any]], split: str,
                                     frozen: Mapping[str, Mapping[str, Any]]) -> None:
    if not frozen:
        return
    ids = [_record_id(row) for row in messages]
    expected = {rid for rid, row in frozen.items() if row["split"] == split}
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError("Prediction cohort differs from frozen global-group membership")
    for row in messages:
        original = frozen[_record_id(row)]
        for key in ("source", "split", "session_id", "eval_group", "duplicate_component"):
            if row.get(key) != original[key]:
                raise ValueError(f"Prediction changed frozen global-group field: {key}")


def cluster_components(rows: Sequence[Mapping[str, Any]], *,
                       unit: str = "message") -> dict[str, str]:
    """Reuse frozen all-split eval_group IDs, preserving cross-source links.

    Genuine SCCD session and duplicate edges are redundant on verified frozen
    cohorts. They also keep small standalone fixtures structurally coherent.
    No observed-label filter is applied before dependency grouping.
    """
    uf, nodes = _UnionFind(), {}
    for row in rows:
        source = _source(row)
        rid = _record_id(row)
        group = row.get("eval_group")
        if group is None:
            raise ValueError(f"{source} requires a genuine eval_group")
        node = f"global-eval-group:{group}"
        nodes[rid] = node
        uf.find(node)
        if source == "sccd":
            uf.union(node, f"true-sccd-session:{row['session_id']}")
        duplicate = row.get("duplicate_component")
        if duplicate is not None:
            uf.union(node, f"duplicate:{duplicate}")
    if unit == "message":
        return {rid: uf.find(node) for rid, node in nodes.items()}
    if unit != "conversation":
        raise ValueError("unit must be message or conversation")
    result = {}
    for row in rows:
        if _source(row) == "sccd":
            sid = str(row["session_id"])
            result[sid] = uf.find(nodes[_record_id(row)])
    return result


@dataclass
class TaskPrediction:
    record_ids: list[str]
    labels: np.ndarray
    scores: np.ndarray
    threshold: float
    clusters: list[str]
    lengths: np.ndarray | None = None


def _weighted_metrics(y: np.ndarray, p: np.ndarray, threshold: float,
                      weights: np.ndarray) -> np.ndarray:
    """Exact weighted AP/AUC/F1/P/R, including score ties, in bootstrap batches."""
    w = np.atleast_2d(weights).astype(float, copy=False)
    order = np.argsort(-p, kind="stable")
    sy, sp, sw = y[order], p[order], w[:, order]
    ends = np.r_[np.flatnonzero(sp[:-1] != sp[1:]), len(sp) - 1]
    tp = np.cumsum(sw * sy, axis=1)[:, ends]
    fp = np.cumsum(sw * (1 - sy), axis=1)[:, ends]
    positives, negatives = tp[:, -1], fp[:, -1]
    valid = (positives > 0) & (negatives > 0)
    d_tp = np.diff(np.column_stack([np.zeros(len(w)), tp]), axis=1)
    d_fp = np.diff(np.column_stack([np.zeros(len(w)), fp]), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision_at_end = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        ap = np.sum(precision_at_end * d_tp, axis=1) / positives
        # Descending order: negatives ranked below the current tied group.
        auc = np.sum(d_tp * (negatives[:, None] - fp + .5 * d_fp), axis=1) / (positives * negatives)
    pred = p >= threshold
    ctp, cfp = w @ (y * pred), w @ ((1 - y) * pred)
    cfn, ctn = positives - ctp, negatives - cfp
    def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.divide(a, b, out=np.zeros_like(a), where=b > 0)
    f1 = .5 * (ratio(2 * ctp, 2 * ctp + cfp + cfn) + ratio(2 * ctn, 2 * ctn + cfp + cfn))
    result = np.column_stack([ap, auc, f1, ratio(ctp, ctp + cfp), ratio(ctp, positives)])
    result[~valid] = np.nan
    return result


def bootstrap_paired(predictions: Mapping[str, Sequence[TaskPrediction]], *,
                     count: int = 2000, seed: int = 20261014,
                     batch_size: int = 32) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One shared cluster resampling stream for every method and model seed."""
    if count <= 0 or batch_size <= 0 or not predictions:
        raise ValueError("Positive bootstrap count/batch_size and predictions required")
    reference = next(iter(predictions.values()))[0]
    cluster_names = sorted(set(reference.clusters))
    cluster_index = {name: i for i, name in enumerate(cluster_names)}
    positions = np.array([cluster_index[name] for name in reference.clusters], dtype=int)
    for method, seeds in predictions.items():
        if not seeds:
            raise ValueError(f"No seeds for {method}")
        for current in seeds:
            if current.record_ids != reference.record_ids or current.clusters != reference.clusters:
                raise ValueError("Paired bootstrap cohorts/groups do not match")
            if not np.array_equal(current.labels, reference.labels):
                raise ValueError("Paired bootstrap labels do not match")
    rng = np.random.default_rng(seed)
    results = {method: np.full((count, len(METRICS)), np.nan) for method in predictions}
    n_group = len(cluster_names)
    if not n_group:
        raise ValueError("No evaluation clusters")
    sizes = np.bincount(positions, minlength=n_group)
    shares = sizes / sizes.sum()
    concentration = {"n_clusters": n_group,
                     "largest_cluster_n": int(sizes.max()),
                     "largest_cluster_fraction": float(shares.max()),
                     "effective_cluster_count": float(1 / np.sum(shares ** 2)),
                     "size_concentration_effective_cluster_count": float(1 / np.sum(shares ** 2)),
                     "effective_cluster_count_definition": "Kish inverse sum of squared record-count shares; concentration diagnostic only, not degrees of freedom",
                     "n_records": len(reference.labels), "random_seed": seed,
                     "resampling": "sample merged dependency clusters with replacement; preserve all members",
                     "seed_aggregation": "mean of per-seed metrics, never metric of averaged scores",
                     "CI": "paired percentile 95%; exploratory, no multiplicity adjustment"}
    if n_group < 2:
        return results, {**concentration, "requested": count, "attempted": 0,
                         "valid": 0, "degenerate": 0,
                         "status": "insufficient_clusters", "interval_available": False,
                         "reason": "Fewer than two independent dependency clusters; confidence interval is N/A, not zero-width"}
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        sampled = rng.multinomial(n_group, np.full(n_group, 1 / n_group), size=stop - start)
        weights = sampled[:, positions]
        for method, seeds in predictions.items():
            per_seed = [_weighted_metrics(x.labels, x.scores, x.threshold, weights) for x in seeds]
            results[method][start:stop] = np.mean(np.stack(per_seed, axis=0), axis=0)
    valid = np.isfinite(next(iter(results.values()))[:, 0])
    return results, {**concentration, "requested": count, "attempted": count,
                     "valid": int(valid.sum()), "degenerate": int((~valid).sum()),
                     "status": "estimated" if valid.any() else "all_degenerate",
                     "interval_available": bool(valid.any())}


def top3_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    sessions: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if _source(row) == "sccd":
            sessions.setdefault(str(row["session_id"]), []).append(row)
    positive_sessions = hit = dangerous_total = dangerous_found = selected_total = 0
    for messages in sessions.values():
        gold = {str(_record_id(r)) for r in messages
                if not r.get("is_fallback", False)
                and _category(r, "observed_mask", "attack", False)
                and int(_category(r, "labels", "attack", 0)) == 1}
        # Fallback rows retain the conversation but have no genuine evidence.
        candidates = [r for r in messages if not r.get("is_fallback", False)]
        ranked = sorted(candidates, key=lambda r: (
            -float(_category(r, "scores", "attack")),
            int(r.get("order", 0) or 0), str(r.get("message_id") or _record_id(r)),
            _record_id(r)))[:3]
        selected = {_record_id(r) for r in ranked}
        selected_total += len(selected)
        dangerous_total += len(gold)
        dangerous_found += len(gold & selected)
        if gold:
            positive_sessions += 1
            hit += bool(gold & selected)
    return {"n_sessions": len(sessions), "positive_gold_sessions": positive_sessions,
            "hit_sessions": hit, "hit_at_3": hit / positive_sessions if positive_sessions else None,
            "dangerous_gold_comments": dangerous_total, "dangerous_selected_comments": dangerous_found,
            "dangerous_comment_recall_at_3": dangerous_found / dangerous_total if dangerous_total else None,
            "selected_comments_all_sessions": selected_total,
            "ranking": "attack score descending; original order, message_id, record_id tie breaks; up to 3 per session",
            "tie_policy": "Preserve original browsing order for tied scores; message_id then record_id break identical order values. Legacy missing order defaults to 0."}


def _threshold(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], category: str) -> float:
    saved = manifest.get("thresholds", {}).get(category)
    values = set()
    for row in rows:
        value = row.get("threshold")
        if isinstance(value, Mapping):
            value = value.get(category)
        if value is not None:
            values.add(float(value))
    if saved is not None:
        values.add(float(saved))
    if len(values) != 1:
        raise ValueError(f"Expected one frozen {category} threshold, got {sorted(values)}")
    return next(iter(values))


def _extract_tasks(manifest: Mapping[str, Any], messages: list[dict], sessions: list[dict]) -> dict[str, TaskPrediction]:
    clusters = cluster_components(messages)
    result = {}
    for source, category in MESSAGE_TASKS:
        selected = sorted((row for row in messages if _source(row) == source
                           and bool(_category(row, "observed_mask", category, False))), key=_record_id)
        if not selected:
            raise ValueError(f"No observed labels for {source}.{category}")
        ids = [_record_id(row) for row in selected]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate prediction IDs in {source}.{category}")
        result[f"{source}.{category}"] = TaskPrediction(
            ids, np.array([int(_category(row, "labels", category)) for row in selected]),
            np.array([float(_category(row, "scores", category)) for row in selected]),
            _threshold(manifest, selected, category), [clusters[rid] for rid in ids],
            np.array([int(row["text_length"]) for row in selected]))
    session_clusters = cluster_components(messages, unit="conversation")
    selected_sessions = sorted(sessions, key=lambda row: str(row["session_id"]))
    ids = [str(row["session_id"]) for row in selected_sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate session prediction IDs")
    if set(ids) != set(session_clusters):
        raise ValueError("Session predictions and message sessions are incomplete or inconsistent")
    for category, score_field, threshold_field in (("conversation", "score", "threshold"),
                                                  ("conversation_common16", "common16_score", "common16_threshold")):
        if category.endswith("common16") and not all(score_field in row for row in selected_sessions):
            raise ValueError("Missing common16 session control predictions")
        threshold_rows = [{"threshold": row.get(threshold_field)} for row in selected_sessions]
        result[f"sccd.{category}"] = TaskPrediction(
            ids, np.array([int(row["label"]) for row in selected_sessions]),
            np.array([float(row[score_field]) for row in selected_sessions]),
            _threshold(manifest, threshold_rows, category), [session_clusters[rid] for rid in ids])
    return result


def _summary(rows: Sequence[Mapping[str, Any]], methods: Sequence[str], seeds: Sequence[int]) -> list[dict]:
    result = []
    for split in ("dev", "test"):
        for task in (*MAIN_TASKS, "sccd.conversation_common16"):
            for method in methods:
                selected = [row for row in rows if row["split"] == split and row["task"] == task and row["method"] == method]
                if len(selected) != len(seeds):
                    raise ValueError(f"Incomplete {split}/{task}/{method} seeds")
                counts = {(row["n"], row["positive"]) for row in selected}
                if len(counts) != 1:
                    raise ValueError("Cohort changed between seeds")
                first = selected[0]
                aggregate = {"split": split, "task": task, "method": method,
                             "n": first["n"], "positive": first["positive"],
                             "prevalence": first["prevalence"], "n_seeds": len(seeds)}
                for name in METRICS:
                    values = [r[name] for r in selected if r[name] is not None]
                    aggregate[name + "_mean"] = float(np.mean(values)) if values else None
                    aggregate[name + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else (0. if values else None)
                result.append(aggregate)
    return result


def _cost_table(costs: Sequence[Mapping[str, Any]], methods: Sequence[str]) -> list[dict[str, Any]]:
    """Flatten the measured cost contract; unavailable values remain explicit."""
    lookup = {str(row["method"]): row for row in costs}
    fields = ("training_seconds_cold", "training_seconds_cached", "shared_encoder_training_seconds",
              "model_load_seconds", "startup_warmup_seconds", "raw_dev_inference_preparation_seconds", "first_inference_seconds", "full_test_inference_seconds",
              "sccd_session_median_seconds", "sccd_session_p95_seconds", "model_bytes",
              "encoder_asset_bytes", "runtime_dependency_bytes", "peak_tree_rss_bytes",
              "training_peak_tree_rss_bytes", "inference_peak_tree_rss_bytes", "GPU_required")
    result = []
    for method in methods:
        cost = lookup.get(method, {})
        row = {"method": method, "seed": 42,
               "measurement_status": "measured" if cost else "unavailable"}
        for name in fields:
            row[name] = cost[name] if cost.get(name) is not None else "N/A"
        warm = cost.get("warm_total_seconds")
        if isinstance(warm, list) and warm:
            if len(warm) != 3 or not np.isfinite(warm).all():
                raise ValueError("Cost CSV expects exactly three finite warm inference repetitions")
            row["warm_total_seconds_repetitions"] = json.dumps(warm, ensure_ascii=False)
            row["warm_total_seconds_mean"] = float(np.mean(warm))
            row["warm_total_seconds_sd"] = float(np.std(warm, ddof=1))
        else:
            row.update(warm_total_seconds_repetitions="N/A", warm_total_seconds_mean="N/A",
                       warm_total_seconds_sd="N/A")
        throughput = cost.get("comment_throughput_per_second_by_source")
        row["comment_throughput_per_second_by_source"] = (
            json.dumps(throughput, ensure_ascii=False, sort_keys=True) if throughput is not None else "N/A")
        for name in ("cache_policy", "cold_training_definition", "memory_definition",
                     "dependency_sizing_scope", "cold_start_definition", "startup_warmup_definition",
                     "raw_dev_inference_preparation_definition"):
            row[name] = cost.get(name) if cost.get(name) is not None else "N/A"
        row["model_load_seconds_definition"] = cost.get("model_load_seconds_definition", (
            "joblib.load model-package deserialization only; excludes lazy encoder initialization and first-call compilation"))
        row["first_inference_seconds_definition"] = cost.get("first_inference_seconds_definition", (
            "first complete SCCD test-conversation prediction after deserialization; includes lazy encoder initialization and method-specific first-call compilation when triggered, plus first-conversation inference; not the entire test set"))
        row["warm_total_seconds_definition"] = cost.get("warm_total_seconds_definition", (
            "three separate complete-test resident inference repetitions without text-embedding or prediction cache; the startup and first-inference definitions in this row describe the method-specific warm-up and must not be interpreted as the same cold request across methods"))
        row["training_seconds_cold_definition"] = (
            "measured cached fitting/validation time plus measured first-time train+dev encoding time; shared encoder work actually ran once but is attributed once to each independent semantic system; not a separate full cold-start end-to-end rerun, not a cleared-OS-cache measurement, and per-method values must not be summed as campaign elapsed time")
        result.append(row)
    return result


def _analysis_summary(result: Mapping[str, Any]) -> str:
    display = {"lda": "LDA", "btm": "BTM", "gsdmm": "GSDMM", "nmf": "NMF",
               "bertopic": "BERTopic", "cst_v1_1": "CST v1.1", "minilm_lr": "MiniLM＋LR"}
    tasks = {"sccd.attack": "SCCD攻击评论", "conda.attack": "CONDA攻击",
             "hatebr.attack": "HateBR冒犯", "toxicn.attack": "ToxiCN冒犯",
             "toxicn.hate": "ToxiCN仇恨", "scamspot.spam_scam": "ScamSpot垃圾／诈骗",
             "sccd.conversation": "SCCD会话"}
    def number(value: Any, *, signed: bool = False) -> str:
        if value is None or isinstance(value, str) or not np.isfinite(value):
            return "N/A"
        return f"{float(value):+.4f}" if signed else f"{float(value):.4f}"
    test = {(row["task"], row["method"]): row for row in result["summary"] if row["split"] == "test"}
    lines = ["# 五来源全量公平比较：分析摘要", "",
        f"已汇总{len(result['methods'])}种方法、{len(result['seeds'])}个种子的结果。下面均为各种子分别计算指标后取均值，未集成预测分数。阈值仅由验证集选择；消息结果和会话结果分别解释。", "",
        "| 任务 | 有效测试数／阳性数 | CST测试AP | 验证集所选主题基线 | 基线测试AP | CST－基线 | 配对95%区间 |",
        "|---|---:|---:|---|---:|---:|---|"]
    if result.get("evaluation_revision"):
        lines[2:2] = [
            "本表采用固定训练权重的原文推理修订评估：21个训练模型未改，三种语义方法统一训练侧预热后重新完成开发阈值冻结和测试预测，传统四方法继承已验收原文结果。原缓存评估与BERTopic一致性失败记录独立保留；此次工程修订未根据测试成绩选择配置。训练仍使用原缓存表示，不将此次重放称为重新训练。", ""]
    directions = []
    for task in result["main_tasks"]:
        if (task, "cst_v1_1") not in test:
            continue
        cst = test[task, "cst_v1_1"]
        baseline = result["selected_baselines"][task]["method"]
        reference = test[task, baseline]
        difference, low, high = None, None, None
        for contrast in result["contrasts"]:
            if contrast["task"] != task or contrast["metric"] != "AP":
                continue
            if contrast["left"] == "cst_v1_1" and contrast["right"] == baseline:
                difference, low, high = contrast["difference"], contrast["ci_low"], contrast["ci_high"]
            elif contrast["right"] == "cst_v1_1" and contrast["left"] == baseline:
                difference = -contrast["difference"]
                low = -contrast["ci_high"] if contrast["ci_high"] is not None else None
                high = -contrast["ci_low"] if contrast["ci_low"] is not None else None
        if difference is not None:
            directions.append(difference)
        interval = f"[{number(low, signed=True)}, {number(high, signed=True)}]" if low is not None and high is not None else "N/A"
        lines.append(f"| {tasks.get(task, task)} | {cst['n']:,}／{cst['positive']:,} | {number(cst['AP_mean'])} | {display.get(baseline, baseline)} | {number(reference['AP_mean'])} | {number(difference, signed=True)} | {interval} |")
    lines += ["", "各任务保留自身的自然阳性比例，AP只能在同一任务内比较。ScamSpot的正类合并垃圾评论与诈骗评论；HateBR提供冒犯监督，不能将其称为已验证的仇恨识别。"]
    if directions:
        gains = sum(value > 0 for value in directions)
        losses = sum(value < 0 for value in directions)
        tied = len(directions) - gains - losses
        lines += ["", f"在表中{len(directions)}个任务的平均AP点估计上，CST相对验证集所选主题基线有{gains}项较高、{losses}项较低、{tied}项相同。这一计数只描述方向，不是跨任务总分，也不代表所有比较均有统计证据。"]
    diagnostic = []
    for task in result["main_tasks"]:
        if (task, "minilm_lr") in test and (task, "cst_v1_1") in test:
            first, second = test[task, "cst_v1_1"]["AP_mean"], test[task, "minilm_lr"]["AP_mean"]
            if first is not None and second is not None:
                diagnostic.append(f"{tasks.get(task, task)} {number(first-second, signed=True)}")
    if diagnostic:
        lines += ["", "CST相对MiniLM＋LR的同任务AP差值为：" + "；".join(diagnostic) + "。两者采用相同的PCA与回复聚合结构，该参照主要检查字符通道及其风险分数传播在语义编码基础上的增量；差值本身不能单独证明MIL、回复结构或聚合机制的因果作用。"]
    boot = next(iter(result["bootstrap"].values()))
    lines += ["", f"区间来自{boot['attempted']:,}次配对依赖簇bootstrap，各种子共享同一次抽样。SCCD按会话、CONDA按对局、HateBR按帖子，跨组重复连接为同簇；退化抽样数见完整统计记录。区间条件于已训练的这些种子，不包含验证集挑选基线的不确定性；全部两两区间均为探索性的未校正95%区间，不作全称显著性保证。", "",
        "优先检查CST差值为负或区间跨零的任务，再结合精确率、召回率、≤30字符分层与同一条预测核验误报和漏报。common16会话对照用于检查聚合表示的影响。分类成绩不等于主题发现质量，本次不报告没有金标准支撑的主题准确率。", ""]
    if result.get("dependence_protocol", {}).get("verified"):
        lines += ["统计协议纠正：依赖簇直接使用准备阶段已冻结的eval_group。这些ID由全部保留的训练、验证、测试元数据建立传递闭包，包含未知标签评论及跨来源重复连接，再对各测试任务取子集；不在测试分区内重新定义独立簇。先前SCCD仅在测试内部重建为55簇，本次按冻结全局口径为47簇，旧口径输出保存在statistical_revisions/test_only_clusters_55。此项事后纠正源于分组口径不一致，未按区间宽度或方法输赢择优；模型、预测、阈值、8项验证参照、点估计与成本记录均保持原样。", ""]
    boundaries = []
    for task in ("sccd.attack", "sccd.conversation", "hatebr.attack"):
        meta = result["bootstrap"].get(task)
        if meta is None:
            continue
        unit = "会话" if task == "sccd.conversation" else "条已标注评论"
        boundaries.append(f"{tasks[task]}为{meta['n_clusters']}个合并依赖簇，最大簇含{meta['largest_cluster_n']:,}{unit}（{meta['largest_cluster_fraction']:.1%}），记录数集中度对应的Kish诊断值为{meta['size_concentration_effective_cluster_count']:.2f}")
    if boundaries:
        lines += ["依赖结构边界：" + "；".join(boundaries) + "。Kish值仅描述簇大小集中程度，不是真正自由度或独立样本数估计。SCCD的大簇和HateBR有限的帖子簇会限制区间的稳定性及外推；即使探索性区间未跨零，也不能据此宣称整体稳定显著优越。不足2簇的任务区间标为N/A，不报告零宽区间。", ""]
    if result["missing_costs"]:
        lines.append("成本记录尚缺：" + "、".join(display.get(m, m) for m in result["missing_costs"]) + "；精确成本表将未知值标为N/A，不能据此认定任何方法更快或更省内存。")
    else:
        startup = ("原文修订的语义方法将延迟加载的编码器初始化、固定训练批次预热及首次编译（如触发）计入startup_warmup_seconds；first_inference_seconds是完成该预热后首条完整SCCD测试会话的耗时。传统方法继承的原测量未采用该语义预热定义，不能将两者首次推理列直接当成相同冷请求。额外原文DEV加载、预热与重放准备在raw_dev_inference_preparation_seconds单列，不计入原训练时间，亦不含阈值扫描和审计。" if result.get("evaluation_revision") else "延迟加载的编码器初始化以及BERTopic首次编译（如触发）计入first_inference_seconds，该首次推理只覆盖第一条完整SCCD会话。")
        lines.append("成本仅使用种子42同机逐方法测量。training_seconds_cold是缓存拟合及验证选择的实测时间，加上训练／验证首次编码的实测时间；同一共享编码器实际只执行一次，为每套独立语义系统的比较各计入一次。该值不是另行完整冷启动端到端重跑，也不表示清空了操作系统缓存，不能将各方法该列相加当作campaign总耗时。驻留推理均值来自3次完整测试。model_load_seconds只计joblib模型包反序列化，不包含完整编码器启动。" + startup + "后续驻留测量不再加载编码器，仍逐次编码完整原文。峰值RSS包括编码、拟合、推理及子进程，不能解释为最低服务器内存。依赖大小按已列包的范围统计，不代表完整Python或操作系统环境。冷启动／加载／首次推理定义、包统计范围、会话中位数／P95和分来源吞吐量见成本表。")
    lines += ["", "计时范围为统一风险预测入口：文本表示、消息风险头及main/common16会话评分。传统主题基线的主题成员变换作为文本表示已计入；动态讨论主题聚类与短语命名未计入。因此，这些时间不等于演示系统完整主题分析、页面渲染或导出的端到端延迟。计时机器同时存在桌面应用负载，并非独占受控硬件基准；成本为当前环境的条件性实测，不能据此推断无资源竞争时的最低延迟。", "",
        "本次评价的是v1.1架构的全量重训版本。与旧权重相比的数据规模和监督协议变化均需单列；既往查看过的测试部分不描述为首次未见测试。", "", "## 数据与结果入口", "",
        "- [数据清洗与划分清单](../prepared/manifest.json)",
        "- [全部指标均值及种子标准差](summary_metrics.csv)",
        "- [逐方法、逐种子精确指标](per_seed_metrics.csv)",
        "- [所有配对差值及区间](paired_contrasts.csv)",
        "- [长度分层](length_strata.csv)、[SCCD Top-3及危险评论召回](top3.csv)",
        "- [精确成本表](costs.csv)、[独立复算与来源指纹](verification.json)",
        "- [完整机器可读汇总](analysis.json)", "",
        f"## 全部{len(result['methods']) * len(result['seeds'])}次训练结果", "",
        "| 方法 | 各种子的模型配置与完整测试预测 |", "|---|---|"]
    for method in result["methods"]:
        links = []
        for seed in result["seeds"]:
            prefix = f"../runs/{method}/seed_{seed}"
            links.append(f"种子{seed}：[配置]({prefix}/run_manifest.json)／[消息预测]({prefix}/predictions_message_test.jsonl)／[会话预测]({prefix}/predictions_session_test.jsonl)")
        lines.append(f"| {display.get(method, method)} | " + "；".join(links) + " |")
    return "\n".join(lines) + "\n"


def summarize(output: str | Path, bootstrap: int = 2000, *,
              methods: Sequence[str] = METHODS, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Analyze completed runs; fail closed if any requested cohort/run is missing."""
    root = Path(output)
    frozen_groups, grouping_audit = _prepared_global_groups(root)
    destination = root / "analysis"
    destination.mkdir(parents=True, exist_ok=True)
    predictions: dict[tuple[str, int, str], dict[str, TaskPrediction]] = {}
    per_seed, strata, top3, hashes, costs = [], [], [], {}, []
    if grouping_audit["verified"]:
        for path in (root / "prepared/manifest.json", root / "prepared/records.jsonl"):
            hashes[str(path.relative_to(root))] = _sha(path)
    missing_costs = []
    for method, seed in itertools.product(methods, seeds):
        run = root / "runs" / method / f"seed_{seed}"
        manifest_path = run / "run_manifest.json"
        manifest = _json(manifest_path)
        if manifest.get("method", method) != method or int(manifest.get("seed", seed)) != seed:
            raise ValueError(f"Run identity mismatch: {run}")
        hashes[str(manifest_path.relative_to(root))] = _sha(manifest_path)
        for split in ("dev", "test"):
            message_path = run / f"predictions_message_{split}.jsonl"
            session_path = run / f"predictions_session_{split}.jsonl"
            messages, sessions = _jsonl(message_path), _jsonl(session_path)
            _verify_prediction_global_groups(messages, split, frozen_groups)
            for path in (message_path, session_path):
                hashes[str(path.relative_to(root))] = _sha(path)
            task_predictions = _extract_tasks(manifest, messages, sessions)
            predictions[method, seed, split] = task_predictions
            for task, pred in task_predictions.items():
                metric = independent_metrics(pred.labels, pred.scores, pred.threshold)
                per_seed.append({"method": method, "seed": seed, "split": split, "task": task, **metric})
                if pred.lengths is not None:
                    for name, mask in (("le_30_chars", pred.lengths <= 30), ("gt_30_chars", pred.lengths > 30)):
                        strata.append({"method": method, "seed": seed, "split": split, "task": task,
                                       "stratum": name, **independent_metrics(pred.labels[mask], pred.scores[mask], pred.threshold)})
            top3.append({"method": method, "seed": seed, "split": split, **top3_metrics(messages)})
        for task in predictions[method, seed, "dev"]:
            first, second = predictions[method, seed, "dev"][task], predictions[method, seed, "test"][task]
            if first.threshold != second.threshold:
                raise ValueError("Validation and test thresholds differ")
            if set(first.record_ids) & set(second.record_ids):
                raise ValueError(f"Dev/test prediction ID overlap for {task}")
        if seed == 42:
            cost_path = run / "cost.json"
            if cost_path.exists():
                cost = {"method": method, "seed": seed, **_json(cost_path)}
                provenance_path = run / "dev_provenance.json"
                if (root / "revision_manifest.json").exists():
                    provenance = _json(provenance_path)
                    hashes[str(provenance_path.relative_to(root))] = _sha(provenance_path)
                    startup = provenance.get("startup") or {}
                    cost["raw_dev_inference_preparation_seconds"] = (
                        float(provenance.get("replay_seconds", 0))
                        + float(startup.get("model_load_seconds", 0))
                        + float(startup.get("startup_warmup_seconds", 0)))
                    cost["raw_dev_inference_preparation_definition"] = (
                        "additional revision DEV raw prediction plus package loading and fixed TRAIN warm-up; excludes threshold scanning and audit; inherited traditional DEV is not re-executed and has zero additional replay cost; not included in the inherited original training timings")
                costs.append(cost)
                hashes[str(cost_path.relative_to(root))] = _sha(cost_path)
            else:
                missing_costs.append(method)
    summary = _summary(per_seed, methods, seeds)
    baselines = [m for m in TOPIC_BASELINES if m in methods]
    if not baselines:
        raise ValueError("At least one topic baseline is required")
    selected = {}
    for task in (*MAIN_TASKS, "sccd.conversation_common16"):
        candidates = [row for row in summary if row["split"] == "dev" and row["task"] == task and row["method"] in baselines]
        winner = max(candidates, key=lambda row: (row["AP_mean"] if row["AP_mean"] is not None else -1,
                                                  -baselines.index(row["method"])))
        selected[task] = {"method": winner["method"], "validation_mean_AP": winner["AP_mean"],
                          "rule": "highest three-seed mean validation AP; fixed method order breaks exact ties"}
    contrasts, bootstrap_meta = [], {}
    for task_index, task in enumerate((*MAIN_TASKS, "sccd.conversation_common16")):
        paired = {m: [predictions[m, seed, "test"][task] for seed in seeds] for m in methods}
        distribution, metadata = bootstrap_paired(paired, count=bootstrap, seed=20261014 + task_index)
        bootstrap_meta[task] = metadata
        np.savez_compressed(destination / f"bootstrap_{task.replace('.', '_')}.npz", **distribution)
        observed = {m: np.array([np.mean([independent_metrics(x.labels, x.scores, x.threshold)[metric]
                                         for x in paired[m]]) for metric in METRICS]) for m in methods}
        for left, right in itertools.combinations(methods, 2):
            difference = distribution[left] - distribution[right]
            for j, metric in enumerate(METRICS):
                valid = difference[:, j][np.isfinite(difference[:, j])]
                lo, hi = np.quantile(valid, [.025, .975]).tolist() if len(valid) else [None, None]
                contrasts.append({"task": task, "metric": metric, "left": left, "right": right,
                                  "difference": float(observed[left][j] - observed[right][j]),
                                  "ci_low": lo, "ci_high": hi, "valid_bootstrap": len(valid),
                                  "attempted_bootstrap": metadata["attempted"], "selected_dev_baseline": selected[task]["method"],
                                  "interval": "paired cluster percentile 95%; unadjusted exploratory"})
    result = {"schema_version": 1, "status": "complete", "methods": list(methods), "seeds": list(seeds),
              "main_tasks": list(MAIN_TASKS), "metrics": list(METRICS), "summary": summary,
              "selected_baselines": selected, "contrasts": contrasts, "bootstrap": bootstrap_meta,
              "costs": costs, "missing_costs": missing_costs,
              "analysis_source_sha256": _sha(Path(__file__)),
              "dependence_protocol": {**grouping_audit,
                  "posthoc_correction": grouping_audit["verified"],
                  "reason": "Resolve global prepared eval_group versus TEST-only reconstruction mismatch; not selected by comparative results",
                  "previous_statistical_scope": "SCCD session graph reconstructed within TEST only (55 clusters in this cohort)",
                  "corrected_statistical_scope": "Frozen all-source train/dev/test eval_group closure (47 SCCD TEST clusters in this cohort)",
                  "previous_outputs_archive": "statistical_revisions/test_only_clusters_55/",
                  "unchanged": "models, predictions, thresholds, validation references, point metrics and cost records"},
              "interpretation": {"pooled_score": "not produced: source prevalences and labels differ",
                  "AP": "recall-increment weighted precision; natural source prevalence retained",
                  "uncertainty": "95% paired cluster intervals conditional on the three trained seeds; selection uncertainty not included",
                  "topic_quality": "classification is not topic quality; no topic accuracy is inferred",
                  "test_exposure": "Previously inspected held-out portions are not described as untouched test data"}}
    revision_path = root / "revision_manifest.json"
    if revision_path.exists():
        revision = _json(revision_path)
        result["evaluation_revision"] = {"identity": revision["identity"],
            "base_output": revision["base_output"], "scope": revision["scope"],
            "manifest_sha256": _sha(revision_path), "trained_weights_unchanged": True}
    for name, rows in (("per_seed_metrics.csv", per_seed), ("summary_metrics.csv", summary),
                       ("length_strata.csv", strata), ("top3.csv", top3), ("paired_contrasts.csv", contrasts),
                       ("costs.csv", _cost_table(costs, methods))):
        _write_csv(destination / name, rows)
    _write_json(destination / "selected_baselines.json", selected)
    _write_json(destination / "analysis.json", result)
    _write_json(destination / "verification.json", {"status": "passed", "independent_metric_recomputation": True,
        "cohort_identity_labels_clusters_checked": True, "validation_thresholds_frozen": True,
        "run_count": len(methods) * len(seeds), "no_seed_score_ensemble": True,
        "analysis_source_sha256": _sha(Path(__file__)),
        "dependence_protocol": grouping_audit,
        "prediction_and_manifest_hashes": hashes, "missing_costs": missing_costs})
    (destination / "analysis_summary.md").write_text(_analysis_summary(result), encoding="utf-8")
    return result
