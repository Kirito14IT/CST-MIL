"""Validated comparison artifacts for completed, fixed-cohort SCCD runs."""
# Markdown templates deliberately keep complete paragraphs on a single source line.
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .common import file_hash, fingerprint, read_json, read_jsonl, write_json
from .dataset import BENCHMARK, load_suite
from .evaluation import binary_metrics, bootstrap_metrics, paired_differences

BASELINES = ("lda", "btm", "gsdmm", "nmf", "bertopic")
EXPECTED_COUNTS = {"pilot300": (300, 17312, 180, 60, 60),
                   "full677": (677, 38872, 406, 135, 136)}
EXPECTED_TEST_SUBSETS = {"pilot300": (60, 0), "full677": (60, 76)}
SHARED_CONFIG_KEYS = (
    "seed", "n_topics", "threads", "window_size", "stride", "membership_threshold",
    "relative_threshold", "strong_threshold", "evidence_top_k", "session_c", "session_C",
)


def _number(value: Any, digits: int = 4) -> str:
    return "NA" if value is None else f"{float(value):.{digits}f}"


def _topic_count_distribution(records: list[dict]) -> dict[str, Any]:
    """Describe retained output counts, without treating them as semantic truth."""
    counts = [len(record.get("topics", [])) for record in records]
    bins = {str(number): counts.count(number) for number in range(4)}
    bins[">3"] = sum(number > 3 for number in counts)
    return {"n_sessions": len(records), "bins": bins,
            "maximum": max(counts) if counts else None,
            "mean": float(np.mean(counts)) if counts else None,
            "exact_counts": {str(number): count for number, count in sorted(Counter(counts).items())},
            "unit": "retained discussion topics per complete session",
            "has_semantic_count_ground_truth": False}


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(cell(value) for value in row) + " |" for row in rows),
    ])


def _completed_run_errors(runs: list[dict]) -> list[dict]:
    """Keep interruption evidence from validated runs, separate from final failures."""
    records = []
    for run in runs:
        for path in sorted(run["path"].glob("error_*.json")):
            error = read_json(path)
            message = str(error.get("message", ""))
            if "CPU affinity" in message:
                category = "cpu_affinity_resource_rejection"
                stage = "inference" if "inference" in message.lower() else "training"
            elif error.get("type") == "KeyboardInterrupt":
                category, stage = "user_keyboard_interrupt", "see_traceback"
            else:
                category, stage = "other_recorded_error", "see_traceback"
            records.append({"method": run["name"], "path": str(path),
                            "category": category, "stage": stage,
                            "final_status": "complete", "error": error})
    return records


def _full_subset_section(subsets: dict) -> str:
    parts = []
    for name, label in [("fresh_test", "新增测试 fresh"),
                        ("previously_seen_test", "已查看测试 seen")]:
        values = subsets[name]
        metrics = values["methods"]
        n_sessions = values["selected"]["n_sessions"]
        parts.append(f"### {label}{n_sessions}：各方法风险指标\n\n" + _table(
            ["方法", "会话数", "macro-F1", "CB Recall", "CB Precision", "AP", "ROC-AUC"],
            [[method, row["n_sessions"], *[_number(row[key]) for key in (
                "macro_f1", "recall_cb", "precision_cb", "ap", "roc_auc",
            )]] for method, row in metrics.items()],
        ))
        paired = values["paired_difference"]
        parts.append("选定 CST 相对本轮验证集最强基线的 macro-F1 差值为 "
                     f"{_number(paired['delta_macro_f1'])}，配对会话 bootstrap 95% 区间为 "
                     f"[{_number(paired['ci95'][0])}, {_number(paired['ci95'][1])}]。")
        if name == "fresh_test":
            leaders = []
            for key, title in [("macro_f1", "macro-F1"), ("ap", "AP")]:
                available = {method: row[key] for method, row in metrics.items()
                             if row[key] is not None}
                if not available:
                    continue
                maximum = max(available.values())
                names = [method for method, value in available.items()
                         if np.isclose(value, maximum, rtol=0, atol=1e-12)]
                leaders.append(f"{title} 点值最高为 " + "、".join(f"`{method}`" for method in names)
                               + ("（并列）" if len(names) > 1 else ""))
            parts.append("新增测试子集：" + "；".join(leaders) + "。排名只描述该固定测试子集，"
                         "不使用测试指标重选版本；AP 衡量分数排序，macro-F1 还受已冻结阈值影响，二者不能互相替代。")
    return "\n\n".join(parts) + "\n"


def _assert_metrics(actual: dict, saved: dict, path: Path) -> None:
    for key, value in actual.items():
        if key not in saved:
            raise ValueError(f"Missing {key} in metrics: {path}")
        if value is None:
            equal = saved[key] is None
        else:
            equal = saved[key] is not None and np.isclose(value, saved[key], rtol=1e-8, atol=1e-10)
        if not equal:
            raise ValueError(f"Recomputed {key} differs from saved metrics: {path}")


def _load_run(path: Path, manifest: dict, sessions: list[dict]) -> dict:
    """Reject edited, partial, duplicated or cross-cohort prediction artifacts."""
    saved = read_json(path / "run_manifest.json")
    binding = saved["binding"]
    if fingerprint(binding) != saved["binding_sha256"]:
        raise ValueError(f"Run binding was modified: {path}")
    if binding["suite"] != manifest["suite"] or binding["cohort"] != manifest["content_fingerprint"]:
        raise ValueError(f"Run has a different cohort: {path}")
    expected_name = (f"cst_mil_{binding['revision']}" if binding["method"] == "cst_mil"
                     else binding["method"])
    if path.name != expected_name:
        raise ValueError(f"Method identity differs from run directory: {path}")
    trained = read_json(path / "TRAINED.json")
    completed = read_json(path / "COMPLETED.json")
    if completed.get("status") != "complete" or completed["completed"] != len(sessions):
        raise ValueError(f"Run is incomplete: {path}")
    model_path = path / "model.joblib"
    if (file_hash(model_path) != trained["model_sha256"]
            or completed["model_sha256"] != trained["model_sha256"]
            or model_path.stat().st_size != trained["model_bytes"]):
        raise ValueError(f"Model hash or size differs: {path}")
    predictions = read_jsonl(path / "predictions.jsonl")
    expected = {session["post_id"]: session for session in sessions}
    if (len(predictions) != len(expected)
            or {row["post_id"] for row in predictions} != set(expected)):
        raise ValueError(f"Predictions are missing, duplicated, or from another cohort: {path}")
    for row in predictions:
        original = expected[row["post_id"]]
        for key in ("split", "label", "pilot_member", "fresh_test"):
            if row[key] != original[key]:
                raise ValueError(f"Prediction {key} differs from cohort: {path}/{row['post_id']}")
        if row["prediction"] != int(row["risk_probability"] >= row["threshold"]):
            raise ValueError(f"Prediction decision/threshold mismatch: {path}/{row['post_id']}")
    by_id = {row["post_id"]: row for row in predictions}
    wrappers = []
    for wrapper_path in sorted((path / "sessions").glob("*.json")):
        wrapper = read_json(wrapper_path)
        row = wrapper["prediction"]
        if (wrapper["binding"] != saved["binding_sha256"]
                or wrapper["model"] != trained["model_sha256"]
                or fingerprint(row) != wrapper["prediction_sha256"]
                or row["post_id"] not in by_id
                or fingerprint(row) != fingerprint(by_id[row["post_id"]])):
            raise ValueError(f"Per-session result binding/content mismatch: {wrapper_path}")
        wrappers.append(wrapper)
    if (len(wrappers) != len(sessions)
            or {w["prediction"]["post_id"] for w in wrappers} != set(expected)):
        raise ValueError(f"Per-session checkpoints are incomplete/duplicated: {path}")
    tests = [row for row in predictions if row["split"] == "test"]
    dev_rows = [row for row in predictions if row["split"] == "dev"]
    metrics = read_json(path / "metrics.json")
    dev = read_json(path / "dev_metrics.json")
    _assert_metrics(binary_metrics(tests), metrics["test"], path / "metrics.json")
    _assert_metrics(binary_metrics(dev_rows), dev["risk"], path / "dev_metrics.json")
    summary = read_json(path / "training_summary.json")
    if summary.get("message_labels_used_for_training") is not False:
        raise ValueError(f"Training supervision declaration is missing: {path}")
    if set(summary.get("training_session_ids", [])) != {
        session["post_id"] for session in sessions if session["split"] == "train"
    }:
        raise ValueError(f"Training IDs do not match the training partition: {path}")
    return {
        "name": path.name, "path": path, "binding": binding, "trained": trained,
        "summary": summary, "dev": dev, "metrics": metrics, "predictions": predictions,
        "tests": tests, "wrappers": wrappers,
    }


def _external_footprint(run: dict) -> dict[str, Any]:
    """Count declared, currently present external assets separately from the pickle."""
    info = run["trained"].get("backend_info", {})
    encoder_bytes, runtime_bytes, cache_bytes = 0, 0, 0
    sources: list[str] = []
    manifest_name = info.get("encoder_manifest")
    if run["binding"]["method"] == "bertopic" and not manifest_name:
        if info.get("n_topics") != 0:
            raise ValueError("BERTopic encoder manifest is missing; disk footprint is unverifiable")
    if manifest_name:
        path = Path(manifest_name)
        encoder = read_json(path)
        if info.get("encoder_fingerprint") != encoder["model_fingerprint"]:
            raise ValueError("BERTopic encoder manifest fingerprint differs from trained model")
        for relative, record in encoder["files"].items():
            asset = (path.parent / relative).resolve()
            if not asset.is_relative_to(path.parent.resolve()):
                raise ValueError("Encoder asset path leaves its snapshot")
            if (not asset.is_file() or asset.stat().st_size != record["bytes"]
                    or file_hash(asset) != record["sha256"]):
                raise ValueError(f"External encoder file missing or changed: {asset}")
            encoder_bytes += asset.stat().st_size
        sources.append(str(path))
        cache = Path(run["binding"]["config"].get("bertopic", {}).get(
            "embedding_cache_dir", path.parent.parents[1] / "cache" / "bertopic_embeddings"
        ))
        if cache.is_dir():
            cache_bytes = sum(item.stat().st_size for item in cache.rglob("*.npy"))
    java = info.get("build", {}).get("java")
    if java:
        java_path = Path(java)
        if not java_path.is_file():
            raise ValueError(f"Declared Java runtime is missing: {java_path}")
        runtime_root = java_path.parent.parent
        runtime_bytes = sum(item.stat().st_size for item in runtime_root.rglob("*")
                            if item.is_file())
        sources.append(str(runtime_root))
    return {"external_encoder_bytes": encoder_bytes, "external_runtime_bytes": runtime_bytes,
            "shared_embedding_cache_bytes": cache_bytes, "external_asset_sources": sources,
            "known_deployed_disk_bytes": run["trained"]["model_bytes"] + encoder_bytes + runtime_bytes}


def _row(run: dict, selected_name: str) -> dict:
    metric = run["metrics"]
    trained = run["trained"]
    backend = trained.get("backend_info", {})
    timings = run["summary"].get("timings", {})
    footprint = _external_footprint(run)
    result = {
        "method": run["name"], "selected": run["name"] == selected_name,
        "dev_macro_f1": run["dev"]["risk"]["macro_f1"],
        **{f"test_{key}": value for key, value in metric["test"].items()},
        **{f"topic_{key}": metric["topics"].get(key) for key in (
            "npmi", "topic_diversity", "coverage", "n_topics", "degenerate_topic_fraction",
        )},
        **{f"evidence_{key}": metric.get("evidence", {}).get(key) for key in (
            "comment_ap", "hit_rate_at_3", "mean_recall_at_3", "n_labeled_messages",
            "n_sessions_with_positive_comments",
        )},
        "train_seconds": trained["training_seconds"],
        "inference_all_seconds": sum(float(w["inference_seconds"]) for w in run["wrappers"]),
        "inference_test_seconds": sum(float(w["inference_seconds"]) for w in run["wrappers"]
                                      if w["prediction"]["split"] == "test"),
        "peak_rss_bytes": max([trained["training_peak_rss_bytes"],
                               *(w["peak_rss_bytes"] for w in run["wrappers"])]),
        "model_bytes": trained["model_bytes"],
        "formal_fit_attempts": trained["formal_fit_attempts"],
        "risk_classifier_fit_count": run["summary"].get("risk_classifier_fit_count"),
        "topic_fit_seconds": timings.get("topic_fit_seconds"),
        "risk_fit_seconds": timings.get("risk_fit_seconds"),
        "preparation_seconds": timings.get("preparation_seconds"),
        "train_encoding_cold_seconds": backend.get("encoding_cold_seconds"),
        "train_encoding_cached_seconds": backend.get("encoding_cached_seconds"),
        "train_encoder_load_seconds": backend.get("encoder_load_seconds"),
        "inference_encoding_counters_available": all(
            "encoding_counters" in wrapper for wrapper in run["wrappers"]
        ),
        "bertopic_training_noise_fraction": backend.get("training_noise_fraction"),
        "model_sha256": trained["model_sha256"],
        "binding_sha256": fingerprint(run["binding"]),
        **footprint,
    }
    for key in ("encoding_cold_seconds", "encoding_cached_seconds", "encoder_load_seconds",
                "encoded_texts", "cached_texts"):
        result[f"inference_{key}"] = (
            sum(wrapper["encoding_counters"].get(key, 0) for wrapper in run["wrappers"])
            if result["inference_encoding_counters_available"] else None
        )
    ci = metric.get("test_ci", {}).get("macro_f1")
    if ci is None:
        ci = bootstrap_metrics(run["tests"])["macro_f1"]
    result["test_macro_f1_ci_low"] = ci["low"]
    result["test_macro_f1_ci_high"] = ci["high"]
    return result


def _figures(rows: list[dict], destination: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    labels = [row["method"].replace("cst_mil_", "CST-") + (" *" if row["selected"] else "")
              for row in rows]
    positions = np.arange(len(rows))
    colors = ["#D55E00" if row["selected"] else "#56B4E9" for row in rows]
    filenames: list[str] = []

    def finish(fig: Any, name: str) -> None:
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(destination / filename, dpi=300, bbox_inches="tight")
            filenames.append(filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, max(3.4, len(rows) * .43)),
                             layout="constrained", sharey=True)
    means = np.asarray([row["test_macro_f1"] for row in rows])
    low = np.asarray([row["test_macro_f1_ci_low"] for row in rows])
    high = np.asarray([row["test_macro_f1_ci_high"] for row in rows])
    axes[0].barh(positions, means, color=colors, edgecolor="black", linewidth=.4)
    axes[0].errorbar(means, positions, xerr=[np.maximum(means - low, 0),
                                          np.maximum(high - means, 0)],
                     fmt="none", ecolor="black", capsize=3)
    axes[1].barh(positions, [row["test_recall_cb"] for row in rows], color=colors,
                 edgecolor="black", linewidth=.4)
    for ax, label in zip(axes, ["Test macro-F1 (95% session bootstrap CI)", "Test CB recall"],
                         strict=True):
        ax.set_xlabel(label)
        ax.set_xlim(0, 1)
        ax.set_yticks(positions, labels)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    finish(fig, "risk_comparison")

    fig, axes = plt.subplots(1, 3, figsize=(12, max(3.4, len(rows) * .43)),
                             layout="constrained", sharey=True)
    for ax, key, label, limits in zip(axes,
            ["topic_npmi", "topic_topic_diversity", "topic_coverage"],
            ["Test-window NPMI", "Top-10 word diversity", "Message assignment coverage"],
            [(-1, 1), (0, 1), (0, 1)], strict=True):
        ax.barh(positions, [row[key] for row in rows], color=colors,
                edgecolor="black", linewidth=.4)
        ax.set_xlabel(label)
        ax.set_xlim(*limits)
        ax.set_yticks(positions, labels)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    finish(fig, "topic_quality")

    fig, axes = plt.subplots(2, 2, figsize=(12, max(7, len(rows) * .7)), layout="constrained")
    specifications = [
        ("train_seconds", "Training + dev threshold selection (seconds)", 1),
        ("inference_all_seconds", "Inference, all saved sessions (seconds)", 1),
        ("peak_rss_bytes", "Peak process-tree RSS (MiB)", 1024**2),
        ("known_deployed_disk_bytes", "Model + provisioned external resources (MiB)", 1024**2),
    ]
    for ax, (key, label, scale) in zip(axes.flat, specifications, strict=True):
        ax.barh(positions, [row[key] / scale for row in rows], color=colors,
                edgecolor="black", linewidth=.4)
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
        ax.grid(axis="x", alpha=.2)
        ax.set_axisbelow(True)
    finish(fig, "time_memory")
    return filenames


def _cases(selected: dict, baseline: dict, messages: list[dict]) -> str:
    originals = {(message["post_id"], message["comment_id"]): message for message in messages}
    reference = {row["post_id"]: row for row in baseline["tests"]}
    selected_tests = selected["tests"]
    failures = sorted([row for row in selected_tests if row["prediction"] != row["label"]],
                      key=lambda row: (-abs(row["risk_probability"] - .5), row["post_id"]))
    multi = sorted(selected_tests, key=lambda row: (-len(row.get("topics", [])), row["post_id"]))
    chosen = []
    for row in (failures[:1] + multi[:1]):
        if row["post_id"] not in {item["post_id"] for item in chosen}:
            chosen.append(row)
    parts = ["案例采用预定规则：先列风险分数离 0.5 最远的一条误判（若存在），再列主题数量最多的测试会话。"
             "它们只帮助检查模型输出，不是随机样本或额外人工标注。"]
    for row in chosen:
        other = reference[row["post_id"]]
        parts.append(f"\n会话 `{row['post_id']}`，数据集标签 `{row['label']}`："
                     f"选定 CST 风险分数 {_number(row['risk_probability'])}、"
                     f"预测 {row['prediction']}；参照 {baseline['name']} 风险分数 "
                     f"{_number(other['risk_probability'])}、预测 {other['prediction']}。")
        for name, prediction in [(selected["name"], row), (baseline["name"], other)]:
            topic_text = "; ".join(str(topic.get("title", ""))
                                   for topic in prediction.get("topics", [])[:3])
            parts.append(f"\n{name} 输出 {len(prediction.get('topics', []))} 个主题；"
                         f"展示至多前三个抽取式标题：{topic_text or '无保留主题'}。")
            for evidence in prediction.get("evidence", [])[:2]:
                key = (prediction["post_id"], evidence["comment_id"])
                original = originals.get(key)
                if original is None:
                    # Root-fallback sessions have no source comment record.
                    parts.append(f"原帖回退证据 `{evidence['comment_id']}`，见逐会话 JSON。")
                    continue
                raw = str(original.get("raw_text", ""))
                if raw != str(evidence.get("raw_text", "")):
                    raise ValueError(f"Evidence quotation differs from source message: {key}")
                quote = raw.replace("\n", " ").replace("\r", " ")
                excerpt = quote[:180] + ("…（节选）" if len(quote) > 180 else "")
                parts.append(f"\n原文证据 `{evidence['comment_id']}` / 匿名发言人 "
                             f"`{original.get('speaker_id')}` / 风险分数 "
                             f"{_number(evidence.get('risk_probability', evidence.get('risk_score')))}："
                             f"\n\n> {excerpt}")
    return "\n".join(parts)


def create_comparison(suite: str) -> dict:
    """Write a report only after all six methods and selected revision are complete."""
    if suite not in EXPECTED_COUNTS:
        raise ValueError("Formal comparison supports pilot300 and full677 only")
    manifest, sessions, messages = load_suite(suite)
    expected = EXPECTED_COUNTS[suite]
    counts = Counter(session["split"] for session in sessions)
    observed = (len(sessions), len(messages), counts["train"], counts["dev"], counts["test"])
    if observed != expected or manifest["n_sessions"] != len(sessions):
        raise ValueError(f"Unexpected SCCD cohort counts: {observed}; expected {expected}")
    old_test = sum(session["split"] == "test" and session["pilot_member"] for session in sessions)
    fresh_test = sum(session["split"] == "test" and session["fresh_test"] for session in sessions)
    if ((old_test, fresh_test) != EXPECTED_TEST_SUBSETS[suite]
            or any(session["pilot_member"] and session["fresh_test"] for session in sessions)):
        raise ValueError("Previously seen and fresh test partitions do not match the protocol")
    selection_path = BENCHMARK / "selection.json"
    if not selection_path.exists():
        raise ValueError("No frozen dev-only CST selection exists")
    selection = read_json(selection_path)
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("Frozen selection does not declare dev-only choice")
    selected_name = f"cst_mil_{selection['revision']}"
    run_root = BENCHMARK / "runs" / suite
    required = {*BASELINES, selected_name}
    available = {path.name for path in run_root.iterdir()
                 if path.is_dir() and (path / "COMPLETED.json").exists()} if run_root.exists() else set()
    missing = sorted(required - available)
    if missing:
        raise ValueError(f"Incomplete comparison; missing complete methods: {', '.join(missing)}")
    names = list(BASELINES) + sorted(name for name in available if name.startswith("cst_mil_"))
    runs = [_load_run(run_root / name, manifest, sessions) for name in names]
    reference_runtime = fingerprint(runs[0]["binding"]["runtime"])
    shared = {key: runs[0]["binding"]["config"].get(key) for key in SHARED_CONFIG_KEYS}
    for run in runs:
        binding = run["binding"]
        if fingerprint(binding["runtime"]) != reference_runtime:
            raise ValueError("Cannot compare mixed source or dependency environments")
        if {key: binding["config"].get(key) for key in SHARED_CONFIG_KEYS} != shared:
            raise ValueError("Cannot compare different seeds, windows, thresholds, or capacities")
        if binding["config"].get("seed") != 42:
            raise ValueError("This protocol requires the one preregistered seed, 42")
    selected = next(run for run in runs if run["name"] == selected_name)
    if fingerprint(selected["binding"]["config"]) != fingerprint(selection["config"]):
        raise ValueError("Selected revision configuration does not match frozen selection")
    baseline = sorted([run for run in runs if run["name"] in BASELINES], key=lambda run: (
        -run["dev"]["risk"]["macro_f1"], -run["dev"]["topics"]["npmi"],
        run["summary"].get("timings", {}).get("dev_inference_seconds", float("inf")),
        run["name"],
    ))[0]
    if suite == "pilot300" and selection.get("baseline_reference") != baseline["name"]:
        raise ValueError("Frozen baseline reference differs from recomputed dev ordering")
    rows = [_row(run, selected_name) for run in runs]
    primary = paired_differences(selected["tests"], baseline["tests"])
    comparisons = {"primary": {"left": selected_name, "right": baseline["name"], **primary},
                   "revision_contrasts": [], "full_test_subsets": {}}
    for run in runs:
        if run["name"].startswith("cst_mil_") and run["name"] != selected_name:
            comparisons["revision_contrasts"].append({
                "left": selected_name, "right": run["name"], "exploratory": True,
                **paired_differences(selected["tests"], run["tests"]),
            })
    if suite == "full677":
        for name, key in [("fresh_test", "fresh_test"), ("previously_seen_test", "pilot_member")]:
            left = [row for row in selected["tests"] if row[key]]
            right = [row for row in baseline["tests"] if row[key]]
            comparisons["full_test_subsets"][name] = {
                "selected": binary_metrics(left), "baseline": binary_metrics(right),
                "paired_difference": paired_differences(left, right),
                "methods": {run["name"]: binary_metrics([
                    row for row in run["tests"] if row[key]
                ]) for run in runs},
            }
    incomplete = []
    for path in sorted(run_root.glob("cst_mil_*")):
        if path.name in available:
            continue
        errors = [read_json(error) for error in sorted(path.glob("error_*.json"))]
        incomplete.append({"name": path.name, "status": "incomplete_not_compared", "errors": errors})
    for campaign_path in (BENCHMARK / "campaign.json", BENCHMARK / "campaign_state.json",
                          BENCHMARK / "pilot_campaign.json"):
        if campaign_path.exists():
            comparisons["campaign_record"] = read_json(campaign_path)
            comparisons["campaign_record_path"] = str(campaign_path)
    comparisons["incomplete_revisions"] = incomplete
    comparisons["completed_run_errors"] = _completed_run_errors(runs)
    comparisons["selection"] = selection
    comparisons["selection_source_suite"] = "pilot300"
    comparisons["trained_cst_revisions"] = [run["name"] for run in runs
                                            if run["name"].startswith("cst_mil_")]
    comparisons["selected_topic_count_distribution"] = {
        "all_test": _topic_count_distribution(selected["tests"]),
        "non_cb_test": _topic_count_distribution([
            record for record in selected["tests"] if record["label"] == 0
        ]),
        "cb_test": _topic_count_distribution([
            record for record in selected["tests"] if record["label"] == 1
        ]),
    }
    comparisons["cohort_fingerprint"] = manifest["content_fingerprint"]
    comparisons["runtime_fingerprint"] = reference_runtime
    comparisons["rows"] = rows
    # Validate evidence before creating any output artifacts.
    cases = _cases(selected, baseline, messages)
    destination = BENCHMARK / "comparison" / suite
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "statistics.json", comparisons)
    with (destination / "comparisons.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                         else value for key, value in row.items()} for row in rows)
    figures = _figures(rows, destination / "figures")
    risk_table = _table(["方法", "选定", "验证 macro-F1", "测试 macro-F1 [95% CI]",
                         "CB Recall", "CB Precision", "AP", "ROC-AUC"], [[
        row["method"], "是" if row["selected"] else "", _number(row["dev_macro_f1"]),
        f"{_number(row['test_macro_f1'])} [{_number(row['test_macro_f1_ci_low'])}, "
        f"{_number(row['test_macro_f1_ci_high'])}]", _number(row["test_recall_cb"]),
        _number(row["test_precision_cb"]), _number(row["test_ap"]), _number(row["test_roc_auc"]),
    ] for row in rows])
    quality_table = _table(["方法", "全局主题数", "NPMI", "主题词多样性", "覆盖率", "退化主题比例",
                            "评论 AP", "Top-3命中", "Top-3召回"], [[
        row["method"], row["topic_n_topics"], *[_number(row[key]) for key in (
            "topic_npmi", "topic_topic_diversity", "topic_coverage", "topic_degenerate_topic_fraction",
            "evidence_comment_ap", "evidence_hit_rate_at_3", "evidence_mean_recall_at_3",
        )],
    ] for row in rows])
    cost_table = _table(["方法", "训练含验证(s)", "全部推理(s)", "测试推理(s)",
                         "峰值RSS(MiB)", "模型(MiB)", "外部编码器(MiB)", "外部运行时(MiB)"], [[
        row["method"], _number(row["train_seconds"], 2), _number(row["inference_all_seconds"], 2),
        _number(row["inference_test_seconds"], 2), *[_number(row[key] / 1024**2, 2) for key in (
            "peak_rss_bytes", "model_bytes", "external_encoder_bytes", "external_runtime_bytes",
        )],
    ] for row in rows])
    selected_row = next(row for row in rows if row["selected"])
    bertopic_row = next(row for row in rows if row["method"] == "bertopic")
    topic_count_table = _table([
        "测试会话分组", "会话数", "0主题", "1主题", "2主题", "3主题", ">3主题", "最大值", "平均值",
    ], [[name, values["n_sessions"], *[values["bins"][key] for key in ("0", "1", "2", "3", ">3")],
         values["maximum"], _number(values["mean"], 3)]
        for name, values in comparisons["selected_topic_count_distribution"].items()])
    direction = "高于" if primary["delta_macro_f1"] >= 0 else "低于"
    delta = abs(primary["delta_macro_f1"])
    caveat = ("这 60 个测试会话在旧版本分析中已经被查看，不能称为完全未接触的新测试集。"
              if suite == "pilot300" else
              "136 个测试会话中，60 个曾在旧版本中查看，76 个为新增测试会话；两部分分开报告。")
    fresh_section = ""
    if suite == "full677":
        fresh_section = _full_subset_section(comparisons["full_test_subsets"])
    failed_text = ("\n".join(f"- {item['name']}：未完整完成，不进入数值排名；"
                              f"错误记录 {len(item['errors'])} 条。" for item in incomplete)
                   or "没有额外的未完成 CST 运行目录。")
    history_table = _table(["修订", "起点", "更改", "替换当前最佳", "达到停止目标"], [[
        item.get("revision"), item.get("base_revision") or "初始",
        item.get("change", "r0 完整改造"), str(item.get("accepted")),
        str(item.get("meets_stop_target")),
    ] for item in selection.get("history", [])])
    completed_errors = comparisons["completed_run_errors"]
    if completed_errors:
        error_labels = {"cpu_affinity_resource_rejection": "CPU affinity 资源限制拒绝",
                        "user_keyboard_interrupt": "用户中断（KeyboardInterrupt）",
                        "other_recorded_error": "其他错误记录（见原记录）"}
        stage_labels = {"training": "训练", "inference": "推理", "see_traceback": "见原始堆栈"}
        recovery_text = "当前有效运行目录中保留的错误/中断记录如下；这些运行随后均已成功完成，不能将以下记录误称为算法最终失败。\n\n" + _table(
            ["方法", "阶段", "记录类型", "最终状态", "保留证据"], [[
                item["method"], stage_labels[item["stage"]], error_labels[item["category"]],
                "成功完成", f"`{item['path']}`",
            ] for item in completed_errors],
        )
        recovery_text += "\n\n中断记录及原始堆栈仍保存在运行目录，statistics.json 同时汇总原记录；最终结果与这些尝试记录分开解释，不能由成功运行的阶段时间推算所有失败/中断尝试的总消耗。"
    else:
        recovery_text = "当前有效且已完成的运行目录中没有 error_*.json 记录。"
    resource_text = "资源政策排除记录：`benchmark/quarantine/pre_cpu_affinity_20260913/` 保留了初轮 LDA/BTM/GSDMM/NMF 训练与验证、被中断的 BERTopic，以及当时的烟雾记录。初轮发现环境线程参数未能限制实际 CPU 使用，因此不用于本表计时或选版；没有解锁测试，也未执行 CST 修订。当前表格来自修复后统一 CPU 亲和性限制的替代运行。这不是额外随机种子重复，不能把两轮合成均值或标准差。"
    revision_text = "所有完整 CST 修订均列入表格；选定标记来自冻结的 selection.json，不能用测试排名反向更换版本。未改善版本仍然保留。"
    stop_text = f"冻结时是否达到预定停止目标：`{selection.get('stop_target_met')}`。若为 False，表示迭代预算内仍未达到全部质量门槛，不能解释为全面超过基线。"
    next_text = "下一步：保留当前冻结版本，并使用准备好的逐方法全量命令扩大到 677 个会话；全量阶段重新在 406 个训练会话拟合，不能直接把本轮模型当成全量训练结果。新的 76 个测试会话应独立报告。"
    if suite == "full677":
        resource_text = "历史 pilot300 资源政策排除记录：`benchmark/quarantine/pre_cpu_affinity_20260913/` 保留的是早期试验及烟雾运行，当时环境线程参数未能限制实际 CPU 使用；这些记录未用于 pilot300 选版或计时，当时没有解锁测试，也未执行 CST 修订。它们不是本轮 full677 的失败运行，更不是额外随机种子重复。本表来自当前 full677 完成的有效运行；当前运行的资源限制拒绝和中断另列如下，不与历史试验混合统计。"
        trained_revisions = "、".join(f"`{name}`" for name in comparisons["trained_cst_revisions"])
        revision_text = ("下表 r0–r3 的修改、接受/拒绝和停止判据记录来自 pilot300 验证集选版，不是 full677 的再次迭代。"
                         f"本轮 full677 已重训的 CST 配置为 {trained_revisions}；"
                         f"冻结版本 `{selected_name}` 使用本轮 {counts['train']} 个训练会话重新拟合，"
                         f"在 {counts['dev']} 个验证会话确定阈值。本轮没有根据测试结果重新选版；"
                         "pilot300 的未改善修订及其结果仍保留在原目录。")
        stop_text = (f"pilot300 冻结选版时的 `stop_target_met={selection.get('stop_target_met')}`："
                     "若为 False，表示当时迭代预算内未达到全部质量门槛；这不是 full677 上重新检验停止目标的结论。"
                     "当前全量结果只评价已冻结版本，不能把试验阶段的停止状态当成本轮成功或失败判定。")
        next_text = "下一步：补充多随机种子实验、人工主题及边界标注，并分析冻结阈值的影响与漏检案例；不依据本轮测试结果回调现有模型。"
    phase_table = _table([
        "方法", "真实风险头拟合数", "正式训练尝试", "主题拟合(s)", "风险拟合(s)",
        "训练冷编码(s)", "训练缓存读取(s)", "推理冷编码(s)", "推理缓存读取(s)",
    ], [[row["method"], row["risk_classifier_fit_count"], row["formal_fit_attempts"],
         *[_number(row[key], 3) for key in (
             "topic_fit_seconds", "risk_fit_seconds", "train_encoding_cold_seconds",
             "train_encoding_cached_seconds", "inference_encoding_cold_seconds",
             "inference_encoding_cached_seconds",
         )]] for row in rows])
    report = f"""# {suite}：CST-MIL 与五个主题基线的固定会话比较

本报告回答：在同一 SCCD 会话划分上，经过验证集选择的 CST 是否在会话风险识别、主题输出和成本之间取得有用表现。主指标是测试 macro-F1，比较对象由验证集确定。

本轮冻结版本为 `{selected_name}`，验证集最强基线为 `{baseline['name']}`。选定 CST 的测试 macro-F1 为 {_number(selected_row['test_macro_f1'])}，{direction}该基线 {_number(delta)}；配对会话 bootstrap 的差值 95% 区间为 [{_number(primary['ci95'][0])}, {_number(primary['ci95'][1])}]。这描述一次固定拟合的结果，不是多随机种子的算法优越性结论。

## 数据与实验口径

完整会话数 {len(sessions)}；清洗后评论数 {len(messages):,}；训练/验证/测试为 {counts['train']}/{counts['dev']}/{counts['test']}。随机种子只有 42，一个基线一次正式配置；CST 修订版不是独立重复实验。交叉拟合折数属于训练内部步骤。{caveat}

全量 SCCD 可用规模为 677 个会话、38,872 条清洗后评论（原始 38,999 条）；3000 个完整会话不在现有数据中。所有数值性能仅在测试分区统计，训练会话的预测仅用于完整输出和断点验收。所有模型使用训练会话标签，评论标签只作测试证据评价。主题词表/模型只在训练分区拟合，阈值和版本选择只使用验证分区。

会话划分不等于用户隔离或所有重复文本隔离：{manifest.get('split_caveat', '未建立跨用户独立性保证。')} CPU 线程上限 {shared.get('threads')}；原生算法可能实际单线程。BERTopic 额外使用预训练多语句向量，CST 没有预训练模型。基线通过主题特征接共同风险头，CST 另有字符、回复和 MIL 通道，因此这不是纯主题算法的单变量消融。

## 风险识别比较

{risk_table}
{fresh_section}
![风险比较](figures/risk_comparison.png)

误差线是固定模型、按完整会话进行分层重采样得到的 95% 区间；不表示训练随机性。图中应同时观察宏平均表现和欺凌类召回，不能根据单个点值忽略漏检。星号表示验证集冻结的 CST 版本。

## 主题与证据

{quality_table}

证据评价分母按各方法实际记录保存于 CSV。选定 `{selected_name}` 的评论 AP 使用 {selected_row['evidence_n_labeled_messages']} 条有标签测试消息；Top-3 命中率与召回率仅在 {selected_row['evidence_n_sessions_with_positive_comments']} 个含正例评论的会话上宏平均。这个分母既不是全部测试会话数，也不是会话标签为 CB 的数量，Non-CB 会话也可能有正例评论。Top-3 召回率先用每会话检索到的正例数除以该会话全部正例评论数，再平均；长会话含较多正例时，该指标会受到固定只返回3条证据的限制。

![主题质量](figures/topic_quality.png)

NPMI、词语多样性和消息分配覆盖率分别描述共现、词表重复和分配行为；没有主题数量/边界人工真值。较高覆盖率可能包含错误分配，较高 NPMI 也不能证明主题属于风险。0/1/2/3 主题构造检查只验证接口与分段。系统输出的主题强度和消息风险分数不是已校准的类别置信度；本数据只监督 CB/Non-CB，不能据此声称已验证三个指定风险类别的多标签识别。

覆盖率按全部消息微平均（保留主题支持的消息数/全部消息数），不是先平均各会话比例。全局主题容量和单会话实际输出主题数不同。BERTopic 原生训练窗口离群比例为 {_number(bertopic_row['bertopic_training_noise_fraction'])}；离群窗口没有被当成额外风险主题。

冻结 CST 在真实测试会话中的保留主题数量如下。分组依据是 SCCD 的会话 CB/Non-CB 标签，不是主题标签：

{topic_count_table}

这些数字直接统计保存的 topics 数组，只能说明输出数量可变；没有人工主题数量真值，不能称为主题数量准确率。Non-CB 会话也可以包含多个正常讨论主题，因此“输出主题”不等于“检出风险”。超过3个主题可能来自真实多话题、同一话题被拆分或一般词语形成的因子，需要人工核对，不能单凭数量认定错误或正确。抽取式标题可能含“人”“没”等过于一般的词，标题可读性和过细分风险仍需验证；也不能将这些标题直接对应为网络欺凌、辱骂、言语暴力三个风险类别。该分布未用于更改冻结模型或重新选版。

{cases}

## 时间、内存与依赖成本

{cost_table}

![时间内存](figures/time_memory.png)

训练时间包含主题拟合、风险交叉拟合、最终重拟合、验证推理和阈值选择；具体阶段及实际风险分类器拟合次数见 CSV。推理耗时求和自逐会话检查点，包含该次真实编码/缓存行为，未用消息或窗口数冒充会话数。缓存会影响运行顺序和重启成本；已记录的逐会话冷编码/缓存计数器在统计附录单列，缺失计数器记 NA。

原生 LDA、BTM、GSDMM 的端到端时间包含子进程启动、模型文件读写和接口转换开销；这些数值描述本地适配流水线，不等于算法核心计算的复杂度比较。所有方法均按完整会话提交结果。

模型文件大小与外部资源分别列出。BERTopic 的编码器没有嵌入模型文件，运行仍需该编码器及 PyTorch 等依赖。Java 基线的运行时列统计本机预置完整 JDK 目录，包含编译器等开发文件，不代表仅推理所需的最小运行时。图中磁盘柱为模型加本机已计量的预置外部资源，不含共享 Python 环境、可重建嵌入缓存和训练日志，不能称为最小部署体积或完整安装体积。编码器逐文件按清单核对大小和 SHA-256；JDK 此处仅检查声明的 java 路径存在并汇总目录文件大小，未进行同等级的逐文件哈希验证。外部资源路径、模型指纹和共享缓存大小见 statistics.json。

## 修订、失败与允许的结论

{resource_text}

{recovery_text}

{revision_text}

{history_table}

{stop_text}

{failed_text}

允许的结论：`{selected_name}` 在本次固定会话测试中取得表内风险指标，并能生成数量可变的主题及原文证据。依据为已校验的逐会话预测、metrics.json 和冻结版本记录。

禁止扩大的结论：尚不能声称普遍优于全部方法、跨种子稳定领先、真实多风险标签识别已达标、主题通道带来因果收益、风险分数已校准或长对话语义理解已经充分验证。仍需独立用户/事件划分、多随机种子、风险主题及边界标注和适当消融。

{next_text}

## 可复核文件

- [逐项数值 CSV](comparisons.csv)、[统计细节](stats-appendix.md)、[图目录](figure-catalog.md)、[机器可读统计](statistics.json)。
- 原运行目录：`{run_root}`；每个运行保存模型、源码/依赖绑定、训练摘要、逐会话原子结果。
- 冻结选择：`{selection_path}`；数据指纹：`{manifest['content_fingerprint']}`。

本报告由 results-analysis 的证据与不确定性规则生成；没有进行 Obsidian 写回。
"""
    appendix = f"""# 统计附录：{suite}

分析单位是完整 post_id 会话；测试 n={counts['test']}，随机种子集合={{42}}。每个配置只有一个拟合结果，无法估计跨训练重复的均值±标准差。三折交叉拟合不能当成三个独立实验。

主比较是验证集选定 `{selected_name}` 对验证集最强基线 `{baseline['name']}`。效果量为 macro-F1 的绝对差值 {_number(primary['delta_macro_f1'])}，1000 次按类别分层、配对完整会话 bootstrap 的 95% 百分位区间为 [{_number(primary['ci95'][0])}, {_number(primary['ci95'][1])}]。每次为两模型抽取完全相同的会话索引，保留类规模。它以已拟合模型为条件，未估计训练/调参不确定性。

没有采用 t 检验、ANOVA 或将窗口当成独立样本；因此不需要以正态性检验为这些未实施的参数检验背书，也不提供无根据的 p 值。各配置的测试 CI 为边际区间；修订版本差值 CI 属探索性、未经同时覆盖校正，不用于显著性或赢家判断。没有把多个 CI 不重叠当成显著性检验。

会话重采样假定测试会话可交换。SCCD 可能共享用户、文本和讨论事件，独立性没有完全验证，因此区间可能偏窄。未根据离群长度删除样本；长会话和极端分数均保留。AP 采用非插值 average precision；ROC-AUC 对单类子集缺失，记 NA。

主指标 macro-F1、CB召回/精确率、AP、ROC-AUC 越高越好。NPMI 范围 [-1,1]，多样性/覆盖率范围 [0,1]；它们不是有标签主题准确率。评论 AP 使用全部可匹配的标注测试消息；Top-3 指标只在含正例评论的会话上宏平均，分母逐方法保存在 evidence_n_sessions_with_positive_comments。原文引用在生成报告前与消息源核对。资源指标越低越省成本，但模型质量与预训练依赖不同，不能单独据时间宣布优劣。

NPMI 的共现单位是对应评价分区的统一5消息窗口；每词对共同出现概率为0时记-1，全窗口恒共现时按实现约定记0。主题内词对先平均，再按主题平均。退化主题定义为Top-10不足2个不同词，或其中在参考语料出现的词不足2个；退化主题NPMI记-1。词语各自出现但没有共同窗口可能得到-1而不算退化。多样性为所有Top-10不同词数/实际列出的词数。消息覆盖率为被保留主题支持的消息总数/模型输入消息总数；另存会话宏平均但不用它选版。无评论会话的原帖回退按一条模型输入消息计数。

正式进程树由操作系统限制在相同4个逻辑CPU内，并保留库级线程设置；辅助线程数不等于可占用CPU数。原生子进程的亲和性宽度也被采样核验。外部句向量缓存跨进程复用，包含工程试验已计算的训练/验证窗口；这不改变冻结编码器与训练数据边界，但使部分编码处于缓存命中状态，因此不能从该次总时间推断完全无缓存的冷启动成本。测试推理耗时和编码/缓存分项另存于数值表。

完整数值与配对结果见 [statistics.json](statistics.json)，运行绑定及模型指纹已逐一核验。观察到的效应并未升级为跨数据/跨种子的因果解释。

## 全部配置数值

{risk_table}

## 推理、编码与拟合口径

{phase_table}

训练冷编码/缓存读取来自训练阶段 backend 计数器；推理计数来自逐会话检查点差值。NA 表示不使用编码器或缺少字段，不是零成本。外部模型下载/环境安装属于提前准备，不计入拟合时间。每个配置运行一次，所以时间没有重复实验误差线。
"""
    catalog = f"""# 图目录：{suite}

数据源均为同目录 comparisons.csv / statistics.json，并可回溯原运行的逐会话结果。每图提供 PNG 预览和 PDF 矢量版本；横向柱状图从零起（NPMI 明确保留 [-1,1]）。主图星号表示冻结选择，不表示显著性。

## risk_comparison.png / risk_comparison.pdf

目的：比较固定测试分区的宏平均分类能力与 CB 召回。横轴分别为 macro-F1 和召回，纵轴为所有基线及保留修订版，测试会话数 {counts['test']}。仅 macro-F1 误差线为 1000 次类别分层会话 bootstrap 的 95% CI。

观察：选定 CST 相对验证最强基线的 macro-F1 差值为 {_number(primary['delta_macro_f1'])}。解释：这是固定拟合的条件差值；检查召回是否发生明显代价。决策影响：描述冻结方法的取舍，不使用测试图重新选版。检查清单：核对星号、样本数、区间含义；不能把误差线当成多次训练标准差。

## topic_quality.png / topic_quality.pdf

目的：并列检查 NPMI、多样性与覆盖率，避免把单一主题代理指标当成语义正确性。观察：选定 CST 的三项值依次为 {_number(selected_row['topic_npmi'])}、{_number(selected_row['topic_topic_diversity'])}、{_number(selected_row['topic_coverage'])}。

解释：高覆盖和高共现不等于风险标签正确；没有主题人工真值。决策影响：确定后续人工标注和主题消融需要检查的方面。图无误差线，因为只有一次拟合且未建立这些代理量的采样不确定性模型；不是宣称零方差。

## time_memory.png / time_memory.pdf

目的：展示训练、全部会话推理、峰值进程树 RSS 与本机预置模型及外部资源的占用。观察：选定 CST 训练 {_number(selected_row['train_seconds'], 2)} 秒，全部推理 {_number(selected_row['inference_all_seconds'], 2)} 秒，峰值 RSS {_number(selected_row['peak_rss_bytes'] / 1024**2, 2)} MiB。

解释：训练含交叉拟合和验证；推理含实际编码或缓存读取；预训练权重与 JDK 不能隐藏在小 pickle 后面。JDK 计量的是本机完整预置目录，不是最小推理运行时。决策影响：逐方法安排手动全量运行和本机资源预算。检查清单：核对单位、缓存条件、模型与运行时的组成；单次耗时无重复误差线；不含共享 Python 环境，不可称为最小或完整安装体积。
"""
    for name, content in [("analysis-report.md", report), ("stats-appendix.md", appendix),
                          ("figure-catalog.md", catalog)]:
        (destination / name).write_text(content, encoding="utf-8", newline="\n")
    return {"suite": suite, "output_dir": str(destination), "methods": names,
            "selected": selected_name, "baseline_reference": baseline["name"],
            "test_sessions": counts["test"], "delta_macro_f1": primary["delta_macro_f1"],
            "figures": figures, "incomplete_revisions": len(incomplete)}
