from __future__ import annotations

import csv

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cst_mil.bench.common import build_windows
from cst_mil.bench.evaluation import binary_metrics

from .io import METHODS, OUT, PROJECT, RUNS, inputs, protect, read, rows, save, table


def ranking_at_k(records: list[dict], ks=(10, 20, 30)) -> list[dict]:
    ordered = sorted(records, key=lambda r: (-r["risk_probability"], r["post_id"]))
    positives = sum(r["label"] for r in ordered)
    result = []
    for k in ks:
        top = ordered[:k]
        found = sum(r["label"] for r in top)
        result.append(
            dict(
                k=k,
                reviewed=len(top),
                found=found,
                precision=found / len(top) if top else None,
                recall=found / positives if positives else None,
            )
        )
    return result


def evidence_details(predictions: list[dict], messages: list[dict], k=3) -> dict:
    truth = {(m["post_id"], m["comment_id"]): m.get("comment_label") for m in messages}
    hits, recalls, details, y, scores = [], [], [], [], []
    missing_top = 0
    for record in predictions:
        ranked = sorted(
            record.get("message_scores", []), key=lambda m: (m.get("rank", 10**9), m["comment_id"])
        )
        labels = [truth.get((record["post_id"], m["comment_id"])) for m in ranked]
        positives = sum(value == 1 for value in labels)
        found = sum(value == 1 for value in labels[:k])
        missing_top += sum(value is None for value in labels[:k])
        for m, value in zip(ranked, labels, strict=True):
            if value is not None:
                y.append(value)
                scores.append(m["risk_probability"])
        if positives:
            hits.append(int(found > 0))
            recalls.append(found / positives)
            details.append(
                dict(
                    post_id=record["post_id"],
                    positives=positives,
                    found=found,
                    hit=int(found > 0),
                    recall=found / positives,
                )
            )
    return dict(
        hit_count=sum(hits),
        n_sessions_with_positive_comments=len(hits),
        hit_rate_at_3=float(np.mean(hits)) if hits else None,
        mean_recall_at_3=float(np.mean(recalls)) if recalls else None,
        comment_ap=float(average_precision_score(y, scores)) if sum(y) else None,
        n_labeled_messages=len(y),
        unlabeled_top_k=missing_top,
        per_session=details,
    )


def paired_bootstrap(y, scores: dict[str, np.ndarray], repeats=2000, seed=42) -> list[dict]:
    y = np.asarray(y)
    groups = [np.flatnonzero(y == value) for value in (0, 1)]
    if any(not len(group) for group in groups):
        raise ValueError("Both classes are needed")
    if any(len(value) != len(y) for value in scores.values()):
        raise ValueError("Scores and labels must align")
    functions = {"ap": average_precision_score, "roc_auc": roc_auc_score}
    rng = np.random.default_rng(seed)
    samples = {name: {metric: [] for metric in functions} for name in scores}
    for _ in range(repeats):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for name, values in scores.items():
            for metric, function in functions.items():
                samples[name][metric].append(function(y[indices], values[indices]))
    result = []
    for name, values in scores.items():
        if name == "cst_mil_r0":
            continue
        for metric, function in functions.items():
            delta = np.asarray(samples["cst_mil_r0"][metric]) - samples[name][metric]
            result.append(
                dict(
                    baseline=name,
                    metric=metric,
                    delta=float(function(y, scores["cst_mil_r0"]) - function(y, values)),
                    ci_low=float(np.quantile(delta, 0.025)),
                    ci_high=float(np.quantile(delta, 0.975)),
                    # Bonferroni simultaneous 95% intervals over 10 contrasts.
                    family_ci_low=float(np.quantile(delta, 0.0025)),
                    family_ci_high=float(np.quantile(delta, 0.9975)),
                    repeats=repeats,
                    seed=seed,
                    unit="complete_session; fixed fitted models",
                )
            )
    return result


def profile_existing() -> list[dict]:
    source = PROJECT / "benchmark/comparison/full677/comparisons.csv"
    with source.open(encoding="utf-8-sig") as f:
        original = {row["method"]: row for row in csv.DictReader(f)}
    result = []
    for method in METHODS:
        durations, cold, cached, loads = [], 0.0, 0.0, 0.0
        for p in (RUNS / method / "sessions").glob("*.json"):
            record = read(p)
            if record["prediction"]["split"] != "test":
                continue
            durations.append(record["inference_seconds"])
            counters = record.get("encoding_counters", {})
            cold += counters.get("encoding_cold_seconds", 0)
            cached += counters.get("encoding_cached_seconds", 0)
            loads += counters.get("encoder_load_seconds", 0)
        if len(durations) != 136:
            raise ValueError(f"Incomplete measured sessions: {method}")
        old = original[method]
        result.append(
            dict(
                method=method,
                n_sessions=len(durations),
                train_seconds=float(old["train_seconds"]),
                inference_test_seconds=sum(durations),
                latency_mean_seconds=float(np.mean(durations)),
                latency_p50_seconds=float(np.median(durations)),
                latency_p95_seconds=float(np.quantile(durations, 0.95)),
                peak_rss_mib=float(old["peak_rss_bytes"]) / 2**20,
                model_mib=float(old["model_bytes"]) / 2**20,
                encoder_mib=float(old["external_encoder_bytes"]) / 2**20,
                runtime_mib=float(old["external_runtime_bytes"]) / 2**20,
                deployed_mib=float(old["known_deployed_disk_bytes"]) / 2**20,
                test_encoding_cold_seconds=cold,
                test_encoding_cached_seconds=cached,
                test_encoder_load_seconds=loads,
                local_gpu_required=False,
                protocol="historical measured per-session end-to-end; mixed cache state",
                cold_start_total_seconds="not separately recorded",
                train_attempts=int(old["formal_fit_attempts"]),
            )
        )
    table(OUT / "costs.csv", result)
    save(OUT / "costs.json", result)
    return result


def run() -> dict:
    protect()
    manifest, sessions, messages, grouped = inputs()
    test = sorted([s for s in sessions if s["split"] == "test"], key=lambda s: s["post_id"])
    ids = [s["post_id"] for s in test]
    y = np.array([s["label"] for s in test])
    performance, topk, evidence, topics, predictions, scores = [], [], [], [], {}, {}
    for method in METHODS:
        records = sorted(
            [r for r in rows(RUNS / method / "predictions.jsonl") if r["split"] == "test"],
            key=lambda r: r["post_id"],
        )
        if [r["post_id"] for r in records] != ids or [r["label"] for r in records] != y.tolist():
            raise ValueError(f"Incomparable IDs/labels: {method}")
        metrics = binary_metrics(records)
        old = read(RUNS / method / "metrics.json")["test"]
        for key in ["ap", "roc_auc", "macro_f1", "precision_cb", "recall_cb"]:
            if not np.isclose(metrics[key], old[key], atol=1e-12, rtol=0):
                raise ValueError(f"Metric mismatch: {method} {key}")
        performance.append(dict(method=method, **metrics))
        topk += [dict(method=method, **row) for row in ranking_at_k(records)]
        e = evidence_details(records, messages)
        table(OUT / "evidence_sessions" / f"{method}.csv", e.pop("per_session"))
        evidence.append(dict(method=method, **e))
        counts = [len(r["topics"]) for r in records]
        topics.append(
            dict(
                method=method,
                minimum=min(counts),
                maximum=max(counts),
                mean=float(np.mean(counts)),
                overlapping_messages=sum(
                    len(m["topic_ids"]) > 1 for r in records for m in r["message_scores"]
                ),
                n_messages=sum(len(r["message_scores"]) for r in records),
                semantic_gold_available=False,
            )
        )
        predictions[method] = records
        scores[method] = np.array([r["risk_probability"] for r in records])
        print(f"audit: {method} AP={metrics['ap']:.6f}, Top3={e['hit_count']}/101", flush=True)
    paired = paired_bootstrap(y, scores)
    for name, data in [
        ("performance", performance),
        ("ranking_at_k", topk),
        ("evidence", evidence),
        ("topic_counts", topics),
        ("paired", paired),
    ]:
        table(OUT / f"{name}.csv", data)
        save(OUT / f"{name}.json", data)
    case_record = next(r for r in predictions["cst_mil_r0"] if r["post_id"] == "RTQyLUlX1bbGeE4m")
    target = next(m for m in case_record["message_scores"] if m["comment_id"] == "vlDBQDbQuuc8hcHA")
    session = next(s for s in sessions if s["post_id"] == case_record["post_id"])
    local = grouped[session["post_id"]]
    windows = build_windows(session, local)
    case = dict(
        session_id=session["post_id"],
        target=target,
        parent=next((m for m in local if m["comment_id"] == target["parent_id"]), None),
        windows=[w for w in windows if target["comment_id"] in w["message_ids"]],
        topics=[t for t in case_record["topics"] if t["topic_id"] in target["topic_ids"]],
        assessment="Suspected window-context propagation; no gold topic annotation; "
        "generic acknowledgement is not independent evidence for two semantics.",
    )
    save(OUT / "topic_failure_case.json", case)
    costs = profile_existing()
    make_figures(predictions, paired, evidence, costs)
    summary = dict(
        data_fingerprint=manifest["content_fingerprint"],
        n_test=136,
        n_positive=71,
        n_negative=65,
        n_message=8028,
        bootstrap=2000,
        statistical_scope="retrospective fixed-seed fitted-model session uncertainty",
        protected=protect(verify=True),
    )
    save(OUT / "audit_summary.json", summary)
    return summary


def make_figures(predictions, paired, evidence, costs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    folder = OUT / "figures"
    folder.mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, records in predictions.items():
        y = [r["label"] for r in records]
        s = [r["risk_probability"] for r in records]
        p, r, _ = precision_recall_curve(y, s)
        fpr, tpr, _ = roc_curve(y, s)
        label = name.replace("cst_mil_r0", "CST-MIL").upper()
        axes[0].step(r, p, where="post", label=label)
        axes[1].plot(fpr, tpr, label=label)
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Session risk ranking: PR")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / "ranking_curves.pdf")
    fig.savefig(folder / "ranking_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    labels = [r["method"].replace("cst_mil_r0", "CST") for r in evidence]
    axes[0].bar(labels, [r["hit_rate_at_3"] for r in evidence], color="#377e91")
    axes[1].bar(labels, [r["mean_recall_at_3"] for r in evidence], color="#c48045")
    axes[0].set(title="Top-3 hit rate (101 sessions)", ylim=(0, 1))
    axes[1].set(title="Mean within-session recall at 3", ylim=(0, 0.15))
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(folder / "evidence.pdf")
    fig.savefig(folder / "evidence.png", dpi=180)
    plt.close(fig)
