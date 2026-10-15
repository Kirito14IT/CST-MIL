"""Fresh-process local timing and train-only independent dataset experiments."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import joblib
import numpy as np

from cst_mil.metrics import binary_classification_metrics

from ..io import inputs
from .core import (
    OUT,
    PROJECT,
    RUNS,
    code_sources,
    digest,
    guard_output,
    jsonl_read,
    jsonl_write,
    read,
    save,
    source_hashes,
    stable,
    verify_sources,
)


def fitting_data(sessions, messages):
    ids = [s["post_id"] for s in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate session across partitions")
    fitting = [s for s in sessions if s["split"] in ["train", "dev"]]
    allowed = {s["post_id"] for s in fitting}
    return fitting, [m for m in messages if m["post_id"] in allowed]


def evaluate(rows, threshold, grouped=None):
    if not rows:
        return dict(n_samples=0)
    raw = binary_classification_metrics(
        [r["label"] for r in rows], [r["risk_probability"] for r in rows], threshold=threshold
    )
    raw.update(
        ap=raw.pop("pr_auc"),
        auc=raw.pop("roc_auc"),
        precision=raw.pop("precision_cb"),
        recall=raw.pop("recall_cb"),
    )
    if grouped is not None:
        hits, denominator, recalled, total = 0, 0, 0, 0
        for r in rows:
            gold = {m["comment_id"] for m in grouped[r["post_id"]] if m.get("comment_label") == 1}
            chosen = {e["comment_id"] for e in r.get("evidence", [])[:3]}
            hits += bool(gold & chosen)
            denominator += bool(gold)
            recalled += len(gold & chosen)
            total += len(gold)
        raw.update(
            top3_hits=hits,
            top3_eligible_sessions=denominator,
            top3_hit_rate=hits / denominator if denominator else None,
            dangerous_comments_recalled=recalled,
            dangerous_comments_total=total,
            dangerous_comment_recall=recalled / total if total else None,
        )
    return raw


def latency(values):
    return dict(
        total_seconds=float(sum(values)),
        median_seconds=float(np.median(values)),
        p95_seconds=float(np.quantile(values, 0.95)),
        n=len(values),
    )


def measure_local(method, out=OUT):
    from cst_mil.bench.model import predict_session
    from cst_mil.bench.runner import ResourceMonitor

    folder = guard_output(out) / "local_timing" / method
    model_path = RUNS / method / "model.joblib"
    selection_path = out / "gemini/selection.json"
    binding = code_sources(Path(__file__), model_path, selection_path)
    if (folder / "summary.json").exists():
        previous = read(folder / "summary.json")
        if previous["sources"] != binding:
            raise ValueError("Timing resume sources changed")
        return previous
    _, sessions, _, grouped = inputs()
    lookup = {s["post_id"]: s for s in sessions}
    selected = [
        lookup[r["post_id"]] for r in read(selection_path)["sessions"] if r["split"] == "test"
    ]
    old = {r["post_id"]: r for r in jsonl_read(RUNS / method / "predictions.jsonl")}
    start = time.perf_counter()
    with ResourceMonitor() as load_monitor:
        model = joblib.load(model_path)
    load_seconds = time.perf_counter() - start
    # Isolate embedding writes; the new cache starts empty for the first pass.
    if hasattr(model.topic_model, "cache_dir"):
        cache = folder / "embedding_cache"
        if cache.exists() and any(cache.rglob("*.npy")):
            raise ValueError(
                "Interrupted timing has a populated cache; preserve it and use revision output"
            )
        model.topic_model.cache_dir = cache
    passes = []
    for repeat in range(4):
        predictions, times = [], []
        before = copy.deepcopy(getattr(model.topic_model, "info", {}))
        with ResourceMonitor() as monitor:
            for s in selected:
                started = time.perf_counter()
                prediction = predict_session(model, s, grouped[s["post_id"]])
                times.append(time.perf_counter() - started)
                predictions.append(prediction)
        parity = max(
            abs(r["risk_probability"] - old[r["post_id"]]["risk_probability"]) for r in predictions
        )
        record = dict(
            repeat=repeat,
            phase="first_pass" if repeat == 0 else "resident_repeat",
            seconds_by_session=times,
            session_ids=[s["post_id"] for s in selected],
            latency=latency(times),
            peak_rss_bytes=monitor.peak,
            peak_affinity_width=monitor.peak_affinity_width,
            max_saved_prediction_difference=parity,
            metrics=evaluate(predictions, model.threshold, grouped),
            backend_counters_before=before,
            backend_counters_after=copy.deepcopy(getattr(model.topic_model, "info", {})),
        )
        jsonl_write(folder / f"predictions_pass{repeat}.jsonl", predictions)
        save(folder / f"pass{repeat}.json", record)
        passes.append(record)
        print(f"{method} pass{repeat}: {sum(times):.3f}s; parity={parity:.3g}", flush=True)
        if monitor.peak_affinity_width > 4 or parity > 1e-6:
            raise RuntimeError("CPU cap or frozen model parity check failed")
    verify_sources(binding)
    result = dict(
        method=method,
        n_test=30,
        model_load_seconds=load_seconds,
        load_peak_rss_bytes=load_monitor.peak,
        model_bytes=model_path.stat().st_size,
        first_session_seconds=passes[0]["seconds_by_session"][0],
        passes=passes,
        resident_latency=latency([t for p in passes[1:] for t in p["seconds_by_session"]]),
        sources=binding,
        measurement="serial complete-session predict_session; 4 CPU cores; loading separate",
    )
    save(folder / "summary.json", result)
    return result
