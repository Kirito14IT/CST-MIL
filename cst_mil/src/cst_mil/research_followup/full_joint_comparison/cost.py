"""Uncached end-to-end CPU inference costs; no paid or remote services."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import time
from collections import defaultdict
from pathlib import Path

from . import CATEGORIES
from .common import (
    TreeMonitor,
    digest,
    identity,
    read,
    read_jsonl,
    save,
    source_hashes,
    verify_hashes,
)
from .execution import _check_complete, verify_experiment


def dependency_sizes(method: str) -> tuple[list[dict], int]:
    names = [
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "jieba",
        "psutil",
        "pandas",
        "threadpoolctl",
    ]
    if method in {"bertopic", "cst_v1_1", "minilm_lr"}:
        names += ["torch", "sentence-transformers", "transformers", "tokenizers", "safetensors"]
    if method == "bertopic":
        names += ["bertopic", "umap-learn", "hdbscan", "numba", "llvmlite"]
    rows = []
    for name in names:
        dist = importlib.metadata.distribution(name)
        paths = {Path(dist.locate_file(p)).resolve() for p in (dist.files or [])}
        if name == "bertopic":
            spec = importlib.util.find_spec("bertopic")
            if spec and spec.submodule_search_locations:
                for location in spec.submodule_search_locations:
                    paths.update(p.resolve() for p in Path(location).rglob("*") if p.is_file())
        rows.append(
            {
                "distribution": name,
                "version": dist.version,
                "installed_bytes": sum(
                    p.stat().st_size for p in paths if p.is_file() and p.suffix != ".pyc"
                ),
            }
        )
    runtime = 0
    if method == "gsdmm":
        from cst_mil.research_followup.shorttext_nmf_stable.native_topics import portable_java

        root = portable_java().parents[1]
        runtime = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        rows.append(
            {"distribution": "portable-Java-runtime", "path": str(root), "installed_bytes": runtime}
        )
    return rows, runtime


def inference_bundles(records: list[dict], sessions: list[dict]):
    """Preserve full-session UMAP batches and canonical 64-comment source batches."""
    grouped = defaultdict(list)
    other = defaultdict(list)
    for r in records:
        (grouped[r["session_id"]] if r["source"] == "sccd" else other[r["source"]]).append(r)
    result = []
    for s in sorted(sessions, key=lambda s: s["session_id"]):
        rows = sorted(grouped[s["session_id"]], key=lambda r: (r["order"], r["message_id"]))
        if not rows:
            raise ValueError("Missing SCCD session records")
        result.append(("sccd", s["session_id"], rows, [s]))
    for source in sorted(other):
        rows = sorted(other[source], key=lambda r: r["record_id"])
        for start in range(0, len(rows), 64):
            result.append((source, f"{source}:batch:{start}", rows[start : start + 64], []))
    return result


def measure_one(output: Path, method: str, seed: int = 42, threads: int = 4) -> dict:
    import joblib
    import numpy as np

    from .data import load_prepared

    if seed != 42:
        raise ValueError("Cost protocol is fixed to seed42")
    frozen = verify_experiment(output)
    if threads != frozen["threads"] or method not in frozen["methods"]:
        raise ValueError("Cost CPU budget/method differs from frozen experiment")
    folder = output / "runs" / method / f"seed_{seed}"
    manifest = read(folder / "run_manifest.json")
    run_id = identity({"experiment": identity(frozen), "method": method, "seed": seed})
    if not _check_complete(folder / "train_state.json", run_id):
        raise ValueError("Cost requires sealed training")
    if not _check_complete(folder / "test_state.json", run_id):
        raise ValueError("Cost requires sealed test evaluation")
    test_state = read(folder / "test_state.json")
    if test_state["status"] != "complete":
        raise ValueError("Cost requires complete frozen test evaluation")
    cost_id = identity(
        {
            "run": manifest["identity"],
            "model": manifest["model_sha256"],
            "cost_code": digest(Path(__file__)),
            "repetitions": 3,
            "threads": threads,
            "batch": "SCCD-session/other64",
            "test_seal_sha256": digest(folder / "test_state.json"),
            "score_tolerance": 1e-4,
        }
    )
    target = folder / "cost.json"
    if digest(folder / "model.joblib") != manifest["model_sha256"]:
        raise ValueError("Frozen model changed")
    if target.exists():
        result = read(target)
        if result["identity"] != cost_id:
            raise ValueError("Existing cost identity changed")
        seal = read(folder / "cost_state.json")
        if seal["identity"] != cost_id or seal["status"] != "complete":
            raise ValueError("Cost seal mismatch")
        verify_hashes(seal["artifacts"])
        return result
    records, sessions, _ = load_prepared(output)
    records = [r for r in records if r["split"] == "test"]
    sessions = [s for s in sessions if s["split"] == "test"]
    bundles = inference_bundles(records, sessions)
    reference = {r["record_id"]: r for r in read_jsonl(folder / "predictions_message_test.jsonl")}
    reference_sessions = {
        r["session_id"]: r for r in read_jsonl(folder / "predictions_session_test.jsonl")
    }
    rows, totals, cold = [], [], None
    max_message_diff, max_session_diff, threshold_disagreements = 0.0, 0.0, 0
    with TreeMonitor() as monitor:
        started = time.perf_counter()
        package = joblib.load(folder / "model.joblib")
        model_load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        package.predict(bundles[0][2], bundles[0][3], embeddings=None)
        cold = time.perf_counter() - started
        for repetition in range(1, 4):
            elapsed_total = 0.0
            for i, (source, identifier, group_records, group_sessions) in enumerate(bundles):
                started = time.perf_counter()
                prediction = package.predict(group_records, group_sessions, embeddings=None)
                elapsed = time.perf_counter() - started
                elapsed_total += elapsed
                rows.append(
                    {
                        "source": source,
                        "batch_id": identifier,
                        "repetition": repetition,
                        "n_records": len(group_records),
                        "seconds": elapsed,
                        "kind": "complete_session" if source == "sccd" else "comment_batch",
                    }
                )
                for row in prediction["message"]:
                    scores = row["scores"]
                    if isinstance(scores, dict):
                        scores = [scores[c] for c in CATEGORIES]
                    ref = reference[row["record_id"]]["scores"]
                    ref = [ref[c] for c in CATEGORIES] if isinstance(ref, dict) else ref
                    max_message_diff = max(
                        max_message_diff, float(np.max(np.abs(np.asarray(scores) - ref)))
                    )
                    threshold_disagreements += sum(
                        (a >= package.thresholds[c]) != (b >= package.thresholds[c])
                        for c, a, b in zip(CATEGORIES, scores, ref, strict=True)
                    )
                for row in prediction["session"]:
                    ref = reference_sessions[row["session_id"]]
                    max_session_diff = max(
                        max_session_diff,
                        abs(row["score"] - ref["score"]),
                        abs(row["common16_score"] - ref["common16_score"]),
                    )
                if (i + 1) % 20 == 0 or i == len(bundles) - 1:
                    save(
                        folder / "cost_progress.json",
                        {
                            "repetition": repetition,
                            "completed_bundles": i + 1,
                            "total_bundles": len(bundles),
                            "seconds": elapsed_total,
                            "method": method,
                        },
                    )
                    print(
                        f"COST {method} repetition={repetition} "
                        f"{i + 1}/{len(bundles)} {elapsed_total:.1f}s",
                        flush=True,
                    )
            totals.append(elapsed_total)
    dependencies, runtime_bytes = dependency_sizes(method)
    sccd_times = [r["seconds"] for r in rows if r["source"] == "sccd"]
    throughput = {}
    for source in sorted({r["source"] for r in rows if r["source"] != "sccd"}):
        subset = [r for r in rows if r["source"] == source]
        throughput[source] = sum(r["n_records"] for r in subset) / sum(r["seconds"] for r in subset)
    result = {
        "identity": cost_id,
        "method": method,
        "seed": seed,
        "repetitions": 3,
        "training_seconds_cached": manifest["training_seconds_cached"],
        "training_seconds_cold": manifest["training_seconds_cold"],
        "shared_encoder_training_seconds": manifest["shared_encoder_training_seconds"],
        "cold_training_definition": manifest["cold_training_definition"],
        "model_load_seconds": model_load_seconds,
        "first_inference_seconds": cold,
        "full_test_inference_seconds": float(np.mean(totals)),
        "warm_total_seconds": totals,
        "sccd_session_median_seconds": float(np.median(sccd_times)),
        "sccd_session_p95_seconds": float(np.quantile(sccd_times, 0.95)),
        "comment_throughput_per_second_by_source": throughput,
        "peak_tree_rss_bytes": max(monitor.peak, manifest["cold_training_peak_tree_rss_bytes"]),
        "inference_peak_tree_rss_bytes": monitor.peak,
        "training_peak_tree_rss_bytes": manifest["cold_training_peak_tree_rss_bytes"],
        "model_bytes": manifest["model_bytes"],
        "encoder_asset_bytes": manifest["encoder_asset_bytes"],
        "runtime_dependency_bytes": sum(r["installed_bytes"] for r in dependencies),
        "dependency_sizing_scope": (
            "listed installed distributions including editable BERTopic source; "
            "not a minimal or transitive-complete environment; Java counted once"
        ),
        "java_runtime_bytes": runtime_bytes,
        "deployment_dependencies": dependencies,
        "GPU_required": False,
        "paid_API_calls": 0,
        "resources": monitor.result(),
        "cache_policy": (
            "resident model/encoder; no text embedding or prediction cache; "
            "Java launch and native temporary I/O included"
        ),
        "cold_start_definition": "fresh Python process, OS filesystem cache not cleared",
        "memory_definition": (
            "sampled benchmark process tree RSS including loaded input corpus, "
            "encoder and child JVM; not minimum server memory"
        ),
        "prediction_consistency": {
            "max_message_abs_diff": max_message_diff,
            "max_session_abs_diff": max_session_diff,
            "head_threshold_disagreements_all_repetitions": int(threshold_disagreements),
        },
        "rows": rows,
    }
    verify_experiment(output)
    if monitor.max_affinity > threads:
        raise RuntimeError("A child exceeded the frozen CPU affinity budget")
    if max(max_message_diff, max_session_diff) > 1e-4:
        result["status"] = "invalid_prediction_consistency"
        save(folder / "cost_invalid.json", result)
        raise RuntimeError(
            "Uncached predictions differ from frozen scores beyond 1e-4; cost not accepted"
        )
    result["status"] = "complete"
    save(target, result)
    save(
        folder / "cost_state.json",
        {"identity": cost_id, "status": "complete", "artifacts": source_hashes([target])},
    )
    return result
