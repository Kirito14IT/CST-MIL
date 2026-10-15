"""Raw-input resident cost, measured separately from shared representation banks."""

from __future__ import annotations

import gc
import platform
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import psutil

from cst_mil.research_followup.full_joint_comparison.model import canonical_batches

from .common import (
    OUTPUT,
    TreeMonitor,
    campaign_lock,
    digest,
    guard,
    identity,
    load_data,
    read,
    read_jsonl,
    save,
    source_hashes,
    verify_hashes,
    write_jsonl,
)
from .execution import folder_for, selected
from .representations import fixed_warmup


def measure(output: Path = OUTPUT) -> dict:
    """Three resident passes at seed42, sequential CPU4; no cached input vectors."""
    selection = selected(output)
    variant = selection["selected_architecture"]
    guard(output)
    folder = folder_for(output, variant, 42)
    root = output / "cost"
    training_seal = read(folder / "complete.json")
    reference_seal = read(folder / "test_complete.json")
    verify_hashes(training_seal["artifacts"])
    verify_hashes(reference_seal["artifacts"])
    if (
        reference_seal["binding"]["model_sha256"] != digest(folder / "model.joblib")
        or reference_seal["binding"]["selection_identity"] != selection["identity"]
    ):
        raise RuntimeError("Raw cost reference belongs to a different model or selection")
    freeze = {
        "selection_identity": selection["identity"],
        "model_sha256": digest(folder / "model.joblib"),
        "reference_artifacts": reference_seal["artifacts"],
        "reference_seal_sha256": digest(folder / "test_complete.json"),
        "code_hashes": source_hashes([Path(__file__)]),
        "repeats": 3,
        "warmup_policy": "fixed64TRAINCONDA before each pass, excluded from timing",
    }
    freeze["identity"] = identity(freeze)
    if (root / "freeze.json").exists() and read(root / "freeze.json") != freeze:
        raise RuntimeError("Cost freeze changed; preserve earlier measurement")
    save(root / "freeze.json", freeze)
    records_all, _ = load_data()
    records, sessions = load_data(("test",))
    expected = read_jsonl(folder / "predictions_message_test.jsonl")
    expected_by_id = {r["record_id"]: r for r in expected}
    expected_sessions = {
        r["session_id"]: r for r in read_jsonl(folder / "predictions_session_test.jsonl")
    }
    with campaign_lock(output):
        with TreeMonitor() as startup_monitor:
            started = time.perf_counter()
            package = joblib.load(folder / "model.joblib")
            package_load = time.perf_counter() - started
            started = time.perf_counter()
            encoder = package.encoder()
            # Explicitly load once; first encode includes model initialization.
            initial_warm = fixed_warmup(encoder, records_all)
            encoder_start = time.perf_counter() - started
        results = []
        for repeat in range(1, 4):
            target = root / f"repeat_{repeat}.json"
            if target.exists():
                old = read(target)
                if old["identity"] != freeze["identity"]:
                    raise RuntimeError("Cost resume identity mismatch")
                verify_hashes(old["artifacts"])
                if old.get("status") != "passed":
                    raise RuntimeError("Earlier failed raw pass retained; do not skip it on resume")
                results.append(old)
                continue
            verify_hashes(freeze["code_hashes"])
            verify_hashes(freeze["reference_artifacts"])
            warm = fixed_warmup(encoder, records_all)
            units, raw_messages, raw_sessions = [], [], []
            per_source = defaultdict(float)
            with TreeMonitor() as monitor:
                started = time.perf_counter()
                for index, ids in enumerate(canonical_batches(records)):
                    current = [records[i] for i in ids]
                    session_ids = {r["session_id"] for r in current if r["source"] == "sccd"}
                    current_sessions = [s for s in sessions if s["session_id"] in session_ids]
                    tick = time.perf_counter()
                    embeddings = encoder.encode([r["text"] for r in current], batch_size=32)
                    encoded = time.perf_counter() - tick
                    payload = package.predict_payload(current, current_sessions, embeddings)
                    elapsed = time.perf_counter() - tick
                    raw_messages.extend(payload["message"])
                    raw_sessions.extend(payload["session"])
                    source = current[0]["source"]
                    per_source[source] += elapsed
                    units.append(
                        {
                            "source": source,
                            "record_ids": [r["record_id"] for r in current],
                            "session_id": current[0]["session_id"],
                            "count": len(current),
                            "seconds": elapsed,
                            "encoding_seconds": encoded,
                        }
                    )
                    if index % 40 == 0:
                        print(f"COST repeat={repeat} units={index + 1}", flush=True)
                total = time.perf_counter() - started
            score_error = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(row["scores"]) - expected_by_id[row["record_id"]]["scores"]
                        )
                    )
                )
                for row in raw_messages
            )
            if sorted(row["record_id"] for row in raw_messages) != sorted(expected_by_id):
                raise RuntimeError("Raw prediction IDs incomplete or duplicated")
            if sorted(row["session_id"] for row in raw_sessions) != sorted(expected_sessions):
                raise RuntimeError("Raw conversation IDs incomplete or duplicated")
            flips = sum(
                (row["scores"][j] >= package.thresholds[c])
                != (expected_by_id[row["record_id"]]["scores"][j] >= package.thresholds[c])
                for row in raw_messages
                for j, c in enumerate(("attack", "hate", "spam_scam"))
            )
            session_error = max(
                abs(row[key] - expected_sessions[row["session_id"]][key])
                for row in raw_sessions
                for key in ("score", "common16_score")
            )
            flips += sum(
                (row[key] >= row[threshold])
                != (expected_sessions[row["session_id"]][key] >= row[threshold])
                for row in raw_sessions
                for key, threshold in (
                    ("score", "threshold"),
                    ("common16_score", "common16_threshold"),
                )
            )
            units_path = root / f"repeat_{repeat}_units.jsonl"
            write_jsonl(units_path, units)
            write_jsonl(root / f"repeat_{repeat}_messages.jsonl", raw_messages)
            write_jsonl(root / f"repeat_{repeat}_sessions.jsonl", raw_sessions)
            sccd_times = [row["seconds"] for row in units if row["source"] == "sccd"]
            result = {
                "identity": freeze["identity"],
                "repeat": repeat,
                "total_seconds": total,
                "source_seconds": dict(per_source),
                "record_count": len(raw_messages),
                "SCCD_conversation_count": len(sccd_times),
                "SCCD_median_seconds": float(np.median(sccd_times)),
                "SCCD_P95_seconds": float(np.percentile(sccd_times, 95)),
                "resources": monitor.result(),
                "warmup": warm,
                "raw_agreement": {
                    "message_max_absolute_difference": score_error,
                    "conversation_max_absolute_difference": session_error,
                    "decision_flips": flips,
                    "tolerance": 1e-4,
                },
                "artifacts": source_hashes(
                    root / f"repeat_{repeat}_{name}.jsonl"
                    for name in ("units", "messages", "sessions")
                ),
            }
            result["status"] = (
                "passed" if max(score_error, session_error) <= 1e-4 and flips == 0 else "failed"
            )
            save(target, result)
            if result["status"] != "passed":
                raise RuntimeError("Raw/cached inference agreement failed; failed pass preserved")
            results.append(result)
            guard(output)
            verify_hashes(freeze["code_hashes"])
            verify_hashes(freeze["reference_artifacts"])
        summary = {
            "variant": variant,
            "seed": 42,
            "raw_repeats": results,
            "total_seconds_mean": float(np.mean([r["total_seconds"] for r in results])),
            "model_package_load_seconds": package_load,
            "encoder_initial_load_and_first_warmup_seconds": encoder_start,
            "startup_resources": startup_monitor.result(),
            "initial_warmup": initial_warm,
            "encoder_asset_bytes": read(output / "protocol.json")["encoder"]["asset_bytes"],
            "model_bytes": (folder / "model.joblib").stat().st_size,
            "CPU": platform.processor(),
            "RAM_bytes": psutil.virtual_memory().total,
            "threads": 4,
            "source_record_counts": dict(Counter(r["source"] for r in records)),
            "source_mean_records_per_second": {
                source: count / float(np.mean([r["source_seconds"][source] for r in results]))
                for source, count in Counter(r["source"] for r in records).items()
            },
            "GPU_required": False,
            "API_cost": 0,
            "interpretation_cost_included": False,
            "scope": (
                "complete frozen TEST raw text; canonical SCCD session/other64 batches; "
                "no topic naming"
            ),
            "training_cost": (
                "new fit cached times in run manifests; "
                "frozen encoder TRAIN bank cost separately inherited"
            ),
            "deployment_dependencies": [
                "Python",
                "PyTorch CPU",
                "sentence-transformers",
                "scikit-learn",
                "SciPy",
                "NumPy",
            ],
            "paid_API_calls": 0,
        }
        save(root / "summary.json", summary)
        del package, encoder
        gc.collect()
        return summary
