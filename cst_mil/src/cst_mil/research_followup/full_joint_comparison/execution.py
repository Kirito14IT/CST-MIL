"""Sequential full-corpus execution with explicit freezes and resumable stages."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import CATEGORIES, METHODS, PROTOCOL, SEEDS
from .common import (
    PROJECT,
    TreeMonitor,
    digest,
    identity,
    participating_code,
    read,
    save,
    source_hashes,
    verify_hashes,
    write_jsonl,
)


def _encoder_metadata() -> dict:
    from cst_mil.unified_semantic_v1_1 import encoder as encoder_module
    from cst_mil.unified_semantic_v1_1.encoder import (
        DEFAULT_ENCODER_PATH,
        ENCODING_VERSION,
        LocalSemanticEncoder,
    )

    enc = LocalSemanticEncoder(threads=4)
    manifest = enc.validate_assets()
    return {
        "encoder_path": str(DEFAULT_ENCODER_PATH),
        "fingerprint": enc.fingerprint,
        "encoding_version": ENCODING_VERSION,
        "encoder_code_sha256": digest(Path(encoder_module.__file__)),
        "asset_bytes": sum(x["bytes"] for x in manifest["files"].values()),
        "asset_hashes": {
            str(DEFAULT_ENCODER_PATH / p): v["sha256"] for p, v in manifest["files"].items()
        },
    }


def freeze_experiment(output: Path, threads: int) -> dict:
    from .data import load_prepared

    _, _, data = load_prepared(output)
    encoder = _encoder_metadata()
    value = {
        "protocol": PROTOCOL,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "threads": threads,
        "dataset_sha256": digest(output / "prepared/manifest.json"),
        "dataset_fingerprint": data["fingerprint"],
        "code_hashes": participating_code(),
        "encoder": encoder,
        "test_exposure_disclosed": True,
        "supervision": "joint observed comment gold; SCCD session gold; no bag label copying",
        "prediction_batch_policy": "SCCD complete session; other sources canonical batches of 64",
        "threshold_selection": "validation macro-F1, lower threshold on ties",
        "unsupervised_oof_policy": (
            "whole training-only representation fixed; held session labels "
            "and duplicate components excluded from heads"
        ),
    }
    path = output / "experiment.json"
    if path.exists():
        if read(path) != value:
            raise RuntimeError(
                "Experiment code/data/encoder/CPU policy changed; use a new revision"
            )
    else:
        save(path, value)
    return value


def verify_experiment(output: Path) -> dict:
    frozen = read(output / "experiment.json")
    verify_hashes(frozen["code_hashes"])
    if frozen["code_hashes"] != participating_code():
        raise RuntimeError("Participating code inventory changed")
    if digest(output / "prepared/manifest.json") != frozen["dataset_sha256"]:
        raise RuntimeError("Prepared cohort changed")
    from .data import load_prepared

    load_prepared(output)
    verify_hashes(frozen["encoder"]["asset_hashes"])
    return frozen


def encode_split(output: Path, split: str, threads: int = 4) -> dict:
    """Encode one split only; chunk checkpoints survive interruption without refitting."""
    import numpy as np

    from cst_mil.unified_semantic_v1_1.encoder import LocalSemanticEncoder

    from .data import load_prepared

    if (output / "experiment.json").exists():
        verify_experiment(output)
    records, _, _ = load_prepared(output)
    rows = [r for r in records if r["split"] == split]
    metadata = _encoder_metadata()
    folder = output / "cache" / split
    folder.mkdir(parents=True, exist_ok=True)
    expected = {
        "record_ids": [r["record_id"] for r in rows],
        "text_sha256": [identity(r["text"]) for r in rows],
        "encoder": metadata,
        "split": split,
        "threads": threads,
        "prepared_manifest_sha256": digest(output / "prepared/manifest.json"),
    }
    binding = folder / "binding.json"
    if binding.exists() and read(binding) != expected:
        raise RuntimeError("Embedding cohort/encoder fingerprint changed")
    if not binding.exists():
        save(binding, expected)
    final = folder / "complete.json"
    if final.exists():
        state = read(final)
        verify_hashes(state["artifacts"])
        return state
    encoder = LocalSemanticEncoder(
        metadata["encoder_path"], metadata["fingerprint"], threads=threads
    )
    arrays, cold_seconds, load_seconds = [], 0.0, 0.0
    monitor = TreeMonitor()
    monitor.__enter__()
    for start in range(0, len(rows), 1024):
        block = rows[start : start + 1024]
        path = folder / f"part_{start:06d}.npz"
        timing_path = path.with_suffix(".timing.json")
        if path.exists() and not timing_path.exists():
            raise RuntimeError(
                "Embedding part has no timing/fingerprint seal; retain it and use a new revision"
            )
        if timing_path.exists():
            timing = read(timing_path)
            if not path.exists() or digest(path) != timing["sha256"]:
                raise RuntimeError("Embedding part changed")
        started = time.perf_counter()
        before = dict(encoder.info)
        array = encoder.encode([r["text"] for r in block], cache_path=path, batch_size=32)
        elapsed = time.perf_counter() - started
        if not timing_path.exists():
            timing = {
                "sha256": digest(path),
                "cold_seconds": elapsed,
                "encoder_load_seconds": encoder.info["encoder_load_seconds"]
                - before["encoder_load_seconds"],
                "encoded_chunks": encoder.info["encoded_chunks"] - before["encoded_chunks"],
            }
            save(timing_path, timing)
        cold_seconds += timing["cold_seconds"]
        load_seconds += timing["encoder_load_seconds"]
        arrays.append(array)
        print(f"encode {split}: {start + len(block)}/{len(rows)}", flush=True)
    combined = np.concatenate(arrays) if arrays else np.zeros((0, 384), dtype=np.float32)
    target = folder / "embeddings.npy"
    temp = target.with_suffix(".pending.npy")
    np.save(temp, combined, allow_pickle=False)
    temp.replace(target)
    monitor.__exit__(None, None, None)
    if _encoder_metadata() != metadata:
        raise RuntimeError("Encoder code/assets changed during encoding")
    load_prepared(output)
    if (output / "experiment.json").exists():
        verify_experiment(output)
    result = {
        "split": split,
        "n_records": len(rows),
        "cold_seconds": cold_seconds,
        "encoder_load_seconds": load_seconds,
        "encoder": metadata,
        "resources": monitor.result(),
        "record_ids": [r["record_id"] for r in rows],
        "artifacts": source_hashes(
            [binding, target, *folder.glob("part_*.npz"), *folder.glob("part_*.timing.json")]
        ),
    }
    save(final, result)
    return result


def embeddings_for(output: Path, records: list[dict], *, include_test: bool = False):
    import numpy as np

    matrix = np.zeros((len(records), 384), dtype=np.float32)
    positions = {r["record_id"]: i for i, r in enumerate(records)}
    if len(positions) != len(records):
        raise ValueError("Duplicate requested record IDs")
    cold = {}
    covered = set()
    encoding_peak = 0
    for split in ("train", "dev", "test") if include_test else ("train", "dev"):
        folder = output / "cache" / split
        state = read(folder / "complete.json")
        verify_hashes(state["artifacts"])
        binding = read(folder / "binding.json")
        if state["record_ids"] != binding["record_ids"] or len(set(state["record_ids"])) != len(
            state["record_ids"]
        ):
            raise RuntimeError("Embedding record IDs differ from frozen binding")
        if (
            binding["prepared_manifest_sha256"] != digest(output / "prepared/manifest.json")
            or state["encoder"] != _encoder_metadata()
        ):
            raise RuntimeError("Embedding dataset or encoder drift")
        values = np.load(folder / "embeddings.npy", allow_pickle=False)
        if values.shape != (len(state["record_ids"]), 384) or not np.isfinite(values).all():
            raise RuntimeError("Embedding shape mismatch")
        for j, rid in enumerate(state["record_ids"]):
            if rid in positions:
                original = records[positions[rid]]
                if (
                    original["split"] != split
                    or identity(original["text"]) != binding["text_sha256"][j]
                ):
                    raise RuntimeError("Embedding text/split binding mismatch")
                matrix[positions[rid]] = values[j]
                covered.add(rid)
        cold[split] = state["cold_seconds"]
        encoding_peak = max(encoding_peak, state.get("resources", {}).get("peak_tree_rss_bytes", 0))
    required = {r["record_id"] for r in records if include_test or r["split"] != "test"}
    if covered != required:
        raise RuntimeError("Missing or surplus embedding rows; never substitute silent zeros")
    meta = _encoder_metadata()
    meta["encoding_seconds_by_split"] = cold
    meta["encoding_peak_tree_rss_bytes"] = encoding_peak
    return matrix, meta


def _canonical_predictions(pred: dict, records: list[dict]) -> dict:
    lookup = {r["record_id"]: r for r in records}
    for row in pred["message"]:
        original = lookup[row["record_id"]]
        for key in (
            "source",
            "split",
            "session_id",
            "message_id",
            "labels",
            "observed_mask",
            "duplicate_component",
            "is_fallback",
            "eval_group",
        ):
            if key in original:
                row[key] = original[key]
        row["evaluation_group"] = row.get("eval_group", original.get("evaluation_group"))
        row["text_length"] = len(original["text"])
        if not isinstance(row["scores"], dict):
            row["scores"] = dict(zip(CATEGORIES, map(float, row["scores"]), strict=True))
    return pred


def _save_predictions(folder: Path, split: str, pred: dict, records: list[dict]) -> None:
    pred = _canonical_predictions(pred, records)
    write_jsonl(folder / f"predictions_message_{split}.jsonl", pred["message"])
    write_jsonl(folder / f"predictions_session_{split}.jsonl", pred["session"])


def _check_complete(path: Path, expected_identity: str) -> bool:
    if not path.exists():
        return False
    state = read(path)
    if state["identity"] != expected_identity:
        raise RuntimeError("Run/stage fingerprint changed; refusing resume")
    if state["status"] == "complete":
        verify_hashes(state["artifacts"])
        return True
    return False


def train_one(output: Path, method: str, seed: int, threads: int = 4) -> dict:
    import joblib

    from .data import load_prepared
    from .model import train_method

    frozen = verify_experiment(output)
    if (
        threads != frozen["threads"]
        or method not in frozen["methods"]
        or seed not in frozen["seeds"]
    ):
        raise ValueError("Run method/seed/CPU budget differs from frozen experiment")
    folder = output / "runs" / method / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    run_id = identity({"experiment": identity(frozen), "method": method, "seed": seed})
    state_path = folder / "train_state.json"
    if _check_complete(state_path, run_id):
        return read(folder / "run_manifest.json")
    save(state_path, {"identity": run_id, "status": "started", "started_at": time.time()})
    records, sessions, _ = load_prepared(output)
    embeddings, metadata = (None, _encoder_metadata())
    if method in {"bertopic", "cst_v1_1", "minilm_lr"}:
        embeddings, metadata = embeddings_for(output, records)
    print(f"TRAIN {method} seed={seed}", flush=True)
    with TreeMonitor() as monitor:
        started = time.perf_counter()
        previous = sorted(folder.glob("attempt_*"))
        attempt = folder / f"attempt_{len(previous) + 1:03d}"
        attempt.mkdir(exist_ok=False)
        if previous:
            partial = attempt / "previous_partial_exports"
            partial.mkdir()
            for old_file in folder.iterdir():
                if old_file.is_file():
                    shutil.copy2(old_file, partial / old_file.name)
        package, result = train_method(
            records,
            sessions,
            method,
            seed,
            attempt,
            embeddings=embeddings,
            encoder_metadata=metadata,
            threads=threads,
        )
        elapsed = time.perf_counter() - started
        package.model_version = (
            "cst-unified-semantic-v1.1-full"
            if method == "cst_v1_1"
            else method + "-joint-supervised-full"
        )
        package.training_summary["run_identity"] = run_id
        model_path = folder / "model.joblib"
        joblib.dump(package, model_path, compress=3)
    if monitor.max_affinity > threads:
        save(folder / "invalid_resource_policy.json", monitor.result())
        raise RuntimeError("Training process tree exceeded the frozen CPU affinity budget")
    verify_experiment(output)
    dev_records = [r for r in records if r["split"] == "dev"]
    _save_predictions(folder, "dev", result.pop("dev_predictions"), dev_records)
    save(folder / "fit_details.json", result)
    save(folder / "metrics.json", result.get("metrics", {}))
    thresholds = dict(package.thresholds)
    thresholds.update(
        conversation=package.conversation_threshold,
        conversation_common16=package.common16_threshold,
    )
    shared_encoding = sum(metadata.get("encoding_seconds_by_split", {}).values())
    manifest = {
        "protocol": PROTOCOL,
        "identity": run_id,
        "method": method,
        "seed": seed,
        "model_version": package.model_version,
        "training_attempt": str(attempt.relative_to(output)),
        "thresholds": thresholds,
        "model_sha256": digest(model_path),
        "model_bytes": model_path.stat().st_size,
        "encoder_asset_bytes": metadata["asset_bytes"] if embeddings is not None else 0,
        "training_seconds_cached": elapsed,
        "shared_encoder_training_seconds": shared_encoding,
        "training_seconds_cold": elapsed + shared_encoding,
        "cold_training_definition": "fit plus validation selection; includes train+dev encoding",
        "cold_training_peak_tree_rss_bytes": max(
            monitor.peak, metadata.get("encoding_peak_tree_rss_bytes", 0)
        ),
        "resources": monitor.result(),
        "parameters": result.get("parameters"),
        "old_test_exposure_disclosed": True,
        "training_labels": "observed comment masks; SCCD session gold only in bag layer",
        "comparison_type": "adapted joint strongly supervised systems",
    }
    save(folder / "run_manifest.json", manifest)
    artifacts = source_hashes(
        [
            model_path,
            folder / "run_manifest.json",
            folder / "fit_details.json",
            folder / "metrics.json",
            folder / "predictions_message_dev.jsonl",
            folder / "predictions_session_dev.jsonl",
        ]
    )
    save(state_path, {"identity": run_id, "status": "complete", "artifacts": artifacts})
    print(f"COMPLETE {method} seed={seed} fit={elapsed:.1f}s", flush=True)
    return manifest


def evaluate_one(output: Path, method: str, seed: int, threads: int = 4) -> dict:
    import joblib

    from .data import load_prepared

    frozen = verify_experiment(output)
    if (
        threads != frozen["threads"]
        or method not in frozen["methods"]
        or seed not in frozen["seeds"]
    ):
        raise ValueError("Run method/seed/CPU budget differs from frozen experiment")
    folder = output / "runs" / method / f"seed_{seed}"
    run_id = identity({"experiment": identity(frozen), "method": method, "seed": seed})
    if not _check_complete(folder / "train_state.json", run_id):
        raise RuntimeError("Training must be complete before opening test")
    state_path = folder / "test_state.json"
    if _check_complete(state_path, run_id):
        return read(state_path)
    model = joblib.load(folder / "model.joblib")
    manifest = read(folder / "run_manifest.json")
    if digest(folder / "model.joblib") != manifest["model_sha256"]:
        raise RuntimeError("Model differs from validation freeze")
    save(
        folder / "test_freeze.json",
        {
            "model_sha256": manifest["model_sha256"],
            "thresholds": manifest["thresholds"],
            "identity": run_id,
        },
    )
    save(state_path, {"identity": run_id, "status": "started"})
    records, sessions, _ = load_prepared(output)
    records = [r for r in records if r["split"] == "test"]
    sessions = [s for s in sessions if s["split"] == "test"]
    embeddings = None
    if model.eval_embeddings_needed:
        embeddings, _ = embeddings_for(output, records, include_test=True)
    with TreeMonitor() as monitor:
        started = time.perf_counter()
        predictions = model.predict(records, sessions, embeddings=embeddings)
        elapsed = time.perf_counter() - started
    _save_predictions(folder, "test", predictions, records)
    verify_experiment(output)
    result = {
        "identity": run_id,
        "status": "complete",
        "inference_seconds_cached": elapsed,
        "resources": monitor.result(),
        "artifacts": source_hashes(
            [
                folder / "predictions_message_test.jsonl",
                folder / "predictions_session_test.jsonl",
                folder / "test_freeze.json",
            ]
        ),
    }
    save(state_path, result)
    print(f"TEST COMPLETE {method} seed={seed} {elapsed:.1f}s", flush=True)
    return result


def launch_stage(
    output: Path,
    stage: str,
    threads: int,
    method: str | None = None,
    seed: int | None = None,
    split: str | None = None,
) -> None:
    """Every model starts in a fresh process; full logs survive a failed stage."""
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "cst_mil.research_followup.full_joint_comparison",
        "--output",
        str(output),
        "--threads",
        str(threads),
        "_" + stage,
    ]
    if method is not None:
        command += ["--method", method, "--seed", str(seed)]
    if split is not None:
        command += ["--split", split]
    log_root = output / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    name = f"{stage}_{method or split or 'all'}_{seed if seed is not None else ''}.log"
    print("START", stage, method or split or "", seed or "", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    started = time.time()
    with (log_root / name).open("a", encoding="utf-8") as handle:
        handle.write(f"\nSTAGE START {started}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=PROJECT.parent,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        import psutil

        child = psutil.Process(process.pid)
        active = output / "active_child.json"
        save(
            active,
            {
                "pid": child.pid,
                "created": child.create_time(),
                "stage": stage,
                "method": method,
                "seed": seed,
            },
        )
        process.wait()
        if active.exists() and read(active).get("pid") == process.pid:
            active.unlink()
    if process.returncode:
        save(
            output / "failure.json",
            {
                "stage": stage,
                "method": method,
                "seed": seed,
                "returncode": process.returncode,
                "log": str(log_root / name),
                "time": time.time(),
                "no_fallback_applied": True,
            },
        )
        raise RuntimeError(f"Stage failed ({process.returncode}): {log_root / name}")
    print("DONE", stage, method or split or "", seed or "", flush=True)


def status(output: Path) -> dict:
    runs = []
    for method in METHODS:
        for seed in SEEDS:
            folder = output / "runs" / method / f"seed_{seed}"
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "train": read(folder / "train_state.json").get("status")
                    if (folder / "train_state.json").exists()
                    else "pending",
                    "test": read(folder / "test_state.json").get("status")
                    if (folder / "test_state.json").exists()
                    else "pending",
                    "cost": (folder / "cost.json").exists(),
                }
            )
    return {
        "runs": runs,
        "training_complete": sum(r["train"] == "complete" for r in runs),
        "test_complete": sum(r["test"] == "complete" for r in runs),
    }
