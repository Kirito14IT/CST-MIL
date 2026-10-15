"""Isolated raw-input evaluation of the original, sealed training weights.

This revision never edits a base model, base threshold manifest or base result.
All semantic processes warm up once on the same label-independent TRAIN CONDA
batch, then predict complete dev/test inference bundles with embeddings=None.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import CATEGORIES, METHODS, SEEDS
from .common import (
    TreeMonitor,
    campaign_lock,
    digest,
    identity,
    read,
    read_jsonl,
    save,
    source_hashes,
    verify_hashes,
    write_jsonl,
)

SEMANTIC = ("bertopic", "cst_v1_1", "minilm_lr")
TRADITIONAL = tuple(m for m in METHODS if m not in SEMANTIC)
REVISION_NAME = "raw_replay_v2"
SCORE_TOLERANCE = 1e-4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def revision_root(base: Path) -> Path:
    return Path(base).resolve() / "evaluation_revisions" / REVISION_NAME


def _copy(source: Path, target: Path) -> None:
    """Copy evidence once; resume checks bytes and never overwrites a mismatch."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if digest(source) != digest(target):
            raise RuntimeError(f"Revision copy already differs: {target}")
        return
    temporary = target.with_name(target.name + f".{os.getpid()}.pending")
    shutil.copyfile(source, temporary)
    if digest(source) != digest(temporary):
        raise RuntimeError("Evidence copy verification failed")
    temporary.replace(target)


def _base_run(base: Path, method: str, seed: int) -> Path:
    return base / "runs" / method / f"seed_{seed}"


def _view_run(revision: Path, method: str, seed: int) -> Path:
    return revision / "runs" / method / f"seed_{seed}"


def _verify_base_training(base: Path, experiment: dict, method: str, seed: int) -> dict:
    folder = _base_run(base, method, seed)
    state, manifest = read(folder / "train_state.json"), read(folder / "run_manifest.json")
    expected = identity({"experiment": identity(experiment), "method": method, "seed": seed})
    if (
        state.get("status") != "complete"
        or state["identity"] != expected
        or manifest["identity"] != expected
    ):
        raise RuntimeError("Original training identity/seal is incomplete or changed")
    required = [
        folder / name
        for name in (
            "model.joblib",
            "run_manifest.json",
            "fit_details.json",
            "metrics.json",
            "predictions_message_dev.jsonl",
            "predictions_session_dev.jsonl",
        )
    ]
    if not {str(p.resolve()) for p in required} <= set(state["artifacts"]):
        raise RuntimeError("Original training seal omitted required artifacts")
    verify_hashes(state["artifacts"])
    if digest(folder / "model.joblib") != manifest["model_sha256"]:
        raise RuntimeError("Original model weight fingerprint changed")
    return {
        "method": method,
        "seed": seed,
        "base_run_identity": expected,
        "base_model_path": str((folder / "model.joblib").resolve()),
        "model_sha256": manifest["model_sha256"],
        "base_run_manifest_sha256": digest(folder / "run_manifest.json"),
        "base_train_state_sha256": digest(folder / "train_state.json"),
    }


def _ensure_views(destination: Path, frozen: dict) -> None:
    """Repair interrupted initialization, never resetting existing DEV overrides."""
    base = Path(frozen["base_output"])
    for inherited in frozen["base_training"]:
        method, seed = inherited["method"], inherited["seed"]
        path = _view_run(destination, method, seed) / "run_manifest.json"
        original = read(_base_run(base, method, seed) / "run_manifest.json")
        if path.exists():
            existing = read(path)
            if (
                existing.get("manifest_role") != "evaluation_view"
                or existing.get("evaluation_revision_identity") != frozen["identity"]
            ):
                raise RuntimeError("Existing evaluation view belongs to another revision")
            if any(existing.get(key) != value for key, value in inherited.items()):
                raise RuntimeError("Existing view changed its inherited training binding")
            if any(
                existing.get(key) != value for key, value in original.items() if key != "thresholds"
            ):
                raise RuntimeError("Existing view changed original training metadata")
            continue
        view = {
            **original,
            **inherited,
            "manifest_role": "evaluation_view",
            "base_output": str(base),
            "evaluation_revision_identity": frozen["identity"],
            "inherited_training": True,
            "evaluation_version": REVISION_NAME,
            "prediction_policy": "raw_after_fixed_train_warmup"
            if method in SEMANTIC
            else "inherited_valid_original",
            "threshold_status": "pending_raw_dev" if method in SEMANTIC else "original_dev_frozen",
        }
        save(path, view)


def _archive_incomplete(folder: Path, stage: str) -> Path | None:
    """Keep every unsealed attempt; complete evidence is handled before this call."""
    names = {
        "dev": (
            "dev_state.json",
            "dev_provenance.json",
            "validation_freeze.json",
            "predictions_message_dev.jsonl",
            "predictions_session_dev.jsonl",
            "dev_progress.json",
        ),
        "test": (
            "test_state.json",
            "test_provenance.json",
            "test_freeze.json",
            "predictions_message_test.jsonl",
            "predictions_session_test.jsonl",
            "test_progress.json",
        ),
        "cost": (
            "cost_state.json",
            "cost.json",
            "cost_provenance.json",
            "cost_invalid.json",
            "cost_progress.json",
        ),
    }[stage]
    existing = [folder / name for name in names if (folder / name).exists()]
    if not existing:
        return None
    attempt = (
        folder
        / "attempts"
        / (stage + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    attempt.mkdir(parents=True, exist_ok=False)
    if (folder / "run_manifest.json").exists():
        shutil.copyfile(folder / "run_manifest.json", attempt / "run_manifest.snapshot.json")
    for path in existing:
        path.replace(attempt / path.name)
    save(
        attempt / "archive_receipt.json",
        {
            "stage": stage,
            "archived_at_utc": utc_now(),
            "reason": "Incomplete/unsealed stage attempt preserved before recovery",
            "artifacts": source_hashes(attempt.glob("*")),
        },
    )
    return attempt


def _refuse_rejected_cost_retry(folder: Path, revision_identity: str) -> None:
    for path in [folder / "cost_invalid.json", *folder.glob("attempts/cost_*/cost_invalid.json")]:
        if path.exists():
            invalid = read(path)
            if (
                invalid.get("evaluation_revision_identity") == revision_identity
                and invalid.get("status") == "invalid_prediction_consistency"
            ):
                raise RuntimeError(
                    "A cost was rejected under this revision identity; "
                    "automatic retry-until-pass is prohibited"
                )


def _consistent_cost(proof: dict) -> bool:
    return (
        max(proof["max_message_abs_diff"], proof["max_session_abs_diff"]) <= SCORE_TOLERANCE
        and proof["head_threshold_disagreements_all_repetitions"] == 0
        and not any(proof["session_threshold_disagreements_all_repetitions"].values())
    )


def select_warmup(records: list[dict]) -> list[dict]:
    """Exactly the first 64 retained TRAIN CONDA record IDs, independent of gold."""
    chosen = sorted(
        (r for r in records if r["source"] == "conda" and r["split"] == "train"),
        key=lambda r: r["record_id"],
    )[:64]
    if len(chosen) != 64:
        raise ValueError("Fixed TRAIN CONDA warm-up requires 64 records")
    return chosen


def init_revision(base: Path, threads: int = 4) -> dict:
    from .data import load_prepared
    from .execution import verify_experiment

    base = Path(base).resolve()
    frozen = verify_experiment(base)
    if (
        frozen["threads"] != threads
        or frozen["methods"] != list(METHODS)
        or frozen["seeds"] != list(SEEDS)
    ):
        raise ValueError(
            "Raw revision requires the unchanged seven-method, three-seed CPU protocol"
        )
    records, sessions, _ = load_prepared(base)
    warmup = select_warmup(records)
    destination = revision_root(base)
    manifest_path = destination / "revision_manifest.json"
    if manifest_path.exists():
        existing = verify_revision(destination)
        _ensure_views(destination, existing)
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    training = [
        _verify_base_training(base, frozen, method, seed) for method in METHODS for seed in SEEDS
    ]
    protocol = {
        "protocol": "raw-input-evaluation-revision-v1",
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "semantic_methods": list(SEMANTIC),
        "traditional_methods": list(TRADITIONAL),
        "threads": threads,
        "model_weights_policy": (
            "All original sealed model files are read-only; no fitting or weight edits"
        ),
        "raw_policy": (
            "package.predict(records, sessions, embeddings=None) in canonical "
            "complete SCCD sessions / other-source batches of 64"
        ),
        "warmup": {
            "source": "conda",
            "split": "train",
            "count": 64,
            "selection": (
                "first 64 retained TRAIN CONDA record IDs in lexicographic order; "
                "independent of labels and scores"
            ),
            "record_ids": [r["record_id"] for r in warmup],
            "text_binding": identity(
                [{"record_id": r["record_id"], "text_sha256": identity(r["text"])} for r in warmup]
            ),
            "repetitions_per_process": 1,
            "predictions_discarded": True,
            "shared_by_stages": ["dev", "test", "cost"],
        },
        "threshold_policy": (
            "DEV macro-F1 on all observed source masks; lower threshold for ties; "
            "separate override metadata, never model-file mutation"
        ),
        "reference_policy": (
            "five topic baselines by three-seed mean DEV AP; fixed method order resolves exact ties"
        ),
        "test_gate": (
            "all 21 dev seals including nine semantic raw DEV replays and validation "
            "reference freeze must precede any revised test replay"
        ),
        "cost_policy": {
            "seed": 42,
            "passes": 3,
            "score_tolerance": SCORE_TOLERANCE,
            "head_threshold_disagreements_required": 0,
            "main_and_common16_threshold_disagreements_required": 0,
            "invalid_consistency_retry_policy": (
                "never automatically retry a rejected cost under the same revision identity; "
                "interrupted infrastructure attempts may resume"
            ),
            "first_inference": (
                "first test conversation after the same TRAIN warm-up; "
                "startup warm-up time separately reported"
            ),
            "traditional_costs": "inherit previously sealed valid costs with original provenance",
        },
        "test_counts": {
            "messages": sum(r["split"] == "test" for r in records),
            "sessions": sum(s["split"] == "test" for s in sessions),
        },
        "known_boundary": (
            "Training feature caches and fitted weights are unchanged; this is an "
            "evaluation/calibration revision, not retraining. Original results remain preserved."
        ),
        "runtime_revision": (
            "Windows venv redirector startup repair only: child validates the active "
            "launcher PID/create-time in its real ancestor chain and the owning parent "
            "lock PID/create-time plus stage/method/seed. Scientific protocol unchanged."
        ),
    }
    protocol_path = destination / "revision_protocol.json"
    if protocol_path.exists() and read(protocol_path) != protocol:
        raise RuntimeError("Pending raw revision protocol differs; use a separate revision")
    if not protocol_path.exists():
        save(protocol_path, protocol)
    base_prepared = read(base / "prepared/manifest.json")
    for name in (*base_prepared["artifacts"], "manifest.json"):
        _copy(base / "prepared" / name, destination / "prepared" / name)
    _copy(base / "experiment.json", destination / "experiment.json")
    code_paths = [
        Path(__file__),
        *[
            Path(__file__).with_name(name)
            for name in (
                "verification.py",
                "cost.py",
                "execution.py",
                "model.py",
                "data.py",
                "common.py",
            )
        ],
    ]
    payload = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "base_output": str(base),
        "base_experiment_sha256": digest(base / "experiment.json"),
        "base_experiment_identity": identity(frozen),
        "prepared_manifest_sha256": digest(base / "prepared/manifest.json"),
        "protocol_sha256": digest(protocol_path),
        "code_hashes": source_hashes(code_paths),
        "base_training": training,
        "scope": "evaluation view; original training seals and weights remain authoritative",
        "revision_name": REVISION_NAME,
        "model_configurations_changed": False,
        "model_files_modified": False,
    }
    payload["identity"] = identity(payload)
    save(manifest_path, payload)
    _ensure_views(destination, payload)
    return payload


def verify_revision(revision: Path) -> dict:
    from .execution import verify_experiment

    revision = Path(revision).resolve()
    frozen = read(revision / "revision_manifest.json")
    payload = {key: value for key, value in frozen.items() if key != "identity"}
    if frozen["identity"] != identity(payload):
        raise RuntimeError("Raw revision identity changed")
    verify_hashes(frozen["code_hashes"])
    if digest(revision / "revision_protocol.json") != frozen["protocol_sha256"]:
        raise RuntimeError("Raw revision protocol changed")
    base = Path(frozen["base_output"])
    experiment = verify_experiment(base)
    if (
        digest(base / "experiment.json") != frozen["base_experiment_sha256"]
        or identity(experiment) != frozen["base_experiment_identity"]
    ):
        raise RuntimeError("Base experiment identity changed")
    if digest(revision / "prepared/manifest.json") != frozen["prepared_manifest_sha256"]:
        raise RuntimeError("Revision prepared-data view changed")
    prepared = read(revision / "prepared/manifest.json")
    for name, expected in prepared["artifacts"].items():
        if digest(revision / "prepared" / name) != expected:
            raise RuntimeError("Revision prepared artifact changed")
    for row in frozen["base_training"]:
        folder = _base_run(base, row["method"], row["seed"])
        for name, expected in (
            ("model.joblib", row["model_sha256"]),
            ("train_state.json", row["base_train_state_sha256"]),
            ("run_manifest.json", row["base_run_manifest_sha256"]),
        ):
            if digest(folder / name) != expected:
                raise RuntimeError("Inherited training evidence changed")
    return frozen


def _stage_identity(revision: Path, method: str, seed: int, stage: str) -> str:
    return identity(
        {
            "revision": read(revision / "revision_manifest.json")["identity"],
            "method": method,
            "seed": seed,
            "stage": stage,
            "run_manifest_sha256": digest(_view_run(revision, method, seed) / "run_manifest.json"),
        }
    )


def _complete(revision: Path, method: str, seed: int, stage: str) -> bool:
    folder = _view_run(revision, method, seed)
    path = folder / f"{stage}_state.json"
    if not path.exists():
        return False
    seal = read(path)
    if seal.get("status") != "complete":
        return False
    if seal["identity"] != _stage_identity(revision, method, seed, stage):
        raise RuntimeError("Raw evaluation stage identity changed")
    verify_hashes(seal["artifacts"])
    return True


def _seal(revision: Path, method: str, seed: int, stage: str, paths: list[Path], **extra) -> dict:
    result = {
        "identity": _stage_identity(revision, method, seed, stage),
        "status": "complete",
        "method": method,
        "seed": seed,
        "stage": stage,
        "completed_at_utc": utc_now(),
        "artifacts": source_hashes(paths),
        **extra,
    }
    save(_view_run(revision, method, seed) / f"{stage}_state.json", result)
    return result


def _thresholds(messages: list[dict], sessions: list[dict]) -> tuple[dict, dict]:
    from .verification import category, independent_dev_threshold

    result, proof = {}, {}
    if any(row["split"] != "dev" for row in messages + sessions):
        raise ValueError("Only complete DEV predictions may select thresholds")
    for name in CATEGORIES:
        observed = [r for r in messages if category(r, "observed_mask", name)]
        proof[name] = independent_dev_threshold(
            [category(r, "labels", name) for r in observed],
            [category(r, "scores", name) for r in observed],
        )
    for name, score in (("conversation", "score"), ("conversation_common16", "common16_score")):
        proof[name] = independent_dev_threshold(
            [r["label"] for r in sessions], [r[score] for r in sessions]
        )
    result = {name: value["threshold"] for name, value in proof.items()}
    return result, proof


def _override_session_thresholds(predictions: dict, thresholds: dict) -> None:
    for row in predictions["session"]:
        row["threshold"] = thresholds["conversation"]
        row["common16_threshold"] = thresholds["conversation_common16"]


def raw_predict(
    package, records: list[dict], sessions: list[dict], *, progress_path: Path | None = None
) -> dict:
    from .cost import inference_bundles
    from .execution import _canonical_predictions

    result = {"message": [], "session": []}
    bundles = inference_bundles(records, sessions)
    started = time.perf_counter()
    for index, (_, _, group_records, group_sessions) in enumerate(bundles, 1):
        prediction = package.predict(group_records, group_sessions, embeddings=None)
        prediction = _canonical_predictions(prediction, group_records)
        result["message"].extend(prediction["message"])
        result["session"].extend(prediction["session"])
        if index % 20 == 0 or index == len(bundles):
            receipt = {
                "completed_bundles": index,
                "total_bundles": len(bundles),
                "messages": len(result["message"]),
                "sessions": len(result["session"]),
                "elapsed_seconds": time.perf_counter() - started,
                "updated_at_utc": utc_now(),
            }
            if progress_path is not None:
                save(progress_path, receipt)
            print(
                f"RAW replay {index}/{len(bundles)} messages={receipt['messages']} "
                f"{receipt['elapsed_seconds']:.1f}s",
                flush=True,
            )
    if len(result["message"]) != len(records) or len(result["session"]) != len(sessions):
        raise RuntimeError("Raw replay omitted an inference record or session")
    return result


def _load_package_and_warmup(
    base: Path, revision: Path, method: str, seed: int, records: list[dict]
):
    import joblib

    manifest = read(_view_run(revision, method, seed) / "run_manifest.json")
    if digest(Path(manifest["base_model_path"])) != manifest["model_sha256"]:
        raise RuntimeError("Model differs from original training weights")
    started = time.perf_counter()
    package = joblib.load(manifest["base_model_path"])
    load_seconds = time.perf_counter() - started
    warmup = select_warmup(records)
    protocol = read(revision / "revision_protocol.json")["warmup"]
    if [r["record_id"] for r in warmup] != protocol["record_ids"] or identity(
        [{"record_id": r["record_id"], "text_sha256": identity(r["text"])} for r in warmup]
    ) != protocol["text_binding"]:
        raise RuntimeError("Fixed training warm-up inputs changed")
    started = time.perf_counter()
    package.predict(warmup, [], embeddings=None)
    startup = time.perf_counter() - started
    return package, {
        "model_load_seconds": load_seconds,
        "startup_warmup_seconds": startup,
        "warmup_record_ids": protocol["record_ids"],
        "warmup_split": "train",
        "warmup_source": "conda",
        "warmup_repetitions": 1,
        "warmup_predictions_discarded": True,
    }


def replay_dev(revision: Path, method: str, seed: int) -> dict:
    from .data import load_prepared
    from .verification import verify_dev_thresholds, verify_prediction_cohort

    frozen = verify_revision(revision)
    if _complete(revision, method, seed, "dev"):
        return read(_view_run(revision, method, seed) / "dev_state.json")
    if (revision / "validation_reference_freeze.json").exists():
        raise RuntimeError("Cannot alter DEV outputs after validation reference freeze")
    base, folder = Path(frozen["base_output"]), _view_run(revision, method, seed)
    _archive_incomplete(folder, "dev")
    save(
        folder / "dev_state.json",
        {
            "identity": _stage_identity(revision, method, seed, "dev"),
            "status": "started",
            "started_at_utc": utc_now(),
        },
    )
    records, sessions, _ = load_prepared(base)
    dev_records, dev_sessions = (
        [r for r in records if r["split"] == "dev"],
        [s for s in sessions if s["split"] == "dev"],
    )
    manifest = read(folder / "run_manifest.json")
    if method in TRADITIONAL:
        original = _base_run(base, method, seed)
        state = read(original / "train_state.json")
        verify_hashes(state["artifacts"])
        paths = []
        for name in ("predictions_message_dev.jsonl", "predictions_session_dev.jsonl"):
            _copy(original / name, folder / name)
            paths.append(folder / name)
        prediction = {"message": read_jsonl(paths[0]), "session": read_jsonl(paths[1])}
        proof = verify_dev_thresholds(
            prediction["message"], prediction["session"], manifest["thresholds"]
        )
        provenance = {
            "kind": "inherited_sealed_dev",
            "base_train_state_sha256": digest(original / "train_state.json"),
            "source_artifacts": source_hashes([original / p.name for p in paths]),
            "no_prediction_reexecution": True,
        }
        resources, startup = {}, None
    else:
        with TreeMonitor() as monitor:
            package, startup = _load_package_and_warmup(base, revision, method, seed, records)
            started = time.perf_counter()
            prediction = raw_predict(
                package, dev_records, dev_sessions, progress_path=folder / "dev_progress.json"
            )
            replay_seconds = time.perf_counter() - started
        thresholds, selection = _thresholds(prediction["message"], prediction["session"])
        _override_session_thresholds(prediction, thresholds)
        manifest.update(
            thresholds=thresholds,
            threshold_status="raw_dev_frozen",
            threshold_override_only=True,
            model_file_thresholds_unchanged=True,
        )
        save(folder / "run_manifest.json", manifest)
        proof = verify_dev_thresholds(prediction["message"], prediction["session"], thresholds)
        proof["selection_details"] = selection
        provenance = {
            "kind": "semantic_raw_dev",
            "embeddings_argument": None,
            "startup": startup,
            "replay_seconds": replay_seconds,
            "base_model_sha256": manifest["model_sha256"],
            "original_model_file_modified": False,
        }
        resources = monitor.result()
        if monitor.max_affinity > read(revision / "revision_protocol.json")["threads"]:
            raise RuntimeError("Raw DEV exceeded CPU budget")
        for name, rows in (
            ("predictions_message_dev.jsonl", prediction["message"]),
            ("predictions_session_dev.jsonl", prediction["session"]),
        ):
            write_jsonl(folder / name, rows)
    verify_prediction_cohort(
        records,
        sessions,
        prediction["message"],
        prediction["session"],
        manifest["thresholds"],
        "dev",
    )
    save(folder / "dev_provenance.json", provenance)
    save(
        folder / "validation_freeze.json",
        {
            "created_at_utc": utc_now(),
            "method": method,
            "seed": seed,
            "revision_identity": frozen["identity"],
            "thresholds": manifest["thresholds"],
            "model_sha256": manifest["model_sha256"],
            "threshold_selection": proof,
            "dev_prediction_hashes": source_hashes(
                [folder / "predictions_message_dev.jsonl", folder / "predictions_session_dev.jsonl"]
            ),
            "only_dev_used_for_threshold_selection": True,
        },
    )
    verify_revision(revision)
    return _seal(
        revision,
        method,
        seed,
        "dev",
        [
            folder / p
            for p in (
                "run_manifest.json",
                "predictions_message_dev.jsonl",
                "predictions_session_dev.jsonl",
                "validation_freeze.json",
                "dev_provenance.json",
            )
        ],
        resources=resources,
        prediction_policy=manifest["prediction_policy"],
    )


def freeze_validation(revision: Path) -> dict:
    import numpy as np

    from .verification import task_metrics

    frozen = verify_revision(revision)
    target = revision / "validation_reference_freeze.json"
    if target.exists():
        return verify_validation_freeze(revision)
    values, bindings = {}, {}
    for method in METHODS:
        values[method] = {}
        for seed in SEEDS:
            if not _complete(revision, method, seed, "dev"):
                raise RuntimeError(
                    "All 21 DEV seals are required before reference selection or testing"
                )
            folder = _view_run(revision, method, seed)
            manifest = read(folder / "run_manifest.json")
            metric, _ = task_metrics(
                read_jsonl(folder / "predictions_message_dev.jsonl"),
                read_jsonl(folder / "predictions_session_dev.jsonl"),
                manifest["thresholds"],
            )
            values[method][str(seed)] = metric
            bindings.update(
                source_hashes(
                    [
                        folder / name
                        for name in (
                            "dev_state.json",
                            "run_manifest.json",
                            "validation_freeze.json",
                            "predictions_message_dev.jsonl",
                            "predictions_session_dev.jsonl",
                        )
                    ]
                )
            )
    selected = {}
    for task in values[METHODS[0]][str(SEEDS[0])]:
        candidate = {
            m: {"seed_AP": {str(s): values[m][str(s)][task]["AP"] for s in SEEDS}}
            for m in METHODS[:5]
        }
        for row in candidate.values():
            if any(v is None for v in row["seed_AP"].values()):
                raise RuntimeError("Validation reference selection needs observed AP for all seeds")
            row["mean_AP"] = float(np.mean(list(row["seed_AP"].values())))
        ordered = sorted(METHODS[:5], key=lambda m: (-candidate[m]["mean_AP"], METHODS.index(m)))
        winner = ordered[0]
        selected[task] = {
            "method": winner,
            "validation_mean_AP": candidate[winner]["mean_AP"],
            "candidates": candidate,
            "runner_up": ordered[1],
            "winner_minus_runner_up_AP": candidate[winner]["mean_AP"]
            - candidate[ordered[1]]["mean_AP"],
            "rule": "highest three-seed mean validation AP; fixed method order breaks exact ties",
        }
    result = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "status": "frozen_validation_reference",
        "revision_identity": frozen["identity"],
        "selected_baselines": selected,
        "references": selected,
        "all_21_run_validation_metrics": values,
        "all_dev_artifacts": bindings,
        "test_used_for_decisions": False,
        "only_comparison_references_frozen": True,
        "model_files_modified": False,
    }
    result["identity"] = identity(result)
    save(target, result)
    return result


def verify_validation_freeze(revision: Path) -> dict:
    frozen = verify_revision(revision)
    value = read(revision / "validation_reference_freeze.json")
    if (
        identity({k: v for k, v in value.items() if k != "identity"}) != value["identity"]
        or value["revision_identity"] != frozen["identity"]
    ):
        raise RuntimeError("Raw DEV comparison freeze identity changed")
    verify_hashes(value["all_dev_artifacts"])
    for method in METHODS:
        for seed in SEEDS:
            if not _complete(revision, method, seed, "dev"):
                raise RuntimeError("Raw test gate lacks complete DEV seals")
    return value


def replay_test(revision: Path, method: str, seed: int) -> dict:
    from .data import load_prepared
    from .verification import verify_prediction_cohort

    frozen = verify_revision(revision)
    reference = verify_validation_freeze(revision)
    if _complete(revision, method, seed, "test"):
        return read(_view_run(revision, method, seed) / "test_state.json")
    base, folder = Path(frozen["base_output"]), _view_run(revision, method, seed)
    _archive_incomplete(folder, "test")
    save(
        folder / "test_state.json",
        {
            "identity": _stage_identity(revision, method, seed, "test"),
            "status": "started",
            "started_at_utc": utc_now(),
        },
    )
    records, sessions, _ = load_prepared(base)
    test_records, test_sessions = (
        [r for r in records if r["split"] == "test"],
        [s for s in sessions if s["split"] == "test"],
    )
    manifest = read(folder / "run_manifest.json")
    save(
        folder / "test_freeze.json",
        {
            "created_at_utc": utc_now(),
            "method": method,
            "seed": seed,
            "revision_identity": frozen["identity"],
            "model_sha256": manifest["model_sha256"],
            "thresholds": manifest["thresholds"],
            "run_manifest_sha256": digest(folder / "run_manifest.json"),
            "validation_reference_freeze_sha256": digest(
                revision / "validation_reference_freeze.json"
            ),
            "validation_reference_identity": reference["identity"],
        },
    )
    if method in TRADITIONAL:
        original = _base_run(base, method, seed)
        state = read(original / "test_state.json")
        if state["status"] != "complete" or state["identity"] != manifest["base_run_identity"]:
            raise RuntimeError("Traditional original test seal is invalid")
        verify_hashes(state["artifacts"])
        for name in ("predictions_message_test.jsonl", "predictions_session_test.jsonl"):
            _copy(original / name, folder / name)
        prediction = {
            "message": read_jsonl(folder / "predictions_message_test.jsonl"),
            "session": read_jsonl(folder / "predictions_session_test.jsonl"),
        }
        provenance = {
            "kind": "inherited_sealed_test",
            "base_test_state_sha256": digest(original / "test_state.json"),
            "source_artifacts": source_hashes(
                [
                    original / "predictions_message_test.jsonl",
                    original / "predictions_session_test.jsonl",
                ]
            ),
            "no_prediction_reexecution": True,
        }
        resources = {}
    else:
        with TreeMonitor() as monitor:
            package, startup = _load_package_and_warmup(base, revision, method, seed, records)
            started = time.perf_counter()
            prediction = raw_predict(
                package, test_records, test_sessions, progress_path=folder / "test_progress.json"
            )
            elapsed = time.perf_counter() - started
        _override_session_thresholds(prediction, manifest["thresholds"])
        for name, rows in (
            ("predictions_message_test.jsonl", prediction["message"]),
            ("predictions_session_test.jsonl", prediction["session"]),
        ):
            write_jsonl(folder / name, rows)
        provenance = {
            "kind": "semantic_raw_test",
            "embeddings_argument": None,
            "startup": startup,
            "replay_seconds": elapsed,
            "base_model_sha256": manifest["model_sha256"],
            "original_model_file_modified": False,
        }
        resources = monitor.result()
        if monitor.max_affinity > read(revision / "revision_protocol.json")["threads"]:
            raise RuntimeError("Raw test exceeded CPU budget")
    verify_prediction_cohort(
        records,
        sessions,
        prediction["message"],
        prediction["session"],
        manifest["thresholds"],
        "test",
    )
    save(folder / "test_provenance.json", provenance)
    verify_validation_freeze(revision)
    return _seal(
        revision,
        method,
        seed,
        "test",
        [
            folder / p
            for p in (
                "run_manifest.json",
                "predictions_message_test.jsonl",
                "predictions_session_test.jsonl",
                "test_freeze.json",
                "test_provenance.json",
            )
        ],
        resources=resources,
        validation_reference_freeze_sha256=digest(revision / "validation_reference_freeze.json"),
    )


def measure_cost(revision: Path, method: str, seed: int = 42) -> dict:
    import numpy as np

    from .cost import dependency_sizes, inference_bundles
    from .data import load_prepared

    frozen = verify_revision(revision)
    verify_validation_freeze(revision)
    if seed != 42 or not _complete(revision, method, seed, "test"):
        raise ValueError("Raw cost requires seed42 and sealed revised test predictions")
    folder, base = _view_run(revision, method, seed), Path(frozen["base_output"])
    if _complete(revision, method, seed, "cost"):
        return read(folder / "cost.json")
    _refuse_rejected_cost_retry(folder, frozen["identity"])
    _archive_incomplete(folder, "cost")
    save(
        folder / "cost_state.json",
        {
            "identity": _stage_identity(revision, method, seed, "cost"),
            "status": "started",
            "started_at_utc": utc_now(),
        },
    )
    manifest = read(folder / "run_manifest.json")
    if method in TRADITIONAL:
        original = _base_run(base, method, seed)
        state, cost = read(original / "cost_state.json"), read(original / "cost.json")
        if state["status"] != "complete" or cost["status"] != "complete":
            raise RuntimeError("Only accepted traditional costs may be inherited")
        verify_hashes(state["artifacts"])
        original_manifest = read(original / "run_manifest.json")
        original_test = read(original / "test_state.json")
        if (
            original_test["status"] != "complete"
            or original_test["identity"] != manifest["base_run_identity"]
        ):
            raise RuntimeError("Inherited cost original test identity differs")
        verify_hashes(original_test["artifacts"])
        expected_cost = identity(
            {
                "run": original_manifest["identity"],
                "model": original_manifest["model_sha256"],
                "cost_code": digest(Path(__file__).with_name("cost.py")),
                "repetitions": 3,
                "threads": read(revision / "revision_protocol.json")["threads"],
                "batch": "SCCD-session/other64",
                "test_seal_sha256": digest(original / "test_state.json"),
                "score_tolerance": SCORE_TOLERANCE,
            }
        )
        if not state["identity"] == cost["identity"] == expected_cost:
            raise RuntimeError("Inherited cost is not bound to the original model/test seal/code")
        if (
            max(
                cost["prediction_consistency"]["max_message_abs_diff"],
                cost["prediction_consistency"]["max_session_abs_diff"],
            )
            > SCORE_TOLERANCE
        ):
            raise RuntimeError("Inherited traditional cost failed score consistency")
        consistency = cost["prediction_consistency"]
        if (
            consistency["head_threshold_disagreements_all_repetitions"] != 0
            or consistency["max_message_abs_diff"] != 0
            or consistency["max_session_abs_diff"] != 0
        ):
            raise RuntimeError(
                "Traditional cost lacks exact-score/zero-class-disagreement proof; "
                "report before deciding whether to remeasure"
            )
        for name in ("predictions_message_test.jsonl", "predictions_session_test.jsonl"):
            if digest(original / name) != digest(folder / name):
                raise RuntimeError("Inherited traditional cost prediction reference differs")
        _copy(original / "cost.json", folder / "cost.json")
        provenance = {
            "kind": "inherited_valid_cost",
            "source_cost_sha256": digest(original / "cost.json"),
            "source_cost_state_sha256": digest(original / "cost_state.json"),
            "source_cost_identity": cost["identity"],
            "no_cost_remeasurement": True,
        }
        save(folder / "cost_provenance.json", provenance)
        _seal(
            revision,
            method,
            seed,
            "cost",
            [folder / "cost.json", folder / "cost_provenance.json"],
            inherited_cost_identity=cost["identity"],
        )
        return cost
    records, sessions, _ = load_prepared(base)
    test_records, test_sessions = (
        [r for r in records if r["split"] == "test"],
        [s for s in sessions if s["split"] == "test"],
    )
    bundles = inference_bundles(test_records, test_sessions)
    expected = read(revision / "revision_protocol.json")["test_counts"]
    if len(test_records) != expected["messages"] or len(test_sessions) != expected["sessions"]:
        raise RuntimeError("Raw cost corpus count differs from revision freeze")
    reference = {r["record_id"]: r for r in read_jsonl(folder / "predictions_message_test.jsonl")}
    reference_sessions = {
        r["session_id"]: r for r in read_jsonl(folder / "predictions_session_test.jsonl")
    }
    rows, totals, pass_counts = [], [], []
    max_message_diff = max_session_diff = 0.0
    threshold_disagreements = 0
    session_threshold_disagreements = {"main": 0, "common16": 0}
    with TreeMonitor() as monitor:
        package, startup = _load_package_and_warmup(base, revision, method, seed, records)
        started = time.perf_counter()
        package.predict(bundles[0][2], bundles[0][3], embeddings=None)
        first = time.perf_counter() - started
        for repetition in (1, 2, 3):
            total, seen_message, seen_session = 0.0, set(), set()
            for i, (source, identifier, group_records, group_sessions) in enumerate(bundles):
                started = time.perf_counter()
                prediction = package.predict(group_records, group_sessions, embeddings=None)
                elapsed = time.perf_counter() - started
                total += elapsed
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
                    rid = row["record_id"]
                    if rid in seen_message:
                        raise RuntimeError("Raw cost duplicated a record")
                    seen_message.add(rid)
                    actual = row["scores"]
                    actual = [actual[c] for c in CATEGORIES] if isinstance(actual, dict) else actual
                    target = reference[rid]["scores"]
                    target = [target[c] for c in CATEGORIES] if isinstance(target, dict) else target
                    max_message_diff = max(
                        max_message_diff, float(np.max(np.abs(np.asarray(actual) - target)))
                    )
                    threshold_disagreements += sum(
                        (a >= manifest["thresholds"][c]) != (b >= manifest["thresholds"][c])
                        for c, a, b in zip(CATEGORIES, actual, target, strict=True)
                    )
                for row in prediction["session"]:
                    sid = row["session_id"]
                    if sid in seen_session:
                        raise RuntimeError("Raw cost duplicated a conversation")
                    seen_session.add(sid)
                    target = reference_sessions[sid]
                    max_session_diff = max(
                        max_session_diff,
                        abs(row["score"] - target["score"]),
                        abs(row["common16_score"] - target["common16_score"]),
                    )
                    for kind, field, threshold in (
                        ("main", "score", "conversation"),
                        ("common16", "common16_score", "conversation_common16"),
                    ):
                        session_threshold_disagreements[kind] += (
                            row[field] >= manifest["thresholds"][threshold]
                        ) != (target[field] >= manifest["thresholds"][threshold])
                if (i + 1) % 20 == 0 or i == len(bundles) - 1:
                    save(
                        folder / "cost_progress.json",
                        {
                            "method": method,
                            "repetition": repetition,
                            "completed_bundles": i + 1,
                            "total_bundles": len(bundles),
                            "seconds": total,
                        },
                    )
                    print(
                        f"RAW COST {method} rep={repetition} {i + 1}/{len(bundles)} {total:.1f}s",
                        flush=True,
                    )
            if set(reference) != seen_message or set(reference_sessions) != seen_session:
                raise RuntimeError("Raw cost did not cover the complete test corpus")
            pass_counts.append(
                {
                    "repetition": repetition,
                    "messages": len(seen_message),
                    "sessions": len(seen_session),
                }
            )
            totals.append(total)
    dependencies, java = dependency_sizes(method)
    sccd = [r["seconds"] for r in rows if r["source"] == "sccd"]
    throughput = {
        source: sum(r["n_records"] for r in rows if r["source"] == source)
        / sum(r["seconds"] for r in rows if r["source"] == source)
        for source in sorted({r["source"] for r in rows if r["source"] != "sccd"})
    }
    cost_identity = identity(
        {
            "revision": frozen["identity"],
            "method": method,
            "seed": seed,
            "model_sha256": manifest["model_sha256"],
            "run_manifest_sha256": digest(folder / "run_manifest.json"),
            "test_state_sha256": digest(folder / "test_state.json"),
            "raw_replay_code_sha256": digest(Path(__file__)),
            "passes": 3,
            "score_tolerance": SCORE_TOLERANCE,
        }
    )
    result = {
        "identity": cost_identity,
        "status": "complete",
        "method": method,
        "seed": seed,
        "evaluation_revision_identity": frozen["identity"],
        "repetitions": 3,
        **{
            key: manifest[key]
            for key in (
                "training_seconds_cached",
                "training_seconds_cold",
                "shared_encoder_training_seconds",
                "cold_training_definition",
                "model_bytes",
                "encoder_asset_bytes",
            )
        },
        **startup,
        "first_inference_seconds": first,
        "model_load_seconds_definition": (
            "joblib model-package deserialization only; encoder loads during fixed TRAIN warm-up"
        ),
        "first_inference_seconds_definition": (
            "first test conversation after the fixed TRAIN warm-up; "
            "excludes startup warm-up and is not whole-test cold inference"
        ),
        "startup_warmup_definition": (
            "one raw prediction of fixed TRAIN CONDA first64 before all DEV/test/cost "
            "quality and timing passes; encoder initialization and first-call compilation included"
        ),
        "full_test_inference_seconds": float(np.mean(totals)),
        "warm_total_seconds": totals,
        "sccd_session_median_seconds": float(np.median(sccd)),
        "sccd_session_p95_seconds": float(np.quantile(sccd, 0.95)),
        "comment_throughput_per_second_by_source": throughput,
        "peak_tree_rss_bytes": max(monitor.peak, manifest["cold_training_peak_tree_rss_bytes"]),
        "inference_peak_tree_rss_bytes": monitor.peak,
        "training_peak_tree_rss_bytes": manifest["cold_training_peak_tree_rss_bytes"],
        "runtime_dependency_bytes": sum(row["installed_bytes"] for row in dependencies),
        "dependency_sizing_scope": (
            "listed installed distributions, not minimal or transitive-complete environment; "
            "Java counted once"
        ),
        "java_runtime_bytes": java,
        "deployment_dependencies": dependencies,
        "GPU_required": False,
        "paid_API_calls": 0,
        "resources": monitor.result(),
        "cache_policy": (
            "resident model/encoder after identical TRAIN warm-up; "
            "no text embedding or prediction cache; native temporary I/O included"
        ),
        "cold_start_definition": (
            "fresh Python process, OS filesystem cache not cleared; "
            "startup warm-up separately timed"
        ),
        "memory_definition": (
            "process-tree RSS including input corpus, encoder and children; "
            "not minimum serving memory"
        ),
        "prediction_consistency": {
            "max_message_abs_diff": max_message_diff,
            "max_session_abs_diff": max_session_diff,
            "head_threshold_disagreements_all_repetitions": int(threshold_disagreements),
            "session_threshold_disagreements_all_repetitions": session_threshold_disagreements,
            "thresholds_from_revision_manifest": True,
        },
        "pass_counts": pass_counts,
        "rows": rows,
        "test_state_sha256": digest(folder / "test_state.json"),
        "run_manifest_sha256": digest(folder / "run_manifest.json"),
    }
    verify_validation_freeze(revision)
    if monitor.max_affinity > read(revision / "revision_protocol.json")["threads"]:
        result["status"] = "invalid_cpu_budget"
    if not _consistent_cost(result["prediction_consistency"]):
        result["status"] = "invalid_prediction_consistency"
    if result["status"] != "complete":
        save(folder / "cost_invalid.json", result)
        raise RuntimeError(
            "Raw-replay cost failed the frozen acceptance rules; invalid evidence retained"
        )
    save(folder / "cost.json", result)
    save(
        folder / "cost_provenance.json",
        {
            "kind": "semantic_raw_remeasured_cost",
            "cost_identity": cost_identity,
            "base_model_sha256": manifest["model_sha256"],
            "revision_test_state_sha256": digest(folder / "test_state.json"),
            "same_training_warmup_as_dev_and_test": True,
            "threshold_override_only": True,
        },
    )
    _seal(
        revision,
        method,
        seed,
        "cost",
        [folder / "cost.json", folder / "cost_provenance.json"],
        cost_identity=cost_identity,
    )
    return result


def _launch(base: Path, stage: str, method: str, seed: int, threads: int) -> None:
    import psutil

    revision = revision_root(base)
    logs = revision / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "cst_mil.research_followup.full_joint_comparison.raw_replay_v2",
        "--output",
        str(base),
        "--threads",
        str(threads),
        "--stage",
        stage,
        "--method",
        method,
        "--seed",
        str(seed),
        "--child",
    ]
    env = dict(
        os.environ,
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        NUMBA_NUM_THREADS=str(threads),
    )
    log = logs / f"{stage}_{method}_{seed}.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\nSTART {utc_now()}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        save(
            revision / "active_child.json",
            {
                "pid": process.pid,
                "created": psutil.Process(process.pid).create_time(),
                "stage": stage,
                "method": method,
                "seed": seed,
            },
        )
        returncode = process.wait()
    active = revision / "active_child.json"
    if active.exists() and read(active)["pid"] == process.pid:
        active.unlink()
    if returncode:
        save(
            revision / "failure.json",
            {
                "stage": stage,
                "method": method,
                "seed": seed,
                "returncode": returncode,
                "log": str(log),
                "time": utc_now(),
                "no_fallback_applied": True,
            },
        )
        raise RuntimeError(f"Raw revision child failed: {log}")


def _lineage_handshake_proof(
    pid: int, receipt: dict, owner: dict, *, stage: str, method: str, seed: int
) -> dict | None:
    """Verify a real child/redirector/owner chain, including PID reuse protection."""
    import psutil

    current = psutil.Process(pid)
    lineage = [current, *current.parents()]
    by_pid = {process.pid: (index, process) for index, process in enumerate(lineage)}
    launcher_pid, owner_pid = receipt.get("pid"), owner.get("pid")
    if launcher_pid not in by_pid:
        return None  # The parent may not yet have replaced an old dead receipt.
    if owner_pid not in by_pid or owner_pid == pid:
        raise RuntimeError("Campaign lock owner is not a live parent ancestor of this child")
    launcher_index, launcher = by_pid[launcher_pid]
    owner_index, parent = by_pid[owner_pid]
    if owner_index <= launcher_index:
        raise RuntimeError("Recorded launcher is not a descendant of the lock owner")
    for process, recorded, role in ((launcher, receipt, "launcher"), (parent, owner, "owner")):
        if (
            not process.is_running()
            or abs(process.create_time() - float(recorded["created"])) >= 0.1
        ):
            raise RuntimeError(f"Recorded {role} PID/create-time no longer identifies that process")
    if (receipt.get("stage"), receipt.get("method"), receipt.get("seed")) != (stage, method, seed):
        raise RuntimeError("Active child stage/method/seed do not match the requested stage")
    return {
        "python_pid": pid,
        "launcher_pid": launcher_pid,
        "parent_owner_pid": owner_pid,
        "pid_match_kind": "self" if launcher_pid == pid else "ancestor_redirector",
        "launcher_create_time_verified": True,
        "parent_lock_create_time_verified": True,
        "owner_is_live_ancestor": True,
        "stage": stage,
        "method": method,
        "seed": seed,
    }


def _await_child_receipt(
    revision: Path, pid: int, timeout: float = 5.0, *, stage: str, method: str, seed: int
) -> dict:
    """Wait only for receipt publication, not to hide a launcher PID mismatch."""
    deadline = time.monotonic() + timeout
    while True:
        lock_path = revision / "campaign.lock"
        if not lock_path.exists():
            raise RuntimeError("Internal child has no owning parent campaign lock")
        receipt = revision / "active_child.json"
        if receipt.exists():
            proof = _lineage_handshake_proof(
                pid, read(receipt), read(lock_path), stage=stage, method=method, seed=seed
            )
            if proof is not None:
                return proof
        if time.monotonic() >= deadline:
            raise RuntimeError("Parent did not publish the matching child startup receipt")
        time.sleep(0.02)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Original full-joint campaign, not revision directory",
    )
    parser.add_argument(
        "--stage", choices=("init", "dev", "freeze", "test", "cost", "all"), required=True
    )
    parser.add_argument("--threads", choices=range(1, 5), type=int, default=4)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", choices=SEEDS, type=int)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    from cst_mil.bench.resources import configure_cpu

    configure_cpu(args.threads)
    base, revision = args.output.resolve(), revision_root(args.output)
    if args.child:
        if args.method is None or args.seed is None or args.stage not in {"dev", "test", "cost"}:
            raise ValueError("Internal child stage requires one method and seed")
        if read(revision / "revision_protocol.json")["threads"] != args.threads:
            raise ValueError("Child CPU threads differ from frozen revision")
        handshake = _await_child_receipt(
            revision, os.getpid(), stage=args.stage, method=args.method, seed=args.seed
        )
        print(f"RAW child startup handshake: {handshake}", flush=True)
        functions = {"dev": replay_dev, "test": replay_test, "cost": measure_cost}
        functions[args.stage](revision, args.method, args.seed)
        return 0
    methods = [args.method] if args.method else list(METHODS)
    seeds = [args.seed] if args.seed else list(SEEDS)
    revision.mkdir(parents=True, exist_ok=True)
    with campaign_lock(revision):
        init_revision(base, args.threads)
        if args.stage == "init":
            print(str(revision))
            return 0
        for stage in ("dev", "freeze", "test", "cost") if args.stage == "all" else (args.stage,):
            if stage == "freeze":
                freeze_validation(revision)
                continue
            for method in methods:
                for seed in [42] if stage == "cost" else seeds:
                    _launch(base, stage, method, seed, args.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
