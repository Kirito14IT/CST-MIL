"""Sequential, resumable bounded fitting and validation-only architecture choice."""

from __future__ import annotations

import gc
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from cst_mil.research_followup.full_joint_comparison.model import (
    evaluate_predictions,
    select_threshold_fast,
)

from .common import (
    ABLATIONS,
    BASE,
    CATEGORIES,
    MAIN_TASKS,
    OUTPUT,
    REVISION,
    SEEDS,
    TreeMonitor,
    campaign_lock,
    digest,
    freeze_execution,
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
from .model import FollowupPackage, VariantConfig, fit_variant
from .representations import embeddings_for


def folder_for(output: Path, variant: str, seed: int) -> Path:
    return output / "runs" / variant / f"seed_{seed}"


def binding(output: Path, config: VariantConfig, seed: int) -> dict:
    return {
        "execution_identity": read(output / "execution_freeze.json")["identity"],
        "config": asdict(config),
        "seed": seed,
        "dev_bank_sha256": digest(output / "raw_embeddings/dev/complete.json"),
    }


def _save_model(path: Path, package) -> None:
    temporary = path.with_suffix(".pending.joblib")
    joblib.dump(package, temporary, compress=3)
    temporary.replace(path)


def _thresholds(package, payload) -> None:
    for j, category in enumerate(CATEGORIES):
        rows = [r for r in payload["message"] if r["observed_mask"][j]]
        package.thresholds[category] = select_threshold_fast(
            [r["labels"][j] for r in rows], [r["scores"][j] for r in rows]
        )[0]
    for attr, field in (
        ("conversation_threshold", "score"),
        ("common16_threshold", "common16_score"),
    ):
        rows = payload["session"]
        setattr(
            package,
            attr,
            select_threshold_fast([r["label"] for r in rows], [r[field] for r in rows])[0],
        )
    for row in payload["session"]:
        row.update(
            threshold=package.conversation_threshold, common16_threshold=package.common16_threshold
        )


def _import_full(output: Path, seed: int, records, sessions, embeddings):
    old = joblib.load(BASE / "runs/cst_v1_1" / f"seed_{seed}/model.joblib")
    package = FollowupPackage(VariantConfig(), seed)
    for key in (
        "heads",
        "thresholds",
        "conversation_classifier",
        "common16_classifier",
        "conversation_threshold",
        "common16_threshold",
        "pca",
        "char_vectorizer",
        "encoder_metadata",
        "threads",
        "training_summary",
    ):
        setattr(package, key, getattr(old, key))
    dev = [i for i, r in enumerate(records) if r["split"] == "dev"]
    payload = package.predict_payload(
        [records[i] for i in dev], [s for s in sessions if s["split"] == "dev"], embeddings[dev]
    )
    _thresholds(package, payload)
    sealed = REVISION / "runs/cst_v1_1" / f"seed_{seed}"
    previous = {r["record_id"]: r for r in read_jsonl(sealed / "predictions_message_dev.jsonl")}
    # Audited raw replay stores named scores, whereas the training API returns lists.
    for row in previous.values():
        if isinstance(row["scores"], dict):
            row["scores"] = [row["scores"][category] for category in CATEGORIES]
    score_diff = max(
        abs(np.asarray(r["scores"]) - previous[r["record_id"]]["scores"]).max()
        for r in payload["message"]
    )
    old_sessions = {
        r["session_id"]: r for r in read_jsonl(sealed / "predictions_session_dev.jsonl")
    }
    session_diff = max(
        abs(r["score"] - old_sessions[r["session_id"]]["score"]) for r in payload["session"]
    )
    old_thresholds = read(sealed / "run_manifest.json")["thresholds"]
    flips = sum(
        (r["scores"][j] >= package.thresholds[c])
        != (previous[r["record_id"]]["scores"][j] >= old_thresholds[c])
        for r in payload["message"]
        for j, c in enumerate(CATEGORIES)
    )
    agreement = {
        "message_max_absolute_difference": float(score_diff),
        "conversation_max_absolute_difference": float(session_diff),
        "message_decision_flips": flips,
        "tolerance": 1e-4,
        "reference": str(sealed),
        "thresholds_selected_on_this_DEV_bank": True,
    }
    if max(score_diff, session_diff) > 1e-4 or flips:
        save(output / "qa" / f"base_replay_seed_{seed}_failure.json", agreement)
        raise RuntimeError(
            "Frozen full model raw DEV agreement failed; do not silently relax tolerance"
        )
    original_details = read(BASE / "runs/cst_v1_1" / f"seed_{seed}/fit_details.json")
    details = {
        "metrics": evaluate_predictions(payload, package.thresholds),
        "dev_predictions": payload,
        "fit_audit": {"inherited": True, "base_replay": agreement},
        "stage_times": {},
        "parameters": {"config": asdict(package.config), "seed": seed},
        "inherited_original_fit_details_sha256": digest(
            BASE / "runs/cst_v1_1" / f"seed_{seed}/fit_details.json"
        ),
        "inherited_original_stage_times": original_details.get("stage_times", {}),
    }
    return package, details, []


def fit_one(
    output: Path, variant: str, seed: int, architecture_base: str, records, sessions, embeddings
) -> dict:
    guard(output)
    config = VariantConfig(architecture_base, variant if variant in ABLATIONS else None)
    folder = folder_for(output, variant, seed)
    expected = binding(output, config, seed)
    complete = folder / "complete.json"
    if complete.exists():
        state = read(complete)
        if state["binding"] != expected:
            raise RuntimeError("Resume configuration/source identity mismatch")
        verify_hashes(state["artifacts"])
        print(f"RESUME {variant} {seed}", flush=True)
        return state
    folder.mkdir(parents=True, exist_ok=True)
    previous = folder / "binding.json"
    if previous.exists() and read(previous) != expected:
        raise RuntimeError("Partial run binding differs; preserve it")
    save(previous, expected)
    print(f"FIT {variant} seed={seed} base={architecture_base}", flush=True)
    started = time.perf_counter()
    try:
        with TreeMonitor() as monitor:
            if variant == "full":
                package, details, oof = _import_full(output, seed, records, sessions, embeddings)
            else:
                package, details, oof = fit_variant(
                    records,
                    embeddings,
                    sessions,
                    config=config,
                    seed=seed,
                    encoder_metadata=read(output / "protocol.json")["encoder"],
                    threads=4,
                )
        guard(output)
        payload = details.pop("dev_predictions")
        _save_model(folder / "model.joblib", package)
        for level, rows in payload.items():
            write_jsonl(folder / f"predictions_{level}_dev.jsonl", rows)
        write_jsonl(folder / "oof_message_train.jsonl", oof)
        save(folder / "fit_details.json", details)
        thresholds = {
            **package.thresholds,
            "conversation": package.conversation_threshold,
            "conversation_common16": package.common16_threshold,
        }
        manifest = {
            **expected,
            "variant": variant,
            "config": asdict(config),
            "seed": seed,
            "thresholds": thresholds,
            "conversation_threshold": package.conversation_threshold,
            "common16_threshold": package.common16_threshold,
            "training_seconds_cached": details.get("stage_times", {}).get("total_fit_seconds"),
            "wall_stage_seconds": time.perf_counter() - started,
            "resources": monitor.result(),
            "model_bytes": (folder / "model.joblib").stat().st_size,
            "inherited_training": variant == "full",
            "TEST_used_for_selection": False,
        }
        save(folder / "manifest.json", manifest)
        artifacts = source_hashes(
            folder / name
            for name in (
                "model.joblib",
                "fit_details.json",
                "oof_message_train.jsonl",
                "manifest.json",
                "predictions_message_dev.jsonl",
                "predictions_session_dev.jsonl",
            )
        )
        state = {"status": "complete", "binding": expected, "artifacts": artifacts}
        save(complete, state)
        print(f"DONE {variant} {seed} seconds={manifest['wall_stage_seconds']:.1f}", flush=True)
        return state
    except BaseException:
        attempt = len(list(folder.glob("failure_*.json"))) + 1
        save(
            folder / f"failure_{attempt:03d}.json",
            {
                "binding": expected,
                "seconds": time.perf_counter() - started,
                "error": traceback.format_exc(),
            },
        )
        raise
    finally:
        gc.collect()


def _task_values(folder: Path) -> dict:
    metrics = read(folder / "fit_details.json")["metrics"]
    result = {}
    for task in MAIN_TASKS:
        source, category = task.split(".")
        if category == "conversation":
            result[task] = metrics["sessions"]["dev"]["main"]
        else:
            result[task] = metrics["messages"]["dev"][source][category]
    return result


def choose_candidate(values: dict[str, dict]) -> dict:
    baseline = values["full"]
    checks = {}
    for variant in ("context_a", "context_b"):
        current = values[variant]
        differences = {task: current[task]["AP"] - baseline[task]["AP"] for task in MAIN_TASKS}
        conditions = {
            "primary_gain": differences["sccd.attack"] >= 0.01 - 1e-12,
            "other_message_AP": all(
                differences[t] >= -0.01 - 1e-12
                for t in MAIN_TASKS
                if t not in {"sccd.attack", "sccd.conversation"}
            ),
            "session_AP": differences["sccd.conversation"] >= -0.01 - 1e-12,
            "session_AUC": current["sccd.conversation"]["roc_auc"]
            - baseline["sccd.conversation"]["roc_auc"]
            >= -0.01 - 1e-12,
            "comment_macro_F1": current["sccd.attack"]["macro_f1"]
            - baseline["sccd.attack"]["macro_f1"]
            >= -0.01 - 1e-12,
        }
        checks[variant] = {
            "eligible": all(conditions.values()),
            "conditions": conditions,
            "AP_differences": differences,
            "comment_macro_F1_difference": current["sccd.attack"]["macro_f1"]
            - baseline["sccd.attack"]["macro_f1"],
            "session_AUC_difference": current["sccd.conversation"]["roc_auc"]
            - baseline["sccd.conversation"]["roc_auc"],
        }
    eligible = [v for v in ("context_a", "context_b") if checks[v]["eligible"]]
    selected = "full"
    for candidate in eligible:
        if (
            selected == "full"
            or values[candidate]["sccd.attack"]["AP"]
            > values[selected]["sccd.attack"]["AP"] + 1e-12
        ):
            selected = candidate
    return {
        "selected_architecture": selected,
        "candidate_checks": checks,
        "mean_DEV_metrics": values,
        "TEST_used": False,
        "selection_rule": (
            "preregistered three-seed means and all guardrails; ties prefer A; fallback full"
        ),
    }


def candidates(output: Path = OUTPUT) -> None:
    freeze_execution(output)
    with campaign_lock(output):
        records, sessions = load_data(("train", "dev"))
        embeddings = embeddings_for(records, output)
        for variant in ("full", "context_a", "context_b"):
            for seed in SEEDS:
                fit_one(output, variant, seed, variant, records, sessions, embeddings)


def select(output: Path = OUTPUT) -> dict:
    guard(output)
    values, files = {}, []
    for variant in ("full", "context_a", "context_b"):
        seed_values = []
        for seed in SEEDS:
            folder = folder_for(output, variant, seed)
            verify_hashes(read(folder / "complete.json")["artifacts"])
            seed_values.append(_task_values(folder))
            files.append(folder / "complete.json")
        values[variant] = {
            task: {
                m: float(np.mean([r[task][m] for r in seed_values]))
                for m in ("AP", "roc_auc", "macro_f1", "precision", "recall")
            }
            for task in MAIN_TASKS
        }
    result = {
        **choose_candidate(values),
        "execution_identity": read(output / "execution_freeze.json")["identity"],
        "input_hashes": source_hashes(files),
    }
    result["identity"] = identity(result)
    target = output / "selection_freeze.json"
    if target.exists() and read(target) != result:
        raise RuntimeError("Selection already frozen with different evidence")
    save(target, result)
    print(f"SELECTED {result['selected_architecture']}", flush=True)
    return result


def selected(output: Path = OUTPUT) -> dict:
    value = read(output / "selection_freeze.json")
    verify_hashes(value["input_hashes"])
    if identity({k: v for k, v in value.items() if k != "identity"}) != value["identity"]:
        raise RuntimeError("Selection fingerprint mismatch")
    return value


def ablations(output: Path = OUTPUT) -> None:
    architecture = selected(output)["selected_architecture"]
    with campaign_lock(output):
        records, sessions = load_data(("train", "dev"))
        embeddings = embeddings_for(records, output)
        for variant in ABLATIONS:
            for seed in SEEDS:
                fit_one(output, variant, seed, architecture, records, sessions, embeddings)


def evaluate_test(output: Path = OUTPUT) -> None:
    architecture = selected(output)["selected_architecture"]
    guard(output)
    records, sessions = load_data(("test",))
    embeddings = embeddings_for(records, output)
    for variant in dict.fromkeys(("full", architecture, *ABLATIONS)):
        for seed in SEEDS:
            folder = folder_for(output, variant, seed)
            state = read(folder / "complete.json")
            verify_hashes(state["artifacts"])
            target = folder / "test_complete.json"
            expected = {
                "selection_identity": selected(output)["identity"],
                "model_sha256": digest(folder / "model.joblib"),
                "test_bank_sha256": digest(output / "raw_embeddings/test/complete.json"),
            }
            if target.exists():
                old = read(target)
                if old["binding"] != expected:
                    raise RuntimeError("TEST resume binding mismatch")
                verify_hashes(old["artifacts"])
                continue
            print(f"TEST {variant} {seed}", flush=True)
            package = joblib.load(folder / "model.joblib")
            payload = package.predict_payload(records, sessions, embeddings)
            for level, rows in payload.items():
                write_jsonl(folder / f"predictions_{level}_test.jsonl", rows)
            save(folder / "test_metrics.json", evaluate_predictions(payload, package.thresholds))
            guard(output)
            save(
                target,
                {
                    "status": "complete",
                    "binding": expected,
                    "artifacts": source_hashes(
                        folder / name
                        for name in (
                            "predictions_message_test.jsonl",
                            "predictions_session_test.jsonl",
                            "test_metrics.json",
                        )
                    ),
                },
            )
            del package
            gc.collect()
