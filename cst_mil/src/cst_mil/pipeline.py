"""End-to-end command orchestration for the single formal CST-MIL pilot."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import yaml

from cst_mil.analysis import (
    compute_binary_metrics,
    create_analysis_bundle,
    stratified_bootstrap,
)
from cst_mil.data import download_sccd, prepare_dataset
from cst_mil.metrics import evidence_localisation_metrics
from cst_mil.model import (
    load_model,
    predict_sessions,
    save_model,
    serialise_prediction_value,
    train_cst_mil,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
    return pd.DataFrame.from_records(records)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Pilot config must be a YAML mapping")
    return payload


def _validate_config_contract(config: dict[str, Any]) -> None:
    required = {"seed", "n_sessions", "positive_sessions", "negative_sessions", "split"}
    if missing := required.difference(config):
        raise ValueError(f"Pilot config is missing declared experiment fields: {sorted(missing)}")
    n_sessions = int(config["n_sessions"])
    positive = int(config["positive_sessions"])
    negative = int(config["negative_sessions"])
    if positive != negative or positive + negative != n_sessions:
        raise ValueError("Pilot class counts must be balanced and sum to n_sessions")
    per_class = n_sessions // 2
    expected_split = {
        "train": 2 * int(round(per_class * 0.60)),
        "dev": 2 * int(round(per_class * 0.20)),
    }
    expected_split["test"] = n_sessions - expected_split["train"] - expected_split["dev"]
    declared_split = {str(key): int(value) for key, value in dict(config["split"]).items()}
    if declared_split != expected_split:
        raise ValueError(
            f"Declared split {declared_split} does not match data pipeline split {expected_split}"
        )
    threshold_metric = str(config.get("evaluation", {}).get("threshold_metric", ""))
    if threshold_metric != "macro_f1":
        raise ValueError("The implemented dev-threshold metric is macro_f1")


def _assert_prepared_matches_config(
    prepared_dir: Path,
    sessions: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    cleaning = _read_json(prepared_dir / "cleaning_report.json")
    selection = cleaning.get("selection", {})
    expected_sessions = int(config["n_sessions"])
    if int(selection.get("seed", -1)) != int(config["seed"]):
        raise ValueError("Prepared seed does not match the pilot config")
    if int(selection.get("requested_sessions", -1)) != expected_sessions:
        raise ValueError("Prepared session count does not match the pilot config")
    if len(sessions) != expected_sessions:
        raise ValueError("Prepared sessions.jsonl row count does not match the pilot config")

    expected_split = {str(key): int(value) for key, value in dict(config["split"]).items()}
    actual_split = {str(key): int(value) for key, value in sessions["split"].value_counts().items()}
    if actual_split != expected_split:
        raise ValueError(f"Prepared split counts {actual_split} do not match {expected_split}")
    expected_labels = {
        0: int(config["negative_sessions"]),
        1: int(config["positive_sessions"]),
    }
    actual_labels = {
        int(key): int(value) for key, value in sessions["label"].value_counts().items()
    }
    if actual_labels != expected_labels:
        raise ValueError(f"Prepared label counts {actual_labels} do not match {expected_labels}")
    for split, total in expected_split.items():
        counts = sessions.loc[sessions["split"] == split, "label"].value_counts().to_dict()
        if counts != {0: total // 2, 1: total // 2}:
            raise ValueError(f"Prepared {split} split is not class-balanced: {counts}")
    if cleaning.get("source_commit") != "cf4015b802cabd651885211dc382152c1a270c31":
        raise ValueError("Prepared cleaning report is not tied to the approved SCCD commit")
    if cleaning.get("source_manifest_verified") is not True:
        raise ValueError("Prepared cleaning report does not confirm raw manifest verification")


def _package_versions() -> dict[str, str]:
    packages = [
        "cst-mil",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "jieba",
        "matplotlib",
        "psutil",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _environment_payload() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "captured_at_utc": _utc_now(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": int(memory.total),
        "packages": _package_versions(),
        "uses_gpu": False,
        "uses_llm": False,
        "uses_pretrained_language_model": False,
    }


@dataclass
class PeakRSSMonitor:
    """Sample the current process RSS while a training or inference call runs."""

    interval_seconds: float = 0.02

    def __post_init__(self) -> None:
        self.peak_bytes = int(psutil.Process().memory_info().rss)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        process = psutil.Process()
        while not self._stop.wait(self.interval_seconds):
            self.peak_bytes = max(self.peak_bytes, int(process.memory_info().rss))

    def __enter__(self) -> PeakRSSMonitor:
        self._thread = threading.Thread(target=self._sample, name="peak-rss-monitor", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.peak_bytes = max(self.peak_bytes, int(psutil.Process().memory_info().rss))


def _validate_prepared(prepared_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = _read_jsonl(prepared_dir / "sessions.jsonl")
    messages = _read_jsonl(prepared_dir / "messages.jsonl")
    required_session = {"post_id", "label", "split", "model_text"}
    required_message = {"post_id", "comment_id", "split", "model_text", "context_model_text"}
    if missing := required_session.difference(sessions.columns):
        raise ValueError(f"Prepared sessions missing columns: {sorted(missing)}")
    if missing := required_message.difference(messages.columns):
        raise ValueError(f"Prepared messages missing columns: {sorted(missing)}")
    overlaps: set[str] = set()
    split_ids = {
        split: set(sessions.loc[sessions["split"] == split, "post_id"].astype(str))
        for split in ("train", "dev", "test")
    }
    overlaps.update(split_ids["train"] & split_ids["dev"])
    overlaps.update(split_ids["train"] & split_ids["test"])
    overlaps.update(split_ids["dev"] & split_ids["test"])
    if overlaps:
        raise ValueError(f"Session leakage across splits: {sorted(overlaps)[:3]}")
    if set(messages["post_id"].astype(str)).difference(set(sessions["post_id"].astype(str))):
        raise ValueError("Prepared messages reference sessions outside the split manifest")
    return sessions, messages


def download_data_command(data_dir: Path) -> dict[str, Any]:
    manifest = download_sccd(data_dir)
    print(f"Downloaded/verified SCCD at {data_dir}")
    return manifest


def prepare_command(raw_dir: Path, output: Path, seed: int, sessions: int) -> dict[str, Any]:
    report = prepare_dataset(raw_dir, output, seed=seed, n_sessions=sessions)
    print(f"Prepared {sessions} SCCD sessions at {output}")
    return report


def _prepared_manifest(
    prepared_dir: Path,
    sessions: pd.DataFrame,
    messages: pd.DataFrame,
) -> dict[str, Any]:
    filenames = ("sessions.jsonl", "messages.jsonl", "splits.csv", "cleaning_report.json")
    hashes: dict[str, str] = {}
    for filename in filenames:
        path = prepared_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required prepared artifact is missing: {path}")
        hashes[filename] = _sha256_file(path)
    return {
        "prepared_dir_at_training": str(prepared_dir.resolve()),
        "files_sha256": hashes,
        "session_rows": int(len(sessions)),
        "message_rows": int(len(messages)),
        "split_counts": {
            str(key): int(value) for key, value in sessions["split"].value_counts().items()
        },
    }


def _assert_prepared_matches_manifest(prepared_dir: Path, manifest: dict[str, Any]) -> None:
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Run data manifest does not contain prepared-file hashes")
    mismatches: list[str] = []
    for filename, expected_hash in expected.items():
        path = prepared_dir / str(filename)
        actual_hash = _sha256_file(path) if path.is_file() else "missing"
        if actual_hash != expected_hash:
            mismatches.append(str(filename))
    if mismatches:
        raise ValueError(
            "Prepared data provenance mismatch; evaluation refused for: "
            + ", ".join(sorted(mismatches))
        )


def _source_manifest(config_path: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[2]
    project_files = [project_root / "pyproject.toml", project_root / "uv.lock"]
    project_files.extend(sorted((project_root / "src" / "cst_mil").glob("*.py")))
    hashes: dict[str, str] = {}
    for path in project_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required source artifact is missing: {path}")
        hashes[path.relative_to(project_root).as_posix()] = _sha256_file(path)
    return {
        "project_root_at_training": str(project_root),
        "project_files_sha256": hashes,
        "config_source": str(config_path.resolve()),
        "config_source_sha256": _sha256_file(config_path),
    }


def _assert_source_matches_manifest(manifest: dict[str, Any]) -> None:
    project_root = Path(__file__).resolve().parents[2]
    expected = manifest.get("project_files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Run manifest does not contain source-file hashes")
    mismatches: list[str] = []
    for relative, expected_hash in expected.items():
        path = project_root / str(relative)
        actual_hash = _sha256_file(path) if path.is_file() else "missing"
        if actual_hash != expected_hash:
            mismatches.append(str(relative))
    if mismatches:
        raise ValueError(
            "Source provenance mismatch; evaluation refused for: "
            + ", ".join(sorted(mismatches))
        )
    config_source = Path(str(manifest.get("config_source", "")))
    expected_config_hash = manifest.get("config_source_sha256")
    actual_config_hash = _sha256_file(config_source) if config_source.is_file() else "missing"
    if actual_config_hash != expected_config_hash:
        raise ValueError("Config provenance mismatch; evaluation refused")


def _start_run(run_dir: Path) -> tuple[int, Path]:
    """Atomically reserve a formal run so concurrent training cannot start."""

    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.with_name(run_dir.name + ".train.lock")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    try:
        lock_fd = os.open(lock_path, flags)
    except FileExistsError as error:
        raise FileExistsError(
            f"Another training process owns the formal run lock: {lock_path}"
        ) from error
    try:
        os.write(lock_fd, f"pid={os.getpid()} started_at_utc={_utc_now()}\n".encode())
        completed = run_dir / "COMPLETED.json"
        if completed.exists():
            raise FileExistsError(
                f"Formal run is already complete and will not be overwritten: {run_dir}"
            )
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory is non-empty and cannot be reused for another training: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "RUNNING.json",
            {"status": "training", "started_at_utc": _utc_now()},
        )
    except Exception:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        raise
    return lock_fd, lock_path


def _release_run_lock(lock: tuple[int, Path]) -> None:
    lock_fd, lock_path = lock
    os.close(lock_fd)
    lock_path.unlink(missing_ok=True)


def _assert_run_available(run_dir: Path) -> None:
    lock_path = run_dir.with_name(run_dir.name + ".train.lock")
    if lock_path.exists():
        raise FileExistsError(f"Formal run lock already exists: {lock_path}")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Formal run directory is not empty: {run_dir}")


def _write_artifact_manifest(run_dir: Path) -> str:
    excluded = {"artifact_manifest.json", "COMPLETED.json", "RUNNING.json"}
    files = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and path.name not in excluded
    ]
    payload = {
        "created_at_utc": _utc_now(),
        "files_sha256": {
            path.relative_to(run_dir).as_posix(): _sha256_file(path)
            for path in sorted(files)
        },
    }
    manifest_path = run_dir / "artifact_manifest.json"
    _write_json(manifest_path, payload)
    return _sha256_file(manifest_path)


def train_command(
    config_path: Path,
    prepared_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Fit the only formal model; evaluation remains a separate deterministic step."""

    config = _load_config(config_path)
    _validate_config_contract(config)
    sessions, messages = _validate_prepared(prepared_dir)
    _assert_prepared_matches_config(prepared_dir, sessions, config)
    data_manifest = _prepared_manifest(prepared_dir, sessions, messages)
    source_manifest = _source_manifest(config_path)
    run_lock = _start_run(run_dir)
    _write_json(run_dir / "data_manifest.json", data_manifest)
    environment = _environment_payload()
    start = time.perf_counter()
    with PeakRSSMonitor() as monitor:
        model = train_cst_mil(sessions, messages, config)
    training_seconds = time.perf_counter() - start

    model_path = save_model(model, run_dir / "model.joblib")
    topics = [
        {"topic_id": topic_id, "keywords": keywords}
        for topic_id, keywords in enumerate(model.topic_words)
    ]
    performance = {
        "training_wall_seconds": float(training_seconds),
        "training_peak_rss_bytes": int(monitor.peak_bytes),
        "model_bytes": int(model_path.stat().st_size),
    }
    summary = {
        "training": model.training_summary,
        "performance": performance,
        "run_kind": "single_fixed_pilot",
        "formal_training_runs": 1,
    }
    _write_json(run_dir / "config.json", model.config)
    _write_json(run_dir / "environment.json", environment)
    _write_json(run_dir / "training_summary.json", summary)
    _write_json(run_dir / "topics.json", topics)
    run_manifest = {
        "created_at_utc": _utc_now(),
        "command": [sys.executable, *sys.argv],
        "formal_training_runs": 1,
        "model_sha256": _sha256_file(model_path),
        **source_manifest,
    }
    _write_json(run_dir / "run_manifest.json", run_manifest)
    for filename in ("splits.csv", "cleaning_report.json"):
        source = prepared_dir / filename
        if source.exists():
            shutil.copy2(source, run_dir / filename)
    _write_json(
        run_dir / "RUNNING.json",
        {"status": "trained", "trained_at_utc": _utc_now(), **performance},
    )
    _release_run_lock(run_lock)
    print(f"Trained CST-MIL once and saved model to {model_path}")
    return summary


def evaluate_command(run_dir: Path, prepared_dir: Path) -> dict[str, Any]:
    """Evaluate the fitted formal model without retraining it."""

    if (run_dir / "COMPLETED.json").exists():
        raise FileExistsError(f"Evaluation artifacts are already complete: {run_dir}")
    model_path = run_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"No fitted model found at {model_path}")
    sessions, messages = _validate_prepared(prepared_dir)
    data_manifest = _read_json(run_dir / "data_manifest.json")
    _assert_prepared_matches_manifest(prepared_dir, data_manifest)
    run_manifest = _read_json(run_dir / "run_manifest.json")
    if _sha256_file(model_path) != run_manifest.get("model_sha256"):
        raise ValueError("Model provenance mismatch; evaluation refused")
    _assert_source_matches_manifest(run_manifest)
    test_sessions = sessions.loc[sessions["split"] == "test"].reset_index(drop=True)
    test_ids = set(test_sessions["post_id"].astype(str))
    test_messages = messages.loc[
        messages["post_id"].astype(str).isin(test_ids)
    ].reset_index(drop=True)
    model = load_model(model_path)

    start = time.perf_counter()
    with PeakRSSMonitor() as monitor:
        session_predictions, message_scores = predict_sessions(
            model,
            test_sessions,
            test_messages,
            evidence_top_k=int(model.config["evaluation"]["evidence_top_k"]),
            return_message_scores=True,
        )
    inference_seconds = time.perf_counter() - start
    session_predictions["label"] = session_predictions["label"].astype(int)
    result = compute_binary_metrics(session_predictions)
    bootstrap_samples = int(model.config.get("evaluation", {}).get("bootstrap_samples", 1000))
    bootstrap = stratified_bootstrap(
        session_predictions,
        n_samples=bootstrap_samples,
        seed=int(model.config["seed"]),
    )

    label_columns = ["post_id", "comment_id", "comment_label"]
    message_labels = test_messages[label_columns].dropna(subset=["comment_label"]).copy()
    evidence_metrics = evidence_localisation_metrics(
        message_scores,
        message_labels,
        top_k=int(model.config["evaluation"]["evidence_top_k"]),
    )
    evidence = message_scores.loc[
        message_scores["rank"] <= int(model.config["evaluation"]["evidence_top_k"])
    ].merge(message_labels, on=["post_id", "comment_id"], how="left", validate="one_to_one")
    evidence["comment_label_name"] = evidence["comment_label"].map({0: "Non-CB", 1: "CB"})

    raw_records = session_predictions.to_dict(orient="records")
    _write_jsonl(run_dir / "session_predictions.jsonl", raw_records)
    csv_predictions = session_predictions.copy()
    for column in ("top_topics", "evidence"):
        csv_predictions[column] = csv_predictions[column].map(serialise_prediction_value)
    csv_predictions.to_csv(run_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    evidence.to_csv(run_dir / "evidence_top3.csv", index=False, encoding="utf-8-sig")
    message_scores.to_csv(run_dir / "message_scores.csv", index=False, encoding="utf-8-sig")

    errors = session_predictions.loc[
        session_predictions["label"] != session_predictions["prediction"]
    ].copy()
    for column in ("top_topics", "evidence"):
        errors[column] = errors[column].map(serialise_prediction_value)
    errors.to_csv(run_dir / "error_cases.csv", index=False, encoding="utf-8-sig")

    training_summary = _read_json(run_dir / "training_summary.json")
    efficiency = {
        **training_summary["performance"],
        "inference_wall_seconds": float(inference_seconds),
        "inference_peak_rss_bytes": int(monitor.peak_bytes),
        "inference_ms_per_session": float(inference_seconds * 1000 / len(test_sessions)),
        "inference_sessions_per_second": float(len(test_sessions) / inference_seconds),
    }
    payload: dict[str, Any] = {
        "session_metrics": result.metrics,
        "bootstrap_conditional_ci": bootstrap,
        "evidence_metrics": evidence_metrics,
        "efficiency": efficiency,
        "threshold": float(model.threshold),
        "formal_training_runs": 1,
        "test_unit": "complete post_id session",
        "evaluated_at_utc": _utc_now(),
    }
    _write_json(run_dir / "metrics.json", payload)
    create_analysis_bundle(
        run_dir,
        session_predictions,
        result.metrics,
        bootstrap,
        evidence_metrics,
    )
    running = run_dir / "RUNNING.json"
    if running.exists():
        running.unlink()
    artifact_manifest_hash = _write_artifact_manifest(run_dir)
    _write_json(
        run_dir / "COMPLETED.json",
        {
            "status": "complete",
            "completed_at_utc": _utc_now(),
            "formal_training_runs": 1,
            "predictions": int(len(session_predictions)),
            "artifact_manifest_sha256": artifact_manifest_hash,
        },
    )
    print(f"Evaluated {len(session_predictions)} test sessions at {run_dir}")
    return payload


def run_pilot_command(
    config_path: Path,
    data_dir: Path,
    prepared_dir: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Run the pinned data preparation and exactly one formal model fit."""

    _assert_run_available(run_dir)
    config = _load_config(config_path)
    _validate_config_contract(config)
    download_data_command(data_dir)
    prepare_command(
        data_dir,
        prepared_dir,
        seed=int(config["seed"]),
        sessions=int(config["n_sessions"]),
    )
    train_command(config_path, prepared_dir, run_dir)
    return evaluate_command(run_dir, prepared_dir)
