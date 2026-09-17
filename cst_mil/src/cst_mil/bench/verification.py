"""Five whole sessions: train, process restart, resume, deterministic readback."""

from __future__ import annotations

import subprocess
import sys
import time

import joblib
import numpy as np

from .common import file_hash, read_json, read_jsonl, write_json
from .dataset import BENCHMARK, PROJECT, load_suite
from .runner import resolve_config, run_path


def _equal(left, right, path="root"):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise AssertionError(f"Keys differ at {path}")
        for key in left:
            _equal(left[key], right[key], path + "." + key)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise AssertionError(f"Lengths differ at {path}")
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            _equal(a, b, f"{path}[{i}]")
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not np.isclose(left, right, rtol=1e-6, atol=1e-7):
            raise AssertionError(f"Numerical mismatch at {path}: {left} vs {right}")
    elif left != right:
        raise AssertionError(f"Value mismatch at {path}")


def verify_method(method: str, revision: str | None = None) -> dict:
    from .model import predict_session

    _, revision = resolve_config(method, revision, "smoke5")
    run_dir = run_path("smoke5", method, revision)
    log_dir = BENCHMARK / "verification"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{method}_{revision or 'baseline'}.log"
    evidence_path = log.with_suffix(".json")
    previous = read_json(evidence_path) if evidence_path.exists() else None
    starting_completed = (
        read_json(run_dir / "status.json")["completed"] if (run_dir / "status.json").exists() else 0
    )
    if starting_completed > 2 and not previous:
        raise ValueError(
            "Existing complete smoke has no partial-resume proof; "
            "archive this smoke run before a fresh verification"
        )
    command = [
        sys.executable,
        "-m",
        "cst_mil.bench.cli",
        "run",
        "--suite",
        "smoke5",
        "--method",
        method,
        "--resume",
    ]
    if revision:
        command += ["--revision", revision]
    stamps = []
    for limit in ("2", "all", "all"):
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"\nVERIFY LIMIT {limit}\n")
            handle.flush()
            result = subprocess.run(
                command + ["--limit", limit],
                cwd=PROJECT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        if result.returncode:
            raise RuntimeError(f"{method} smoke subprocess failed; see {log}")
        stamps.append(
            {
                "model_sha256": file_hash(run_dir / "model.joblib"),
                "model_mtime": (run_dir / "model.joblib").stat().st_mtime_ns,
                "status": read_json(run_dir / "status.json"),
            }
        )
    if (
        len({s["model_sha256"] for s in stamps}) != 1
        or len({s["model_mtime"] for s in stamps}) != 1
    ):
        raise AssertionError("Smoke resume retrained or rewrote model")
    counts = [item["status"]["completed"] for item in stamps]
    partial_proof = (
        stamps
        if counts == [2, 5, 5]
        else previous.get("partial_resume_checkpoints", previous["checkpoints"])
    )
    if [item["status"]["completed"] for item in partial_proof] != [2, 5, 5] or any(
        item["model_sha256"] != stamps[0]["model_sha256"] for item in partial_proof
    ):
        raise AssertionError("Missing valid same-model 2-to-5 partial-resume evidence")
    _, sessions, messages = load_suite("smoke5")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    if len(predictions) != 5 or len({r["post_id"] for r in predictions}) != 5:
        raise AssertionError("Smoke did not finish exactly five sessions")
    model = joblib.load(run_dir / "model.joblib")
    lookup = {p["post_id"]: p for p in predictions}
    for session in sessions:
        expected = predict_session(
            model, session, [m for m in messages if m["post_id"] == session["post_id"]]
        )
        expected.update(pilot_member=session["pilot_member"], fresh_test=session["fresh_test"])
        _equal(expected, lookup[session["post_id"]])
    payload = {
        "method": method,
        "revision": revision,
        "passed": True,
        "source_sessions": 5,
        "source_full_partition": "train",
        "n_comments": len(messages),
        "independent_process_limits": [2, "all", "all"],
        "checkpoints": stamps,
        "starting_completed": starting_completed,
        "partial_resume_checkpoints": partial_proof,
        "labels_topics_evidence_match": True,
        "tolerance": {"rtol": 1e-6, "atol": 1e-7},
        "completed_at_unix": time.time(),
        "engineering_only": True,
    }
    write_json(log_dir / f"{method}_{revision or 'baseline'}.json", payload)
    return payload
