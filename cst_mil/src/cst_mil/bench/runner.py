"""Single-method durable training and exactly-once session inference."""

from __future__ import annotations

import copy
import importlib.metadata
import os
import shutil
import sys
import threading
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import psutil
from threadpoolctl import threadpool_limits

from .common import (
    build_windows,
    file_hash,
    fingerprint,
    read_json,
    write_json,
    write_jsonl,
)
from .dataset import BENCHMARK, PROJECT, load_suite
from .evaluation import binary_metrics, bootstrap_metrics, evidence_metrics, topic_quality

METHODS = ("lda", "btm", "gsdmm", "nmf", "bertopic", "cst_mil")
DEFAULT_CONFIG = {
    "seed": 42,
    "n_topics": 32,
    "threads": 4,
    "window_size": 5,
    "stride": 2,
    "word_features": False,
    "risk_weighted_topics": False,
    "multiscale": False,
    "char_n_features": 65536,
    "mil_rounds": 3,
    "positive_top_fraction": 0.20,
    "instance_c": 4.0,
    "session_c": 1.0,
    "membership_threshold": 0.25,
    "relative_threshold": 0.5,
    "strong_threshold": 0.6,
    "evidence_top_k": 3,
    "lda": {"em_max_iter": 100, "var_max_iter": 20},
    "btm": {"iterations": 100, "alpha": 50 / 32, "beta": 0.005},
    "gsdmm": {"iterations": 100, "alpha": 0.1, "beta": 0.1},
    "nmf": {"max_iter": 300, "tol": 1e-4},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_fingerprint() -> dict:
    sources = [
        PROJECT / "src/cst_mil/bench" / name
        for name in (
            "common.py",
            "model.py",
            "interpretation.py",
            "native_topics.py",
            "bertopic_backend.py",
            "runner.py",
            "dataset.py",
            "evaluation.py",
            "campaign.py",
            "cli.py",
            "verification.py",
            "resources.py",
        )
    ]
    sources += [PROJECT / "src/cst_mil" / name for name in ("preprocess.py", "metrics.py")]
    sources += [PROJECT / name for name in ("pyproject.toml", "uv.lock")]
    packages = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "jieba",
        "joblib",
        "bertopic",
        "sentence-transformers",
        "torch",
        "transformers",
        "umap-learn",
        "hdbscan",
    ]
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    native_sources = [
        p
        for name in ("LDA", "BTM", "GSDMM")
        for p in (PROJECT.parent / "baseline" / name).rglob("*")
        if p.suffix in {".c", ".cpp", ".h", ".java", ".jar"}
    ]
    native_tools = [
        p for p in (BENCHMARK / "tools/native").rglob("*") if p.suffix in {".exe", ".class"}
    ]
    return {
        "source_sha256": {
            p.relative_to(PROJECT).as_posix(): file_hash(p) for p in sources if p.exists()
        },
        "native_source_sha256": {
            p.relative_to(PROJECT.parent).as_posix(): file_hash(p) for p in native_sources
        },
        "native_tool_sha256": {
            p.relative_to(BENCHMARK).as_posix(): file_hash(p) for p in native_tools
        },
        "packages": versions,
        "python": sys.version.split()[0],
    }


def ensure_protocol() -> dict:
    path = BENCHMARK / "protocol.json"
    protocol = {
        "schema_version": 1,
        "config": DEFAULT_CONFIG,
        "pilot_ids": load_suite("pilot300")[0]["content_fingerprint"],
        "full_ids": load_suite("full677")[0]["content_fingerprint"],
        "iterations": {"r1": "word_features", "r2": "risk_weighted_topics", "r3": "multiscale"},
        "selection_data": "pilot300 dev only",
        "test_unlock": "selection.json",
        "threads": 4,
        "test_order": "train then dev then test",
        "smoke_only_prior_folds": True,
    }
    if path.exists():
        existing = read_json(path)
        if fingerprint(existing) != fingerprint(protocol):
            raise ValueError("Frozen protocol mismatch; create a new experiment protocol")
        return existing
    write_json(path, protocol)
    return protocol


def functional_checks_passed() -> bool:
    path = BENCHMARK / "functional_checks.json"
    if not path.exists():
        return False
    evidence = read_json(path)
    return bool(
        evidence.get("passed", False)
        and evidence.get("runtime_sha256") == fingerprint(runtime_fingerprint())
    )


@contextmanager
def run_lock(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ".lock"
    if path.exists():
        info = read_json(path)
        pid = int(info["pid"])
        alive = False
        if psutil.pid_exists(pid):
            try:
                alive = (
                    abs(psutil.Process(pid).create_time() - float(info["process_created"])) < 0.01
                )
            except psutil.NoSuchProcess:
                pass
        if alive:
            raise RuntimeError(f"Run already active under PID {pid}: {run_dir}")
        path.rename(run_dir / f"stale_lock_{time.time_ns()}.json")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        payload = {
            "pid": os.getpid(),
            "process_created": psutil.Process().create_time(),
            "started": utc_now(),
        }
        import json

        os.write(fd, json.dumps(payload).encode())
        os.close(fd)
        fd = -1
        yield
    finally:
        if fd >= 0:
            os.close(fd)
        path.unlink(missing_ok=True)


class ResourceMonitor:
    def __init__(self):
        self.peak = 0
        self.peak_affinity_width = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop.is_set():
            rss = 0
            process = psutil.Process()
            for item in [process, *process.children(recursive=True)]:
                try:
                    rss += item.memory_info().rss
                    self.peak_affinity_width = max(
                        self.peak_affinity_width, len(item.cpu_affinity())
                    )
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            self.peak = max(self.peak, rss)
            self.stop.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def run_path(suite: str, method: str, revision: str | None = None) -> Path:
    return (
        BENCHMARK
        / "runs"
        / suite
        / (f"cst_mil_{revision or 'r0'}" if method == "cst_mil" else method)
    )


def resolve_config(method: str, revision: str | None, suite: str) -> tuple[dict, str | None]:
    ensure_protocol()
    config = copy.deepcopy(DEFAULT_CONFIG)
    if method != "cst_mil":
        config["instance_c"] = 1.0
    if method == "cst_mil":
        if revision is None and (BENCHMARK / "selection.json").exists():
            selected = read_json(BENCHMARK / "selection.json")
            config = selected["config"]
            revision = selected["revision"]
        else:
            revision = revision or "r0"
            config_path = BENCHMARK / "revisions" / f"{revision}.json"
            if config_path.exists():
                config = read_json(config_path)["config"]
            elif revision != "r0":
                raise ValueError("Revision must be preregistered from current dev-selected best")
    if suite == "smoke5":
        config = {**config, "engineering_smoke": True}
    return config, revision


def _write_progress(
    run_dir: Path, manifest: dict, sessions: list[dict], records: list[dict], status: str
):
    done = {r["post_id"] for r in records}
    progress = {
        "status": status,
        "updated_at": utc_now(),
        "suite": manifest["suite"],
        "completed": len(done),
        "total": len(sessions),
        "test_completed": sum(s["post_id"] in done and s["split"] == "test" for s in sessions),
        "test_total": sum(s["split"] == "test" for s in sessions),
        "fresh_test_completed": sum(
            s["post_id"] in done and s.get("fresh_test", False) for s in sessions
        ),
        "fresh_test_total": sum(s.get("fresh_test", False) for s in sessions),
    }
    write_json(run_dir / "status.json", progress)
    return progress


def run_method(
    suite: str,
    method: str,
    *,
    revision: str | None = None,
    limit: str = "all",
    resume: bool = False,
    stage: str = "all",
) -> dict:
    if method not in METHODS or stage not in {"all", "train"}:
        raise ValueError("Unknown method or stage")
    manifest, sessions, messages = load_suite(suite)
    config, revision = resolve_config(method, revision, suite)
    from .resources import configure_cpu

    cpu_policy = configure_cpu(int(config["threads"]))
    target = len(sessions) if limit == "all" else int(limit)
    if not 0 <= target <= len(sessions):
        raise ValueError(f"limit must be between 0 and {len(sessions)}, or all")
    if (
        suite == "pilot300"
        and stage != "train"
        and any(s["split"] == "test" for s in sessions[:target])
    ):
        if not (BENCHMARK / "selection.json").exists():
            raise ValueError(
                "Pilot test inference locked until dev-only revision selection is frozen"
            )
    if suite == "full677" and not (BENCHMARK / "selection.json").exists():
        raise ValueError("Full experiment requires frozen pilot selection")
    path = run_path(suite, method, revision)
    if path.exists() and (path / "run_manifest.json").exists() and not resume:
        raise FileExistsError(f"Run exists; pass --resume: {path}")
    from .native_topics import prepare_native_tools

    for native_method in ("lda", "btm", "gsdmm"):
        prepare_native_tools(native_method)
    source = runtime_fingerprint()
    binding = {
        "suite": suite,
        "cohort": manifest["content_fingerprint"],
        "method": method,
        "revision": revision,
        "config": config,
        "runtime": source,
        "cpu_affinity": cpu_policy["assigned_cpus"],
    }
    by_post = defaultdict(list)
    for message in messages:
        by_post[message["post_id"]].append(message)
    from .model import predict_session, train_model

    with run_lock(path), threadpool_limits(limits=int(config["threads"])):
        try:
            saved_path = path / "run_manifest.json"
            if saved_path.exists():
                saved = read_json(saved_path)
                if saved["binding_sha256"] != fingerprint(binding):
                    raise ValueError(
                        "Run config/source/environment/data changed; "
                        "use a new version, cannot merge predictions"
                    )
            else:
                saved = {
                    "binding": binding,
                    "binding_sha256": fingerprint(binding),
                    "created_at": utc_now(),
                    "formal_fit_attempts": 0,
                    "cpu_policy": cpu_policy,
                }
                write_json(saved_path, saved)
                snapshot = path / "source_snapshot"
                for relative in source["source_sha256"]:
                    destination = snapshot / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(PROJECT / relative, destination)
                for name in ("pyproject.toml", "uv.lock"):
                    shutil.copy2(PROJECT / name, snapshot / name)
                for relative in source["native_source_sha256"]:
                    destination = snapshot / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(PROJECT.parent / relative, destination)
                write_json(path / "config.json", config)
            model_path = path / "model.joblib"
            trained_path = path / "TRAINED.json"
            if trained_path.exists():
                trained = read_json(trained_path)
                if file_hash(model_path) != trained["model_sha256"]:
                    raise ValueError("Model fingerprint mismatch; refusing resume")
                model = joblib.load(model_path)
                if hasattr(model.topic_model, "validate_assets"):
                    model.topic_model.validate_assets()
            else:
                saved["formal_fit_attempts"] += 1
                write_json(saved_path, saved)
                _write_progress(path, manifest, sessions, [], "training")
                print(
                    f"[{suite}/{method}/{revision or '-'}] training attempt "
                    f"{saved['formal_fit_attempts']}",
                    flush=True,
                )
                fitting = [s for s in sessions if s["split"] in {"train", "dev"}]
                fitting_ids = {s["post_id"] for s in fitting}
                fitting_messages = [m for m in messages if m["post_id"] in fitting_ids]
                start = time.perf_counter()
                with ResourceMonitor() as monitor:
                    model = train_model(
                        fitting,
                        fitting_messages,
                        method,
                        config,
                        path / "assets" / f"attempt_{saved['formal_fit_attempts']}",
                    )
                seconds = time.perf_counter() - start
                if monitor.peak_affinity_width > int(config["threads"]):
                    raise RuntimeError("Observed process CPU affinity exceeds frozen resource cap")
                temporary = model_path.with_name("model.pending.joblib")
                joblib.dump(model, temporary, compress=3)
                temporary.replace(model_path)
                dev_sessions = [s for s in sessions if s["split"] == "dev"]
                docs = [
                    w["text"] for s in dev_sessions for w in build_windows(s, by_post[s["post_id"]])
                ]
                quality = topic_quality(
                    model.topic_model.topic_words(10), docs, model.dev_predictions
                )
                dev = {
                    "risk": binary_metrics(model.dev_predictions),
                    "topics": quality,
                    "functional_checks_passed": functional_checks_passed(),
                }
                write_json(path / "dev_metrics.json", dev)
                write_jsonl(path / "dev_predictions.jsonl", model.dev_predictions)
                write_json(path / "training_summary.json", model.training_summary)
                write_json(path / "topics.json", model.topic_model.topic_words(10))
                trained = {
                    "model_sha256": file_hash(model_path),
                    "training_seconds": seconds,
                    "training_peak_rss_bytes": monitor.peak,
                    "training_peak_affinity_width": monitor.peak_affinity_width,
                    "model_bytes": model_path.stat().st_size,
                    "trained_at": utc_now(),
                    "formal_fit_attempts": saved["formal_fit_attempts"],
                    "backend_info": model.topic_model.info,
                    "cpu_policy": cpu_policy,
                }
                write_json(trained_path, trained)
                print(
                    f"[{suite}/{method}] trained in {seconds:.2f}s; "
                    f"dev macro-F1={dev['risk']['macro_f1']:.4f}",
                    flush=True,
                )
            files_dir = path / "sessions"
            records = []
            for file in sorted(files_dir.glob("*.json")):
                wrapper = read_json(file)
                if (
                    wrapper["binding"] != saved["binding_sha256"]
                    or wrapper["model"] != trained["model_sha256"]
                ):
                    raise ValueError("Prediction binding mismatch")
                if fingerprint(wrapper["prediction"]) != wrapper["prediction_sha256"]:
                    raise ValueError("Prediction content was modified")
                records.append(wrapper["prediction"])
            done = {r["post_id"] for r in records}
            if len(done) != len(records) or done - {s["post_id"] for s in sessions}:
                raise ValueError("Invalid/duplicate completed session IDs")
            if stage == "train":
                return _write_progress(path, manifest, sessions, records, "trained")
            for session in sessions[:target]:
                if session["post_id"] in done:
                    continue
                start = time.perf_counter()
                counters = (
                    "encoding_cold_seconds",
                    "encoding_cached_seconds",
                    "encoder_load_seconds",
                    "encoded_texts",
                    "cached_texts",
                )
                before = {k: model.topic_model.info.get(k, 0) for k in counters}
                with ResourceMonitor() as monitor:
                    prediction = predict_session(model, session, by_post[session["post_id"]])
                prediction.update(
                    pilot_member=session["pilot_member"], fresh_test=session["fresh_test"]
                )
                if monitor.peak_affinity_width > int(config["threads"]):
                    raise RuntimeError(
                        "Observed inference process CPU affinity exceeds resource cap"
                    )
                wrapper = {
                    "binding": saved["binding_sha256"],
                    "model": trained["model_sha256"],
                    "prediction": prediction,
                    "prediction_sha256": fingerprint(prediction),
                    "inference_seconds": time.perf_counter() - start,
                    "peak_rss_bytes": monitor.peak,
                    "peak_affinity_width": monitor.peak_affinity_width,
                    "encoding_counters": {
                        k: model.topic_model.info.get(k, 0) - before[k] for k in counters
                    },
                }
                write_json(files_dir / (fingerprint(session["post_id"]) + ".json"), wrapper)
                records.append(prediction)
                done.add(session["post_id"])
                _write_progress(path, manifest, sessions, records, "predicting")
                if len(done) % 10 == 0 or len(done) == target:
                    print(f"[{suite}/{method}] completed {len(done)}/{len(sessions)}", flush=True)
            positions = {s["post_id"]: i for i, s in enumerate(sessions)}
            records.sort(key=lambda r: positions[r["post_id"]])
            write_jsonl(path / "predictions.jsonl", records)
            test_ids = {s["post_id"] for s in sessions if s["split"] == "test"}
            if test_ids <= done:
                tests = [r for r in records if r["split"] == "test"]
                docs = [
                    w["text"]
                    for s in sessions
                    if s["split"] == "test"
                    for w in build_windows(s, by_post[s["post_id"]])
                ]
                metrics = {
                    "test": binary_metrics(tests),
                    "evidence": evidence_metrics(tests, messages),
                    "topics": topic_quality(model.topic_model.topic_words(10), docs, tests),
                    "test_ci": bootstrap_metrics(tests),
                    "engineering_only": suite == "smoke5",
                    "fresh_test": binary_metrics([r for r in tests if r["fresh_test"]]),
                    "previously_seen_test": binary_metrics([r for r in tests if r["pilot_member"]]),
                }
                write_json(path / "metrics.json", metrics)
            status = _write_progress(
                path,
                manifest,
                sessions,
                records,
                "complete" if len(done) == len(sessions) else "paused",
            )
            if status["status"] == "complete":
                write_json(
                    path / "COMPLETED.json", {**status, "model_sha256": trained["model_sha256"]}
                )
            return status
        except BaseException as exc:
            write_json(
                path / f"error_{time.time_ns()}.json",
                {
                    "at": utc_now(),
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
