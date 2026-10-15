"""Serial six-method training with durable per-comment inference and bindings."""

from __future__ import annotations

import gc
import importlib.metadata
import platform
import sys
import time
import traceback
from pathlib import Path

from .core import (
    METHODS,
    OUT,
    PROJECT,
    bind_or_verify,
    digest,
    guard_output,
    identity,
    read,
    read_jsonl,
    save,
    source_hashes,
    verify_sources,
    write_jsonl,
)


def _deployment(model, checkpoint):
    topic = model.topic_model
    result = dict(
        model_bytes=checkpoint.stat().st_size,
        encoder_bytes=0,
        java_runtime_bytes=0,
        requires_gpu=False,
        gpu_used=False,
        python=sys.version.split()[0],
        platform=platform.platform(),
        external_file_hashes={},
        python_environment_bytes=None,
        scope="checkpoint plus separately provisioned encoder/JDK; shared Python environment not measured per run",
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "joblib", "jieba", "psutil")
        },
    )
    if hasattr(topic, "model_dir"):
        manifest_path = Path(topic.model_dir) / "benchmark_model_manifest.json"
        manifest = read(manifest_path)
        result.update(
            encoder_bytes=sum(v["bytes"] for v in manifest["files"].values()),
            encoder_path=str(topic.model_dir),
            encoder_revision=manifest["revision"],
        )
        result["external_file_hashes"] = {
            str(Path(topic.model_dir) / path): item["sha256"]
            for path, item in manifest["files"].items()
        }
        result["external_file_hashes"][str(manifest_path)] = digest(manifest_path)
    if getattr(topic, "method", None) == "gsdmm":
        from .native_topics import portable_java

        executable = portable_java().resolve()
        root = executable.parent.parent
        result.update(
            java_executable=str(executable),
            java_runtime_path=str(root),
            java_runtime_bytes=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()),
        )
        result["external_file_hashes"].update(source_hashes([executable, root / "release"]))
    return result


def _binding(dataset, method, out, manifest, config):
    from sklearn.decomposition import _nmf

    source = Path(__file__).parent
    paths = [
        source / n
        for n in (
            "core.py",
            "common.py",
            "model.py",
            "native_topics.py",
            "numerics.py",
            "fork_provenance.json",
            "bertopic_backend.py",
            "data.py",
            "execution.py",
        )
    ]
    paths += [Path(_nmf.__file__)]
    paths += list(source.glob("*.py"))
    paths += [PROJECT / "src/cst_mil" / n for n in ("preprocess.py", "metrics.py")]
    paths += [
        PROJECT / "src/cst_mil/bench" / n
        for n in ("interpretation.py", "resources.py", "runner.py", "common.py")
    ]
    paths += [
        PROJECT / "src/cst_mil/research_followup/round2/execution.py",
        PROJECT / "src/cst_mil/research_followup/runtime_conditions.py",
    ]
    paths += [
        PROJECT / "pyproject.toml",
        PROJECT / "uv.lock",
        out / "prepared" / dataset / "manifest.json",
        PROJECT / "benchmark/runs/full677" / method / "config.json",
    ]
    paths += list((out / "prepared" / dataset).rglob("*.jsonl"))
    if method in {"lda", "btm", "gsdmm"}:
        paths += [
            p
            for p in (PROJECT.parent / "baseline" / method.upper()).rglob("*")
            if p.is_file() and p.suffix in {".c", ".h", ".cpp", ".java", ".jar"}
        ]
        paths += [
            p
            for p in (PROJECT / "benchmark/tools/native" / method).rglob("*")
            if p.is_file() and p.suffix in {".exe", ".class", ".json"}
        ]
    return dict(
        dataset=dataset,
        method=method,
        sources=source_hashes(paths),
        config=config,
        cohort_fingerprint=manifest["cohort_fingerprint"],
        python=sys.version,
        tokenizer="unicode-han-jieba-words-v1",
        unit="single_comment",
    )


def _valid_wrapper(row, session, messages, bound, model_sha):
    if row["binding_sha256"] != identity(bound) or row["model_sha256"] != model_sha:
        raise ValueError("Prediction fingerprint differs")
    if row["input_sha256"] != identity(dict(session=session, messages=messages)):
        raise ValueError("Prediction input differs")
    prediction = row["prediction"]
    if row["prediction_sha256"] != identity(prediction):
        raise ValueError("Prediction changed")
    if prediction["post_id"] != session["post_id"] or prediction["label"] != session["label"]:
        raise ValueError("Prediction identity differs")
    if not 0 <= prediction["risk_probability"] <= 1 or row["seconds"] < 0:
        raise ValueError("Invalid prediction probability/latency")


def _completed(folder, bound, selected, grouped):
    from cst_mil.research_followup.round2.execution import evaluate, latency

    summary = read(folder / "summary.json")
    trained = read(folder / "trained.json")
    if summary["binding_sha256"] != identity(bound) or trained["binding_sha256"] != identity(bound):
        raise ValueError("Completed result binding changed")
    if summary["training"] != trained or summary["threshold"] != trained["threshold"]:
        raise ValueError("Completed training or threshold changed")
    verify_sources(summary["artifact_hashes"])
    verify_sources(trained["artifact_hashes"])
    verify_sources(summary["dependencies"].get("external_file_hashes", {}))
    expected = {identity(s["post_id"]) + ".json" for s in selected}
    if {p.name for p in (folder / "sessions").glob("*.json")} != expected:
        raise ValueError("Completed prediction inventory changed")
    rows, times = [], []
    for session in selected:
        wrapper = read(folder / "sessions" / (identity(session["post_id"]) + ".json"))
        _valid_wrapper(
            wrapper, session, grouped[session["post_id"]], bound, trained["model_sha256"]
        )
        rows.append(wrapper["prediction"])
        times.append(wrapper["seconds"])
    if rows != read_jsonl(folder / "test_predictions.jsonl"):
        raise ValueError("Aggregate predictions differ from checkpoints")
    if evaluate(rows, trained["threshold"]) != summary["test"]["metrics"]:
        raise ValueError("Completed metric changed")
    if latency(times) != summary["test"]["latency"]:
        raise ValueError("Completed latency changed")
    return summary


def run_one(dataset, method, out=OUT, seed=42, threads=4):
    from cst_mil.bench.resources import configure_cpu

    if not 1 <= threads <= 4:
        raise ValueError("Maximum 4 threads")
    cpu = configure_cpu(threads)
    import joblib
    from threadpoolctl import threadpool_limits

    from cst_mil.bench.runner import ResourceMonitor, run_lock
    from cst_mil.research_followup.runtime_conditions import conditions
    from cst_mil.research_followup.round2.execution import evaluate, latency

    from .data import load_dataset
    from .model import predict_session, train_model

    if method not in METHODS:
        raise ValueError("Unknown method")
    out = guard_output(out)
    manifest, sessions, messages = load_dataset(dataset, out)
    if not manifest.get("formal_comparison_allowed", True):
        raise ValueError("Dataset failed formal comparison gate")
    grouped = {s["post_id"]: [] for s in sessions}
    for m in messages:
        grouped[m["post_id"]].append(m)
    if len(grouped) != len(sessions) or any(len(ms) != 1 for ms in grouped.values()):
        raise ValueError("New tasks require exactly one unique comment per sample")
    if any(m.get("parent_id") is not None for m in messages):
        raise ValueError("Single comment cannot invent reply edges")
    folder = out / "runs" / dataset / method
    folder.mkdir(parents=True, exist_ok=True)
    config = read(PROJECT / "benchmark/runs/full677" / method / "config.json")
    config.update(seed=seed, threads=threads)
    if method == "bertopic":
        config["bertopic"] = {
            **config.get("bertopic", {}),
            "embedding_cache_dir": str(folder / "embedding_cache"),
        }
    bound = _binding(dataset, method, out, manifest, config)
    bind_or_verify(folder / "binding.json", bound)
    verify_sources(manifest["source_hashes"])
    verify_sources(bound["sources"])
    fitted = [s for s in sessions if s["split"] in {"train", "dev"}]
    allowed = {s["post_id"] for s in fitted}
    fit_messages = [m for m in messages if m["post_id"] in allowed]
    model_path = folder / "model.joblib"
    with run_lock(folder), threadpool_limits(limits=threads):
        try:
            if (folder / "summary.json").exists():
                selected = [s for s in sessions if s["split"] == "test"]
                result = _completed(folder, bound, selected, grouped)
                print(
                    f"{dataset}/{method}: completed checkpoint verified; original timing retained",
                    flush=True,
                )
                return result
            if (folder / "trained.json").exists():
                trained = read(folder / "trained.json")
                if trained["binding_sha256"] != identity(bound):
                    raise ValueError("Training fingerprint changed")
                verify_sources(trained["artifact_hashes"])
            else:
                if model_path.exists() or (folder / "assets").exists():
                    raise ValueError(
                        "Incomplete training assets exist; retain failure and use new revision"
                    )
                print(f"{dataset}/{method}: training", flush=True)
                start = time.perf_counter()
                with ResourceMonitor() as monitor:
                    model = train_model(
                        fitted,
                        fit_messages,
                        "cst_mil" if method == "cst_mil_r0" else method,
                        config,
                        folder / "assets",
                    )
                elapsed = time.perf_counter() - start
                if monitor.peak_affinity_width > threads:
                    raise RuntimeError("CPU cap exceeded in training")
                verify_sources(bound["sources"])
                temp = model_path.with_suffix(".pending.joblib")
                joblib.dump(model, temp, compress=3)
                temp.replace(model_path)
                write_jsonl(folder / "dev_predictions.jsonl", model.dev_predictions)
                asset_paths = [p for p in (folder / "assets").rglob("*") if p.is_file()]
                trained = dict(
                    binding_sha256=identity(bound),
                    training_seconds=elapsed,
                    peak_rss_bytes=monitor.peak,
                    peak_affinity_width=monitor.peak_affinity_width,
                    model_sha256=digest(model_path),
                    threshold=model.threshold,
                    dev_metrics=evaluate(model.dev_predictions, model.threshold),
                    training_summary=model.training_summary,
                    artifact_hashes=source_hashes(
                        [model_path, folder / "dev_predictions.jsonl", *asset_paths]
                    ),
                )
                save(folder / "trained.json", trained)
                del model
                gc.collect()
            with ResourceMonitor() as load_monitor:
                load_started = time.perf_counter()
                model = joblib.load(model_path)
                load_seconds = time.perf_counter() - load_started
            if not (folder / "loading.json").exists():
                save(
                    folder / "loading.json",
                    dict(
                        seconds=load_seconds,
                        peak_rss_bytes=load_monitor.peak,
                        policy="freshly reloaded saved checkpoint; OS caches not cleared; encoder may remain resident after training",
                    ),
                )
            deps = _deployment(model, model_path)
            save(folder / "dependencies.json", deps)
            selected = [s for s in sessions if s["split"] == "test"]
            rows, times = [], []
            test_conditions = conditions()
            with ResourceMonitor() as test_monitor:
                for index, session in enumerate(selected):
                    dest = folder / "sessions" / (identity(session["post_id"]) + ".json")
                    if dest.exists():
                        row = read(dest)
                    else:
                        verify_sources(bound["sources"])
                        started = time.perf_counter()
                        pred = predict_session(model, session, grouped[session["post_id"]])
                        seconds = time.perf_counter() - started
                        pred["prediction_name"] = "risk" if pred["prediction"] else "non-risk"
                        row = dict(
                            prediction=pred,
                            seconds=seconds,
                            binding_sha256=identity(bound),
                            model_sha256=trained["model_sha256"],
                            input_sha256=identity(
                                dict(session=session, messages=grouped[session["post_id"]])
                            ),
                            prediction_sha256=identity(pred),
                        )
                        save(dest, row)
                    _valid_wrapper(
                        row, session, grouped[session["post_id"]], bound, trained["model_sha256"]
                    )
                    rows.append(row["prediction"])
                    times.append(row["seconds"])
                    if index % 50 == 0:
                        print(f"{dataset}/{method}: test {index + 1}/{len(selected)}", flush=True)
            if test_monitor.peak_affinity_width > threads:
                raise RuntimeError("CPU cap exceeded in inference")
            write_jsonl(folder / "test_predictions.jsonl", rows)
            summary = dict(
                status="complete",
                dataset=dataset,
                method=method,
                binding_sha256=identity(bound),
                sources=bound["sources"],
                config=config,
                threshold=trained["threshold"],
                training=trained,
                dependencies=deps,
                loading=read(folder / "loading.json"),
                cpu_policy=cpu,
                test=dict(
                    metrics=evaluate(rows, model.threshold),
                    latency=latency(times),
                    seconds_by_session=times,
                    session_ids=[s["post_id"] for s in selected],
                    peak_rss_bytes=test_monitor.peak,
                    conditions=test_conditions,
                ),
                task_limits="balanced single-comment risk comparison; no reply/MIL/Top3 gold",
            )
            artifacts = [
                folder / "test_predictions.jsonl",
                folder / "trained.json",
                folder / "dependencies.json",
                folder / "loading.json",
            ]
            summary["artifact_hashes"] = source_hashes(artifacts)
            verify_sources(bound["sources"])
            verify_sources(manifest["source_hashes"])
            save(folder / "summary.json", summary)
            print(
                f"{dataset}/{method}: AP={summary['test']['metrics']['ap']:.4f}; train={trained['training_seconds']:.1f}s",
                flush=True,
            )
            return summary
        except BaseException:
            save(folder / "failure.json", dict(traceback=traceback.format_exc(), binding=bound))
            raise
