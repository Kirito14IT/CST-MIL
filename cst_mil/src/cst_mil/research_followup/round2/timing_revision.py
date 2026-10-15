"""BERTopic-only timing revision with bounded drift and exact decision parity.

Run as a fresh process. Numerical imports are deferred until CPU affinity is set.
The original timing runner and every v1 artifact remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from itertools import groupby
from pathlib import Path

PROBABILITY_TOLERANCE = 1e-5


def validate_parity(predictions: list[dict], reference: list[dict]) -> dict:
    """Reject changed decisions or ranking, including changes to score ties."""
    current = {row["post_id"]: row for row in predictions}
    original = {row["post_id"]: row for row in reference}
    if (
        not original
        or len(current) != len(predictions)
        or len(original) != len(reference)
        or current.keys() != original.keys()
    ):
        raise ValueError("Timing session population differs from the reference")
    for row in [*predictions, *reference]:
        score = row["risk_probability"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("Invalid probability in timing parity inputs")
    maximum = max(
        abs(row["risk_probability"] - original[sid]["risk_probability"])
        for sid, row in current.items()
    )
    if maximum > PROBABILITY_TOLERANCE:
        raise ValueError("BERTopic probability drift exceeds the authorized 1e-5 limit")

    def ranking(rows: list[dict]) -> list[tuple[str, ...]]:
        ordered = sorted(rows, key=lambda row: (-row["risk_probability"], row["post_id"]))
        return [
            tuple(row["post_id"] for row in tied)
            for _, tied in groupby(ordered, key=lambda row: row["risk_probability"])
        ]

    if ranking(predictions) != ranking(reference):
        raise ValueError("Session ranking or probability tie groups changed")
    for sid, row in current.items():
        old = original[sid]
        if row["prediction"] != old["prediction"]:
            raise ValueError(f"Session classification changed: {sid}")
        if [item["comment_id"] for item in row["evidence"][:3]] != [
            item["comment_id"] for item in old["evidence"][:3]
        ]:
            raise ValueError(f"Ordered Top-3 evidence changed: {sid}")
        if [item["topic_id"] for item in row["topics"]] != [
            item["topic_id"] for item in old["topics"]
        ]:
            raise ValueError(f"Ordered topic IDs changed: {sid}")
    return dict(
        accepted=True,
        probability_tolerance=PROBABILITY_TOLERANCE,
        max_saved_prediction_difference=maximum,
        ranking_equal=True,
        ranking_ties_equal=True,
        classification_equal=True,
        ordered_top3_equal=True,
        topic_ids_equal=True,
    )


def measure_bertopic(out: Path, cpu_policy: dict) -> dict:
    """Measure the same 30 sessions: one cold pass and three resident passes."""
    import joblib

    from cst_mil.bench.model import predict_session
    from cst_mil.bench.runner import ResourceMonitor

    from ..io import inputs
    from . import execution
    from .core import (
        RUNS,
        code_sources,
        guard_output,
        jsonl_read,
        jsonl_write,
        read,
        save,
        verify_sources,
    )
    from .execution import evaluate, latency

    out = guard_output(out)
    folder = out / "local_timing_v2" / "bertopic"
    original = out / "local_timing" / "bertopic"
    model_path = RUNS / "bertopic" / "model.joblib"
    reference_path = RUNS / "bertopic" / "predictions.jsonl"
    selection_path = out / "gemini" / "selection.json"
    parity_path = original / "parity_diagnostic.json"
    embedding_path = original / "embedding_diagnostic.json"
    binding = code_sources(
        Path(__file__),
        Path(execution.__file__),
        model_path,
        reference_path,
        selection_path,
        parity_path,
        embedding_path,
        original / "pass0.json",
        original / "predictions_pass0.jsonl",
    )
    verify_sources(binding)
    if (folder / "summary.json").exists():
        previous = read(folder / "summary.json")
        if previous["sources"] != binding:
            raise ValueError("BERTopic timing revision sources changed")
        return previous
    if folder.exists() and any(folder.rglob("*")):
        raise ValueError("Interrupted timing revision exists; preserve it and use revision output")

    parity_basis, embedding_basis = read(parity_path), read(embedding_path)
    if (
        not all(
            parity_basis[key] is True
            for key in ("classification_equal", "evidence_equal", "topic_ids_equal", "order_equal")
        )
        or not 0 <= parity_basis["maxdiff"] <= PROBABILITY_TOLERANCE
    ):
        raise ValueError("Original parity diagnostic does not support the authorized revision")
    if embedding_basis["missing"] != 0:
        raise ValueError("Original embedding diagnostic has unmatched cache entries")
    revision = dict(
        method="bertopic",
        original_failed_path=str(original),
        original_probability_tolerance=1e-6,
        probability_tolerance=PROBABILITY_TOLERANCE,
        acceptance="exact ranking including ties, classification, ordered Top-3 and topic IDs",
        diagnostic_basis=dict(
            parity_path=str(parity_path),
            embedding_path=str(embedding_path),
            parity=parity_basis,
            embeddings=embedding_basis,
        ),
        sources=binding,
    )
    save(folder / "revision.json", revision)
    save(folder / "cpu_policy.json", cpu_policy)
    _, sessions, _, grouped = inputs()
    lookup = {session["post_id"]: session for session in sessions}
    selected = [
        lookup[row["post_id"]] for row in read(selection_path)["sessions"] if row["split"] == "test"
    ]
    if len(selected) != 30 or len({row["post_id"] for row in selected}) != 30:
        raise ValueError("Exactly 30 distinct frozen test sessions are required")
    old = {row["post_id"]: row for row in jsonl_read(reference_path)}
    reference = [old[row["post_id"]] for row in selected]
    verify_sources(binding)
    started = time.perf_counter()
    with ResourceMonitor() as load_monitor:
        model = joblib.load(model_path)
    load_seconds = time.perf_counter() - started
    if model.method != "bertopic":
        raise ValueError("The numerical tolerance revision is authorized only for BERTopic")
    if load_monitor.peak_affinity_width > 4:
        raise RuntimeError("Model loading exceeded the four-core CPU cap")
    model.topic_model.cache_dir = folder / "embedding_cache"
    reference_metrics = evaluate(reference, model.threshold, grouped)
    passes = []
    for repeat in range(4):
        verify_sources(binding)
        predictions, times = [], []
        before = copy.deepcopy(getattr(model.topic_model, "info", {}))
        with ResourceMonitor() as monitor:
            for session in selected:
                started = time.perf_counter()
                prediction = predict_session(model, session, grouped[session["post_id"]])
                times.append(time.perf_counter() - started)
                predictions.append(prediction)
        verify_sources(binding)
        error = None
        try:
            acceptance = validate_parity(predictions, reference)
        except ValueError as exc:
            error = str(exc)
            acceptance = dict(accepted=False, reason=error)
        metrics = evaluate(predictions, model.threshold, grouped)
        if metrics != reference_metrics:
            error = "Frozen prediction metrics changed"
            acceptance.update(accepted=False, reason=error)
        if monitor.peak_affinity_width > 4:
            error = "Timing exceeded the four-core CPU cap"
            acceptance.update(accepted=False, reason=error)
        record = dict(
            repeat=repeat,
            phase="first_pass" if repeat == 0 else "resident_repeat",
            seconds_by_session=times,
            session_ids=[session["post_id"] for session in selected],
            latency=latency(times),
            peak_rss_bytes=monitor.peak,
            peak_affinity_width=monitor.peak_affinity_width,
            max_saved_prediction_difference=max(
                abs(row["risk_probability"] - old[row["post_id"]]["risk_probability"])
                for row in predictions
            ),
            acceptance=acceptance,
            metrics=metrics,
            reference_metrics=reference_metrics,
            backend_counters_before=before,
            backend_counters_after=copy.deepcopy(getattr(model.topic_model, "info", {})),
        )
        jsonl_write(folder / f"predictions_pass{repeat}.jsonl", predictions)
        save(folder / f"pass{repeat}.json", record)
        passes.append(record)
        print(
            f"bertopic revision pass{repeat}: {sum(times):.3f}s; "
            f"parity={record['max_saved_prediction_difference']:.3g}",
            flush=True,
        )
        if error:
            raise RuntimeError(error)
    verify_sources(binding)
    result = dict(
        **revision,
        n_test=30,
        model_load_seconds=load_seconds,
        load_peak_rss_bytes=load_monitor.peak,
        model_bytes=model_path.stat().st_size,
        first_session_seconds=passes[0]["seconds_by_session"][0],
        passes=passes,
        resident_latency=latency(
            [t for record in passes[1:] for t in record["seconds_by_session"]]
        ),
        measurement="serial complete-session predict_session; 4 CPU cores; loading separate",
    )
    save(folder / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    from cst_mil.bench.resources import configure_cpu

    cpu = configure_cpu(4)
    from cst_mil.bench.runner import run_lock

    from .core import OUT, guard_output, read, verify_sources

    out = guard_output(args.out or OUT)
    preservation = read(out / "preservation.json")
    verify_sources(preservation)
    with run_lock(out):
        measure_bertopic(out, cpu)
    verify_sources(preservation)


if __name__ == "__main__":
    main()
