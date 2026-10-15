"""Derive pilot comparisons from durable receipts and same-cohort predictions."""

from __future__ import annotations

import csv

from ..io import inputs
from .core import METHODS, OLD, OUT, jsonl_read, jsonl_write, read, save, stable
from .execution import evaluate, latency


def write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def pilot(out=OUT):
    folder = out / "gemini"
    selection = read(folder / "selection.json")
    summary = read(folder / "summary.json")
    _, sessions, _, grouped = inputs()
    lookup = {s["post_id"]: s for s in sessions}
    selected = {r["post_id"]: r for r in selection["sessions"]}
    rows, times = [], []
    errors = []
    review = []
    prompt_tokens = completion_tokens = 0
    all_prompt_tokens = all_completion_tokens = 0
    test_cost = 0
    for r in summary["results"]:
        sid = r["session_id"]
        s = lookup[sid]
        for a in r["attempts"]:
            usage = a.get("usage") or {}
            all_prompt_tokens += usage.get("prompt_tokens", 0)
            all_completion_tokens += usage.get("completion_tokens", 0)
        if r["split"] != "test":
            continue
        times.append(sum(a.get("seconds", 0) for a in r["attempts"]))
        test_cost += sum(
            a.get("cost_estimate_usd")
            if a.get("cost_estimate_usd") is not None
            else a.get("reserved_usd", 0)
            for a in r["attempts"]
        )
        for a in r["attempts"]:
            usage = a.get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens", 0)
            completion_tokens += usage.get("completion_tokens", 0)
        record = dict(
            post_id=sid,
            label=s["label"],
            status=r["status"],
            classification_error=None,
            nonexistent_id_count=None,
            fabricated_quote_count=None,
        )
        if r["status"] == "success":
            output = r["output"]
            row = dict(
                post_id=sid,
                split="test",
                label=s["label"],
                risk_probability=output["risk_score"],
                prediction=int(output["risk_score"] >= 0.5),
                threshold=0.5,
                evidence=[
                    dict(comment_id=e["message_id"], quote=e["quote"], reason=e["reason"])
                    for e in output["top3_evidence"]
                ],
            )
            rows.append(row)
            record.update(classification_error=row["prediction"] != row["label"], **r["validation"])
            review.append(
                dict(
                    post_id=sid,
                    model_score=output["risk_score"],
                    evidence=output["top3_evidence"],
                    messages=[
                        {k: m.get(k) for k in ["comment_id", "parent_id", "order", "raw_text"]}
                        for m in grouped[sid]
                    ],
                )
            )
        errors.append(record)
    jsonl_write(folder / "test_predictions.jsonl", rows)
    jsonl_write(folder / "review_packets.jsonl", review)
    write_table(folder / "errors.csv", errors)
    n_test = sum(r["split"] == "test" for r in selected.values())
    result = dict(
        method="gemini-2.5-flash",
        n_test=n_test,
        n_valid=len(rows),
        effective_output_ratio=len(rows) / n_test,
        complete_top3_sessions=sum(r.get("insufficient_evidence_count") == 0 for r in errors),
        metrics=evaluate(rows, 0.5, grouped),
        latency=latency(times) if times else None,
        all_attempts=summary["attempts"],
        all_cost_upper_rate_usd=summary["accounted_upper_rate_usd"],
        test_cost_upper_rate_usd=test_cost,
        account_debit_usd=None,
        test_prompt_tokens=prompt_tokens,
        test_completion_tokens=completion_tokens,
        all_prompt_tokens=all_prompt_tokens,
        all_completion_tokens=all_completion_tokens,
        errors=errors,
        model_size=None,
        server_gpu=None,
        server_memory=None,
    )
    save(folder / "evaluation.json", result)
    local = []
    previous_cost = {r["method"]: r for r in read(OLD / "costs.json")}
    for method in METHODS:
        timing_path = out / "local_timing_v2" / method / "summary.json"
        if not timing_path.exists():
            timing_path = out / "local_timing" / method / "summary.json"
        timing = read(timing_path)
        measured = timing["passes"][0]["metrics"]
        local.append(
            dict(
                method=method,
                ap=measured["ap"],
                auc=measured["auc"],
                macro_f1=measured["macro_f1"],
                top3_hits=measured["top3_hits"],
                top3_eligible=measured["top3_eligible_sessions"],
                dangerous_comment_recall=measured["dangerous_comment_recall"],
                threshold=measured["threshold"],
                load_seconds=timing["model_load_seconds"],
                first_session_seconds=timing["first_session_seconds"],
                first_pass_total_seconds=timing["passes"][0]["latency"]["total_seconds"],
                resident_median_seconds=timing["resident_latency"]["median_seconds"],
                resident_p95_seconds=timing["resident_latency"]["p95_seconds"],
                resident_mean_total_seconds=timing["resident_latency"]["total_seconds"] / 3,
                model_mib=timing["model_bytes"] / 2**20,
                peak_rss_mib=max(p["peak_rss_bytes"] for p in timing["passes"]) / 2**20,
                historical_training_seconds=previous_cost[method]["train_seconds"],
                historical_extra_dependencies_mib=previous_cost[method]["encoder_mib"]
                + previous_cost[method]["runtime_mib"],
                incremental_api_usd=0,
            )
        )
    m = result["metrics"]
    combined = local + [
        dict(
            method="Gemini",
            ap=m.get("ap"),
            auc=m.get("auc"),
            macro_f1=m.get("macro_f1"),
            top3_hits=m.get("top3_hits"),
            top3_eligible=m.get("top3_eligible_sessions"),
            dangerous_comment_recall=m.get("dangerous_comment_recall"),
            threshold=0.5,
            first_pass_total_seconds=result["latency"]["total_seconds"] if times else None,
            resident_median_seconds=None,
            resident_p95_seconds=None,
            api_median_seconds=result["latency"]["median_seconds"] if times else None,
            api_p95_seconds=result["latency"]["p95_seconds"] if times else None,
            incremental_api_usd=test_cost,
        )
    ]
    write_table(out / "pilot_comparison.csv", combined)
    save(out / "pilot_comparison.json", combined)
    return result, local


def review_stub(out=OUT):
    """Export full context before independent textual review; no automated semantic verdict."""
    packets = jsonl_read(out / "gemini/review_packets.jsonl")
    for index, packet in enumerate(packets):
        path = (
            out / "gemini/review_packets" / f"{index + 1:02d}_{stable(packet['post_id'])[:8]}.json"
        )
        save(path, packet)
    return len(packets)


def export_reviews(out=OUT):
    _, _, _, grouped = inputs()
    review = []
    for r in read(out / "gemini/summary.json")["results"]:
        if r["split"] == "test" and r["status"] == "success":
            review.append(
                dict(
                    post_id=r["session_id"],
                    model_score=r["output"]["risk_score"],
                    evidence=r["output"]["top3_evidence"],
                    messages=[
                        {k: m.get(k) for k in ["comment_id", "parent_id", "order", "raw_text"]}
                        for m in grouped[r["session_id"]]
                    ],
                )
            )
    jsonl_write(out / "gemini/review_packets.jsonl", review)
    return review_stub(out)
