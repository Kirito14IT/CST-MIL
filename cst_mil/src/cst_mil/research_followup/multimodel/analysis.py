"""Auditable metrics and blind review packets; no inference and no semantic auto-labels."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

from ..io import inputs, read, save, table
from ..round2.execution import evaluate, latency
from .core import MODELS, OLD, OUT


def wilson(events: int, total: int) -> list[float] | None:
    if not total:
        return None
    z = 1.959963984540054
    p = events / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0, center - half), min(1, center + half)]


def _token(usage: dict, path: tuple[str, ...]) -> int | None:
    value = usage
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if type(value) is int and value >= 0 else None


def analyze(out: Path = OUT) -> dict:
    read(out / "selection.json")
    _, sessions, _, grouped = inputs()
    lookup = {s["post_id"]: s for s in sessions}
    all_rows, errors, review_map, blind = [], [], {}, []
    evaluations = {}
    sources = [(m, out / m) for m in MODELS] + [("gemini-2.5-flash", OLD)]
    for model, folder in sources:
        if not (folder / "summary.json").exists():
            continue
        summary = read(folder / "summary.json")
        predictions, test_results, usage_rows = [], [], []
        for result in summary["results"]:
            sid = result["session_id"]
            for attempt in result["attempts"]:
                u = attempt.get("usage") or {}
                usage_rows.append(
                    dict(
                        model=model,
                        session_id=sid,
                        split=result["split"],
                        attempt=attempt.get("attempt"),
                        seconds=attempt.get("seconds"),
                        prompt_tokens=_token(u, ("prompt_tokens",)),
                        completion_tokens=_token(u, ("completion_tokens",)),
                        reasoning_tokens=_token(
                            u, ("completion_tokens_details", "reasoning_tokens")
                        ),
                        cached_tokens=_token(u, ("prompt_tokens_details", "cached_tokens")),
                        cost_estimate_upper_rate_usd=attempt.get("cost_estimate_usd"),
                        account_debit_usd=None,
                        http_status=attempt.get("http_status"),
                    )
                )
            if result["split"] != "test":
                continue
            test_results.append(result)
            error = dict(
                model=model,
                session_id=sid,
                status=result["status"],
                returned_model=result.get("returned_model"),
                classification_error=None,
                nonexistent_id_count=None,
                fabricated_quote_count=None,
            )
            if result["status"] == "success":
                value = result["output"]
                # Re-audit the durable output against current frozen text, including old Gemini.
                from ..api import validate_evidence

                validation = validate_evidence(value, grouped[sid])
                validation["unsupported_reason_status"] = (
                    "requires blind Codex source-support review"
                )
                error.update(validation)
                row = dict(
                    post_id=sid,
                    label=lookup[sid]["label"],
                    threshold=0.5,
                    risk_probability=value["risk_score"],
                    prediction=int(value["risk_score"] >= 0.5),
                    evidence=[
                        dict(comment_id=e["message_id"], quote=e["quote"], reason=e["reason"])
                        for e in value["top3_evidence"]
                    ],
                )
                predictions.append(row)
                error["classification_error"] = row["prediction"] != row["label"]
                rid = hashlib.sha256(("blind-review-v1|" + model + "|" + sid).encode()).hexdigest()[
                    :20
                ]
                # Packet intentionally omits model identity, score, gold label and prior reviews.
                packet = dict(
                    review_id=rid,
                    evidence=value["top3_evidence"],
                    messages=[
                        {k: m.get(k) for k in ("comment_id", "parent_id", "order", "raw_text")}
                        for m in grouped[sid]
                    ],
                )
                blind.append(packet)
                review_map[rid] = dict(model=model, session_id=sid)
            errors.append(error)
        usage_path = out / "derived" / model
        table(usage_path / "usage.csv", usage_rows)
        save(usage_path / "test_predictions.json", predictions)
        valid_errors = [r for r in errors if r["model"] == model and r["status"] == "success"]
        n_valid = len(valid_errors)
        evidence_n = sum(len(p["evidence"]) for p in predictions)
        audit = {}
        for field in ("nonexistent_id_count", "fabricated_quote_count", "duplicate_id_count"):
            count = sum(r[field] for r in valid_errors)
            sessions_with_error = sum(r[field] > 0 for r in valid_errors)
            audit[field] = dict(
                events=count,
                evidence_denominator=evidence_n,
                evidence_rate=count / evidence_n if evidence_n else None,
                evidence_wilson95=wilson(count, evidence_n),
                sessions_with_error=sessions_with_error,
                valid_session_denominator=n_valid,
                session_rate=sessions_with_error / n_valid if n_valid else None,
                session_wilson95=wilson(sessions_with_error, n_valid),
            )
        times = [sum(a.get("seconds", 0) for a in r["attempts"]) for r in test_results]
        test_usage = [r for r in usage_rows if r["split"] == "test"]
        known_cost = sum(
            r["cost_estimate_upper_rate_usd"]
            for r in test_usage
            if r["cost_estimate_upper_rate_usd"] is not None
        )
        unknown_cost = sum(r["cost_estimate_upper_rate_usd"] is None for r in test_usage)
        metrics = evaluate(predictions, 0.5, grouped) if predictions else {}
        evaluation = dict(
            model=model,
            expected_test=30,
            n_completed=len(test_results),
            n_valid=n_valid,
            effective_output_ratio=n_valid / 30,
            metrics=metrics,
            latency=latency(times) if times else None,
            automatic_evidence_audit=audit,
            usage_known_cost_sum_usd=known_cost,
            cost_unknown_attempts=unknown_cost,
            test_cost_upper_rate_usd=known_cost if not unknown_cost else None,
            account_debit_usd=None,
            requests_all_splits=summary.get("attempts"),
            model_size=None,
            server_gpu=None,
            server_memory=None,
            classification_errors=sum(r["classification_error"] is True for r in valid_errors),
            quote_verification=(
                "Exact contiguous source substring; semantic support separately reviewed"
            ),
            ci_note="Wilson intervals are descriptive; same-session evidence items are dependent",
        )
        evaluations[model] = evaluation
        save(usage_path / "evaluation.json", evaluation)
        t = evaluation["latency"] or {}
        all_rows.append(
            dict(
                method=model,
                n_test=30,
                n_valid=n_valid,
                ap=metrics.get("ap"),
                auc=metrics.get("auc"),
                macro_f1=metrics.get("macro_f1"),
                precision=metrics.get("precision"),
                recall=metrics.get("recall"),
                threshold=0.5,
                top3_hits=metrics.get("top3_hits"),
                top3_eligible=metrics.get("top3_eligible_sessions"),
                dangerous_comment_recall=metrics.get("dangerous_comment_recall"),
                api_total_seconds=t.get("total_seconds"),
                api_median_seconds=t.get("median_seconds"),
                api_p95_seconds=t.get("p95_seconds"),
                incremental_api_usd=evaluation["test_cost_upper_rate_usd"],
                account_debit_usd=None,
                cost_unknown_attempts=unknown_cost,
                nonexistent_ids=audit["nonexistent_id_count"]["events"],
                fabricated_quotes=audit["fabricated_quote_count"]["events"],
                evidence_denominator=evidence_n,
                classification_errors=evaluation["classification_errors"],
                train_seconds=None,
                model_mib=None,
                peak_rss_mib=None,
                deployment="Remote API; server resource use unknown",
            )
        )
    import csv

    local_path = OLD.parent / "pilot_comparison.csv"
    with local_path.open(encoding="utf-8-sig", newline="") as handle:
        local_rows = [r for r in csv.DictReader(handle) if r["method"] != "Gemini"]
    if len(local_rows) != 6:
        raise ValueError("Expected six frozen local methods on same cohort")
    for row in local_rows:
        row.update(
            n_test=30,
            source="20260929 same-cohort timing reused",
            api_cost_basis="No per-request API fee; hardware and training costs remain",
        )
    table(out / "comparison.csv", local_rows + all_rows)
    table(out / "errors.csv", errors)
    save(out / "evaluations.json", evaluations)
    blind.sort(key=lambda r: r["review_id"])
    save(out / "blind_review_map.json", review_map)
    save(out / "blind_review_packets.json", blind)
    for i, packet in enumerate(blind, 1):
        save(out / "blind_review" / f"{i:03d}_{packet['review_id']}.json", packet)
    save(
        out / "review_protocol.json",
        dict(
            reviewer="Codex textual source-support review; not independent human gold",
            blinding="Packets omit model identity, score, labels and previous semantic judgments",
            supported="Core reading is supported by full supplied text and reply relationships",
            unsupported="Specific factual claim absent from or contradicted by supplied text",
            uncertain="Ambiguous referent, sarcasm or motive; not counted as unsupported",
            exclusions="Classification disagreements are not automatically hallucinations",
            denominator="Review actual items only; missing items are separate completeness errors",
            limitations="No human gold or inter-rater agreement; 30-test pilot is exploratory",
        ),
    )
    lines = [
        "# 同一30会话的多模型小规模对照",
        "",
        "固定原 Gemini 的6验证+30测试清单、完整上下文、提示词、0.5阈值。"
        "新增调用不用于挑选会话或调参。",
        "费用为公开最高分组按返回usage估算；账户扣款未知。缺usage的尝试费用标记未知，不记为零。",
        "本地模型计时复用上一轮同一30条会话，API延迟属于本次中转接口端到端响应。",
        "自动编号/引文核验与普通分类错误分别报告。理由支持度等待盲化Codex文本复核，不称人工金标。",
        "",
    ]
    for model, e in evaluations.items():
        m, t = e["metrics"], e["latency"] or {}
        lines.append(
            f"- {model}：有效 {e['n_valid']}/30；AP={m.get('ap')}，AUC={m.get('auc')}；"
            f"中位延迟={t.get('median_seconds')}秒，P95={t.get('p95_seconds')}秒；"
            f"测试费用估算={e['test_cost_upper_rate_usd']}美元；"
            f"不存在编号={e['automatic_evidence_audit']['nonexistent_id_count']['events']}，"
            f"伪造摘录={e['automatic_evidence_audit']['fabricated_quote_count']['events']}。"
        )
    lines += [
        "",
        "不预设大模型必然更差。可用的限定结论是比较本次部署下的风险排序、响应时间、逐请求费用和资源需求；"
        "零次自动错误不证明总体幻觉率为零，需同时报告区间和理由来源支持复核。",
    ]
    (out / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return evaluations


def finalize_review(out: Path = OUT) -> list[dict]:
    """Validate each textual review item against its blinded packet before aggregation."""
    mapping = read(out / "blind_review_map.json")
    packets = {p["review_id"]: p for p in read(out / "blind_review_packets.json")}
    reviews = {}
    for path in (out / "review_results").glob("*.json"):
        value = read(path)
        for r in value if isinstance(value, list) else [value]:
            rid = r["review_id"]
            if rid in reviews or rid not in packets:
                raise ValueError("Duplicate or unknown review ID")
            evidence = packets[rid]["evidence"]
            if len(r["items"]) != len(evidence):
                raise ValueError("Review must cover every actual evidence item")
            for index, (item, source) in enumerate(zip(r["items"], evidence, strict=True)):
                if item["message_id"] != source["message_id"]:
                    raise ValueError("Review item source mismatch")
                if item["support_status"] not in ("supported", "unsupported", "uncertain"):
                    raise ValueError("Unknown semantic review label")
                if not isinstance(item["justification"], str) or not item["justification"].strip():
                    raise ValueError("Source-based review justification required")
                item["item_index"] = index
            reviews[rid] = r
    rows, examples = [], []
    for model in [*MODELS, "gemini-2.5-flash"]:
        expected = [rid for rid, route in mapping.items() if route["model"] == model]
        done = [rid for rid in expected if rid in reviews]
        counts = {k: 0 for k in ("supported", "unsupported", "uncertain")}
        for rid in done:
            for item in reviews[rid]["items"]:
                counts[item["support_status"]] += 1
                if item["support_status"] != "supported":
                    evidence = packets[rid]["evidence"][item["item_index"]]
                    examples.append(
                        dict(
                            model=model,
                            session_id=mapping[rid]["session_id"],
                            **item,
                            quote=evidence["quote"],
                            reason=evidence["reason"],
                        )
                    )
        total = sum(counts.values())
        rows.append(
            dict(
                model=model,
                n_expected_sessions=len(expected),
                n_reviewed_sessions=len(done),
                review_complete=len(done) == len(expected),
                evidence_denominator=total,
                **counts,
                unsupported_rate=counts["unsupported"] / total if total else None,
                unsupported_wilson95=wilson(counts["unsupported"], total),
                reviewer="Codex blind textual review, not human gold",
                limitation="Evidence items within each conversation are dependent",
            )
        )
    save(out / "semantic_review_summary.json", rows)
    table(out / "semantic_review_summary.csv", rows)
    save(out / "semantic_review_examples.json", examples)
    save(out / "semantic_review_all.json", reviews)
    return rows
