"""Post-hoc review of frozen classification errors; never fits or changes models.

Run only after all requested predictions and independent analysis are complete:
``python -m cst_mil.research_followup.full_joint_comparison.errors --output PATH``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import CATEGORIES, METHODS, SEEDS

TASKS = (("sccd", "attack"), ("conda", "attack"), ("hatebr", "attack"),
         ("toxicn", "attack"), ("toxicn", "hate"), ("scamspot", "spam_scam"))


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rows(path: Path) -> list[dict]:
    # JSONL boundaries are physical LF/CRLF lines. Unicode separators inside a
    # quoted multilingual string are valid text and must never split a record.
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _value(row: Mapping[str, Any], field: str, category: str) -> Any:
    value = row.get(field)
    if field == "scores" and value is None:
        value = row.get("scores3")
    if field == "observed_mask" and value is None:
        value = row.get("masks")
    if isinstance(value, Mapping):
        return value.get(category)
    if isinstance(value, (list, tuple)) and len(value) == len(CATEGORIES):
        return value[CATEGORIES.index(category)]
    raise ValueError(f"Missing or malformed {field}/{category}")


def _source(row: Mapping[str, Any]) -> str:
    return str(row.get("source", "")).lower().replace("_", "").replace("-", "")


def _score(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("Risk score and threshold must be finite values in [0,1]")
    return result


def _lookup(rows: Sequence[dict], key: str) -> dict[str, dict]:
    result = {str(row[key]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {key} values")
    return result


def _counts(labels: Sequence[int], scores: Sequence[float], threshold: float) -> dict:
    counts = dict.fromkeys(("TP", "FP", "TN", "FN"), 0)
    for label, score in zip(labels, scores, strict=True):
        if label not in (0, 1):
            raise ValueError("Observed gold must be binary")
        pred = _score(score) >= threshold
        counts[("T" if pred == bool(label) else "F") + ("P" if pred else "N")] += 1
    return {"n_observed": len(labels), "positive": sum(labels), **counts}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows or [{"status": "N/A"}])


def _excerpt(value: str, length: int = 160) -> str:
    flat = " ".join(value.split()).replace("|", "\\|").replace("`", "'")
    return flat if len(flat) <= length else flat[:length] + "…"


def _compare(run: Mapping[str, Any], rid: str, category: str, *, unit: str) -> dict:
    if unit == "message":
        value = _score(_value(run["messages"][rid], "scores", category))
        threshold = _score(run["manifest"]["thresholds"][category])
    else:
        value = _score(run["sessions"][rid]["score"])
        threshold = _score(run["manifest"]["thresholds"]["conversation"])
    return {"score": value, "threshold": threshold, "prediction": int(value >= threshold)}


def _top3(sid: str, records: Sequence[dict], cst_run: Mapping[str, Any]) -> list[dict]:
    real = [row for row in records if _source(row) == "sccd" and str(row["session_id"]) == sid
            and not row.get("is_fallback", False)]
    ranked = sorted(real, key=lambda row: (
        -_score(_value(cst_run["messages"][str(row["record_id"])], "scores", "attack")),
        int(row.get("order", 0) or 0), str(row.get("message_id") or row["record_id"]), str(row["record_id"])))[:3]
    result = []
    for rank, row in enumerate(ranked, 1):
        observed = bool(_value(row, "observed_mask", "attack"))
        result.append({"rank": rank, "record_id": row["record_id"], "message_id": row["message_id"],
                       "parent_id": row.get("parent_id"), "order": row.get("order"),
                       "text": row["text"], "gold_observed": observed,
                       "gold_comment_label": int(_value(row, "labels", "attack")) if observed else None,
                       **_compare(cst_run, str(row["record_id"]), "attack", unit="message")})
    return result


def build_error_review(output: str | Path) -> dict:
    """Save error counts and a deterministic seed-42 review set from frozen files."""
    root = Path(output)
    analysis_path = root / "analysis/analysis.json"
    independent = _read(analysis_path)
    if independent.get("status") != "complete":
        raise ValueError("Independent analysis must be complete before error review")
    verification_path = root / "analysis/verification.json"
    verification = _read(verification_path)
    if verification.get("status") != "passed":
        raise ValueError("Independent verification must pass before error review")
    frozen_hashes = verification.get("prediction_and_manifest_hashes", {})
    selected_path = root / "analysis/selected_baselines.json"
    selected = _read(selected_path)
    methods = tuple(independent.get("methods", METHODS))
    seeds = tuple(independent.get("seeds", SEEDS))
    if not {"cst_v1_1", "minilm_lr"}.issubset(methods) or 42 not in seeds:
        raise ValueError("Error review requires CST, MiniLM reference and seed42")
    prepared = root / "prepared"
    manifest_path = prepared / "manifest.json"
    prepared_manifest = _read(manifest_path)
    records_path, sessions_path = prepared / "records.jsonl", prepared / "sessions.jsonl"
    for path in (records_path, sessions_path):
        expected = prepared_manifest.get("artifacts", {}).get(path.name)
        if expected is None or _sha(path) != expected:
            raise ValueError(f"Prepared audit fingerprint unavailable or changed: {path.name}")
    records = [row for row in _rows(records_path) if row["split"] == "test"]
    sessions = [row for row in _rows(sessions_path) if row["split"] == "test"]
    original = _lookup(records, "record_id")
    original_sessions = _lookup(sessions, "session_id")
    hashes = {str(path.relative_to(root)): _sha(path)
              for path in (analysis_path, verification_path, selected_path, manifest_path, records_path, sessions_path)}
    runs, counts = {}, []
    for method in methods:
        for seed in seeds:
            folder = root / "runs" / method / f"seed_{seed}"
            run_manifest_path = folder / "run_manifest.json"
            message_path, session_path = folder / "predictions_message_test.jsonl", folder / "predictions_session_test.jsonl"
            manifest = _read(run_manifest_path)
            if manifest.get("method", method) != method or int(manifest.get("seed", seed)) != seed:
                raise ValueError("Run identity mismatch")
            predicted = _lookup(_rows(message_path), "record_id")
            predicted_sessions = _lookup(_rows(session_path), "session_id")
            if predicted.keys() != original.keys() or predicted_sessions.keys() != original_sessions.keys():
                raise ValueError("Incomplete or surplus frozen test predictions")
            for rid, row in predicted.items():
                raw = original[rid]
                if row.get("split", "test") != "test" or _source(row) != _source(raw):
                    raise ValueError("Frozen prediction split/source differs from original")
                for category in CATEGORIES:
                    if bool(_value(row, "observed_mask", category)) != bool(_value(raw, "observed_mask", category)):
                        raise ValueError("Frozen prediction label mask differs from original")
                    if bool(_value(raw, "observed_mask", category)) and _value(row, "labels", category) != _value(raw, "labels", category):
                        raise ValueError("Frozen prediction gold differs from original")
                    _score(_value(row, "scores", category))
            for sid, row in predicted_sessions.items():
                if row["label"] != original_sessions[sid]["label"]:
                    raise ValueError("Frozen conversation gold differs from original")
            run = {"manifest": manifest, "messages": predicted, "sessions": predicted_sessions}
            runs[method, seed] = run
            for source, category in TASKS:
                candidates = [r for r in records if _source(r) == source]
                observed = [r for r in candidates if bool(_value(r, "observed_mask", category))]
                threshold = _score(manifest["thresholds"][category])
                values = [_score(_value(predicted[str(r["record_id"])], "scores", category)) for r in observed]
                labels = [int(_value(r, "labels", category)) for r in observed]
                counts.append({"method": method, "seed": seed, "task": f"{source}.{category}",
                               "unit": "message", "threshold": threshold, "n_unknown": len(candidates)-len(observed),
                               **_counts(labels, values, threshold)})
            for name, score_key, threshold_key in (("conversation", "score", "conversation"),
                                                  ("conversation_common16", "common16_score", "conversation_common16")):
                threshold = _score(manifest["thresholds"][threshold_key])
                labels = [int(r["label"]) for r in sessions]
                values = [_score(predicted_sessions[str(r["session_id"])][score_key]) for r in sessions]
                counts.append({"method": method, "seed": seed, "task": "sccd." + name,
                               "unit": "conversation", "threshold": threshold, "n_unknown": 0,
                               **_counts(labels, values, threshold)})
            for path in (run_manifest_path, message_path, session_path):
                expected = frozen_hashes.get(str(path.relative_to(root)))
                if expected is None or _sha(path) != expected:
                    raise ValueError("Frozen independent-analysis inputs changed")
                hashes[str(path.relative_to(root))] = _sha(path)
    cst = runs["cst_v1_1", 42]
    review, flat = {}, []
    for task in (*[f"{source}.{category}" for source, category in TASKS], "sccd.conversation"):
        source, category = task.split(".", 1)
        unit = "conversation" if category == "conversation" else "message"
        topic_method = selected[task]["method"]
        if (topic_method, 42) not in runs:
            raise ValueError("Validation-selected topic baseline is not among complete runs")
        if unit == "message":
            observed = [row for row in records if _source(row) == source and bool(_value(row, "observed_mask", category))]
            lookup = {str(row["record_id"]): row for row in observed}
        else:
            lookup = original_sessions
        task_result = {"unit": unit, "selection": "two highest-score FP and two lowest-score FN; record_id resolves ties",
                       "selected_topic_baseline": topic_method, "n_gold": len(lookup), "FP": {}, "FN": {}}
        for error_type in ("FP", "FN"):
            candidates = []
            for rid, raw in lookup.items():
                comparison = _compare(cst, rid, category, unit=unit)
                label = int(raw["label"]) if unit == "conversation" else int(_value(raw, "labels", category))
                if (error_type == "FP" and label == 0 and comparison["prediction"] == 1) or (
                        error_type == "FN" and label == 1 and comparison["prediction"] == 0):
                    candidates.append((rid, raw, label, comparison))
            candidates.sort(key=lambda item: ((-item[3]["score"] if error_type == "FP" else item[3]["score"]), item[0]))
            entries = []
            for rank, (rid, raw, label, comparison) in enumerate(candidates[:2], 1):
                comparators = {method: _compare(runs[method, 42], rid, category, unit=unit)
                               for method in ("minilm_lr", topic_method)}
                if unit == "message":
                    text = raw["text"]
                    evidence = []
                    case = {"record_id": rid, "message_id": raw["message_id"], "session_id": raw["session_id"],
                            "parent_id": raw.get("parent_id"), "order": raw.get("order"), "source_ref": raw.get("source_ref")}
                else:
                    comment_rows = sorted((r for r in records if _source(r) == "sccd" and str(r["session_id"]) == rid
                                           and not r.get("is_fallback", False)),
                                          key=lambda r: (int(r.get("order", 0) or 0), str(r["message_id"]), str(r["record_id"])))
                    text = "\n".join(f"[{r['message_id']}] {r['text']}" for r in comment_rows)
                    evidence = _top3(rid, records, cst)
                    case = {"record_id": rid, "session_id": rid, "session_record": raw,
                            "full_text_definition": "chronological original comment bodies with original message IDs; not a new model input"}
                entry = {"rank": rank, "task": task, "unit": unit, "error_type": error_type,
                         **case, "gold_label": label, "gold_observed": True, **comparison,
                         "full_text": text, "comparators": comparators, "top3_comment_evidence": evidence}
                entries.append(entry)
                flat.append({"task": task, "unit": unit, "error_type": error_type, "rank": rank,
                             "record_id": rid, "message_id": case.get("message_id", "N/A"),
                             "session_id": case["session_id"], "gold_label": label, **comparison,
                             "full_text": text, "comparators_json": json.dumps(comparators, ensure_ascii=False),
                             "top3_comment_evidence_json": json.dumps(evidence, ensure_ascii=False)})
            task_result[error_type] = {"status": "available" if entries else "N/A",
                                      "reason": None if entries else ("no_observed_gold" if not lookup else "no_"+error_type),
                                      "total_error_count": len(candidates), "selected_count": len(entries), "cases": entries}
        review[task] = task_result
    destination = root / "errors"
    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "confusion_counts.csv", counts)
    _write_csv(destination / "cst_seed42_cases.csv", flat)
    payload = {"schema_version": 1, "status": "complete", "seed": 42, "cases": review,
               "methods": list(methods), "seeds": list(seeds),
               "selection_policy": "post-hoc deterministic error review, not a representative sample; no parameter changes",
               "interpretation": "Ordinary classification errors are not hallucinations. Session classification and comment evidence localization are separate. No cause is inferred from text alone.",
               "input_sha256": hashes, "error_review_source_sha256": _sha(Path(__file__))}
    (destination / "cst_seed42_cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    md = ["# CST种子42：可核验分类错误", "",
          "按固定规则列出每项任务最高分误报和最低分漏报各至多2条，同分按record_id排序。错误选择发生在全部冻结预测完成之后，不用于训练、阈值或配置选择；这些极端错误不代表总体错误频率。", "",
          "原文、ID、金标准和各方法分数见[完整JSON](cst_seed42_cases.json)及[完整CSV](cst_seed42_cases.csv)；[全部方法／种子的混淆计数](confusion_counts.csv)仅统计已标注标签。普通分类错误不称为幻觉，也不根据这些文本自行编造错误原因。"]
    for task, result in review.items():
        md += ["", "## " + task, "", f"验证集所选主题基线：{result['selected_topic_baseline']}。"]
        for error_type in ("FP", "FN"):
            group = result[error_type]
            md += ["", f"### {error_type}：共{group['total_error_count']}条，固定展示{group['selected_count']}条", ""]
            if not group["cases"]:
                md.append("N/A（" + group["reason"] + "）")
            for entry in group["cases"]:
                md.append(f"- ID `{entry['record_id']}`；金标准={entry['gold_label']}；CST分数={entry['score']:.6f}，阈值={entry['threshold']:.6f}，预测={entry['prediction']}。")
                md.append("  原文摘录：" + _excerpt(entry["full_text"]))
                for method, comparison in entry["comparators"].items():
                    md.append(f"  {method}：分数={comparison['score']:.6f}，阈值={comparison['threshold']:.6f}，预测={comparison['prediction']}。")
                if entry["unit"] == "conversation":
                    md.append("  对应真实评论Top-3（消息攻击分数排序；独立于会话分类是否正确）：")
                    for evidence in entry["top3_comment_evidence"]:
                        label = evidence["gold_comment_label"] if evidence["gold_observed"] else "N/A"
                        md.append(f"  - 消息 `{evidence['message_id']}`；原评论金标={label}；分数={evidence['score']:.6f}；{_excerpt(evidence['text'])}")
    (destination / "cst_seed42_cases.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_error_review(args.output)
    print(json.dumps({"status": result["status"], "tasks": len(result["cases"]),
                      "output": str(args.output / "errors")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
