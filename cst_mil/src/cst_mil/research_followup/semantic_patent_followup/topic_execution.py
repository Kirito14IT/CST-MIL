"""Frozen-case interpretation regression and anonymous assisted-review packets.

Reuses audited TRAIN/DEV representations for real cases, encodes authored
challenges independently, and never trains a model or opens the TEST bank.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from cst_mil.unified_semantic_v1_1 import topics as frozen_topics

from . import review, topics
from .common import (
    OUTPUT,
    digest,
    guard,
    identity,
    load_data,
    read,
    save,
    source_hashes,
    verify_hashes,
)
from .representations import embeddings_for, fixed_warmup, new_encoder

CONFIGURATIONS = ("frozen_baseline", *topics.VARIANTS)


def convert_messages(records: list[dict]) -> list[dict]:
    """Explicit display allowlist: labels, source names and filenames stay out."""
    messages = []
    for i, record in enumerate(records):
        source_ref = record.get("source_ref") or {}
        messages.append(
            {
                "post_id": str(record["session_id"]),
                "comment_id": str(record["message_id"]),
                "raw_text": str(record["text"]),
                "parent_id": record.get("parent_id"),
                "order": record.get("order", i),
                "comment_time": source_ref.get("comment_time", record.get("comment_time")),
                "speaker_id": source_ref.get("speaker_id", record.get("speaker_id")),
            }
        )
    return messages


def _array_identity(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    return identity(
        {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    )


def _source_fingerprint() -> dict[str, str]:
    return source_hashes(
        [
            Path(__file__),
            Path(topics.__file__),
            Path(review.__file__),
            Path(frozen_topics.__file__),
        ]
    )


def _interpret_case(
    case: dict,
    embeddings: np.ndarray,
    scores: np.ndarray,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    messages = convert_messages(case["messages"])
    expected_risks = scores.max(axis=1) if len(scores) else np.empty(0)
    outputs, audits, seconds = {}, {}, {}
    for configuration in CONFIGURATIONS:
        started = time.perf_counter()
        if configuration == "frozen_baseline":
            value = frozen_topics.interpret_topics(
                messages,
                embeddings,
                scores,
                distance_threshold=0.35,
                category_thresholds=thresholds,
            )
        else:
            value = topics.interpret_topics(
                messages,
                embeddings,
                scores,
                category_thresholds=thresholds,
                variant=configuration,
                distance_threshold=0.35,
            )
        seconds[configuration] = time.perf_counter() - started
        audit = review.machine_quality_audit(messages, value)
        actual_scores = np.asarray([row["category_scores"] for row in value["message_scores"]])
        actual_risks = np.asarray([row["risk_score"] for row in value["message_scores"]])
        score_unchanged = actual_scores.shape == scores.shape and np.array_equal(
            actual_scores, scores
        )
        risk_unchanged = np.array_equal(actual_risks, expected_risks)
        audit["risk_scores_unchanged"] = bool(score_unchanged and risk_unchanged)
        if not audit["risk_scores_unchanged"]:
            audit["passed"] = False
            audit["violations"].append({"kind": "risk_score_changed_in_interpretation"})
        outputs[configuration], audits[configuration] = value, audit

    def visible_identity(value: dict) -> str:
        return identity(
            {
                "topics": [
                    {key: topic[key] for key in ("title", "keywords", "message_ids")}
                    for topic in value["topics"]
                ],
                "assignments": [
                    {"id": item["comment_id"], "topic_ids": item["topic_ids"]}
                    for item in value["message_scores"]
                ],
            }
        )

    full_identity = visible_identity(outputs["full"])
    effects = {
        configuration: {
            "component_exists_in_full": True,
            "visible_output_changed_from_full": visible_identity(outputs[configuration])
            != full_identity,
            "no_change_interpretation": (
                "inactive_or_same_output_for_this_case_not_a_missing_component"
            ),
        }
        for configuration in topics.VARIANTS
        if configuration != "full"
    }
    return {
        "case_id": str(case["id"]),
        "kind": case["kind"],
        "messages": messages,
        "variants": outputs,
        "machine_quality": audits,
        "interpretation_seconds": seconds,
        "category_scores": scores.tolist(),
        "control_effects": effects,
        "semantic_accuracy": "not_available_without_human_theme_gold",
    }


def _aggregate(results: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    totals = {
        variant: {
            "messages": 0,
            "assigned": 0,
            "topics": 0,
            "keywords": 0,
            "supported_keywords": 0,
            "linked": 0,
            "machine_violations": 0,
            "seconds": 0.0,
        }
        for variant in CONFIGURATIONS
    }
    for result in results:
        for variant in CONFIGURATIONS:
            audit = result["machine_quality"][variant]
            timing = result["interpretation_seconds"][variant]
            rows.append(
                {
                    "case_id": result["case_id"],
                    "kind": result["kind"],
                    "configuration": variant,
                    "passed": audit["passed"],
                    "messages": audit["messages"],
                    "topics": audit["topics"],
                    "assigned_messages": audit["assigned_messages"],
                    "coverage": audit["coverage"],
                    "linked_messages": audit["linked_messages"],
                    "keyword_count": audit["keyword_count"],
                    "supported_keyword_count": audit["supported_keyword_count"],
                    "risk_scores_unchanged": audit["risk_scores_unchanged"],
                    "violation_count": len(audit["violations"]),
                    "interpretation_seconds": timing,
                }
            )
            total = totals[variant]
            for target, source in (
                ("messages", "messages"),
                ("assigned", "assigned_messages"),
                ("topics", "topics"),
                ("keywords", "keyword_count"),
                ("supported_keywords", "supported_keyword_count"),
                ("linked", "linked_messages"),
            ):
                total[target] += audit[source]
            total["machine_violations"] += len(audit["violations"])
            total["seconds"] += timing
    for total in totals.values():
        total["coverage"] = total["assigned"] / total["messages"] if total["messages"] else 0.0
        total["original_support_fraction"] = (
            total["supported_keywords"] / total["keywords"] if total["keywords"] else None
        )
    frozen, full = totals["frozen_baseline"], totals["full"]
    drop = frozen["coverage"] - full["coverage"]
    coverage_passed = drop <= 0.05 + 1e-12
    nontrivial = bool(full["topics"] and full["assigned"])
    hard_passed = all(total["machine_violations"] == 0 for total in totals.values())
    return rows, {
        "case_count": len(results),
        "configurations": totals,
        "coverage_gate": {
            "scope": "all_72_fixed_cases_message_weighted",
            "drop_from_frozen": drop,
            "maximum_drop": 0.05,
            "passed": coverage_passed,
            "nontrivial_output": nontrivial,
        },
        "hard_machine_checks_passed": hard_passed,
        "status": "passed" if hard_passed and coverage_passed and nontrivial else "failed",
        "evidence_type": "machine_grounding_and_model_assisted_not_human_gold",
        "theme_accuracy": "not_estimated",
        "risk_inheritance": "forbidden_and_checked",
        "controls": {
            variant: {
                "component_exists_in_full": True,
                "cases_with_visible_change": sum(
                    result["control_effects"][variant]["visible_output_changed_from_full"]
                    for result in results
                ),
                "unchanged_cases_do_not_establish_component_absence": True,
            }
            for variant in topics.VARIANTS
            if variant != "full"
        },
        "classification_and_interpretation": "separate_no_risk_score_changes",
        "score_source": "selected_seed42_model_same_scores_for_all_interpretation_variants",
        "old_case_limit": "regression_cases_already_reviewed",
        "new_case_limit": "DEV_cases_not_first_blind_test_and_with_frozen_dependency_groups",
        "cost_scope": (
            "cached-representation interpretation_only_encoder_and_risk_prediction_separate"
        ),
    }


def _review_packets(results: list[dict], folder: Path) -> dict:
    cases = [
        {
            "case_id": item["case_id"],
            "messages": item["messages"],
            "variants": {variant: item["variants"][variant] for variant in topics.VARIANTS},
        }
        for item in results
        if item["kind"] == "new_dev_assisted_review"
    ]
    manifest = review.export_review_packets(cases, folder / "packets", seed=42)
    save(folder / "packets_ready.json", manifest)
    print(
        f"TOPICS review packets ready: {len(cases)} cases, "
        f"{len(cases) * len(topics.VARIANTS)} assignments per reviewer",
        flush=True,
    )
    return manifest


def run_topics(output: str | Path = OUTPUT, package_path: str | Path | None = None) -> dict:
    output = Path(output)
    protocol = guard(output)
    selection_path, cases_path = output / "selection_freeze.json", output / "prepared/cases.json"
    selection, prepared = read(selection_path), read(cases_path)
    selected = selection["selected_architecture"]
    package_path = (
        Path(package_path)
        if package_path is not None
        else (output / "runs" / selected / "seed_42/model.joblib")
    )
    if not package_path.exists():
        raise FileNotFoundError(f"Selected seed42 model is not available: {package_path}")
    if len(prepared["old_cases"]) != 42 or len(prepared["new_cases"]) != 30:
        raise ValueError("The approved interpretation cohort requires 42 old and 30 new cases")
    cases = [*prepared["new_cases"], *prepared["old_cases"]]
    if len({case["id"] for case in cases}) != 72:
        raise ValueError("Fixed interpretation case IDs are not unique")
    real_records = {}
    for case in cases:
        for row in case["messages"]:
            if row["source"] == "authored_challenge":
                continue
            if row["split"] not in {"train", "dev"}:
                raise ValueError("Interpretation review cannot open TEST records")
            existing = real_records.setdefault(row["record_id"], row)
            if existing != row:
                raise ValueError("A repeated case record ID has different content")
    banks = [output / "raw_embeddings/dev/complete.json"]
    binding = {
        "protocol_identity": protocol["identity"],
        "selection_sha256": digest(selection_path),
        "cases_sha256": digest(cases_path),
        "package_path": str(package_path.resolve()),
        "package_sha256": digest(package_path),
        "selected_architecture": selected,
        "code_hashes": _source_fingerprint(),
        "bank_receipts": source_hashes(banks),
        "configurations": list(CONFIGURATIONS),
        "distance": 0.35,
        "review_seed": 42,
        "human_gold": False,
        "case_order": [str(case["id"]) for case in cases],
    }
    binding["identity"] = identity(binding)
    folder = output / "topics"
    freeze_path = folder / "topic_freeze.json"
    if freeze_path.exists() and read(freeze_path) != binding:
        raise RuntimeError(
            "Topic execution identity changed; preserve old results and use a revision"
        )
    save(freeze_path, binding)
    if (folder / "complete.json").exists():
        complete = read(folder / "complete.json")
        if complete["identity"] != binding["identity"]:
            raise RuntimeError("Completed topic cache has a different identity")
        verify_hashes(complete["artifacts"])
        print("TOPICS accepted completed cache reused", flush=True)
        return read(folder / "summary.json")
    package = joblib.load(package_path)
    if int(package.seed) != 42:
        raise ValueError("Interpretation review uses selected seed42 package only")
    if hasattr(package, "config") and package.config.architecture_base != selected:
        raise ValueError("Selected package architecture does not match frozen choice")
    thresholds = dict(package.thresholds)
    current = list(real_records.values())
    values = embeddings_for(current, output)
    lookup = {row["record_id"]: values[i] for i, row in enumerate(current)}
    encoder, warmup = None, None
    results, case_receipts = [], []
    for number, case in enumerate(cases, 1):
        verify_hashes(binding["code_hashes"])
        records = case["messages"]
        authored = all(row["source"] == "authored_challenge" for row in records)
        if any(row["source"] == "authored_challenge" for row in records) and not authored:
            raise ValueError("Authored and real messages cannot be mixed into one review case")
        cache_path = folder / "cases" / f"case_{number:03d}.json"
        receipt_path = cache_path.with_suffix(".receipt.json")
        case_identity = identity({"topic_identity": binding["identity"], "case": case})
        if cache_path.exists():
            if not receipt_path.exists():
                raise RuntimeError("Unsealed topic case cache cannot be silently resumed")
            receipt = read(receipt_path)
            if receipt.get("case_identity") != case_identity or receipt.get("sha256") != digest(
                cache_path
            ):
                raise RuntimeError("Topic case receipt or content fingerprint changed")
            payload = read(cache_path)
            if payload.get("case_identity") != case_identity:
                raise RuntimeError("Saved topic case belongs to a different identity")
            results.append(payload)
            case_receipts.append(
                {"case_id": str(case["id"]), "path": str(cache_path), "sha256": digest(cache_path)}
            )
            print(f"TOPICS {number}/72 reuse {case['id']}", flush=True)
        else:
            started = time.perf_counter()
            if authored:
                if encoder is None:
                    encoder = new_encoder(output)
                    warmup = fixed_warmup(encoder, load_data(("train",))[0])
                    save(folder / "challenge_encoder_warmup.json", warmup)
                embeddings = encoder.encode([str(row["text"]) for row in records], batch_size=32)
            else:
                embeddings = np.asarray(
                    [lookup[row["record_id"]] for row in records], dtype=np.float32
                )
            encoding_seconds = time.perf_counter() - started
            started = time.perf_counter()
            scores, _ = package.score_records(records, embeddings)
            prediction_seconds = time.perf_counter() - started
            scores = np.asarray(scores, dtype=float)
            payload = _interpret_case(case, embeddings, scores, thresholds)
            payload.update(
                case_identity=case_identity,
                topic_identity=binding["identity"],
                embeddings_identity=_array_identity(embeddings),
                scores_identity=_array_identity(scores),
                representation_source=(
                    "authored_case_raw_encoding" if authored else "audited_train_dev_bank"
                ),
                representation_preparation_seconds=encoding_seconds,
                category_prediction_seconds=prediction_seconds,
                eval_groups=case.get("eval_groups", []),
            )
            save(cache_path, payload)
            save(receipt_path, {"case_identity": case_identity, "sha256": digest(cache_path)})
            results.append(payload)
            case_receipts.append(
                {"case_id": str(case["id"]), "path": str(cache_path), "sha256": digest(cache_path)}
            )
            print(
                f"TOPICS {number}/72 complete {case['id']} "
                f"full={len(payload['variants']['full']['topics'])} topics "
                f"coverage={payload['variants']['full']['coverage']:.3f}",
                flush=True,
            )
        if number == 30:
            _review_packets(results, folder)
    rows, summary = _aggregate(results)
    summary.update(
        identity=binding["identity"],
        selected_architecture=selected,
        seed=42,
        case_receipts=case_receipts,
        review_packets=read(folder / "packets_ready.json"),
        authored_encoding_seconds=sum(
            r["representation_preparation_seconds"]
            for r in results
            if r["representation_source"] == "authored_case_raw_encoding"
        ),
        risk_prediction_seconds=sum(r["category_prediction_seconds"] for r in results),
    )
    csv_path = folder / "machine_quality.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(folder / "summary.json", summary)
    artifacts = [freeze_path, csv_path, folder / "summary.json", folder / "packets_ready.json"]
    artifacts += [Path(item["path"]) for item in case_receipts]
    artifacts += [Path(item["path"]).with_suffix(".receipt.json") for item in case_receipts]
    artifacts += [p for p in (folder / "packets").rglob("*") if p.is_file()]
    if (folder / "challenge_encoder_warmup.json").exists():
        artifacts.append(folder / "challenge_encoder_warmup.json")
    verify_hashes(binding["code_hashes"])
    complete = {
        "status": summary["status"],
        "identity": binding["identity"],
        "artifacts": source_hashes(artifacts),
    }
    save(folder / "complete.json", complete)
    print(
        f"TOPICS complete: {summary['status']}; "
        f"coverage drop={summary['coverage_gate']['drop_from_frozen']:.6f}",
        flush=True,
    )
    return summary
