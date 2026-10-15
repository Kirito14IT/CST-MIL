"""Final artifact, leakage, old-file preservation and disclosure acceptance audit."""

from __future__ import annotations

import csv
import re
import zipfile
from pathlib import Path

from .common import (
    ABLATIONS,
    BASE,
    MODULE,
    OUTPUT,
    REPORT,
    REVISION,
    SEEDS,
    digest,
    guard,
    load_data,
    preservation,
    read,
    read_jsonl,
    save,
    source_hashes,
    verify_hashes,
)
from .execution import folder_for, selected


def train_cache_guard(output: Path = OUTPUT) -> dict:
    before = read(output / "preservation_before.json")
    files = [
        BASE / "cache/train" / name for name in ("complete.json", "binding.json", "embeddings.npy")
    ]
    mapping = {str(p.resolve()): before[str(p.resolve())] for p in files}
    verify_hashes(mapping)
    return {"status": "passed", "cache_artifacts": mapping}


def verify(output: Path = OUTPUT) -> dict:
    guard(output)
    choice = selected(output)
    cache = train_cache_guard(output)
    checks, failures = [], []

    def check(name: str, passed: bool, details=None):
        checks.append({"check": name, "passed": bool(passed), "details": details})
        if not passed:
            failures.append(name)

    records, sessions = load_data()
    expected = {r["record_id"] for r in records if r["split"] == "test"}
    expected_sessions = {s["session_id"] for s in sessions if s["split"] == "test"}
    fit_count = 0
    embedding_identities = set()
    for variant in ("full", "context_a", "context_b", *ABLATIONS):
        for seed in SEEDS:
            folder = folder_for(output, variant, seed)
            verify_hashes(read(folder / "complete.json")["artifacts"])
            details = read(folder / "fit_details.json")
            manifest = read(folder / "manifest.json")
            if variant != "full":
                fit_count += 1
                audit = details["fit_audit"]
                embedding_identities.add(audit["training_embedding_sha256"])
                check(f"no_TEST_fit.{variant}.{seed}", audit["test_records_used_for_fit"] == 0)
                check(f"three_folds.{variant}.{seed}", len(audit["folds"]) == 3)
                check(f"seed.{variant}.{seed}", all(f["seed"] == seed for f in audit["folds"]))
            if variant in {"full", choice["selected_architecture"], *ABLATIONS}:
                verify_hashes(read(folder / "test_complete.json")["artifacts"])
                messages = read_jsonl(folder / "predictions_message_test.jsonl")
                bags = read_jsonl(folder / "predictions_session_test.jsonl")
                check(
                    f"complete_TEST_messages.{variant}.{seed}",
                    len(messages) == len(expected)
                    and {r["record_id"] for r in messages} == expected,
                )
                check(
                    f"complete_TEST_sessions.{variant}.{seed}",
                    len(bags) == len(expected_sessions)
                    and {r["session_id"] for r in bags} == expected_sessions,
                )
            check(f"CPU_cap.{variant}.{seed}", manifest["resources"]["peak_affinity_width"] <= 4)
    check("new_fit_configuration_seed_cap", fit_count <= 24, fit_count)
    check(
        "same_training_representations",
        len(embedding_identities) == 1,
        sorted(embedding_identities),
    )

    # Independently cover both conversation heads, beyond the initial import guard.
    for seed in SEEDS:
        current = read_jsonl(folder_for(output, "full", seed) / "predictions_session_dev.jsonl")
        old = {
            r["session_id"]: r
            for r in read_jsonl(
                REVISION / "runs/cst_v1_1" / f"seed_{seed}/predictions_session_dev.jsonl"
            )
        }
        diffs, flips = [], 0
        for row in current:
            previous = old[row["session_id"]]
            for score, threshold in (
                ("score", "threshold"),
                ("common16_score", "common16_threshold"),
            ):
                diffs.append(abs(row[score] - previous[score]))
                flips += (row[score] >= row[threshold]) != (previous[score] >= previous[threshold])
        check(
            f"full_raw_DEV_both_conversation_heads.{seed}",
            max(diffs) <= 1e-4 and flips == 0,
            {"max_absolute_difference": max(diffs), "decision_flips": int(flips)},
        )

    analysis = read(output / "analysis/summary.json")
    check("independent_metric_analysis_exists", bool(analysis))
    analysis_binding = read(output / "analysis/identity.json")
    verify_hashes(analysis_binding["input_sha256"])
    check(
        "analysis_source_unchanged",
        digest(MODULE / "analysis.py") == analysis_binding["analysis_source_sha256"],
    )
    verify_hashes(
        {str(output / "analysis" / name): sha for name, sha in analysis["artifacts"].items()}
    )
    with (output / "analysis/ablation_effects.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        effects = list(csv.DictReader(handle))
    check("all_ablation_effects_retained", len(effects) == 240, len(effects))
    topic = read(output / "topics/summary.json")
    verify_hashes(read(output / "topics/topic_freeze.json")["code_hashes"])
    verify_hashes(read(output / "topics/complete.json")["artifacts"])
    check("72_fixed_cases", topic["case_count"] == 72)
    check("topic_hard_checks", topic["hard_machine_checks_passed"])
    check(
        "topic_coverage_gate",
        topic["coverage_gate"]["passed"] and topic["coverage_gate"]["nontrivial_output"],
        topic["coverage_gate"],
    )
    reviews = read(output / "topics/assisted_review_summary.json")
    verify_hashes(reviews["input_hashes"])
    check(
        "two_complete_anonymous_assisted_reviews",
        reviews["case_configuration_pairs"] == 150
        and len(reviews["review_records"]) == 2
        and all(len(r) == 150 for r in reviews["review_records"]),
    )
    check("review_not_human_gold", reviews["evidence_type"] == "model_assisted_not_human_gold")
    cost = read(output / "cost/summary.json")
    cost_freeze = read(output / "cost/freeze.json")
    verify_hashes(cost_freeze["code_hashes"])
    verify_hashes(cost_freeze["reference_artifacts"])
    check(
        "cost_reference_seal_unchanged",
        cost_freeze["reference_seal_sha256"]
        == digest(folder_for(output, choice["selected_architecture"], 42) / "test_complete.json"),
    )
    for result in cost["raw_repeats"]:
        verify_hashes(result["artifacts"])
    check(
        "three_raw_resident_passes",
        len(cost["raw_repeats"]) == 3 and all(r["status"] == "passed" for r in cost["raw_repeats"]),
    )
    bundle = read(output / "final_model_bundle/manifest.json")
    verify_hashes(bundle["code_hashes"])
    verify_hashes(bundle["evidence_hashes"])
    check(
        "one_final_seed42_bundle",
        bundle["seed"] == 42
        and digest(output / "final_model_bundle/model.joblib") == bundle["model_sha256"]
        and read(output / "final_model_bundle/explanation_config.json")
        == bundle["explanation_config"],
    )
    crossfit = read(output / "qa/final_crossfit_reuse_audit.json")
    verify_hashes({r["OOF_reference"]: r["OOF_sha256"] for r in crossfit["rows"]})
    check("matching_final_OOF_provenance", crossfit["status"] == "passed")

    delivered = read(REPORT / "交付文件.json")
    docx, md, pdf = (REPORT / delivered[k] for k in ("docx", "md", "pdf"))
    check("formal_outputs_exist", all(p.is_file() for p in (docx, md, pdf)))
    formal = md.read_text(encoding="utf-8-sig")
    banned = re.findall(
        r"(?i)(?<![A-Za-z0-9])r0(?![A-Za-z0-9])|v1\.1|候选[AB]|TO CONFIRM|待补|草稿|占位", formal
    )
    check("no_internal_versions_or_placeholders_in_formal_MD", not banned, banned)
    with zipfile.ZipFile(docx) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    check("editable_Word_math", xml.count("<m:oMath>") >= 10, xml.count("<m:oMath>"))
    document_qa = read(REPORT / "qa/文档自动核验.json")
    check("document_QA_matches_final_PDF", digest(pdf) == document_qa["sha256"])
    check(
        "PDF_fonts_layout_and_formal_hygiene",
        document_qa["all_fonts_embedded"]
        and document_qa["replacement_characters"] == 0
        and document_qa["outside_page_text"] == 0
        and not document_qa["banned_internal_terms"],
    )
    protection = preservation(output, verify=True)
    result = {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "selected_architecture": choice["selected_architecture"],
        "new_configuration_seed_fits": fit_count,
        "train_cache": cache,
        "preservation": protection,
        "paid_API_calls": 0,
        "followup_code_hashes": source_hashes(MODULE.glob("*.py")),
        "artifacts": source_hashes(
            (
                md,
                docx,
                pdf,
                output / "selection_freeze.json",
                output / "analysis/summary.json",
                output / "topics/summary.json",
                output / "topics/assisted_review_summary.json",
                output / "cost/summary.json",
                output / "final_model_bundle/manifest.json",
            )
        ),
    }
    save(output / "completion_audit.json", result)
    if failures:
        raise RuntimeError(f"Final audit failed: {failures}")
    return result
