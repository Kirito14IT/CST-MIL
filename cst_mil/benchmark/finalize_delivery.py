"""Read-only experiment audit plus a generated delivery readiness certificate."""

from __future__ import annotations

import json
import subprocess
import sys
import time

from cst_mil.bench.common import file_hash, fingerprint, read_json, write_json
from cst_mil.bench.dataset import BENCHMARK, PROJECT, load_suite
from cst_mil.bench.runner import METHODS, functional_checks_passed, run_path, runtime_fingerprint


def main() -> None:
    assert functional_checks_passed(), "Functional evidence differs from current runtime"
    runtime = runtime_fingerprint()
    selected = read_json(BENCHMARK / "selection.json")
    assert selected["test_used_for_selection"] is False
    assert 1 <= len(selected["history"]) <= 4
    full_manifest, full, full_messages = load_suite("full677")
    _, pilot, pilot_messages = load_suite("pilot300")
    assert (len(full), len(full_messages), len(pilot), len(pilot_messages)) == (
        677,
        38872,
        300,
        17312,
    )
    full_lookup = {s["post_id"]: s for s in full}
    assert all(full_lookup[s["post_id"]]["split"] == s["split"] for s in pilot)
    assert sum(s["fresh_test"] for s in full) == 76
    assert not list((BENCHMARK / "runs/full677").glob("*/TRAINED.json")), (
        "User-only full training has already started; this is not a pre-launch handoff"
    )
    evidence = []
    for method in METHODS:
        revision = selected["revision"] if method == "cst_mil" else None
        suffix = revision or "baseline"
        proof_path = BENCHMARK / "verification" / f"{method}_{suffix}.json"
        proof = read_json(proof_path)
        smoke = run_path("smoke5", method, revision)
        saved = read_json(smoke / "run_manifest.json")
        assert fingerprint(saved["binding"]["runtime"]) == fingerprint(runtime)
        assert proof["passed"] and proof["source_sessions"] == 5
        assert proof["labels_topics_evidence_match"] and proof["source_full_partition"] == "train"
        assert [p["status"]["completed"] for p in proof["partial_resume_checkpoints"]] == [2, 5, 5]
        assert {p["model_sha256"] for p in proof["partial_resume_checkpoints"]} == {
            file_hash(smoke / "model.joblib")
        }
        formal = run_path("pilot300", method, revision)
        trained = read_json(formal / "TRAINED.json")
        assert trained["cpu_policy"]["affinity_enforced"]
        assert trained["cpu_policy"]["assigned_cpus"] == [0, 1, 2, 3]
        assert trained["training_peak_affinity_width"] <= 4
        assert read_json(formal / "status.json")["completed"] == 300
        assert read_json(formal / "metrics.json")["test"]["n_sessions"] == 60
        evidence.append(
            {
                "method": method,
                "revision": revision,
                "verification": str(proof_path),
                "smoke_passed": True,
                "pilot_run": str(formal),
                "formal_fit_attempts": read_json(formal / "TRAINED.json")["formal_fit_attempts"],
            }
        )
    for item in selected["history"]:
        formal = run_path("pilot300", "cst_mil", item["revision"])
        assert read_json(formal / "status.json")["completed"] == 300
    iteration_audit_path = BENCHMARK / "verification/iteration_audit.json"
    iteration_audit = read_json(iteration_audit_path)
    assert iteration_audit["passed"]
    assert all(check["passed"] for check in iteration_audit["checks"])
    for relative_path, expected_hash in iteration_audit["evidence_sha256"].items():
        assert file_hash(PROJECT / relative_path) == expected_hash, relative_path
    command_audit_path = BENCHMARK / "verification/command_audit.json"
    command_audit = read_json(command_audit_path)
    assert command_audit["passed"]
    assert all(check["passed"] for check in command_audit["checks"])
    assert len(command_audit["commands"]) == len(METHODS)
    for relative_path, expected_hash in command_audit["evidence_sha256"].items():
        assert file_hash(PROJECT / relative_path) == expected_hash, relative_path
    comparison = BENCHMARK / "comparison/pilot300/analysis-report.md"
    assert comparison.is_file()
    result = subprocess.run(
        [sys.executable, str(BENCHMARK / "preserve_legacy.py")],
        cwd=PROJECT,
        capture_output=True,
        text=True,
        check=True,
    )
    protection = json.loads(result.stdout)
    assert not protection["mismatches"]
    payload = {
        "ready": True,
        "created_at_unix": time.time(),
        "selected_cst_revision": selected["revision"],
        "methods": evidence,
        "full677_formal_training_started": False,
        "full677_content_fingerprint": full_manifest["content_fingerprint"],
        "runtime_sha256": fingerprint(runtime),
        "legacy_protection": protection,
        "functional_tests": read_json(BENCHMARK / "functional_checks.json")["counts"],
        "iteration_audit": {
            "path": str(iteration_audit_path),
            "checks": len(iteration_audit["checks"]),
            "evidence_files": len(iteration_audit["evidence_sha256"]),
            "passed": True,
        },
        "command_audit": {
            "path": str(command_audit_path),
            "checks": len(command_audit["checks"]),
            "commands": len(command_audit["commands"]),
            "passed": True,
        },
        "comparison": str(comparison),
        "guide": str(BENCHMARK / "操作指南.md"),
    }
    write_json(BENCHMARK / "READY.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
