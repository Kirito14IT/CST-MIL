"""Light provenance/role tests for the independent raw evaluation audit."""

import copy
import csv

import numpy as np
import pytest

from cst_mil.research_followup.full_joint_comparison import raw_replay_verification as audit
from cst_mil.research_followup.full_joint_comparison.common import digest, identity, save


def view_fixture(tmp_path, method="bertopic"):
    base = tmp_path / "base_run"
    base.mkdir()
    (base / "model.joblib").write_bytes(b"sealed-weight-bytes-not-a-loadable-model")
    original = {
        "identity": "original-training",
        "method": method,
        "seed": 42,
        "model_sha256": digest(base / "model.joblib"),
        "model_bytes": 44,
        "thresholds": dict(
            attack=0.5, hate=0.4, spam_scam=0.6, conversation=0.7, conversation_common16=0.65
        ),
        "training_seconds_cached": 12.3,
        "parameters": {"seed": 42},
    }
    save(base / "run_manifest.json", original)
    save(base / "train_state.json", {"identity": original["identity"], "status": "complete"})
    view = {
        **copy.deepcopy(original),
        "manifest_role": "evaluation_view",
        "base_run_identity": original["identity"],
        "inherited_training": True,
        "base_run_manifest_sha256": digest(base / "run_manifest.json"),
        "base_train_state_sha256": digest(base / "train_state.json"),
        "base_model_path": str(base / "model.joblib"),
        "evaluation_revision_identity": "revision-identity",
    }
    return base, original, view


def test_view_accepts_threshold_only_revision_and_keeps_original_weights(tmp_path):
    base, original, view = view_fixture(tmp_path)
    view["thresholds"]["attack"] = 0.61
    proof = audit.verify_evaluation_view(view, original, base, "revision-identity")
    assert proof["weights_unchanged"] and proof["base_train_identity"] == original["identity"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("manifest_role", "training"),
        ("inherited_training", False),
        ("base_train_state_sha256", "invented"),
        ("model_sha256", "different-weights"),
        ("training_seconds_cached", 1.0),
        ("seed", 43),
    ],
)
def test_view_rejects_fake_training_or_changed_provenance(tmp_path, field, value):
    base, original, view = view_fixture(tmp_path)
    view[field] = value
    with pytest.raises(ValueError):
        audit.verify_evaluation_view(view, original, base, "revision-identity")


def test_inherited_traditional_thresholds_cannot_be_reselected(tmp_path):
    base, original, view = view_fixture(tmp_path, "lda")
    view["thresholds"]["attack"] = 0.61
    with pytest.raises(ValueError, match="traditional thresholds"):
        audit.verify_evaluation_view(view, original, base, "revision-identity")


def stage_fixture(tmp_path, status="complete"):
    folder = tmp_path / "revision_run"
    folder.mkdir()
    save(folder / "run_manifest.json", {"manifest_role": "evaluation_view"})
    files = [folder / f"predictions_{unit}_dev.jsonl" for unit in ("message", "session")]
    for path in files:
        path.write_text("{}\n", encoding="utf-8")
    expected = identity(
        {
            "revision": "revision-identity",
            "method": "bertopic",
            "seed": 42,
            "stage": "dev",
            "run_manifest_sha256": digest(folder / "run_manifest.json"),
        }
    )
    save(
        folder / "dev_state.json",
        {
            "identity": expected,
            "status": status,
            "artifacts": {str(p.resolve()): digest(p) for p in files},
        },
    )
    return folder


def test_stage_seal_is_evaluation_identity_and_checks_prediction_hashes(tmp_path):
    folder = stage_fixture(tmp_path)
    assert (
        audit.verify_revision_stage(folder, "revision-identity", "bertopic", 42, "dev")["status"]
        == "complete"
    )
    (folder / "predictions_message_dev.jsonl").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen files changed"):
        audit.verify_revision_stage(folder, "revision-identity", "bertopic", 42, "dev")


def test_started_stage_is_incomplete_not_passed_or_fake_train_seal(tmp_path):
    folder = stage_fixture(tmp_path, "started")
    with pytest.raises(audit.IncompleteAudit):
        audit.verify_revision_stage(folder, "revision-identity", "bertopic", 42, "dev")
    assert not (folder / "train_state.json").exists()


def test_baseline_selection_uses_three_seed_validation_average_and_fixed_ties():
    methods = ["lda", "btm", "gsdmm", "nmf", "bertopic", "cst_v1_1", "minilm_lr"]
    seeds = [42, 43, 44]
    values = {}
    for method in methods:
        for seed in seeds:
            values[method, seed, "dev", "sccd.attack"] = {
                "AP": 0.8 if method in {"lda", "btm"} else 0.7
            }
            values[method, seed, "test", "sccd.attack"] = {
                "AP": 1.0 if method == "bertopic" else 0.1
            }
    result = audit.expected_selected_baselines(values, methods, seeds)
    assert result["sccd.attack"]["method"] == "lda"
    assert result["sccd.attack"]["validation_mean_AP"] == pytest.approx(0.8)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_metric_tables_accept_exact_recomputation_and_reject_changed_test_metric(tmp_path):
    folder = tmp_path / "analysis"
    folder.mkdir()
    value = audit.independent.independent_metrics([0, 1, 1, 0], [0.1, 0.9, 0.7, 0.3], 0.5)
    key = ("cst_v1_1", 42, "test", "sccd.attack")
    expected = {key: value}
    strata = {(*key, "le_30_chars"): value}
    top3 = {("cst_v1_1", 42, "test"): {"n_sessions": 1, "hit_at_3": 1.0}}
    row = dict(method=key[0], seed=key[1], split=key[2], task=key[3], **value)
    write_csv(folder / "per_seed_metrics.csv", [row])
    write_csv(folder / "length_strata.csv", [{**row, "stratum": "le_30_chars"}])
    write_csv(folder / "top3.csv", [dict(method=key[0], seed=42, split="test", **top3[key[:3]])])
    assert audit.verify_metric_tables(tmp_path, expected, strata, top3)["max_abs_difference"] == 0
    row["AP"] = 0.2
    write_csv(folder / "per_seed_metrics.csv", [row])
    with pytest.raises(ValueError, match="Metric mismatch"):
        audit.verify_metric_tables(tmp_path, expected, strata, top3)


@pytest.mark.parametrize(
    "labels,scores",
    [
        ([0, 1], [0.5, 0.5]),
        ([0, 1], [0.0, 1.0]),
        ([0, 0], [0.0, 0.0]),
        ([1, 1], [1.0, 1.0]),
    ],
)
def test_revision_threshold_vectorized_scan_matches_separate_pointer_scan(labels, scores):
    result = audit.independent_revision_threshold(labels, scores)
    reference = audit.independent.independent_dev_threshold(labels, scores)
    assert result["threshold"] == reference["threshold"]
    assert result["macro_f1"] == reference["macro_f1"]
    assert result["candidate_count"] == reference["candidate_count"]


def test_revision_threshold_512_tied_samples_and_no_observed_branch():
    rng = np.random.default_rng(44)
    labels, scores = rng.integers(0, 2, 512), rng.integers(0, 10, 512) / 9
    result = audit.independent_revision_threshold(labels, scores)
    reference = audit.independent.independent_dev_threshold(labels, scores)
    assert result["threshold"] == reference["threshold"]
    assert result["macro_f1"] == reference["macro_f1"]
    with pytest.raises(audit.IncompleteAudit):
        audit.independent_revision_threshold([], [])


def test_same_label_independent_train_warmup_ids_are_required():
    records = [
        {"record_id": f"conda:{i:03}", "source": "conda", "split": "train", "text": f"body{i}"}
        for i in range(70)
    ]
    chosen = records[:64]
    ids = [r["record_id"] for r in chosen]
    protocol = {
        "warmup": {
            "record_ids": ids,
            "text_binding": identity(
                [{"record_id": r["record_id"], "text_sha256": identity(r["text"])} for r in chosen]
            ),
        }
    }
    startup = {
        "warmup_record_ids": ids,
        "warmup_split": "train",
        "warmup_source": "conda",
        "warmup_repetitions": 1,
        "warmup_predictions_discarded": True,
    }
    assert audit.verify_warmup(startup, records, protocol)["warmup_count"] == 64
    startup["warmup_split"] = "test"
    with pytest.raises(ValueError, match="Warm-up"):
        audit.verify_warmup(startup, records, protocol)


def test_cost_coverage_requires_full_test_three_repetitions_and_cpu_policy():
    records = [{"source": "sccd", "split": "test"}] * 2 + [{"source": "conda", "split": "test"}] * 3
    rows = [
        {"repetition": i, "source": source, "n_records": n, "seconds": seconds}
        for i in (1, 2, 3)
        for source, n, seconds in (("sccd", 2, 2.0), ("conda", 3, 1.0))
    ]
    cost = {
        "status": "complete",
        "method": "bertopic",
        "seed": 42,
        "repetitions": 3,
        "warm_total_seconds": [3.0, 3.0, 3.0],
        "GPU_required": False,
        "paid_API_calls": 0,
        "resources": {"peak_affinity_width": 4},
        "rows": rows,
        "prediction_consistency": {"max_message_abs_diff": 0.0, "max_session_abs_diff": 0.0},
    }
    cost["prediction_consistency"].update(
        head_threshold_disagreements_all_repetitions=0,
        session_threshold_disagreements_all_repetitions={"main": 0, "common16": 0},
    )
    assert audit.verify_cost_coverage(cost, records, "bertopic", 4)["full_test_coverage"]
    cost["rows"] = rows[:-1]
    with pytest.raises(ValueError, match="skipped or duplicated"):
        audit.verify_cost_coverage(cost, records, "bertopic", 4)
    cost["rows"] = rows
    cost["resources"]["peak_affinity_width"] = 8
    with pytest.raises(ValueError, match="CPU"):
        audit.verify_cost_coverage(cost, records, "bertopic", 4)


@pytest.mark.parametrize(
    "method,change,reason",
    [
        ("lda", {"max_message_abs_diff": 1e-8}, "exact score"),
        ("bertopic", {"head_threshold_disagreements_all_repetitions": 1}, "head classifications"),
        (
            "bertopic",
            {"session_threshold_disagreements_all_repetitions": {"main": 1, "common16": 0}},
            "conversation classifications",
        ),
    ],
)
def test_cost_strictly_rejects_inherited_score_drift_and_any_classification_flip(
    method, change, reason
):
    records = [{"source": "conda", "split": "test"}]
    cost = {
        "status": "complete",
        "method": method,
        "seed": 42,
        "repetitions": 3,
        "warm_total_seconds": [1.0, 1.0, 1.0],
        "GPU_required": False,
        "paid_API_calls": 0,
        "resources": {"peak_affinity_width": 4},
        "rows": [
            {"repetition": i, "source": "conda", "n_records": 1, "seconds": 1.0} for i in (1, 2, 3)
        ],
        "prediction_consistency": {
            "max_message_abs_diff": 0.0,
            "max_session_abs_diff": 0.0,
            "head_threshold_disagreements_all_repetitions": 0,
            "session_threshold_disagreements_all_repetitions": {"main": 0, "common16": 0},
            **change,
        },
    }
    with pytest.raises(ValueError, match=reason):
        audit.verify_cost_coverage(cost, records, method, 4)


def test_archived_same_revision_consistency_failure_forbids_acceptance(tmp_path):
    path = tmp_path / "attempts" / "attempt_001" / "cost_invalid.json"
    save(
        path,
        {
            "status": "invalid_prediction_consistency",
            "evaluation_revision_identity": "same-revision",
        },
    )
    with pytest.raises(ValueError, match="forbids retry"):
        audit.verify_no_consistency_retry(tmp_path, "same-revision")
    assert (
        audit.verify_no_consistency_retry(tmp_path, "other-revision")[
            "same_revision_invalid_consistency_retry"
        ]
        is False
    )


def test_raw_dev_preparation_cost_is_reported_separately_with_source_hash(tmp_path):
    provenance = {}
    for method in ("lda", "bertopic"):
        path = tmp_path / "runs" / method / "seed_42/dev_provenance.json"
        payload = (
            {"kind": "inherited_sealed_dev"}
            if method == "lda"
            else {
                "kind": "semantic_raw_dev",
                "replay_seconds": 20.0,
                "startup": {"model_load_seconds": 2.0, "startup_warmup_seconds": 3.0},
            }
        )
        save(path, payload)
        provenance[str(path.relative_to(tmp_path))] = digest(path)
    rows = [
        {
            "method": method,
            "raw_dev_inference_preparation_seconds": seconds,
            "raw_dev_inference_preparation_definition": (
                "excludes threshold scanning and audit; not included in old training timings"
            ),
        }
        for method, seconds in (("lda", 0.0), ("bertopic", 25.0))
    ]
    save(tmp_path / "analysis/analysis.json", {"costs": rows})
    save(tmp_path / "analysis/verification.json", {"prediction_and_manifest_hashes": provenance})
    write_csv(tmp_path / "analysis/costs.csv", rows)
    proof = audit.verify_raw_dev_preparation_costs(tmp_path, ["lda", "bertopic"])
    assert proof["raw_dev_inference_preparation_seconds"] == {"lda": 0.0, "bertopic": 25.0}
    rows[1]["raw_dev_inference_preparation_seconds"] = 20.0
    write_csv(tmp_path / "analysis/costs.csv", rows)
    with pytest.raises(ValueError, match="Metric mismatch"):
        audit.verify_raw_dev_preparation_costs(tmp_path, ["lda", "bertopic"])


def test_uninitialized_raw_revision_audit_is_incomplete_and_writes_only_audit(tmp_path):
    result = audit.audit(tmp_path, partial=True)
    assert result["status"] == "incomplete" and result["completed_runs_audited"] == 0
    assert result["evaluation_revision"] == tmp_path.name
    assert {p.name for p in tmp_path.iterdir()} == {"completion_audit.json"}


@pytest.mark.parametrize(
    "filename,revision_name",
    [
        ("raw_replay.py", "raw_replay_v1"),
        ("raw_replay_v2.py", "raw_replay_v2"),
        ("raw_replay_v37.py", "raw_replay_v37"),
    ],
)
def test_replay_execution_source_is_resolved_from_manifest_not_hardcoded_v1(
    tmp_path, filename, revision_name
):
    path = tmp_path / filename
    path.write_text("# sealed revision execution source\n", encoding="utf-8")
    manifest = {"code_hashes": {str(path.resolve()): digest(path)}}
    result = audit.replay_source_metadata(manifest, module_dir=tmp_path)
    assert result["evaluation_revision"] == revision_name
    assert result["execution_source_path"] == str(path.resolve())
    assert result["execution_source_sha256"] == digest(path)


def test_replay_source_rejects_ambiguous_versions_and_changed_hash(tmp_path):
    paths = [tmp_path / name for name in ("raw_replay.py", "raw_replay_v2.py")]
    for path in paths:
        path.write_text("# immutable source\n", encoding="utf-8")
    manifest = {"code_hashes": {str(path.resolve()): digest(path) for path in paths}}
    with pytest.raises(ValueError, match="exactly one"):
        audit.replay_source_metadata(manifest, module_dir=tmp_path)
    manifest["code_hashes"].pop(str(paths[0].resolve()))
    paths[1].write_text("# changed source\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        audit.replay_source_metadata(manifest, module_dir=tmp_path)
