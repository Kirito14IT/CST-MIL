"""Frozen all-split dependence is reused, never rebuilt from scored TEST rows."""

import copy
import hashlib
import json

import numpy as np
import pytest

from cst_mil.research_followup.full_joint_comparison import analysis
from cst_mil.research_followup.full_joint_comparison import raw_replay_verification as audit


def global_id(*bases):
    return "eval:" + hashlib.sha256("|".join(sorted(bases)).encode()).hexdigest()


def frozen_cohort():
    a, b, bridge, post = (
        "sccd:session:a",
        "sccd:session:b",
        "conda:match:dev-bridge",
        "hatebr:post:post-1",
    )
    joined = global_id(a, b, bridge, post)

    def row(rid, source, split, session, base, component, observed, frozen=None):
        return {
            "record_id": rid,
            "source": source,
            "split": split,
            "session_id": session,
            "base_eval_group": base,
            "duplicate_component": component,
            "eval_group": frozen or global_id(base),
            "observed_mask": observed,
        }

    return [
        row("a", "sccd", "test", "a", a, "alpha", [True, False, False], joined),
        row("b", "sccd", "test", "b", b, "beta", [True, False, False], joined),
        row("bridge-a", "conda", "dev", "bridge-a", bridge, "alpha", [False] * 3, joined),
        row("bridge-b", "conda", "dev", "bridge-b", bridge, "beta", [False] * 3, joined),
        row("post", "hatebr", "test", "post", post, "beta", [True, False, False], joined),
        row("d", "sccd", "test", "d", "sccd:session:d", "delta", [True, False, False]),
        row("t", "toxicn", "test", "t", "toxicn:component:gamma", "gamma", [True, True, False]),
        row(
            "s",
            "scamspot",
            "test",
            "s",
            "scamspot:component:epsilon",
            "epsilon",
            [False, False, True],
        ),
    ]


def test_two_independent_algorithms_verify_frozen_all_split_closure():
    records = frozen_cohort()
    production = analysis.validate_frozen_global_groups(records)
    independent = audit.verify_frozen_global_groups(records[::-1])
    for key, value in production.items():
        assert independent[key] == value
    assert production["cross_source_global_groups"] == 1
    assert production["unknown_records_included"] == 2
    assert production["global_groups"] == 4
    assert production["global_ids_redefined"] is False


@pytest.mark.parametrize(
    "verifier", [analysis.validate_frozen_global_groups, audit.verify_frozen_global_groups]
)
def test_dev_unknown_bridge_cannot_be_discarded_before_closure_validation(verifier):
    records = frozen_cohort()
    with pytest.raises(ValueError, match="complete all-source"):
        verifier([row for row in records if row["split"] == "test"])
    with pytest.raises(ValueError, match="complete all-source"):
        verifier([row for row in records if any(row["observed_mask"])])


def test_test_subset_keeps_dev_bridge_and_cross_source_global_identity():
    messages = [row for row in frozen_cohort() if row["split"] == "test"]
    clusters = analysis.cluster_components(messages)
    assert clusters["a"] == clusters["b"] == clusters["post"]
    assert clusters["a"] != clusters["d"]
    sessions = analysis.cluster_components(messages, unit="conversation")
    assert len(set(sessions.values())) == 2
    counts = audit.frozen_global_cluster_counts(messages)
    assert counts["sccd.attack"] == counts["sccd.conversation"] == 2
    assert counts["sccd.conversation_common16"] == 2
    assert counts["hatebr.attack"] == 1
    profiles = audit.frozen_global_cluster_profiles(messages)
    assert profiles["sccd.conversation"]["largest_cluster_n"] == 2
    assert profiles["sccd.conversation"]["largest_cluster_fraction"] == pytest.approx(2 / 3)
    assert profiles["sccd.conversation"][
        "size_concentration_effective_cluster_count"
    ] == pytest.approx(1.8)


@pytest.mark.parametrize(
    "verifier", [analysis.validate_frozen_global_groups, audit.verify_frozen_global_groups]
)
def test_frozen_id_corruption_or_spurious_overmerge_is_rejected(verifier):
    records = frozen_cohort()
    records[-1]["eval_group"] = records[0]["eval_group"]
    with pytest.raises(ValueError, match="complete all-source"):
        verifier(records)
    records = frozen_cohort()
    records[0]["base_eval_group"] = "sccd:session:invented"
    with pytest.raises(ValueError, match="genuine SCCD session"):
        verifier(records)


def test_prediction_group_binding_uses_exact_frozen_membership():
    records = frozen_cohort()
    frozen = {row["record_id"]: row for row in records}
    messages = [copy.deepcopy(row) for row in records if row["split"] == "test"]
    analysis._verify_prediction_global_groups(messages, "test", frozen)
    messages[0]["eval_group"] = "changed"
    with pytest.raises(ValueError, match="frozen global-group field"):
        analysis._verify_prediction_global_groups(messages, "test", frozen)
    with pytest.raises(ValueError, match="membership"):
        analysis._verify_prediction_global_groups(messages[1:], "test", frozen)


def test_registered_analysis_requires_frozen_prepared_hashes(tmp_path):
    (tmp_path / "revision_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="frozen prepared global groups"):
        analysis._prepared_global_groups(tmp_path)
    folder = tmp_path / "prepared"
    folder.mkdir()
    records = frozen_cohort()
    path = folder / "records.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    manifest = {"artifacts": {"records.jsonl": hashlib.sha256(path.read_bytes()).hexdigest()}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    frozen, proof = analysis._prepared_global_groups(tmp_path)
    assert set(frozen) == {row["record_id"] for row in records}
    assert proof["verified"] and proof["global_groups"] == 4
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint changed"):
        analysis._prepared_global_groups(tmp_path)


def test_frozen_global_single_cluster_is_na_not_zero_width():
    messages = [row for row in frozen_cohort() if row["record_id"] in {"a", "b"}]
    groups = analysis.cluster_components(messages)
    prediction = analysis.TaskPrediction(
        ["a", "b"],
        np.array([0, 1]),
        np.array([0.1, 0.9]),
        0.5,
        [groups["a"], groups["b"]],
    )
    draws, metadata = analysis.bootstrap_paired({"one": [prediction]}, count=20)
    assert metadata["n_clusters"] == 1
    assert metadata["attempted"] == 0
    assert metadata["interval_available"] is False
    assert np.isnan(draws["one"]).all()


def dependency_hash_fixture(tmp_path, separator):
    folder = tmp_path / "prepared"
    folder.mkdir()
    (folder / "records.jsonl").write_bytes(b'{"record_id":"actual-Windows-key"}\r\n')
    (folder / "manifest.json").write_bytes(b'{"artifacts":{}}\r\n')
    hashes = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("records.jsonl", "manifest.json")
    }
    dependence = {
        "prepared_records_sha256": hashes["records.jsonl"],
        "prepared_manifest_sha256": hashes["manifest.json"],
    }
    declaration = {
        "prediction_and_manifest_hashes": {
            "prepared" + separator + name: sha for name, sha in hashes.items()
        }
    }
    return dependence, declaration


@pytest.mark.parametrize("separator", ["\\", "/"])
def test_dependency_hash_binding_accepts_native_windows_and_posix_keys(tmp_path, separator):
    dependence, declaration = dependency_hash_fixture(tmp_path, separator)
    proof = audit.verify_dependency_input_hashes(tmp_path, dependence, declaration)
    assert proof["prepared/records.jsonl"] == dependence["prepared_records_sha256"]
    assert proof["prepared/manifest.json"] == dependence["prepared_manifest_sha256"]


@pytest.mark.parametrize("corruption", ["actual-file", "declared-hash", "missing-key", "alias-key"])
def test_windows_dependency_keys_do_not_relax_hash_or_unique_identity_checks(tmp_path, corruption):
    dependence, declaration = dependency_hash_fixture(tmp_path, "\\")
    hashes = declaration["prediction_and_manifest_hashes"]
    if corruption == "actual-file":
        (tmp_path / "prepared/records.jsonl").write_bytes(b"changed bytes")
    elif corruption == "declared-hash":
        hashes["prepared\\records.jsonl"] = "0" * 64
    elif corruption == "missing-key":
        del hashes["prepared\\records.jsonl"]
    else:
        hashes["prepared/records.jsonl"] = hashes["prepared\\records.jsonl"]
    with pytest.raises(ValueError):
        audit.verify_dependency_input_hashes(tmp_path, dependence, declaration)
