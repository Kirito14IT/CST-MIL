"""Evidence integrity and CPU cost protocol guards for the isolated campaign."""

from __future__ import annotations

import os

import pytest

from cst_mil.research_followup.full_joint_comparison import common, execution
from cst_mil.research_followup.full_joint_comparison.cost import inference_bundles


def test_atomic_json_roundtrip_and_no_pending_file(tmp_path):
    p = tmp_path / "nested/state.json"
    common.save(p, {"内容": "一致", "x": 1})
    assert common.read(p) == {"内容": "一致", "x": 1}
    assert not list(tmp_path.rglob("*.pending"))


def test_guard_rejects_old_evidence_and_manuscripts():
    for name in (
        "data",
        "benchmark/runs/full677",
        "reports",
        "benchmark/research_followup/20261005",
    ):
        with pytest.raises(ValueError):
            common.guard_output(common.PROJECT / name / "replacement")
    assert common.guard_output(common.DEFAULT_OUTPUT) == common.DEFAULT_OUTPUT.resolve()


def test_hash_check_detects_source_mutation(tmp_path):
    p = tmp_path / "source.py"
    p.write_text("first", encoding="utf8")
    frozen = common.source_hashes([p])
    common.verify_hashes(frozen)
    p.write_text("second", encoding="utf8")
    with pytest.raises(RuntimeError, match="changed"):
        common.verify_hashes(frozen)


def test_live_lock_refuses_concurrent_runner(tmp_path):
    with common.campaign_lock(tmp_path):
        with pytest.raises(RuntimeError, match="already running"):
            with common.campaign_lock(tmp_path):
                pytest.fail("Concurrent runner was allowed")
    assert not (tmp_path / "campaign.lock").exists()


def test_stale_lock_is_recovered(tmp_path):
    # Matching PID with wrong creation time is stale even if that PID was reused.
    common.save(tmp_path / "campaign.lock", {"pid": os.getpid(), "created": 0})
    with common.campaign_lock(tmp_path):
        assert common.read(tmp_path / "campaign.lock")["pid"] == os.getpid()


def test_complete_stage_requires_same_identity_and_artifacts(tmp_path):
    artifact = tmp_path / "predictions.jsonl"
    artifact.write_text("one\n", encoding="utf8")
    state = tmp_path / "state.json"
    common.save(
        state,
        {
            "identity": "expected",
            "status": "complete",
            "artifacts": common.source_hashes([artifact]),
        },
    )
    assert execution._check_complete(state, "expected")
    with pytest.raises(RuntimeError, match="fingerprint"):
        execution._check_complete(state, "different")
    artifact.write_text("changed\n", encoding="utf8")
    with pytest.raises(RuntimeError, match="changed"):
        execution._check_complete(state, "expected")


def test_cost_bundles_preserve_real_sessions_and_comment_batches():
    records = [
        {
            "source": "sccd",
            "session_id": "c1",
            "order": i,
            "message_id": str(i),
            "record_id": f"s{i}",
        }
        for i in range(80)
    ]
    records += [
        {"source": "conda", "session_id": str(i), "record_id": f"c{i:03d}"} for i in range(130)
    ]
    bundles = inference_bundles(records[::-1], [{"session_id": "c1"}])
    assert [len(b[2]) for b in bundles] == [80, 64, 64, 2]
    assert [r["order"] for r in bundles[0][2]] == list(range(80))
    assert sum(len(b[2]) for b in bundles) == len(records)
    assert all(not b[3] for b in bundles[1:])


def test_embedding_missing_ids_cannot_be_silent_zero(tmp_path, monkeypatch):
    import numpy as np

    meta = {"frozen": True}
    monkeypatch.setattr(execution, "_encoder_metadata", lambda: meta)
    p = tmp_path / "prepared/manifest.json"
    common.save(p, {})
    for split in ("train", "dev"):
        folder = tmp_path / "cache" / split
        common.save(
            folder / "binding.json",
            {"record_ids": [], "text_sha256": [], "prepared_manifest_sha256": common.digest(p)},
        )
        np.save(folder / "embeddings.npy", np.zeros((0, 384), dtype="float32"))
        common.save(
            folder / "complete.json",
            {"artifacts": {}, "record_ids": [], "encoder": meta, "cold_seconds": 0},
        )
    with pytest.raises(RuntimeError, match="Missing or surplus"):
        execution.embeddings_for(
            tmp_path, [{"record_id": "missing", "split": "train", "text": "hi"}]
        )


def test_train_freeze_evaluate_resume_integration(tmp_path, monkeypatch):
    from test_full_joint_models import fixture_records

    from cst_mil.bench.resources import configure_cpu
    from cst_mil.research_followup.full_joint_comparison import data

    configure_cpu(4)
    records, sessions, vectors = fixture_records()
    frozen = {"threads": 4, "methods": ["minilm_lr"], "seeds": [42, 43, 44]}
    metadata = {"asset_bytes": 0, "fingerprint": "synthetic-test-only"}
    positions = {r["record_id"]: i for i, r in enumerate(records)}
    monkeypatch.setattr(data, "load_prepared", lambda output: (records, sessions, {}))
    monkeypatch.setattr(execution, "verify_experiment", lambda output: frozen)
    monkeypatch.setattr(execution, "_encoder_metadata", lambda: metadata)
    monkeypatch.setattr(
        execution,
        "embeddings_for",
        lambda output, rows, **kwargs: (
            vectors[[positions[r["record_id"]] for r in rows]],
            metadata,
        ),
    )
    manifest = execution.train_one(tmp_path, "minilm_lr", 43)
    assert manifest["seed"] == 43
    assert manifest["training_attempt"].endswith("attempt_001")
    assert execution.train_one(tmp_path, "minilm_lr", 43) == manifest
    state = execution.evaluate_one(tmp_path, "minilm_lr", 43)
    assert state["status"] == "complete"
    assert execution.evaluate_one(tmp_path, "minilm_lr", 43) == state
    folder = tmp_path / "runs/minilm_lr/seed_43"
    predictions = common.read_jsonl(folder / "predictions_message_test.jsonl")
    assert set(predictions[0]["scores"]) == {"attack", "hate", "spam_scam"}
    assert all(r["split"] == "test" for r in predictions)
    original = folder / "model.joblib"
    original.write_bytes(original.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        execution.evaluate_one(tmp_path, "minilm_lr", 43)
