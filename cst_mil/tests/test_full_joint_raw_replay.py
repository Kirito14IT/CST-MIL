from __future__ import annotations

from pathlib import Path

import pytest

from cst_mil.research_followup.full_joint_comparison.common import read, save
from cst_mil.research_followup.full_joint_comparison.raw_replay import (
    _archive_incomplete,
    _await_child_receipt,
    _consistent_cost,
    _ensure_views,
    _override_session_thresholds,
    _refuse_rejected_cost_retry,
    _thresholds,
    raw_predict,
    select_warmup,
)


def _record(source: str, split: str, index: int) -> dict:
    return {
        "record_id": f"{source}-{split}-{index:03d}",
        "message_id": str(index),
        "source": source,
        "split": split,
        "session_id": "s" if source == "sccd" else source,
        "text": f"原文{index}",
        "order": index,
        "labels": [index % 2] * 3,
        "observed_mask": [True] * 3,
        "duplicate_component": str(index),
        "eval_group": str(index),
        "is_fallback": False,
        "parent_id": None,
    }


def test_warmup_is_fixed_first64_train_conda_independent_of_labels() -> None:
    rows = [_record("conda", "train", i) for i in range(70)]
    rows += [_record("conda", "dev", 0), _record("conda", "test", 0), _record("sccd", "train", 0)]
    chosen = select_warmup(rows[::-1])
    assert len(chosen) == 64
    assert [r["record_id"] for r in chosen] == [f"conda-train-{i:03d}" for i in range(64)]
    for row in rows:
        row["labels"] = [1 - label for label in row["labels"]]
    assert [r["record_id"] for r in select_warmup(rows)] == [r["record_id"] for r in chosen]
    with pytest.raises(ValueError, match="requires 64"):
        select_warmup(rows[:10])


def test_all_five_thresholds_are_dev_only_and_override_prediction_metadata() -> None:
    messages = [
        {**_record("conda", "dev", i), "scores": [0.1 if i % 2 == 0 else 0.9] * 3} for i in range(4)
    ]
    sessions = [
        {
            "session_id": f"s{i}",
            "split": "dev",
            "label": i,
            "score": 0.1 if i == 0 else 0.9,
            "common16_score": 0.2 if i == 0 else 0.8,
            "threshold": 0.5,
            "common16_threshold": 0.5,
        }
        for i in range(2)
    ]
    thresholds, proof = _thresholds(messages, sessions)
    assert set(thresholds) == {
        "attack",
        "hate",
        "spam_scam",
        "conversation",
        "conversation_common16",
    }
    assert thresholds["attack"] == 0.9
    assert thresholds["conversation_common16"] == 0.8
    _override_session_thresholds({"session": sessions}, thresholds)
    assert sessions[0]["threshold"] == 0.9
    assert sessions[0]["common16_threshold"] == 0.8
    messages[0]["split"] = "test"
    with pytest.raises(ValueError, match="Only complete DEV"):
        _thresholds(messages, sessions)


def test_interrupted_init_builds_missing_views_without_resetting_thresholds(tmp_path: Path) -> None:
    base, revision = tmp_path / "base", tmp_path / "revision"
    inherited = []
    for method in ("lda", "cst_v1_1"):
        folder = base / "runs" / method / "seed_42"
        original = {
            "method": method,
            "seed": 42,
            "identity": "base-run-" + method,
            "model_sha256": "weights",
            "thresholds": {"attack": 0.5},
        }
        save(folder / "run_manifest.json", original)
        inherited.append(
            {
                "method": method,
                "seed": 42,
                "base_run_identity": original["identity"],
                "model_sha256": "weights",
                "base_model_path": str(folder / "model.joblib"),
                "base_run_manifest_sha256": "manifest",
                "base_train_state_sha256": "train",
            }
        )
    frozen = {"base_output": str(base), "identity": "revision", "base_training": inherited}
    _ensure_views(revision, frozen)
    semantic = revision / "runs/cst_v1_1/seed_42/run_manifest.json"
    view = read(semantic)
    view.update(thresholds={"attack": 0.7}, threshold_status="raw_dev_frozen")
    save(semantic, view)
    traditional = revision / "runs/lda/seed_42/run_manifest.json"
    traditional.unlink()
    _ensure_views(revision, frozen)
    assert traditional.exists()
    assert read(semantic)["thresholds"]["attack"] == 0.7
    assert read(semantic)["threshold_status"] == "raw_dev_frozen"


def test_unsealed_attempt_is_archived_and_rejected_consistency_cannot_retry(tmp_path: Path) -> None:
    save(tmp_path / "run_manifest.json", {"thresholds": {"attack": 0.5}})
    save(
        tmp_path / "cost_invalid.json",
        {"evaluation_revision_identity": "rev", "status": "invalid_prediction_consistency"},
    )
    with pytest.raises(RuntimeError, match="retry-until-pass"):
        _refuse_rejected_cost_retry(tmp_path, "rev")
    archive = _archive_incomplete(tmp_path, "cost")
    assert archive is not None and (archive / "cost_invalid.json").exists()
    assert not (tmp_path / "cost_invalid.json").exists()
    assert (archive / "run_manifest.snapshot.json").exists()
    with pytest.raises(RuntimeError, match="retry-until-pass"):
        _refuse_rejected_cost_retry(tmp_path, "rev")
    _refuse_rejected_cost_retry(tmp_path, "another-revision")


def test_cost_requires_both_score_tolerance_and_zero_class_disagreements() -> None:
    proof = {
        "max_message_abs_diff": 1e-6,
        "max_session_abs_diff": 0.0,
        "head_threshold_disagreements_all_repetitions": 0,
        "session_threshold_disagreements_all_repetitions": {"main": 0, "common16": 0},
    }
    assert _consistent_cost(proof)
    proof["head_threshold_disagreements_all_repetitions"] = 1
    assert not _consistent_cost(proof)
    proof["head_threshold_disagreements_all_repetitions"] = 0
    proof["session_threshold_disagreements_all_repetitions"]["main"] = 1
    assert not _consistent_cost(proof)
    proof["session_threshold_disagreements_all_repetitions"]["main"] = 0
    proof["max_session_abs_diff"] = 0.001
    assert not _consistent_cost(proof)


def test_child_waits_for_own_parent_receipt_and_rejects_wrong_pid(tmp_path: Path) -> None:
    save(tmp_path / "campaign.lock", {"pid": 1, "created": 1})
    save(tmp_path / "active_child.json", {"pid": 2, "created": 2})
    _await_child_receipt(tmp_path, 2, timeout=0.01)
    with pytest.raises(RuntimeError, match="matching child"):
        _await_child_receipt(tmp_path, 3, timeout=0.01)


def test_raw_replay_uses_none_and_complete_frozen_bundles_without_model_mutation(
    tmp_path: Path,
) -> None:
    class FakePackage:
        thresholds = {"attack": 0.51}
        calls = []

        def predict(self, records, sessions, *, embeddings):
            assert embeddings is None
            self.calls.append((len(records), len(sessions)))
            return {
                "message": [
                    {"record_id": r["record_id"], "scores": [0.2, 0.3, 0.4]} for r in records
                ],
                "session": [
                    {
                        **s,
                        "score": 0.6,
                        "common16_score": 0.7,
                        "threshold": 0.5,
                        "common16_threshold": 0.5,
                    }
                    for s in sessions
                ],
            }

    records = [_record("sccd", "dev", 0), *[_record("conda", "dev", i) for i in range(130)]]
    sessions = [{"session_id": "s", "split": "dev", "label": 0}]
    package = FakePackage()
    prediction = raw_predict(package, records, sessions, progress_path=tmp_path / "progress.json")
    assert package.calls == [(1, 1), (64, 0), (64, 0), (2, 0)]
    assert len(prediction["message"]) == 131
    assert prediction["message"][0]["scores"] == {"attack": 0.2, "hate": 0.3, "spam_scam": 0.4}
    assert package.thresholds == {"attack": 0.51}
    assert read(tmp_path / "progress.json")["messages"] == 131
