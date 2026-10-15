"""Protocol checks with synthetic data, never used as research evidence."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import joblib

from cst_mil.metrics import select_threshold
from cst_mil.research_followup.full_joint_comparison import model as joint
from cst_mil.research_followup.full_joint_comparison.topic_backend import AuditedBERTopic, verify_native_tools
from cst_mil.unified_semantic_v1_1.model import ConstantHead


def fixture_records():
    records, sessions = [], []
    for split, count in (("train", 8), ("dev", 4), ("test", 4)):
        for i in range(count):
            sid, label = f"{split}-session-{i}", i % 2
            sessions.append({"session_id": sid, "split": split, "label": label})
            for message in range(2):
                rid = f"{sid}-message-{message}"
                records.append({"record_id": rid, "message_id": rid, "source": "sccd",
                                "session_id": sid, "split": split, "text": f"class {label} 文本 {i} {message}",
                                "labels": [label, None, None], "observed_mask": [True, False, False],
                                "parent_id": f"{sid}-message-0" if message else None,
                                "order": message, "duplicate_component": rid, "eval_group": sid})
        for source, category in (("toxicn", 1), ("scamspot", 2)):
            for i in range(6):
                rid, label = f"{split}-{source}-{i}", i % 2
                labels, masks = [None] * 3, [False] * 3
                labels[category], masks[category] = label, True
                records.append({"record_id": rid, "message_id": rid, "source": source,
                                "session_id": None, "split": split, "text": f"source 文本 {label} {i}",
                                "labels": labels, "observed_mask": masks, "parent_id": None,
                                "order": 0, "duplicate_component": rid, "eval_group": rid})
    embeddings = np.random.default_rng(120).normal(size=(len(records), 384)).astype(np.float32)
    return records, sessions, embeddings


@pytest.mark.parametrize("labels,scores", [
    ([0, 0, 1, 1], [0., .5, .5, 1.]),
    ([1, 1], [0., 0.]), ([0, 0], [1., 1.]),
    ([0, 1], [.5, .5]), ([1], [1.]), ([0], [0.]),
])
def test_threshold_ties_match_frozen_selector(labels, scores):
    assert joint.select_threshold_fast(labels, scores) == pytest.approx(select_threshold(labels, scores))


def test_threshold_random_duplicate_scores_matches_frozen_selector():
    rng = np.random.default_rng(135)
    for n in range(1, 60, 3):
        for _ in range(5):
            labels, scores = rng.integers(0, 2, n), rng.integers(0, 11, n) / 10
            assert joint.select_threshold_fast(labels, scores) == pytest.approx(select_threshold(labels, scores))


def test_masked_labels_and_zero_topic_weighted_prior():
    records, _, _ = fixture_records()
    eligible = np.asarray([i for i, row in enumerate(records) if row["split"] == "train"])
    heads, audit = joint.fit_heads(np.zeros((len(records), 0)), records, eligible, 43)
    assert all(isinstance(head, ConstantHead) for head in heads)
    assert [head.probability for head in heads] == pytest.approx([.5, .5, .5])
    for category, details in enumerate(audit):
        assert all(records[next(i for i, r in enumerate(records) if r["record_id"] == rid)]["observed_mask"][category]
                   for rid in details["observed_record_ids"])
        assert details["seed"] == 43


def test_true_reply_edge_risk_coverage_survives_zero_representations():
    records = [{"record_id": "a", "message_id": "a", "parent_id": None},
               {"record_id": "b", "message_id": "b", "parent_id": "a"},
               {"record_id": "c", "message_id": "c", "parent_id": "b"}]
    representations = np.asarray([[0., 0.], [1., 0.], [0., 1.]])
    scores = np.asarray([[.1, 0, 0], [.7, 0, 0], [.9, 0, 0]])
    main, common, coverage = joint.session_feature_vectors(records, representations, scores)
    assert main.shape == (20,)
    assert common.shape == (16,)
    assert common[-6:-3] == pytest.approx([1., 0., 1.])
    assert common[-3:] == pytest.approx([.4, .6, 2 / 3])
    assert coverage["true_reply_edges"] == 2
    assert coverage["semantic_valid_reply_edges"] == 1
    assert coverage["zero_endpoint_reply_edges"] == 1
    _, common_zero, coverage_zero = joint.session_feature_vectors(records, np.zeros((3, 0)), scores)
    assert common_zero[-6:-3] == pytest.approx([0., 0., 0.])
    assert common_zero[-3:] == pytest.approx([.4, .6, 2 / 3])
    assert coverage_zero["true_reply_edges"] == 2


def test_canonical_batches_respect_split_source_and_conversation():
    records, _, _ = fixture_records()
    batches = joint.canonical_batches(records, 4)
    assert sorted(i for batch in batches for i in batch) == list(range(len(records)))
    for batch in batches:
        assert len({records[i]["split"] for i in batch}) == 1
        assert len({records[i]["source"] for i in batch}) == 1
        if records[batch[0]]["source"] == "sccd":
            assert len({records[i]["session_id"] for i in batch}) == 1
        else:
            assert len(batch) <= 4
            assert [records[i]["record_id"] for i in batch] == sorted(records[i]["record_id"] for i in batch)


@pytest.mark.parametrize("method", ["cst_v1_1", "minilm_lr"])
def test_training_isolation_seed_oof_fallback_and_predict_roundtrip(tmp_path, method):
    records, sessions, embeddings = fixture_records()
    # A conflicting cross-source duplicate of the first held conversation must
    # be excluded from that fold even when its gold belongs to a different head.
    duplicate = next(r for r in records if r["source"] == "toxicn" and r["split"] == "train")
    duplicate["duplicate_component"] = records[0]["duplicate_component"]
    fallback = next(r for r in records if r["source"] == "scamspot" and r["split"] == "train")
    fallback["fit_eligible"] = False
    frozen = copy.deepcopy(records)
    package, result = joint.train_method(records, sessions, method, 44, tmp_path,
                                          embeddings=embeddings, threads=1)
    assert records == frozen
    fit_ids = result["fit_audit"]["representation_fit_record_ids"]
    assert fallback["record_id"] not in fit_ids
    assert all(not rid.startswith("test") for rid in fit_ids)
    assert package.pca.random_state == 44
    assert package.conversation_classifier[-1].random_state == 44
    assert package.common16_classifier[-1].random_state == 44
    assert all(head.random_state == 44 for head in package.heads if not isinstance(head, ConstantHead))
    all_held = []
    for fold in result["fit_audit"]["folds"]:
        assert fold["seed"] == 44
        all_held.extend(fold["held_session_ids"])
        assert not set(fold["held_record_ids"]) & set(fold["allowed_record_ids"])
        assert not set(fold["excluded_cross_source_record_ids"]) & set(fold["allowed_record_ids"])
    assert sorted(all_held) == sorted(s["session_id"] for s in sessions if s["split"] == "train")
    assert result["fit_audit"]["conversation_features"]["main_dimensions"] == 144
    assert result["fit_audit"]["conversation_features"]["common_dimensions"] == 16
    assert all(row["split"] == "dev" for rows in result["dev_predictions"].values() for row in rows)
    dev = [i for i, row in enumerate(records) if row["split"] == "dev"]
    check = package.predict([records[i] for i in dev], [s for s in sessions if s["split"] == "dev"],
                            embeddings[dev])
    assert check == result["dev_predictions"]
    model_path = tmp_path / "model.joblib"
    joblib.dump(package, model_path)
    restored = joblib.load(model_path)
    assert restored.predict([records[i] for i in dev],
                            [s for s in sessions if s["split"] == "dev"], embeddings[dev]) == check
    # Test labels and even nonfinite test embeddings cannot influence training.
    other = copy.deepcopy(records)
    test = [i for i, row in enumerate(other) if row["split"] == "test"]
    for i in test:
        other[i]["text"] = "TEST CHANGED ONLY"
        other[i]["labels"] = [1, 1, 1]
    alternate = embeddings.copy()
    alternate[test] = np.nan
    second, _ = joint.train_method(other, sessions, method, 44, tmp_path / "second",
                                     embeddings=alternate, threads=1)
    assert second.thresholds == package.thresholds
    assert second.conversation_threshold == package.conversation_threshold
    assert np.array_equal(second.heads[0].coef_, package.heads[0].coef_)


def test_all_noise_bertopic_keeps_unknown_representation_not_forced_negative():
    backend = AuditedBERTopic(43, 1, info={"fitted": True}, topic_ids=[])
    assert backend.transform(["anything"], np.ones((1, 384))).shape == (1, 0)


def test_batch_sensitive_nmf_uses_frozen_dev_predictions_for_thresholds(tmp_path, monkeypatch):
    class BatchSensitiveNMF:
        vocabulary_ = {"word": 0}
        n_topics_ = 2
        info = {"fixture": "batch_sensitive_nmf"}

        def transform(self, texts):
            return np.column_stack((
                [2. if "class 1" in text else 1. for text in texts],
                np.full(len(texts), 1 / (len(texts) + 1)),
            ))

    records, sessions, _ = fixture_records()
    monkeypatch.setattr(joint, "fit_topic_backend", lambda *args, **kwargs: BatchSensitiveNMF())
    package, result = joint.train_method(records, sessions, "nmf", 43, tmp_path, threads=1)
    dev_records = [row for row in records if row["split"] == "dev"]
    dev_sessions = [row for row in sessions if row["split"] == "dev"]
    repeated = package.predict(dev_records, dev_sessions)
    assert repeated == result["dev_predictions"]
    for j, category in enumerate(joint.CATEGORY_IDS):
        observed = [row for row in repeated["message"] if row["observed_mask"][j]]
        assert package.thresholds[category] == joint.select_threshold_fast(
            [row["labels"][j] for row in observed], [row["scores"][j] for row in observed])[0]
    _, joint_representation = package._canonical_features([r for r in records if r["split"] != "test"])
    _, dev_representation = package._canonical_features(dev_records)
    ids = {r["record_id"]: i for i, r in enumerate(records) if r["split"] != "test"}
    assert np.array_equal(joint_representation[[ids[r["record_id"]] for r in dev_records]], dev_representation)


@pytest.mark.parametrize("method", ["lda", "btm", "gsdmm"])
def test_existing_native_build_is_read_only_and_matches_sources(method):
    manifest = verify_native_tools(method)
    assert manifest["method"] == method
