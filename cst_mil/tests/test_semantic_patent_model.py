"""Small numerical/protocol tests; no encoder, network, or formal experiment."""
from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy import sparse

from cst_mil.research_followup.full_joint_comparison.model import train_method
from cst_mil.research_followup.semantic_patent_followup.model import (
    ABLATIONS, FollowupPackage, VariantConfig, fit_heads, fit_variant,
    global_class_weights, message_matrix, resolve_parents, session_vectors,
)
from cst_mil.unified_semantic_v1_1.model import message_features, new_char_vectorizer


def row(rid, sid="one", mid=None, parent=None, source="sccd", split="train", label=0):
    return {"record_id": rid, "message_id": mid or rid, "session_id": sid,
            "source": source, "split": split, "parent_id": parent, "order": 0,
            "text": "评论 sample " + rid, "labels": [label, label, label],
            "observed_mask": [True, True, True], "duplicate_component": "dup:" + rid,
            "eval_group": "group:" + rid, "fit_eligible": True}


def fixture():
    records, sessions = [], []
    for split, count in (("train", 8), ("dev", 4), ("test", 2)):
        for i in range(count):
            sid = f"{split}-{i}"
            label = i % 2
            sessions.append({"session_id": sid, "split": split, "label": label})
            records.extend([row(sid + ":0", sid, split=split, label=label),
                            row(sid + ":1", sid, parent=sid + ":0", split=split, label=label)])
    for source in ("toxicn", "scamspot"):
        for split in ("train", "dev"):
            for i in range(4):
                records.append(row(f"{source}-{split}-{i}", None, source=source,
                                   split=split, label=i % 2))
    rng = np.random.default_rng(123)
    return records, rng.normal(size=(len(records), 384)).astype(np.float32), sessions


def block_values(matrix, descriptions, name):
    block = next(value for value in descriptions if value["name"] == name)
    return matrix[:, block["start"]:block["stop"]].toarray()


def test_default_features_exact_frozen_helper():
    records, embeddings, _ = fixture()
    vectorizer = new_char_vectorizer()
    matrix, representations, blocks, _ = message_matrix(records, embeddings, VariantConfig(), vectorizer)
    expected = message_features([r["text"] for r in records], representations, vectorizer)
    assert (matrix != expected).nnz == 0
    assert matrix.dtype == expected.dtype
    assert blocks[-1]["stop"] == 65920


def test_context_edges_reject_self_cycles_cross_session_and_split():
    records = [row("root", mid="root"), row("child", mid="child", parent="root"),
               row("self", mid="self", parent="self"), row("c1", mid="c1", parent="c2"),
               row("c2", mid="c2", parent="c1"), row("foreign", sid="other", parent="root"),
               row("dev-root", mid="root", split="dev"),
               row("missing", parent="absent")]
    parents, audit = resolve_parents(records)
    assert parents.tolist() == [-1, 0, -1, -1, -1, -1, -1, -1]
    assert audit["cycle_record_ids"] == ["c1", "c2"]
    assert audit["valid_edges"] == 1


def test_context_same_id_other_source_is_not_parent():
    records = [row("root", mid="root", source="conda"), row("child", parent="root")]
    assert resolve_parents(records)[0].tolist() == [-1, -1]


def test_duplicate_ids_fail_even_if_record_ids_unique():
    with pytest.raises(ValueError, match="Message IDs"):
        resolve_parents([row("first", mid="same"), row("second", mid="same")])


@pytest.mark.parametrize("architecture", ["context_a", "context_b"])
def test_all_context_features_zero_without_valid_parent(architecture):
    records = [row("root"), row("child", parent="root"), row("missing", parent="nope")]
    embeddings = np.random.default_rng(3).normal(size=(3, 384)).astype(np.float32)
    matrix, _, blocks, _ = message_matrix(records, embeddings, VariantConfig(architecture))
    for block in blocks[2:]:
        values = block_values(matrix, blocks, block["name"])
        assert not values[[0, 2]].any()
    assert block_values(matrix, blocks, "valid_parent").ravel().tolist() == [0, 1, 0]
    assert np.linalg.norm(block_values(matrix, blocks, "parent_semantic")[1]) == pytest.approx(1)
    if architecture == "context_b":
        for name in ("absolute_semantic_difference", "semantic_product"):
            assert np.linalg.norm(block_values(matrix, blocks, name)[1]) == pytest.approx(1)


def test_no_message_semantics_preserves_only_char_and_structural_flag():
    records, embeddings, _ = fixture()
    matrix, _, blocks, _ = message_matrix(
        records, embeddings, VariantConfig("context_b", "no_message_semantics"))
    assert [b["name"] for b in blocks] == ["current_character", "valid_parent"]
    assert matrix.shape[1] == 65537


def test_unknown_mask_and_fallback_never_become_negative_supervision():
    records = [row("negative", label=0), row("positive", label=1), row("unknown", label=9),
               row("fallback", label=0)]
    records[2]["observed_mask"] = [False] * 3
    records[3]["fit_eligible"] = False
    _, audits = fit_heads(sparse.eye(4, format="csr"), records, np.arange(4), 43, VariantConfig())
    for audit in audits:
        assert audit["observed_record_ids"] == ["negative", "positive"]
        assert audit["sample_weight"] == [1, 1]
        assert audit["seed"] == 43


def test_global_class_balance_is_not_source_balance():
    records = [row(str(i), source="a" if i < 4 else "b", label=i % 2) for i in range(6)]
    _, source = fit_heads(sparse.eye(6, format="csr"), records, np.arange(6), 42, VariantConfig())
    _, global_audit = fit_heads(sparse.eye(6, format="csr"), records, np.arange(6), 42,
                               VariantConfig("full", "no_source_balance"))
    assert source[0]["source_weight_totals"] == pytest.approx({"a": 3, "b": 3})
    assert global_audit[0]["source_weight_totals"] == pytest.approx({"a": 4, "b": 2})
    assert global_class_weights(np.array([0, 0, 1])).tolist() == [.75, .75, 1.5]


class SmallPCA:
    def transform(self, values):
        return values[:, :64]


@pytest.mark.parametrize("ablation,main_dim,common_dim", [
    (None, 144, 16), ("no_session_semantics", 16, 16),
    ("no_session_reply", 138, 10), ("no_session_risk", 132, 4),
])
def test_session_ablation_dimensions(ablation, main_dim, common_dim):
    records = [row("root"), row("child", parent="root")]
    embeddings = np.random.default_rng(3).normal(size=(2, 384)).astype(np.float32)
    main, common, coverage = session_vectors(records, embeddings, np.full((2, 3), .5),
                                             SmallPCA(), VariantConfig("full", ablation))
    assert main.shape == (main_dim,)
    assert common.shape == (common_dim,)
    assert coverage["true_reply_edges"] == 1


def test_no_session_risk_removes_derived_reply_risk_too():
    records = [row("root"), row("child", parent="root")]
    embeddings = np.random.default_rng(3).normal(size=(2, 384)).astype(np.float32)
    config = VariantConfig("full", "no_session_risk")
    first = session_vectors(records, embeddings, np.array([[.1] * 3, [.9] * 3]), SmallPCA(), config)
    second = session_vectors(records, embeddings, np.array([[.6] * 3, [.6] * 3]), SmallPCA(), config)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_default_fit_exact_old_architecture_and_thresholds(tmp_path):
    records, embeddings, sessions = fixture()
    old, _ = train_method(records, sessions, "cst_v1_1", 43, tmp_path, embeddings, threads=2)
    new, details, oof = fit_variant(records, embeddings, sessions, seed=43, threads=2)
    dev_indices = [i for i, r in enumerate(records) if r["split"] == "dev"]
    dev_records = [records[i] for i in dev_indices]
    dev_sessions = [s for s in sessions if s["split"] == "dev"]
    before = old.predict(dev_records, dev_sessions, embeddings[dev_indices])
    after = new.predict_payload(dev_records, dev_sessions, embeddings[dev_indices])
    np.testing.assert_array_equal([r["scores"] for r in before["message"]],
                                  [r["scores"] for r in after["message"]])
    np.testing.assert_array_equal([r["score"] for r in before["session"]],
                                  [r["score"] for r in after["session"]])
    assert new.thresholds == old.thresholds
    assert new.conversation_threshold == old.conversation_threshold
    assert details["fit_audit"]["test_records_used_for_fit"] == 0
    assert len(oof) == 16
    assert {r["split"] for r in oof} == {"train"}
    assert all(np.isfinite(r["scores"]).all() for r in oof)


def test_oof_excludes_entire_session_and_duplicate_components_and_seed():
    records, embeddings, sessions = fixture()
    duplicate = row("cross-source-duplicate", None, source="toxicn", label=0)
    duplicate["duplicate_component"] = records[0]["duplicate_component"]
    records.append(duplicate)
    embeddings = np.vstack([embeddings, np.zeros((1, 384), dtype=np.float32)])
    package, details, _ = fit_variant(records, embeddings, sessions, variant="context_b", seed=44, threads=2)
    assert package.config == VariantConfig("context_b")
    assert package.heads[0].random_state == 44
    assert package.pca.random_state == 44
    audit = details["fit_audit"]
    train_ids = {r["record_id"] for r in records if r["split"] == "train"}
    assert set(audit["representation_fit_record_ids"]) == train_ids
    seen = set()
    for fold in audit["folds"]:
        assert not set(fold["held_record_ids"]) & set(fold["allowed_record_ids"])
        seen.update(fold["held_session_ids"])
        if "train-0" in fold["held_session_ids"]:
            assert duplicate["record_id"] not in fold["allowed_record_ids"]
        assert fold["seed"] == 44
    assert seen == {s["session_id"] for s in sessions if s["split"] == "train"}


def test_test_text_labels_embeddings_do_not_affect_fit():
    records, embeddings, sessions = fixture()
    before, _, _ = fit_variant(records, embeddings, sessions, variant="context_a", threads=2)
    altered = copy.deepcopy(records)
    new_embeddings = embeddings.copy()
    for i, r in enumerate(altered):
        if r["split"] == "test":
            r.update(text="unseen test metadata", labels=[9, 9, 9])
            new_embeddings[i] = np.nan
    after, _, _ = fit_variant(altered, new_embeddings, sessions, variant="context_a", threads=2)
    for a, b in zip(before.heads, after.heads, strict=True):
        np.testing.assert_array_equal(a.coef_, b.coef_)
    np.testing.assert_array_equal(before.pca.components_, after.pca.components_)


def test_raw_encode_uses_canonical_sessions_and_source_id_batches():
    records = [row("c", sid="s2"), row("a", sid="s1"), row("b", sid="s1"),
               row("z", sid=None, source="conda"), row("y", sid=None, source="conda")]

    class FakeEncoder:
        def __init__(self):
            self.calls = []

        def encode(self, texts):
            self.calls.append(texts)
            return np.zeros((len(texts), 384), dtype=np.float32)

    encoder = FakeEncoder()
    package = FollowupPackage(VariantConfig(), 42)
    assert package.encode_records(records, encoder).shape == (5, 384)
    assert encoder.calls == [[records[4]["text"], records[3]["text"]],
                             [records[1]["text"], records[2]["text"]], [records[0]["text"]]]


def test_empty_predict_and_invalid_config():
    assert FollowupPackage(VariantConfig(), 42).predict([]).shape == (0, 3)
    with pytest.raises(ValueError):
        VariantConfig("unknown")
    with pytest.raises(ValueError):
        VariantConfig("full", "unknown")
    assert len(ABLATIONS) == 6
