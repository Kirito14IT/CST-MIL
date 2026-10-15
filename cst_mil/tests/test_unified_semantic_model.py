"""Meaningful protocol guards without paid calls or expensive formal training."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from cst_mil.unified_semantic import CATEGORY_IDS, training
from cst_mil.unified_semantic import model as model_module
from cst_mil.unified_semantic.encoder import DEFAULT_ENCODER_PATH, file_sha256, token_chunks
from cst_mil.unified_semantic.model import message_features, new_char_vectorizer, session_features


class FakeEncoder:
    def __init__(self, model_path=None, fingerprint=None, *, threads=4):
        self.model_path = str(Path(model_path or "fixture-encoder").resolve())
        self.fingerprint = "test-encoder-v1"
        self.info = {"encoded_texts": 0, "truncated_content_tokens": 0}

    def validate_assets(self):
        return {"files": {"fixture": {"bytes": 0}}, "model_fingerprint": self.fingerprint}

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            value = np.random.default_rng(seed).normal(0, 0.05, 384)
            value[0] = 1 if "abuse" in text or "promotion" in text else -1
            vectors.append(value / np.linalg.norm(value))
        self.info["encoded_texts"] += len(texts)
        return np.asarray(vectors, dtype=np.float32).reshape((-1, 384))


def _record(source, split, index, label, *, session_id=None, order=0, parent_id=None):
    record_id = f"{source}/{split}/{index}"
    mask = [
        source in {"sccd", "conda", "toxicn", "hatebr"},
        source == "toxicn",
        source == "scamspot",
    ]
    return {
        "record_id": record_id,
        "source": source,
        "split": split,
        "group_id": session_id or record_id,
        "session_id": session_id or record_id,
        "message_id": record_id,
        "order": order,
        "parent_id": parent_id,
        "text": ("abuse promotion forbidden" if label else "friendly study discussion")
        + f" {index}",
        "labels": [label, label, label],
        "observed_mask": mask,
        "original_label": {"fixture": label},
        "source_ref": {},
        "duplicate_component": f"component/{record_id}",
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "LocalSemanticEncoder", FakeEncoder)
    monkeypatch.setattr(model_module, "LocalSemanticEncoder", FakeEncoder)
    records, sessions = [], []
    for split, count in (("train", 8), ("dev", 4), ("test", 4)):
        for i in range(count):
            session_id = f"sccd-{split}-{i}"
            messages = []
            for order in range(2):
                record = _record(
                    "sccd",
                    split,
                    f"{i}-{order}",
                    i % 2,
                    session_id=session_id,
                    order=order,
                    parent_id=messages[-1] if order else None,
                )
                records.append(record)
                messages.append(record["message_id"])
            sessions.append(
                {
                    "session_id": session_id,
                    "split": split,
                    "label": i % 2,
                    "message_ids": messages,
                    "post_text": "original post",
                }
            )
        for source in ("conda", "toxicn", "scamspot", "hatebr"):
            for i in range(count):
                records.append(_record(source, split, str(i), i % 2))
    directory = tmp_path / "prepared"
    directory.mkdir()
    for name, values in (("records.jsonl", records), ("sessions.jsonl", sessions)):
        (directory / name).write_text(
            "".join(json.dumps(r) + "\n" for r in values), encoding="utf-8"
        )
    (directory / "manifest.json").write_text('{"fixture":true}', encoding="utf-8")
    return directory, records, sessions


def test_encoder_default_path_and_lossless_chunks():
    assert DEFAULT_ENCODER_PATH == Path(__file__).resolve().parents[2] / "benchmark" / "models" / (
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    values = list(range(601))
    chunks = token_chunks(values)
    assert all(len(chunk) <= 112 for chunk in chunks)
    assert chunks[1][:16] == chunks[0][-16:]
    assert sorted({token for chunk in chunks for token in chunk}) == values
    assert chunks[-1][-1] == 600
    assert token_chunks([]) == [[]]
    with pytest.raises(ValueError):
        token_chunks(values, 16, 16)


def test_source_class_weights_are_explicit_once():
    sources = ["one"] * 10 + ["two"] * 4
    labels = np.asarray([0] * 9 + [1] + [0] * 2 + [1] * 2)
    weights = training.balanced_source_class_weights(sources, labels)
    assert weights.mean() == pytest.approx(1)
    assert weights[:10].sum() == pytest.approx(weights[10:].sum())
    assert weights[labels == 0].sum() == pytest.approx(weights[labels == 1].sum())
    assert training._new_lr().class_weight is None


def test_unknown_masks_and_duplicate_components_are_excluded(prepared):
    _, records, _ = prepared
    held_id = "sccd-train-0"
    held = next(r for r in records if r["session_id"] == held_id)
    linked = next(r for r in records if r["source"] == "toxicn" and r["split"] == "train")
    linked["duplicate_component"] = held["duplicate_component"]
    allowed, audit = training.oof_exclusions(records, {held_id})
    assert linked["record_id"] in audit["excluded_cross_source_record_ids"]
    assert all(records[i]["session_id"] != held_id for i in allowed)
    assert all(records[i]["split"] == "train" for i in allowed)
    vectors = FakeEncoder().encode([r["text"] for r in records])
    matrix = message_features([r["text"] for r in records], vectors, new_char_vectorizer())
    _, audits = training.fit_heads(matrix, records, allowed)
    for j, entry in enumerate(audits):
        expected = {records[i]["record_id"] for i in allowed if records[i]["observed_mask"][j]}
        assert set(entry["observed_record_ids"]) == expected
    spam = audits[2]
    assert spam["source_counts"] == {"scamspot": 8}


def test_full_training_resume_freeze_and_inference_boundary(prepared, tmp_path):
    directory, _, sessions = prepared
    output = tmp_path / "run"
    result = training.train(output, directory)
    model = joblib.load(result["model_path"])
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert "test" not in metrics["messages"]
    assert "test" not in metrics["conversations"]
    assert metrics["test_deferred"]
    predictions = (output / "message_predictions.jsonl").read_text(encoding="utf-8")
    assert '"split": "test"' not in predictions
    audit = json.loads((output / "oof_folds.json").read_text(encoding="utf-8"))
    held = [session_id for fold in audit for session_id in fold["held_session_ids"]]
    assert set(held) == {s["session_id"] for s in sessions if s["split"] == "train"}
    assert len(held) == len(set(held))
    for fold in audit:
        assert not set(fold["held_record_ids"]) & set(fold["allowed_record_ids"])
    assert training.train(output, directory, resume=True) == result
    with pytest.raises(FileExistsError):
        training.train(output, directory)
    frozen = {
        "model_sha256": file_sha256(output / "model.joblib"),
        "thresholds": model.thresholds,
        "conversation_threshold": model.conversation_threshold,
        "topic_distance": model.topic_distance,
    }
    freeze_path = output / "freeze.json"
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    test_metrics = training.evaluate_test(output, directory, freeze_path)
    assert test_metrics["conversations"]["test"]["n_observed"] == 4
    assert test_metrics["messages"]["test"]["pooled"]["hate"]["n_observed"] == 4
    assert training.evaluate_test(output, directory, freeze_path) == test_metrics
    frozen["topic_distance"] = 0.45
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="different frozen"):
        training.evaluate_test(output, directory, freeze_path)
    message = {
        "comment_id": "m1",
        "raw_text": "abuse forbidden insults",
        "order": 0,
        "label": "FORBIDDEN_ANNOTATION",
        "source": "FORBIDDEN_SOURCE",
        "parent_id": None,
    }
    prediction = model_module.predict_session(model, {"post_id": "new", "label": 1}, [message])
    assert prediction["schemaVersion"] == 2
    assert prediction["label"] is None
    assert prediction["overallEvidenceScore"] == max(
        c["score"] for c in prediction["riskCategories"]
    )
    assert "risk_probability" not in prediction and "threshold" not in prediction
    assert "FORBIDDEN_" not in json.dumps(prediction)
    assert set(prediction["message_scores"][0]["riskCategoryScores"]) == set(CATEGORY_IDS)
    for category in prediction["riskCategories"]:
        for evidence in category["evidence"]:
            assert evidence["position"] == 0 and isinstance(evidence["risk_score"], float)


def test_train_resume_rejects_changed_data(prepared, tmp_path):
    directory, _, _ = prepared
    output = tmp_path / "run"
    training.train(output, directory)
    (directory / "manifest.json").write_text('{"fixture":"changed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        training.train(output, directory, resume=True)


def test_real_reply_features_and_root_fallback(prepared, tmp_path):
    directory, _, _ = prepared
    result = training.train(tmp_path / "run", directory)
    model = joblib.load(result["model_path"])
    messages = [
        {"comment_id": "a", "raw_text": "friendly discussion", "parent_id": None},
        {"comment_id": "b", "raw_text": "abuse forbidden", "parent_id": "a"},
    ]
    embedding = FakeEncoder().encode([m["raw_text"] for m in messages])
    score = model.score_embeddings([m["raw_text"] for m in messages], embedding)
    linked = session_features(messages, embedding, score, model.pca)
    unlinked = session_features(
        [{**m, "parent_id": None} for m in messages], embedding, score, model.pca
    )
    assert linked.shape == (144,)
    assert linked[-1] == 0.5 and unlinked[-1] == 0
    assert np.array_equal(linked[:-6], unlinked[:-6])
    fallback = model_module.predict_session(
        model, {"post_id": "root", "post_raw_text": "friendly discussion"}, []
    )
    assert fallback["message_scores"][0]["comment_id"] == "root:root_fallback"
