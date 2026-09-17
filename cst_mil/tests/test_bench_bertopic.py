"""BERTopic adapter checks independent of model/network availability."""

from __future__ import annotations

import copy
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from cst_mil.bench import bertopic_backend as backend
from cst_mil.bench.bertopic_backend import BERTopicModel, token_chunks
from cst_mil.bench.common import file_hash, write_json


def test_original_token_chunks_preserve_every_token_and_suffix():
    original = list(range(401))
    chunks = token_chunks(original, 126)
    assert [len(chunk) for chunk in chunks] == [126, 126, 126, 23]
    assert [token for chunk in chunks for token in chunk] == original
    assert token_chunks([], 126) == [[]]
    with pytest.raises(ValueError):
        token_chunks([1, 2], 0)


class _MembershipBackend:
    embedding_model = None

    def transform(self, documents, embeddings):
        assert len(documents) == len(embeddings)
        return np.array([0, -1, 1]), np.array([[0.4, 0.2], [0.2, 0.1], [0.1, 0.7]])

    def get_topic(self, topic_id):
        return [(f"word_{topic_id}", 0.5)]


def test_memberships_preserve_noise_and_order(tmp_path, monkeypatch):
    model = BERTopicModel({}, tmp_path)
    model.info["fitted"] = True
    model.topic_ids = [0, 1]
    model.model = _MembershipBackend()
    monkeypatch.setattr(model, "validate_assets", lambda: {})
    monkeypatch.setattr(model, "_embeddings", lambda docs: np.zeros((len(docs), 384)))
    actual = model.transform(["a", "b", "c"])
    np.testing.assert_allclose(actual, [[2 / 3, 1 / 3], [0, 0], [0.125, 0.875]])
    assert model.topic_words() == [["word_0"], ["word_1"]]


def test_all_noise_is_zero_width_not_a_fabricated_topic(tmp_path, monkeypatch):
    model = BERTopicModel({}, tmp_path)
    model.info["fitted"] = True
    monkeypatch.setattr(model, "validate_assets", lambda: {})
    assert model.transform(["unknown", "text"]).shape == (2, 0)
    assert model.topic_words() == []


def test_cached_embedding_keys_include_full_text_and_model(tmp_path, monkeypatch):
    model = BERTopicModel({"bertopic": {"embedding_cache_dir": str(tmp_path / "cache")}},
                         tmp_path / "work")
    manifest = {"model_fingerprint": "revision-a"}
    monkeypatch.setattr(model, "_read_manifest", lambda: manifest)
    calls = []

    def encode(documents):
        calls.append(list(documents))
        values = np.zeros((len(documents), 384), dtype=np.float32)
        for i, text in enumerate(documents):
            values[i, len(text) % 384] = 1
        return values

    monkeypatch.setattr(model, "_encode_uncached", encode)
    inputs = ["prefix", "prefix", "prefix suffix"]
    first = model._embeddings(inputs)
    second = model._embeddings(inputs)
    np.testing.assert_array_equal(first, second)
    assert calls == [["prefix", "prefix suffix"]]
    assert model.info["encoded_texts"] == 2
    assert model.info["cached_texts"] == 3
    manifest["model_fingerprint"] = "revision-b"
    model._embeddings(["prefix"])
    assert calls[-1] == ["prefix"]


def test_pickle_preserves_pipeline_without_global_sentence_encoder(tmp_path):
    model = BERTopicModel({}, tmp_path)
    model.model = _MembershipBackend()
    restored = pickle.loads(pickle.dumps(model))
    assert isinstance(restored.model, _MembershipBackend)
    assert restored.model.embedding_model is None
    assert restored.model_dir == Path(model.model_dir)


def test_official_small_chinese_corpus_retains_text_and_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("bertopic")
    from hdbscan import HDBSCAN
    from umap import UMAP

    documents = ["机器学习算法", "学校课程考试", "聊天侮辱攻击", "足球比赛球队", "历史文化考古"]
    monkeypatch.setattr(BERTopicModel, "validate_assets", lambda self: {})
    monkeypatch.setattr(BERTopicModel, "_embeddings", lambda self, docs: np.eye(len(docs), 384))
    model = BERTopicModel({"threads": 1}, tmp_path).fit(documents)
    assert model.model.language == "multilingual"
    assert "机器" in model.model.vectorizer_model.vocabulary_
    assert model.info["all_noise"]
    assert model.transform(documents).shape == (5, 0)
    restored = pickle.loads(pickle.dumps(model))
    assert isinstance(restored.model.umap_model, UMAP)
    assert isinstance(restored.model.hdbscan_model, HDBSCAN)
    assert restored.model.embedding_model is None


@pytest.fixture
def external_snapshot(tmp_path):
    directory = tmp_path / "encoder"
    directory.mkdir()
    (directory / "model.safetensors").write_bytes(b"weights-a")
    (directory / "config.json").write_bytes(b"{}")
    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in sorted(directory.iterdir())
    }
    manifest = {
        "model_id": backend.MODEL_ID, "revision": backend.MODEL_REVISION,
        "files": files, "model_fingerprint": backend._manifest_digest(files),
        "weights_format": "safetensors", "device": "cpu",
    }
    write_json(directory / backend._MANIFEST_NAME, manifest)
    model = BERTopicModel({"bertopic": {"model_dir": str(directory)}}, tmp_path / "work")
    return model, directory, manifest


def test_external_assets_hash_once_per_process_and_after_stat_change(
    external_snapshot, monkeypatch,
):
    model, directory, manifest = external_snapshot
    hashes = []

    def counted(path):
        hashes.append(Path(path).name)
        return file_hash(path)

    monkeypatch.setattr(backend, "file_hash", counted)
    assert model.validate_assets() == manifest
    assert len(hashes) == 2
    model.validate_assets()
    assert len(hashes) == 2
    # Touching identical bytes triggers a fresh hash, but need not invalidate data.
    weight_path = directory / "model.safetensors"
    stat = weight_path.stat()
    os.utime(weight_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    model.validate_assets()
    assert len(hashes) == 4
    # A new process has no verification-cache entry even if manifest was pickled.
    restored = pickle.loads(pickle.dumps(model))
    backend._VERIFIED_SNAPSHOTS.clear()
    restored.validate_assets()
    assert len(hashes) == 6
    restored.validate_assets()
    assert len(hashes) == 6


def test_restored_model_detects_same_size_external_weight_tampering(external_snapshot):
    model, directory, _ = external_snapshot
    model.validate_assets()
    restored = pickle.loads(pickle.dumps(model))
    weight_path = directory / "model.safetensors"
    old_stat = weight_path.stat()
    weight_path.write_bytes(b"weights-b")
    os.utime(weight_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="snapshot hash mismatch"):
        restored.validate_assets()


def test_manifest_must_be_self_consistent(external_snapshot):
    model, directory, manifest = external_snapshot
    manifest["model_fingerprint"] = "0" * 64
    write_json(directory / backend._MANIFEST_NAME, manifest)
    with pytest.raises(ValueError, match="fingerprint is not self-consistent"):
        model.validate_assets()
    assert model.manifest is None


def test_self_consistent_replacement_manifest_cannot_replace_fitted_identity(external_snapshot):
    model, directory, manifest = external_snapshot
    model.validate_assets()
    restored = pickle.loads(pickle.dumps(model))
    replacement = copy.deepcopy(manifest)
    (directory / "model.safetensors").write_bytes(b"weights-b")
    replacement["files"]["model.safetensors"]["sha256"] = file_hash(directory / "model.safetensors")
    replacement["model_fingerprint"] = backend._manifest_digest(replacement["files"])
    write_json(directory / backend._MANIFEST_NAME, replacement)
    with pytest.raises(ValueError, match="differs from the fitted model manifest"):
        restored.validate_assets()


def test_untracked_external_assets_are_rejected(external_snapshot):
    model, directory, _ = external_snapshot
    model.validate_assets()
    (directory / "adapter_config.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="file inventory differs"):
        model.validate_assets()


@pytest.mark.parametrize("documents", [[], ["a document"]])
def test_all_noise_cannot_skip_external_asset_validation(external_snapshot, documents):
    model, directory, _ = external_snapshot
    model.validate_assets()
    model.info["fitted"] = True
    assert model.transform(documents).shape == (len(documents), 0)
    (directory / "model.safetensors").write_bytes(b"wrong-size-weights")
    with pytest.raises(ValueError, match="changed size"):
        model.transform(documents)
