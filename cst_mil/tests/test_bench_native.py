"""Small synthetic checks; never consume held-out SCCD conversations."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pytest

from cst_mil.bench.native_topics import (
    NativeTopicModel,
    NMFTopicModel,
    portable_java,
)

TRAIN = [
    "苹果 水果 香蕉 苹果 营养", "香蕉 苹果 水果 桃子",
    "篮球 球队 比赛 篮球 冠军", "球队 冠军 篮球 足球",
    "辱骂 威胁 欺凌 打死 威胁", "欺凌 辱骂 威胁 暴力",
]
QUERY = ["苹果 香蕉", "篮球 球队", "威胁 辱骂", "", "zzneverobservedzz"]


def config() -> dict:
    return {"seed": 42, "n_topics": 3, "threads": 1,
            "lda": {"em_max_iter": 8, "var_max_iter": 8},
            "btm": {"iterations": 8}, "gsdmm": {"iterations": 8},
            "nmf": {"max_iter": 80}}


@pytest.mark.parametrize("method", ["lda", "btm", "gsdmm"])
def test_native_frozen_batch_resume_and_oov(method: str, tmp_path: Path) -> None:
    model = NativeTopicModel(method, config(), tmp_path / "first").fit(TRAIN)
    matrix = model.transform(QUERY)
    assert matrix.shape == (5, 3)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix[:3].sum(axis=1), 1)
    np.testing.assert_array_equal(matrix[3:], 0)
    assert len(model.topic_words()) == 3
    assert model.info["vocabulary_size"] == len(model.vocabulary_)
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in model.model_assets_.items()}
    checkpoint = tmp_path / "saved.joblib"
    joblib.dump(model, checkpoint)
    reloaded = joblib.load(checkpoint)
    resumed = np.concatenate([reloaded.transform(QUERY[:2]), reloaded.transform(QUERY[2:])])
    np.testing.assert_allclose(matrix, resumed, atol=1e-12)
    assert hashes == {name: hashlib.sha256(data).hexdigest()
                      for name, data in reloaded.model_assets_.items()}
    # Inference is independent of the original fit directory, not just joblib reload.
    reloaded.work_dir = tmp_path / "never_created"
    np.testing.assert_allclose(matrix, reloaded.transform(QUERY), atol=1e-12)
    second = NativeTopicModel(method, config(), tmp_path / "second").fit(TRAIN)
    np.testing.assert_allclose(matrix, second.transform(QUERY), atol=1e-12)


@pytest.mark.parametrize("method", ["lda", "btm", "gsdmm"])
def test_native_empty_training_and_no_biterm(method: str, tmp_path: Path) -> None:
    empty = NativeTopicModel(method, config(), tmp_path / "empty").fit(["", "啊啊啊"])
    np.testing.assert_array_equal(empty.transform(QUERY), np.zeros((5, 3)))
    one = NativeTopicModel(method, config(), tmp_path / "single").fit(["苹果", "篮球"])
    matrix = one.transform(["苹果", "苹果 篮球", "notinvocabulary"])
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix[:2].sum(axis=1), 1)
    np.testing.assert_array_equal(matrix[2], 0)


def test_gsdmm_repeated_words_change_frozen_posterior(tmp_path: Path) -> None:
    model = NativeTopicModel("gsdmm", config(), tmp_path).fit(TRAIN)
    matrix = model.transform(["苹果", "苹果 苹果 苹果 苹果", "苹果 篮球"])
    assert not np.allclose(matrix[0], matrix[1], atol=1e-8)
    assert np.isfinite(matrix).all()
    # Five very long messages stress log-space inference without truncating tokens.
    long_window = "\n".join(["苹果 篮球 威胁 " * 500] * 5)
    long_values = model.transform([long_window, long_window + "苹果 " * 1000])
    assert np.isfinite(long_values).all()
    np.testing.assert_allclose(long_values.sum(axis=1), 1)


def test_nmf_scale_correction_weighted_fit_and_padding(tmp_path: Path) -> None:
    model = NMFTopicModel(config()).fit(TRAIN, sample_weight=np.arange(1, 7))
    matrix = model.transform(QUERY)
    np.testing.assert_allclose(matrix[:3].sum(axis=1), 1)
    np.testing.assert_array_equal(matrix[3:], 0)
    assert model.info["risk_weighted_topics"] is True
    checkpoint = tmp_path / "nmf.joblib"
    joblib.dump(model, checkpoint)
    np.testing.assert_allclose(matrix, joblib.load(checkpoint).transform(QUERY), atol=1e-12)
    small = NMFTopicModel({"n_topics": 32}).fit(["苹果", "篮球"])
    assert small.transform(["苹果"]).shape == (1, 32)
    assert small.info["effective_topics"] == 2
    np.testing.assert_array_equal(small.transform(["苹果"])[0, 2:], 0)
    # Check membership factor-scale correction against the explicit calculation.
    raw = model.nmf_.transform(model.vectorizer_.transform(QUERY[:3]))
    raw *= model.nmf_.components_.sum(axis=1)
    np.testing.assert_allclose(matrix[:3], raw / raw.sum(axis=1, keepdims=True))


def test_nmf_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        NMFTopicModel(config()).fit(TRAIN, sample_weight=np.array([1, 0, 1, 1, 1, 1]))
    empty = NMFTopicModel(config()).fit(["", "哈哈哈哈"])
    np.testing.assert_array_equal(empty.transform([""]), np.zeros((1, 3)))


def test_gsdmm_java_available() -> None:
    result = subprocess.run([str(portable_java()), "-version"], capture_output=True,
                            text=True, check=True)
    assert "17.0.20.1" in result.stderr or "version" in result.stderr
