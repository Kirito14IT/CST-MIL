"""Small model invariants; no SCCD test labels are used by these fixtures."""

import copy
import json

import joblib
import numpy as np
import pytest

from cst_mil.bench import model as bench_model
from cst_mil.bench.common import build_windows, normalise_rows, topic_tokens
from cst_mil.bench.interpretation import interpret_topics


class TinyTopics:
    """Deterministic lexical backend to isolate risk/interpretation behavior."""

    def fit(self, texts, sample_weight=None):
        self.vocabulary_ = {word: i for i, word in enumerate(sorted({
            word for text in texts for word in topic_tokens(text)
        }))}
        self.words = [[], [], []]
        for word, index in self.vocabulary_.items():
            self.words[index % 3].append(word)
        self.sample_weight = sample_weight
        self.info = {"name": "test-only-lexical-backend"}
        return self

    def transform(self, texts):
        matrix = np.zeros((len(texts), 3))
        for index, text in enumerate(texts):
            for word in topic_tokens(text):
                if word in self.vocabulary_:
                    matrix[index, self.vocabulary_[word] % 3] += 1
        return normalise_rows(matrix)

    def topic_words(self, top_n=10):
        return [words[:top_n] for words in self.words]


@pytest.fixture
def data():
    sessions, messages = [], []
    for index in range(10):
        split = "train" if index < 6 else "dev" if index < 8 else "test"
        label = index % 2
        post_id = f"p{index}"
        sessions.append({"post_id": post_id, "label": label, "split": split,
                         "post_raw_text": "原帖"})
        words = "谩骂 欺凌 威胁 垃圾滚开" if label else "音乐 旅行 天气 朋友快乐"
        if split == "dev":
            words += " DEVONLY_SENTINEL"
        for order in range(7):
            messages.append({
                "post_id": post_id, "comment_id": f"{post_id}m{order}",
                "raw_text": words + f" 文本{order}", "model_text": words + f" 文本{order}",
                "order": order, "parent_id": f"{post_id}m{order - 1}" if order else None,
                "speaker_id": f"speaker{order}", "comment_time": order * 10,
                "comment_label": label, "label": label,
            })
    return sessions, messages


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setattr(bench_model, "_new_topics", lambda *args: TinyTopics())


@pytest.mark.parametrize("number", [0, 1, 2, 3])
def test_variable_topic_count(number):
    messages = [{"comment_id": f"m{i}", "post_id": "p", "raw_text": "旅行 谩骂 音乐",
                 "speaker_id": "anonymous", "order": i} for i in range(6)]
    weights = np.zeros((6, 3))
    for topic in range(number):
        weights[2 * topic:2 * topic + 2, topic] = 1
    result = interpret_topics(messages, weights, np.linspace(0, 1, 6),
                              [["旅行"], ["谩骂"], ["音乐"]])
    assert len(result["topics"]) == number
    assert result["coverage"] == number / 3


def test_soft_overlap_disjoint_segments_original_metadata():
    messages = [{"post_id": "p", "comment_id": f"m{i}", "raw_text": "旅行期间遭到谩骂",
                 "speaker_id": "actor", "order": i * 100, "comment_time": f"t{i}",
                 "comment_label": "not-for-output"} for i in range(4)]
    weights = np.asarray([[0.5, 0.5], [1.0, 0], [0, 1.0], [0.5, 0.5]])
    result = interpret_topics(messages, weights, np.asarray([0.2, 0.4, 0.8, 0.9]),
                              [["旅行"], ["谩骂"]])
    assert result["message_scores"][0]["topic_ids"] == [0, 1]
    topic = next(topic for topic in result["topics"] if topic["topic_id"] == 0)
    assert [segment["message_ids"] for segment in topic["segments"]] == [["m0", "m1"], ["m3"]]
    assert topic["evidence"][0]["comment_id"] == "m3"
    assert topic["support_messages"][0]["speaker_id"] == "actor"
    assert "comment_label" not in json.dumps(result)
    assert topic["title"] in messages[0]["raw_text"]


def test_diffuse_and_weak_single_message_topics_not_forced():
    message = [{"comment_id": "m", "raw_text": "旅行 天气"}]
    assert interpret_topics(message, np.ones((1, 8)), np.asarray([0.8]), [[]] * 8)["topics"] == []
    assert interpret_topics(message, np.asarray([[0.55, 0.45]]),
                            np.asarray([0.8]), [[], []])["topics"] == []


@pytest.mark.parametrize("method", ["nmf", "cst_mil"])
def test_training_is_crossfit_label_independent_and_reload_stable(data, tmp_path, method):
    sessions, messages = data
    config = {"seed": 42, "char_n_features": 1024, "word_features": True,
              "mil_rounds": 1}
    fitted = bench_model.train_model(sessions, messages, method, config, tmp_path)
    summary = fitted.training_summary
    assert summary["message_labels_used_for_training"] is False
    assert len(summary["crossfit_folds"]) == 3
    held_bags = []
    for fold in summary["crossfit_folds"]:
        assert not set(fold["train_bag_indices"]) & set(fold["held_bag_indices"])
        held_bags.extend(fold["held_bag_indices"])
    assert sorted(held_bags) == list(range(6))
    assert "devonly_sentinel" not in fitted.topic_model.vocabulary_
    if method == "cst_mil":
        assert not any("devonly" in token
                       for token in fitted.risk_features.word_vectorizer.vocabulary_)
    changed = copy.deepcopy(messages)
    for message in changed:
        message["label"] = 1 - message["label"]
        message["comment_label"] = "unavailable"
        message["speaker_id"] = "different-private-id"
    second = bench_model.train_model(sessions, changed, method, config, tmp_path)
    assert [p["risk_probability"] for p in fitted.dev_predictions] == [
        p["risk_probability"] for p in second.dev_predictions
    ]
    path = tmp_path / "model.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    prediction = bench_model.predict_session(fitted, sessions[-1], messages)
    assert prediction == bench_model.predict_session(loaded, sessions[-1], messages)
    assert isinstance(prediction["coverage"], float)
    assert set(range(1, 8)) == {row["rank"] for row in prediction["message_scores"]}
    json.dumps(prediction, allow_nan=False)


def test_smoke_single_class_folds_are_explicit_no_in_sample_fallback(data, tmp_path):
    sessions, messages = data
    sessions = [sessions[0], sessions[1], sessions[6], sessions[7], sessions[8]]
    ids = {session["post_id"] for session in sessions}
    messages = [message for message in messages if message["post_id"] in ids]
    fitted = bench_model.train_model(sessions, messages, "cst_mil",
                                    {"char_n_features": 512, "mil_rounds": 1}, tmp_path)
    assert fitted.training_summary["single_class_smoke_fallback"] is True
    assert len(fitted.training_summary["crossfit_folds"]) == 2
    assert all(fold["single_class_constant_fallback"]
               for fold in fitted.training_summary["crossfit_folds"])
    assert fitted.training_summary["risk_classifier_fit_count"] == 2


def test_oov_messages_unassigned_despite_known_neighbors_and_root_fallback(data, tmp_path):
    sessions, messages = data
    fitted = bench_model.train_model(sessions, messages, "nmf", {}, tmp_path)
    session = {"post_id": "new", "post_raw_text": "音乐旅行", "split": "inference"}
    local = [{"post_id": "new", "comment_id": "a", "raw_text": "音乐 旅行", "order": 0},
             {"post_id": "new", "comment_id": "b", "raw_text": "qxzvunknown", "order": 1}]
    prediction = bench_model.predict_session(fitted, session, local)
    assert prediction["message_scores"][1]["topic_strengths"] == [0.0, 0.0, 0.0]
    assert prediction["message_scores"][1]["topic_ids"] == []
    root = bench_model.predict_session(fitted, session, [])
    assert root["message_scores"][0]["comment_id"] == "new:root"
    assert root["message_scores"][0]["raw_text"] == "音乐旅行"


def test_windows_cover_long_tail_and_risk_weights_use_oof(data, tmp_path):
    session = {"post_id": "long"}
    local = [{"post_id": "long", "comment_id": str(i), "raw_text": "长文本" * 1000,
              "order": i} for i in range(18)]
    windows = build_windows(session, local)
    assert {mid for window in windows for mid in window["message_ids"]} == {
        str(i) for i in range(18)
    }
    assert windows[-1]["message_ids"][-1] == "17"
    sessions, messages = data
    model = bench_model.train_model(sessions, messages, "cst_mil", {
        "risk_weighted_topics": True, "multiscale": True,
        "char_n_features": 512, "mil_rounds": 1,
    }, tmp_path)
    assert model.topic_model.sample_weight.min() >= 1
    assert model.topic_model.sample_weight.max() <= 3
    assert model.training_summary["risk_weighted_topics"] is True
    assert len(model.dev_predictions) == 2


def test_class_and_reference_validation(data, tmp_path):
    sessions, messages = data
    broken = [session for session in sessions if session["split"] != "dev" or session["label"] == 1]
    with pytest.raises(ValueError, match="dev must contain both"):
        bench_model.train_model(broken, messages, "nmf", {}, tmp_path)


def test_all_noise_backend_keeps_zero_topics(data, tmp_path, monkeypatch):
    class EmptyTopics(TinyTopics):
        def transform(self, texts):
            return np.zeros((len(texts), 0))

        def topic_words(self, top_n=10):
            return []

    monkeypatch.setattr(bench_model, "_new_topics", lambda *args: EmptyTopics())
    sessions, messages = data
    fitted = bench_model.train_model(sessions, messages, "bertopic", {}, tmp_path)
    assert fitted.training_summary["risk_head_zero_topic_prior"] is True
    assert fitted.training_summary["single_class_smoke_fallback"] is False
    assert fitted.dev_predictions[0]["topics"] == []
    assert fitted.dev_predictions[0]["message_scores"][0]["topic_strengths"] == []
    assert fitted.dev_predictions[0]["coverage"] == 0.0


def test_real_nmf_backend_integrates_with_weighted_cst(data, tmp_path, monkeypatch):
    from cst_mil.bench.native_topics import NMFTopicModel

    monkeypatch.setattr(bench_model, "_new_topics",
                        lambda method, config, path: NMFTopicModel(config))
    sessions, messages = data
    fitted = bench_model.train_model(sessions, messages, "cst_mil", {
        "n_topics": 3, "nmf": {"max_iter": 100}, "risk_weighted_topics": True,
        "char_n_features": 512, "mil_rounds": 0,
    }, tmp_path)
    assert fitted.topic_model.info["backend"] == "sklearn.decomposition.NMF"
    assert fitted.topic_model.info["risk_weighted_topics"] is True
    prediction = bench_model.predict_session(fitted, sessions[-1], messages)
    assert len(prediction["message_scores"]) == 7
    assert len(prediction["message_scores"][0]["topic_strengths"]) == 3
    assert np.isfinite(prediction["risk_probability"])
