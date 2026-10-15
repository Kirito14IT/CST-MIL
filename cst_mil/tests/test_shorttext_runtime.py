import pytest


def test_unicode_words_retained_without_changing_old_tokens():
    from cst_mil.bench.common import topic_tokens as frozen
    from cst_mil.research_followup.shorttext.common import topic_tokens

    assert "não" in topic_tokens("não violência")
    assert "violência" in topic_tokens("não violência")
    assert "привет" in topic_tokens("привет мир")
    assert "नमस्ते" in topic_tokens("नमस्ते दुनिया")
    assert "judi" in topic_tokens("judi online")
    assert "não" not in frozen("não violência")
    assert topic_tokens("这是一条校园评论")
    assert topic_tokens("https://example.com @user 12345") == []


def test_backends_share_isolated_tokenizer():
    from cst_mil.research_followup.shorttext import common, model, native_topics

    assert model.topic_tokens is common.topic_tokens
    assert native_topics.topic_tokens is common.topic_tokens
    vocabulary = native_topics._vocabulary(["ação não crime"])
    assert "ação" in vocabulary and "não" in vocabulary


def test_resume_rejects_changed_source(tmp_path):
    from cst_mil.research_followup.shorttext.core import bind_or_verify

    path = tmp_path / "binding.json"
    bind_or_verify(path, {"source": "a", "seed": 42})
    bind_or_verify(path, {"source": "a", "seed": 42})
    with pytest.raises(ValueError):
        bind_or_verify(path, {"source": "b", "seed": 42})


@pytest.mark.parametrize("method", ["nmf", "cst_mil"])
def test_fit_isolation_metadata_and_checkpoint_parity(tmp_path, method):
    import joblib

    from cst_mil.research_followup.shorttext.model import predict_session, train_model

    sessions, messages = [], []
    for split, n in (("train", 16), ("dev", 6)):
        for i in range(n):
            pid = f"{split}:{i}"
            label = i % 2
            sessions.append(dict(post_id=pid, split=split, label=label, post_raw_text=""))
            text = (
                "violent hate ameaça 危险 攻击 " if label else "friend peaceful amigo 校园 学习 "
            ) + str(i)
            messages.append(
                dict(
                    post_id=pid,
                    comment_id=pid + ":0",
                    order=0,
                    raw_text=text,
                    model_text=text,
                    parent_id=None,
                    comment_label=None,
                )
            )
    config = dict(
        n_topics=4,
        seed=42,
        threads=1,
        nmf={"max_iter": 50},
        char_n_features=1024,
        mil_rounds=1,
        evidence_top_k=3,
    )
    model = train_model(sessions, messages, method, config, tmp_path / "assets")
    assert model.training_summary["training_sessions"] == 16
    assert model.training_summary["topic_backend"]["training_documents"] == 16
    assert not model.training_summary["message_labels_used_for_training"]
    s = dict(post_id="test", split="test", label=0, post_raw_text="")
    m = [
        dict(
            post_id="test",
            comment_id="test:0",
            order=0,
            raw_text="ameaça violent",
            model_text="ameaça violent",
            parent_id=None,
            comment_label=None,
        )
    ]
    p = predict_session(model, s, m)["risk_probability"]
    altered = dict(s, label=1, source_id="risk_scenario", filename="positive", original_label=5)
    assert predict_session(model, altered, m)["risk_probability"] == p
    checkpoint = tmp_path / "model.joblib"
    joblib.dump(model, checkpoint)
    assert predict_session(joblib.load(checkpoint), s, m)["risk_probability"] == p
