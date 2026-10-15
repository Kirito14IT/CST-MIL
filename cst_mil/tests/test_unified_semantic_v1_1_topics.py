import numpy as np
import pytest

from cst_mil.unified_semantic_v1_1.topics import interpret_topics, phrase_candidates


def rows(texts):
    return [
        {"post_id": "p", "comment_id": f"m{i}", "raw_text": text, "order": i, "parent_id": None}
        for i, text in enumerate(texts)
    ]


def test_complete_words_do_not_match_substrings_or_generic_titles():
    phrases = phrase_candidates("下次比赛还去看吗")
    assert "比" not in phrases and "看" not in phrases and "还" not in phrases
    assert "比赛" in phrases


def test_counter_speech_not_grouped_with_confident_attack():
    messages = rows(
        [
            "你这个废物赶紧滚出队伍",
            "这种废物别参加比赛",
            "不要骂队员，停止人身攻击",
            "不要围攻同学，停止辱骂",
        ]
    )
    vectors = np.tile([1.0, 0.0], (4, 1))
    scores = np.array([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1], [0.2, 0.1, 0.1], [0.2, 0.1, 0.1]])
    result = interpret_topics(
        messages,
        vectors,
        scores,
        distance_threshold=0.35,
        category_thresholds={"attack": 0.5, "hate": 0.5, "spam_scam": 0.5},
    )
    assert len(result["topics"]) == 2
    assert {frozenset(topic["message_ids"]) for topic in result["topics"]} == {
        frozenset(["m0", "m1"]),
        frozenset(["m2", "m3"]),
    }
    for topic in result["topics"]:
        assert topic["keyword_supports"]
        for word, supports in topic["keyword_supports"].items():
            assert supports
            for support in supports:
                raw = messages[int(support["message_id"][1:])]["raw_text"]
                assert raw[support["start"] : support["end"]].casefold() == word


def test_singletons_keep_risk_evidence_and_do_not_fake_topics():
    result = interpret_topics(
        rows(["你好同学"]),
        np.array([[1.0, 0.0]]),
        np.array([[0.1, 0.1, 0.1]]),
        distance_threshold=0.35,
        category_thresholds={"attack": 0.5},
    )
    assert result["topics"] == [] and result["coverage"] == 0
    assert len(result["evidence"]) == 1


def test_unicode_original_support_and_sparse_order_segments():
    messages = rows(["não ofenda colegas", "não ofenda colegas"])
    messages[1]["order"] = 20
    result = interpret_topics(
        messages,
        np.array([[1.0, 0.0], [1.0, 0.0]]),
        np.ones((2, 3)) * 0.1,
        distance_threshold=0.35,
        category_thresholds={"attack": 0.5},
    )
    assert len(result["topics"]) == 1
    assert result["topics"][0]["segments"][0]["end_position"] == 1
    assert any("ofenda" in word for word in result["topics"][0]["keywords"])


def test_decomposed_unicode_word_boundaries_preserved():
    phrases = phrase_candidates("na\u0303o ofenda colegas")
    assert "não ofenda" in phrases
    assert "na" not in phrases


def test_invalid_scores_rejected():
    with pytest.raises(ValueError):
        interpret_topics(
            rows(["你好"]),
            np.ones((1, 2)),
            np.array([[2.0, 0.0, 0.0]]),
            distance_threshold=0.35,
            category_thresholds={"attack": 0.5},
        )


def test_decorative_function_words_are_not_theme_names():
    phrases = phrase_candidates("𝓉𝒽𝓮 𝓎𝓸𝓊")
    assert not phrases


def test_extract_names_keep_explicit_negation_and_question_polarity():
    phrases = phrase_candidates("without giving up")
    assert "without giving up" in phrases
    assert "giving up" not in phrases
    phrases = phrase_candidates("不可能惠及老百姓")
    assert "可能惠及老百姓" not in phrases
    assert any(word.startswith("不可能") for word in phrases)
    phrases = phrase_candidates("这样的事会不会发生到学校")
    assert not any(word.startswith("不会") for word in phrases)


def test_short_compound_name_is_not_replaced_by_its_familiar_prefix():
    phrases = phrase_candidates("孟子旋今天来了")
    assert "孟子旋" in phrases
    assert "孟子" not in phrases
    assert "孟子" in phrase_candidates("孟子")


def test_distant_negation_is_not_turned_into_a_positive_assertion():
    assert "逃避能得到赞赏" not in phrase_candidates("没有任何一种逃避能得到赞赏")
    assert not phrase_candidates("不能让外国专家到吗？")
    assert not phrase_candidates("𝓮𝔁𝓽𝓻𝓮𝓶𝓮𝓵𝔂")
    assert "认真训练" in phrase_candidates("不要辱骂队员，认真训练")


def test_closed_class_chinese_words_cannot_stand_alone_as_names():
    from cst_mil.unified_semantic_v1_1.topics import generic_word

    for word in ("非常", "一部", "这几个", "非常一部"):
        assert generic_word(word)
    for word in ("养车费用", "比赛表现", "垃圾赶紧滚出", "别再出来丢人"):
        assert not generic_word(word)
    phrases = phrase_candidates("一部电影非常感人")
    assert "一部" not in phrases and "非常" not in phrases
    assert any("电影" in phrase for phrase in phrases)


def test_function_word_subclasses_and_modality_do_not_become_standalone_titles():
    from cst_mil.unified_semantic_v1_1.topics import generic_word

    for word in ("确实", "应该", "应该是有人", "非常一部", "should", "the you"):
        assert generic_word(word)
    assert "应该送蛋糕" in phrase_candidates("应该送蛋糕")
    assert "认真训练" in phrase_candidates("认真训练")
    assert "without giving up" in phrase_candidates("without giving up")
    assert "should stop bullying" in phrase_candidates("should stop bullying")
