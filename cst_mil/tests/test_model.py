from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from cst_mil.metrics import select_threshold
from cst_mil.model import load_model, predict_sessions, save_model, train_cst_mil


def _synthetic_records() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sessions: list[dict[str, object]] = []
    messages: list[dict[str, object]] = []
    split_sizes = {"train": 12, "dev": 6, "test": 6}
    session_number = 0
    for split, count in split_sizes.items():
        for within_split in range(count):
            label = within_split % 2
            post_id = f"p-{session_number:02d}"
            root = "公开羞辱攻击讨论" if label else "日常友好聊天"
            sessions.append(
                {
                    "post_id": post_id,
                    "raw_text": root,
                    "model_text": root,
                    "label": label,
                    "split": split,
                }
            )
            previous_id: str | None = None
            for order in range(4):
                comment_id = f"{post_id}-c-{order}"
                if label:
                    text = (
                        f"恶意辱骂蠢货攻击滚开 {order}"
                        if order >= 2
                        else f"围攻那个对象 {order}"
                    )
                    comment_label = "CB" if order >= 2 else "Non-CB"
                else:
                    text = f"天气不错友善交流谢谢 {order}"
                    comment_label = "Non-CB"
                context = f"上下文 {text}" if previous_id else text
                messages.append(
                    {
                        "comment_id": comment_id,
                        "post_id": post_id,
                        "raw_text": text,
                        "model_text": text,
                        "context_model_text": context,
                        "parent_id": previous_id,
                        "comment_label": comment_label,
                        "split": split,
                        "order": order,
                    }
                )
                previous_id = comment_id
            session_number += 1
    return sessions, messages


def _small_config() -> dict[str, object]:
    return {
        "seed": 42,
        "char_features": {
            "n_features": 1_024,
            "mil_rounds": 3,
            "positive_top_fraction": 0.2,
        },
        "topics": {"n_components": 4, "max_features": 500, "min_df": 1, "max_df": 1.0},
        "evaluation": {"evidence_top_k": 3},
    }


def test_select_threshold_breaks_macro_f1_ties_toward_lower_value() -> None:
    threshold, score = select_threshold([0, 1], [0.8, 0.8])
    assert threshold == 0.0
    assert np.isclose(score, 1.0 / 3.0)


def test_train_predict_rank_and_joblib_roundtrip(tmp_path: Path) -> None:
    sessions, messages = _synthetic_records()
    model = train_cst_mil(sessions, messages, _small_config())
    model_path = save_model(model, tmp_path / "model.joblib")
    model = load_model(model_path)

    test_sessions = [row for row in sessions if row["split"] == "test"]
    test_messages = [row for row in messages if row["split"] == "test"]
    predictions, message_scores = predict_sessions(
        model,
        test_sessions,
        test_messages,
        return_message_scores=True,
    )

    assert len(predictions) == 6
    assert {"post_id", "label", "prediction", "risk_probability", "evidence"}.issubset(
        predictions.columns
    )
    assert predictions["risk_probability"].between(0.0, 1.0).all()
    assert all(len(items) == 3 for items in predictions["evidence"])
    assert message_scores.groupby("post_id")["rank"].nunique().eq(4).all()
    assert model.training_summary["message_labels_used_for_training"] is False
    assert len(model.training_summary["mil_refinement_rounds"]) == 3
    assert len(model.feature_names) == 12 + 3 * 4


def test_message_labels_cannot_change_training_or_predictions() -> None:
    sessions, messages = _synthetic_records()
    altered_messages = copy.deepcopy(messages)
    for row in altered_messages:
        row["comment_label"] = "CB" if row["comment_label"] == "Non-CB" else "Non-CB"

    model_a = train_cst_mil(sessions, messages, _small_config())
    model_b = train_cst_mil(sessions, altered_messages, _small_config())
    test_sessions = pd.DataFrame([row for row in sessions if row["split"] == "test"])
    test_messages_a = pd.DataFrame([row for row in messages if row["split"] == "test"])
    test_messages_b = pd.DataFrame([row for row in altered_messages if row["split"] == "test"])
    predictions_a = predict_sessions(model_a, test_sessions, test_messages_a)
    predictions_b = predict_sessions(model_b, test_sessions, test_messages_b)

    np.testing.assert_allclose(
        predictions_a["risk_probability"].to_numpy(),
        predictions_b["risk_probability"].to_numpy(),
        rtol=0.0,
        atol=0.0,
    )
    assert predictions_a["prediction"].tolist() == predictions_b["prediction"].tolist()


def test_char_estimator_is_partial_fit_compatible_but_pipeline_is_static() -> None:
    sessions, messages = _synthetic_records()
    model = train_cst_mil(sessions, messages, _small_config())
    update_matrix = model.char_vectorizer.transform(["新的辱骂变体表达"])
    model.instance_classifier.partial_fit(
        update_matrix,
        np.asarray([1]),
        sample_weight=np.asarray([1.0]),
    )
    assert model.training_summary["char_estimator_partial_fit_compatible"] is True
    assert model.training_summary["online_update_pipeline_implemented"] is False
