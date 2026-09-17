"""Core CPU-only CST-MIL training and inference pipeline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cst_mil.features import (
    RecordCollection,
    TopicFeatureExtractor,
    aggregate_session_features,
    build_char_vectorizer,
    canonicalise_messages,
    canonicalise_sessions,
)
from cst_mil.metrics import binary_classification_metrics, select_threshold

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "char_features": {
        "ngram_min": 2,
        "ngram_max": 5,
        "n_features": 65_536,
        "mil_rounds": 3,
        "positive_top_fraction": 0.20,
    },
    "topics": {
        "n_components": 8,
        "max_features": 20_000,
        "min_df": 2,
        "max_df": 0.95,
        "top_words": 12,
    },
    "evaluation": {"evidence_top_k": 3},
}


def _deep_merge(base: Mapping[str, Any], update: Mapping[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in base.items():
        result[key] = _deep_merge(value, None) if isinstance(value, Mapping) else value
    if update:
        for key, value in update.items():
            if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
                result[key] = _deep_merge(result[key], value)
            else:
                result[key] = value
    return result


def _new_instance_classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        class_weight=None,
        max_iter=1_000,
        tol=1e-3,
        shuffle=True,
        random_state=seed,
        average=True,
    )


def _balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    """Return weights equivalent to ``class_weight='balanced'``.

    Explicit weights keep the fitted SGD estimator compatible with a future
    ``partial_fit`` update path; this pilot still performs static fitting only.
    """

    binary = np.asarray(labels, dtype=int)
    counts = np.bincount(binary, minlength=2)
    if len(binary) == 0 or np.any(counts == 0):
        raise ValueError("Balanced sample weights require both classes")
    class_weights = len(binary) / (2.0 * counts.astype(float))
    return class_weights[binary]


def _new_session_classifier(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=seed,
                    solver="liblinear",
                ),
            ),
        ]
    )


@dataclass
class CSTModelBundle:
    """Serializable fitted components and the dev-selected decision rule."""

    char_vectorizer: HashingVectorizer
    instance_classifier: SGDClassifier
    topic_extractor: TopicFeatureExtractor
    session_classifier: Pipeline
    threshold: float
    feature_names: list[str]
    topic_words: list[list[str]]
    config: dict[str, Any]
    training_summary: dict[str, Any] = field(default_factory=dict)


def _validate_training_partitions(sessions: pd.DataFrame) -> None:
    available = set(sessions["split"].unique())
    missing = {"train", "dev"}.difference(available)
    if missing:
        raise ValueError(f"sessions is missing required partitions: {sorted(missing)}")
    for split in ("train", "dev"):
        labels = set(sessions.loc[sessions["split"] == split, "label"].astype(int).unique())
        if labels != {0, 1}:
            raise ValueError(f"{split} split must contain both binary classes")


def _fit_mil_instance_classifier(
    vectorizer: HashingVectorizer,
    train_messages: pd.DataFrame,
    session_labels: Mapping[str, int],
    *,
    seed: int,
    rounds: int,
    positive_top_fraction: float,
) -> tuple[SGDClassifier, list[dict[str, int]]]:
    if rounds < 0:
        raise ValueError("mil_rounds cannot be negative")
    if not 0.0 < positive_top_fraction <= 1.0:
        raise ValueError("positive_top_fraction must lie in (0, 1]")
    inherited_labels = train_messages["post_id"].map(session_labels)
    if inherited_labels.isna().any():
        raise ValueError("A training message references a session without a label")
    inherited = inherited_labels.to_numpy(dtype=int)
    if set(np.unique(inherited)) != {0, 1}:
        raise ValueError("Message instances must cover both training classes")

    matrix = vectorizer.transform(train_messages["context_model_text"].tolist())
    classifier = _new_instance_classifier(seed)
    classifier.fit(matrix, inherited, sample_weight=_balanced_sample_weights(inherited))

    summaries: list[dict[str, int]] = []
    for round_index in range(rounds):
        probabilities = classifier.predict_proba(matrix)[:, 1]
        selected_indices: list[int] = []
        selected_labels: list[int] = []

        negative_indices = np.flatnonzero(inherited == 0)
        selected_indices.extend(negative_indices.tolist())
        selected_labels.extend([0] * len(negative_indices))

        selected_positive = 0
        positive_bags = 0
        for post_id, group_indices in train_messages.groupby("post_id", sort=False).indices.items():
            if int(session_labels[str(post_id)]) != 1:
                continue
            positive_bags += 1
            group_array = np.asarray(group_indices, dtype=int)
            keep = max(1, int(math.ceil(len(group_array) * positive_top_fraction)))
            local_order = np.argsort(-probabilities[group_array], kind="mergesort")[:keep]
            retained = group_array[local_order]
            selected_indices.extend(retained.tolist())
            selected_labels.extend([1] * len(retained))
            selected_positive += len(retained)

        selected_array = np.asarray(selected_indices, dtype=int)
        label_array = np.asarray(selected_labels, dtype=int)
        if set(np.unique(label_array)) != {0, 1}:
            raise ValueError("MIL refinement selected only one class")
        classifier = _new_instance_classifier(seed + round_index + 1)
        classifier.fit(
            matrix[selected_array],
            label_array,
            sample_weight=_balanced_sample_weights(label_array),
        )
        summaries.append(
            {
                "round": round_index + 1,
                "negative_instances": int(len(negative_indices)),
                "positive_bags": int(positive_bags),
                "selected_positive_instances": int(selected_positive),
                "fitted_instances": int(len(selected_array)),
            }
        )
    return classifier, summaries


def _session_feature_table(
    sessions: pd.DataFrame,
    messages: pd.DataFrame,
    *,
    vectorizer: HashingVectorizer,
    instance_classifier: SGDClassifier,
    topic_extractor: TopicFeatureExtractor,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    session_ids = sessions["post_id"].tolist()
    message_subset = messages.loc[messages["post_id"].isin(session_ids)].copy()
    if len(message_subset):
        char_matrix = vectorizer.transform(message_subset["context_model_text"].tolist())
        message_risks = instance_classifier.predict_proba(char_matrix)[:, 1]
        message_topics = topic_extractor.transform(message_subset["model_text"].tolist())
    else:
        message_risks = np.zeros(0, dtype=float)
        message_topics = np.zeros((0, topic_extractor.n_components), dtype=float)

    root_topic_matrix = topic_extractor.transform(sessions["model_text"].tolist())
    root_topics = dict(zip(session_ids, root_topic_matrix, strict=True))
    features, _ = aggregate_session_features(
        message_subset,
        message_risks,
        message_topics,
        session_ids=session_ids,
        root_topics=root_topics,
    )
    return features, message_risks, message_topics


def train_cst_mil(
    sessions: RecordCollection,
    messages: RecordCollection,
    config: Mapping[str, Any] | None = None,
) -> CSTModelBundle:
    """Fit CST-MIL using session labels only.

    Expected prepared fields are ``post_id``, ``label``, ``split`` and
    ``model_text`` for sessions, and ``post_id``, ``comment_id``, ``model_text``,
    ``context_model_text``, ``parent_id`` and ``order`` for messages. Common raw
    SCCD aliases are also accepted. Any message-level annotation columns are
    discarded by :func:`canonicalise_messages` before fitting.
    """

    resolved = _deep_merge(DEFAULT_CONFIG, config)
    seed = int(resolved["seed"])
    session_frame = canonicalise_sessions(sessions, require_label=True, require_split=True)
    message_frame = canonicalise_messages(messages)
    _validate_training_partitions(session_frame)
    unknown_sessions = set(message_frame["post_id"]).difference(session_frame["post_id"])
    if unknown_sessions:
        preview = sorted(unknown_sessions)[:3]
        raise ValueError(f"messages references unknown sessions: {preview}")

    train_sessions = session_frame.loc[session_frame["split"] == "train"].copy()
    dev_sessions = session_frame.loc[session_frame["split"] == "dev"].copy()
    train_ids = set(train_sessions["post_id"])
    train_messages = message_frame.loc[message_frame["post_id"].isin(train_ids)].copy()
    if train_messages.empty:
        raise ValueError("No messages belong to the training sessions")

    char_config = resolved["char_features"]
    vectorizer = build_char_vectorizer(
        ngram_min=int(char_config["ngram_min"]),
        ngram_max=int(char_config["ngram_max"]),
        n_features=int(char_config["n_features"]),
    )
    session_label_map = dict(
        zip(train_sessions["post_id"], train_sessions["label"].astype(int), strict=True)
    )
    instance_classifier, mil_summary = _fit_mil_instance_classifier(
        vectorizer,
        train_messages,
        session_label_map,
        seed=seed,
        rounds=int(char_config["mil_rounds"]),
        positive_top_fraction=float(char_config["positive_top_fraction"]),
    )

    topic_config = resolved["topics"]
    topic_extractor = TopicFeatureExtractor(
        n_components=int(topic_config["n_components"]),
        max_features=int(topic_config["max_features"]),
        min_df=int(topic_config["min_df"]),
        max_df=float(topic_config["max_df"]),
        random_state=seed,
        top_words_count=int(topic_config["top_words"]),
    )
    topic_extractor.fit(train_messages["model_text"].tolist())

    train_features, _, _ = _session_feature_table(
        train_sessions,
        message_frame,
        vectorizer=vectorizer,
        instance_classifier=instance_classifier,
        topic_extractor=topic_extractor,
    )
    dev_features, _, _ = _session_feature_table(
        dev_sessions,
        message_frame,
        vectorizer=vectorizer,
        instance_classifier=instance_classifier,
        topic_extractor=topic_extractor,
    )
    feature_names = train_features.columns.tolist()
    if dev_features.columns.tolist() != feature_names:
        raise RuntimeError("Training and dev session feature schemas differ")

    session_classifier = _new_session_classifier(seed)
    train_labels = (
        train_sessions.set_index("post_id").loc[train_features.index, "label"].astype(int)
    )
    session_classifier.fit(train_features.to_numpy(dtype=float), train_labels.to_numpy())
    dev_probabilities = session_classifier.predict_proba(dev_features.to_numpy(dtype=float))[:, 1]
    dev_labels = dev_sessions.set_index("post_id").loc[dev_features.index, "label"].astype(int)
    threshold, dev_macro_f1 = select_threshold(dev_labels.to_numpy(), dev_probabilities)
    dev_metrics = binary_classification_metrics(
        dev_labels.to_numpy(),
        dev_probabilities,
        threshold=threshold,
    )
    words = topic_extractor.topic_words()
    training_summary = {
        "seed": seed,
        "training_sessions": int(len(train_sessions)),
        "dev_sessions": int(len(dev_sessions)),
        "training_messages": int(len(train_messages)),
        "mil_initialisation": "inherit_session_label",
        "mil_refinement_rounds": mil_summary,
        "selected_threshold": float(threshold),
        "selected_dev_macro_f1": float(dev_macro_f1),
        "dev_metrics": dev_metrics,
        "topic_components_requested": int(topic_extractor.n_components),
        "topic_components_fitted": int(topic_extractor.actual_components_),
        "topic_fallback_analyzer": topic_extractor.fallback_analyzer_,
        "message_labels_used_for_training": False,
        "char_estimator_partial_fit_compatible": True,
        "online_update_pipeline_implemented": False,
    }
    return CSTModelBundle(
        char_vectorizer=vectorizer,
        instance_classifier=instance_classifier,
        topic_extractor=topic_extractor,
        session_classifier=session_classifier,
        threshold=float(threshold),
        feature_names=feature_names,
        topic_words=words,
        config=resolved,
        training_summary=training_summary,
    )


def predict_sessions(
    model: CSTModelBundle,
    sessions: RecordCollection,
    messages: RecordCollection,
    *,
    evidence_top_k: int | None = None,
    return_message_scores: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Predict session risk and rank the highest-risk comment evidence.

    By default, a session-level DataFrame is returned. Set
    ``return_message_scores=True`` to also receive a second DataFrame containing
    every message score and its within-session rank.
    """

    session_frame = canonicalise_sessions(sessions, require_label=False, require_split=False)
    message_frame = canonicalise_messages(messages)
    wanted_ids = set(session_frame["post_id"])
    unknown_sessions = set(message_frame["post_id"]).difference(wanted_ids)
    if unknown_sessions:
        preview = sorted(unknown_sessions)[:3]
        raise ValueError(f"messages references sessions not provided for prediction: {preview}")
    message_frame = message_frame.loc[message_frame["post_id"].isin(wanted_ids)].reset_index(
        drop=True
    )

    features, message_risks, message_topics = _session_feature_table(
        session_frame,
        message_frame,
        vectorizer=model.char_vectorizer,
        instance_classifier=model.instance_classifier,
        topic_extractor=model.topic_extractor,
    )
    if features.columns.tolist() != model.feature_names:
        raise ValueError("Prediction session features do not match the fitted model schema")
    probabilities = model.session_classifier.predict_proba(features.to_numpy(dtype=float))[:, 1]
    predictions = (probabilities >= model.threshold).astype(int)

    message_predictions = message_frame[
        ["post_id", "comment_id", "parent_id", "order", "raw_text"]
    ].copy()
    message_predictions["risk_probability"] = message_risks
    message_predictions["dominant_topic"] = (
        np.argmax(message_topics, axis=1).astype(int)
        if len(message_topics)
        else np.asarray([], dtype=int)
    )
    message_predictions["rank"] = (
        message_predictions.groupby("post_id")["risk_probability"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    top_k = (
        int(model.config["evaluation"]["evidence_top_k"])
        if evidence_top_k is None
        else int(evidence_top_k)
    )
    if top_k < 1:
        raise ValueError("evidence_top_k must be positive")

    topic_lookup = {index: words for index, words in enumerate(model.topic_words)}
    session_rows: list[dict[str, Any]] = []
    for position, session in enumerate(session_frame.itertuples(index=False)):
        group = message_predictions.loc[message_predictions["post_id"] == session.post_id]
        evidence_group = group.sort_values(
            ["risk_probability", "order", "comment_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).head(top_k)
        evidence = [
            {
                "comment_id": row.comment_id,
                "raw_text": row.raw_text,
                "risk_probability": float(row.risk_probability),
                "parent_id": row.parent_id,
                "rank": rank,
            }
            for rank, row in enumerate(evidence_group.itertuples(index=False), start=1)
        ]

        group_indices = group.index.to_numpy(dtype=int)
        if len(group_indices):
            mean_topic = np.mean(message_topics[group_indices], axis=0)
        else:
            mean_topic = model.topic_extractor.transform([session.model_text])[0]
        top_topic_ids = np.argsort(mean_topic, kind="mergesort")[::-1][:2]
        top_topics = [
            {
                "topic_id": int(topic_id),
                "weight": float(mean_topic[topic_id]),
                "keywords": topic_lookup.get(int(topic_id), []),
            }
            for topic_id in top_topic_ids
        ]
        row: dict[str, Any] = {
            "post_id": session.post_id,
            "split": session.split,
            "risk_probability": float(probabilities[position]),
            "prediction": int(predictions[position]),
            "prediction_name": "CB" if predictions[position] else "Non-CB",
            "threshold": float(model.threshold),
            "top_topics": top_topics,
            "evidence": evidence,
        }
        if not pd.isna(session.label):
            row["label"] = int(session.label)
        session_rows.append(row)

    session_predictions = pd.DataFrame.from_records(session_rows)
    if return_message_scores:
        return session_predictions, message_predictions
    return session_predictions


def save_model(model: CSTModelBundle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_model(path: str | Path) -> CSTModelBundle:
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, CSTModelBundle):
        raise TypeError("The serialized object is not a CSTModelBundle")
    return loaded


def serialise_prediction_value(value: Any) -> Any:
    """Convert nested prediction cells to deterministic JSON for CSV output."""

    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value
