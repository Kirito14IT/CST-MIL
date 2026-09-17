"""Shared SCCD benchmark risk learning and fixed-model session inference.

Topic backends fit training windows only. Their unsupervised space is fixed
across supervised cross-fitting; fold-local risk estimators never see labels
of held-out bags. Comment annotations are not read by this module.
"""

from __future__ import annotations

import copy
import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cst_mil.metrics import binary_classification_metrics, select_threshold
from cst_mil.preprocess import normalize_text

from .common import build_windows, canonical_messages, json_safe, normalise_rows, topic_tokens
from .interpretation import interpret_topics


def _new_topics(method: str, config: dict[str, Any], work_dir: Path) -> Any:
    if method == "bertopic":
        from .bertopic_backend import BERTopicModel

        return BERTopicModel(config, work_dir)
    from .native_topics import NativeTopicModel, NMFTopicModel

    if method in {"cst_mil", "nmf"}:
        return NMFTopicModel(config)
    if method in {"lda", "btm", "gsdmm"}:
        return NativeTopicModel(method, config, work_dir)
    raise ValueError(f"Unsupported benchmark method: {method}")


def _messages(session: dict[str, Any], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    post_id = str(session["post_id"])
    result = canonical_messages(session, messages)
    ids = [str(item["comment_id"]) for item in result]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate message IDs in session {post_id}")
    return result


def _risk_texts(messages: list[dict[str, Any]]) -> list[str]:
    lookup = {str(message["comment_id"]): normalize_text(
        message.get("raw_text", message.get("model_text", ""))
    ) for message in messages}
    result = []
    for message in messages:
        current = lookup[str(message["comment_id"])]
        parent = lookup.get(str(message.get("parent_id", "")), "")
        result.append(f"{parent} <reply> {current}" if parent else current)
    return result


def risk_word_tokens(text: str) -> list[str]:
    """Risk vocabulary keeps negation, abuse and repetitions (no topic stop list)."""
    import jieba

    return [token.strip() for token in jieba.lcut(normalize_text(text)) if token.strip()]


@dataclass
class RiskTextFeatures:
    config: dict[str, Any]
    char_vectorizer: Any = None
    word_vectorizer: Any = None

    def fit_transform(self, texts: list[str]) -> sparse.csr_matrix:
        self.char_vectorizer = HashingVectorizer(
            analyzer="char", ngram_range=(2, 5),
            n_features=int(self.config.get("char_n_features", 65536)),
            alternate_sign=False, norm="l2", lowercase=False,
        )
        char_matrix = self.char_vectorizer.transform(texts)
        if not self.config.get("word_features", False):
            return char_matrix.tocsr()
        self.word_vectorizer = TfidfVectorizer(
            tokenizer=risk_word_tokens, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), max_features=30000, sublinear_tf=True,
        )
        try:
            word_matrix = self.word_vectorizer.fit_transform(texts)
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            self.word_vectorizer = None
            return char_matrix.tocsr()
        return sparse.hstack([char_matrix, word_matrix], format="csr")

    def transform(self, texts: list[str]) -> sparse.csr_matrix:
        char_matrix = self.char_vectorizer.transform(texts)
        if self.word_vectorizer is None:
            return char_matrix.tocsr()
        return sparse.hstack([char_matrix, self.word_vectorizer.transform(texts)], format="csr")


@dataclass
class ConstantRisk:
    """Explicit prior for single-class folds or absence of any topic dimensions."""

    probability: float

    def predict_proba(self, matrix: Any) -> np.ndarray:
        values = np.full(matrix.shape[0], self.probability)
        return np.column_stack([1 - values, values])


def _bag_weights(bags: np.ndarray, labels: np.ndarray) -> np.ndarray:
    unique, counts = np.unique(bags, return_counts=True)
    bag_labels = {bag: int(labels[np.flatnonzero(bags == bag)[0]]) for bag in unique}
    class_counts = np.bincount(list(bag_labels.values()), minlength=2)
    weights = np.empty(len(bags), dtype=float)
    for bag, count in zip(unique, counts, strict=True):
        # Every bag of a given class contributes equally, independent of length.
        weights[bags == bag] = 1.0 / (count * max(1, class_counts[bag_labels[bag]]))
    return weights * (len(weights) / weights.sum())


def _fit_risk(
    matrix: Any, labels: np.ndarray, bags: np.ndarray, *, mil: bool, config: dict[str, Any],
) -> tuple[Any, int]:
    if len(np.unique(labels)) == 1:
        return ConstantRisk(float(labels[0])), 0
    if matrix.shape[1] == 0:
        # The two classes receive equal total bag weight. Keep no-topic outputs
        # empty rather than inventing a semantic topic just to fit an estimator.
        return ConstantRisk(0.5), 0
    rounds = int(config.get("mil_rounds", 3)) if mil else 0
    fraction = float(config.get("positive_top_fraction", 0.2))
    if rounds < 0 or not 0 < fraction <= 1:
        raise ValueError("MIL rounds/fraction are invalid")
    indices = np.arange(len(labels))
    classifier = None
    for round_index in range(rounds + 1):
        classifier = LogisticRegression(
            C=float(config.get("instance_C", config.get("instance_c", 4.0 if mil else 1.0))),
            solver="liblinear", max_iter=1000, random_state=int(config.get("seed", 42)),
        )
        classifier.fit(matrix[indices], labels[indices],
                       sample_weight=_bag_weights(bags[indices], labels[indices]))
        if round_index < rounds:
            scores = classifier.predict_proba(matrix)[:, 1]
            selected = np.flatnonzero(labels == 0).tolist()
            for bag in np.unique(bags[labels == 1]):
                candidates = np.flatnonzero(bags == bag)
                count = max(1, math.ceil(len(candidates) * fraction))
                order = np.argsort(-scores[candidates], kind="stable")[:count]
                selected.extend(candidates[order].tolist())
            indices = np.asarray(selected, dtype=int)
    return classifier, rounds + 1


def _crossfit_risk(
    texts: list[str] | None,
    topic_matrix: np.ndarray | None,
    bags: np.ndarray,
    labels_by_bag: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, Any, Any, list[dict[str, Any]], int]:
    is_cst = texts is not None
    counts = np.bincount(labels_by_bag, minlength=2)
    if min(counts) < 2:
        if len(labels_by_bag) != 2:
            raise ValueError("Cross-fitting needs at least two bags per class except smoke2")
        partitions = list(LeaveOneOut().split(labels_by_bag))
    else:
        partitions = list(StratifiedKFold(
            n_splits=min(3, int(min(counts))), shuffle=True,
            random_state=int(config.get("seed", 42)),
        ).split(np.arange(len(labels_by_bag)), labels_by_bag))
    scores = np.full(len(bags), np.nan)
    history: list[dict[str, Any]] = []
    fit_count = 0
    inherited_labels = labels_by_bag[bags]
    for fold, (train_bags, held_bags) in enumerate(partitions):
        train_indices = np.flatnonzero(np.isin(bags, train_bags))
        held_indices = np.flatnonzero(np.isin(bags, held_bags))
        if is_cst:
            features = RiskTextFeatures(config)
            fit_matrix = features.fit_transform([texts[index] for index in train_indices])
            held_matrix = features.transform([texts[index] for index in held_indices])
        else:
            fit_matrix = topic_matrix[train_indices]
            held_matrix = topic_matrix[held_indices]
        classifier, fits = _fit_risk(
            fit_matrix, inherited_labels[train_indices], bags[train_indices],
            mil=is_cst, config=config,
        )
        scores[held_indices] = classifier.predict_proba(held_matrix)[:, 1]
        fit_count += fits
        history.append({
            "fold": fold, "train_bag_indices": train_bags.tolist(),
            "held_bag_indices": held_bags.tolist(), "fitted_instances": len(train_indices),
            "held_instances": len(held_indices), "risk_classifier_fits": fits,
            "single_class_constant_fallback": len(np.unique(inherited_labels[train_indices])) == 1,
            "zero_topic_constant_fallback": fit_matrix.shape[1] == 0,
        })
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Cross-fitting left missing risk scores")
    features = RiskTextFeatures(config) if is_cst else None
    full_matrix = features.fit_transform(texts) if is_cst else topic_matrix
    classifier, fits = _fit_risk(full_matrix, inherited_labels, bags, mil=is_cst, config=config)
    return scores, features, classifier, history, fit_count + fits


def _window_average(
    messages: list[dict[str, Any]], windows: list[dict[str, Any]], values: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    result = np.zeros((len(messages), values.shape[1]), dtype=float)
    counts = np.zeros(len(messages), dtype=float)
    indices = {str(message["comment_id"]): index for index, message in enumerate(messages)}
    for window, row in zip(windows, values, strict=True):
        for message_id in window["message_ids"]:
            index = indices[str(message_id)]
            result[index] += row
            counts[index] += 1
    if np.any(counts == 0):
        raise RuntimeError("Window construction failed to cover every message")
    return result / counts[:, None]


def _known_message_mask(topic_model: Any, messages: list[dict[str, Any]]) -> np.ndarray:
    vocabulary = getattr(topic_model, "vocabulary_", None)
    if vocabulary is None:
        vocabulary = getattr(topic_model, "vocab_", None)
    result = []
    for message in messages:
        tokens = topic_tokens(str(message.get("model_text", message.get("raw_text", ""))))
        result.append(bool(tokens) and (vocabulary is None or any(t in vocabulary for t in tokens)))
    return np.asarray(result, dtype=bool)


def _message_topics(
    topic_model: Any, messages: list[dict[str, Any]], windows: list[dict[str, Any]],
    window_topics: np.ndarray,
) -> np.ndarray:
    result = normalise_rows(_window_average(messages, windows, window_topics))
    result[~_known_message_mask(topic_model, messages)] = 0.0
    return result


def _bag_features(
    messages: list[dict[str, Any]], topics: np.ndarray, risks: np.ndarray, *, structure: bool,
) -> np.ndarray:
    count = len(risks)
    upper = max(1, math.ceil(count * 0.2))
    stats = [
        float(risks.mean()), float(risks.max()), float(risks.std()),
        *np.quantile(risks, [0.25, 0.5, 0.75, 0.9]).tolist(),
        float(np.sort(risks)[-upper:].mean()), float((risks >= 0.5).mean()),
        float(np.log1p(count)),
    ]
    parts = [np.asarray(stats), topics.mean(axis=0), topics.max(axis=0)]
    if structure:
        distances = np.abs(np.diff(topics, axis=0)).sum(axis=1)
        indices = {str(message["comment_id"]): index for index, message in enumerate(messages)}
        pairs = [(index, indices[str(message["parent_id"])])
                 for index, message in enumerate(messages)
                 if str(message.get("parent_id")) in indices]
        reply_distance = [float(np.abs(topics[a] - topics[b]).sum()) for a, b in pairs]
        reply_risk = [abs(float(risks[a] - risks[b])) for a, b in pairs]
        parts.append(np.asarray([
            float(distances.mean()) if len(distances) else 0.0,
            float(distances.max()) if len(distances) else 0.0,
            float(np.mean(reply_distance)) if pairs else 0.0,
            float(np.max(reply_distance)) if pairs else 0.0,
            float(np.mean(reply_risk)) if pairs else 0.0,
            float(len(pairs) / count),
        ]))
    return np.concatenate(parts)


@dataclass
class BenchmarkModel:
    method: str
    config: dict[str, Any]
    topic_model: Any
    risk_classifier: Any
    risk_features: Any
    session_classifier: Any
    threshold: float = 0.5
    training_summary: dict[str, Any] = field(default_factory=dict)
    dev_predictions: list[dict[str, Any]] = field(default_factory=list)


def _session_inputs(
    model: BenchmarkModel, session: dict[str, Any], messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    local = _messages(session, messages)
    windows = build_windows(session, local, size=int(model.config.get("window_size", 5)),
                            stride=int(model.config.get("stride", 2)))
    window_topics = model.topic_model.transform([window["text"] for window in windows])
    topics = _message_topics(model.topic_model, local, windows, window_topics)
    if model.method == "cst_mil":
        matrix = model.risk_features.transform(_risk_texts(local))
        risks = model.risk_classifier.predict_proba(matrix)[:, 1]
        if model.config.get("multiscale", False):
            large = build_windows(session, local, size=10,
                                  stride=int(model.config.get("stride", 2)))
            large_topics = model.topic_model.transform([window["text"] for window in large])
            topics = (topics + _message_topics(model.topic_model, local, large, large_topics)) / 2
    else:
        window_risks = model.risk_classifier.predict_proba(window_topics)[:, 1]
        risks = _window_average(local, windows, window_risks)[:, 0]
    return local, topics, risks


def predict_session(
    model: BenchmarkModel, session: dict[str, Any], messages: list[dict[str, Any]],
) -> dict[str, Any]:
    """Infer one complete session; output is invariant to runner batch boundaries."""
    local, topics, risks = _session_inputs(model, session, messages)
    vector = _bag_features(local, topics, risks, structure=model.method == "cst_mil")
    probability = float(model.session_classifier.predict_proba(vector[None, :])[0, 1])
    interpretation = interpret_topics(
        local, topics, risks, model.topic_model.topic_words(top_n=10),
        membership_threshold=float(model.config.get("membership_threshold", 0.25)),
        relative_threshold=float(model.config.get("relative_threshold", 0.5)),
        strong_threshold=float(model.config.get("strong_threshold", 0.6)),
        evidence_top_k=int(model.config.get("evidence_top_k", 3)),
    )
    return json_safe({
        "post_id": str(session["post_id"]), "split": session.get("split", "inference"),
        "label": int(session["label"]) if session.get("label") is not None else None,
        "prediction": int(probability >= model.threshold),
        "prediction_name": "CB" if probability >= model.threshold else "Non-CB",
        "risk_probability": probability, "threshold": float(model.threshold),
        **interpretation,
    })


def train_model(
    sessions: list[dict[str, Any]], messages: list[dict[str, Any]], method: str,
    config: dict[str, Any], work_dir: Path,
) -> BenchmarkModel:
    """Fit on training bags and select threshold using development bag labels."""
    started = time.perf_counter()
    config = copy.deepcopy(config)
    work_dir = Path(work_dir)
    train = [session for session in sessions if session["split"] == "train"]
    dev = [session for session in sessions if session["split"] in {"dev", "validation"}]
    for split_name, partition in (("train", train), ("dev", dev)):
        if {int(session["label"]) for session in partition} != {0, 1}:
            raise ValueError(f"{split_name} must contain both binary classes")
    ids = [str(session["post_id"]) for session in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("Session IDs must be unique across all splits")
    unknown = {str(message["post_id"]) for message in messages}.difference(ids)
    if unknown:
        raise ValueError(f"Messages reference unknown sessions: {sorted(unknown)[:3]}")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for message in messages:
        grouped[str(message["post_id"])].append(message)
    local_messages = [_messages(session, grouped[str(session["post_id"])]) for session in train]
    windows_by_bag = [build_windows(
        session, local, size=int(config.get("window_size", 5)), stride=int(config.get("stride", 2)),
    ) for session, local in zip(train, local_messages, strict=True)]
    windows = [window for bag_windows in windows_by_bag for window in bag_windows]
    labels = np.asarray([int(session["label"]) for session in train], dtype=int)
    window_bags = np.concatenate([np.full(len(items), index, dtype=int)
                                  for index, items in enumerate(windows_by_bag)])
    message_bags = np.concatenate([np.full(len(items), index, dtype=int)
                                   for index, items in enumerate(local_messages)])
    preparation_seconds = time.perf_counter() - started
    risk_started = time.perf_counter()
    if method == "cst_mil":
        texts = [text for local in local_messages for text in _risk_texts(local)]
        oof, risk_features, risk_classifier, folds, fits = _crossfit_risk(
            texts, None, message_bags, labels, config,
        )
    else:
        oof = risk_features = risk_classifier = folds = fits = None
    risk_seconds = time.perf_counter() - risk_started
    sample_weight = None
    if method == "cst_mil" and config.get("risk_weighted_topics", False):
        sample_weight_values = []
        for bag_index, (local, bag_windows) in enumerate(zip(local_messages, windows_by_bag,
                                                            strict=True)):
            local_risks = oof[message_bags == bag_index]
            lookup = {str(message["comment_id"]): index for index, message in enumerate(local)}
            sample_weight_values.extend(1 + 2 * np.mean([
                local_risks[lookup[str(message_id)]] for message_id in window["message_ids"]
            ]) for window in bag_windows)
        sample_weight = np.asarray(sample_weight_values)
    topic_model = _new_topics(method, config, work_dir / "topics")
    topic_started = time.perf_counter()
    topic_model.fit([window["text"] for window in windows], sample_weight=sample_weight)
    topic_fit_seconds = time.perf_counter() - topic_started
    transform_started = time.perf_counter()
    # Per-session transform matches eventual inference exactly, including UMAP.
    topic_parts = [topic_model.transform([window["text"] for window in bag_windows])
                   for bag_windows in windows_by_bag]
    window_topics = np.concatenate(topic_parts, axis=0)
    topic_transform_seconds = time.perf_counter() - transform_started
    if method != "cst_mil":
        risk_started = time.perf_counter()
        oof, risk_features, risk_classifier, folds, fits = _crossfit_risk(
            None, window_topics, window_bags, labels, config,
        )
        risk_seconds += time.perf_counter() - risk_started
    vectors = []
    for bag_index, (session, local, bag_windows, bag_topics) in enumerate(zip(
        train, local_messages, windows_by_bag, topic_parts, strict=True,
    )):
        topics = _message_topics(topic_model, local, bag_windows, bag_topics)
        if method == "cst_mil":
            risks = oof[message_bags == bag_index]
            if config.get("multiscale", False):
                large = build_windows(session, local, size=10, stride=int(config.get("stride", 2)))
                large_topics = topic_model.transform([window["text"] for window in large])
                topics = (topics + _message_topics(topic_model, local, large, large_topics)) / 2
        else:
            risks = _window_average(local, bag_windows, oof[window_bags == bag_index])[:, 0]
        vectors.append(_bag_features(local, topics, risks, structure=method == "cst_mil"))
    session_started = time.perf_counter()
    classifier = make_pipeline(StandardScaler(), LogisticRegression(
        C=float(config.get("session_C", config.get("session_c", 1.0))),
        solver="liblinear", max_iter=1000,
        class_weight="balanced", random_state=int(config.get("seed", 42)),
    ))
    classifier.fit(np.asarray(vectors), labels)
    session_fit_seconds = time.perf_counter() - session_started
    model = BenchmarkModel(method, config, topic_model, risk_classifier, risk_features, classifier)
    dev_started = time.perf_counter()
    dev_predictions = [predict_session(model, session, grouped[str(session["post_id"])])
                       for session in dev]
    dev_seconds = time.perf_counter() - dev_started
    probabilities = np.asarray([prediction["risk_probability"] for prediction in dev_predictions])
    dev_labels = np.asarray([int(session["label"]) for session in dev])
    model.threshold, dev_macro_f1 = select_threshold(dev_labels, probabilities)
    for prediction in dev_predictions:
        prediction["threshold"] = model.threshold
        prediction["prediction"] = int(prediction["risk_probability"] >= model.threshold)
        prediction["prediction_name"] = "CB" if prediction["prediction"] else "Non-CB"
    model.dev_predictions = dev_predictions
    model.training_summary = {
        "training_sessions": len(train), "dev_sessions": len(dev),
        "training_messages": int(len(message_bags)), "training_windows": len(windows),
        "training_session_ids": [str(session["post_id"]) for session in train],
        "message_labels_used_for_training": False, "speaker_ids_used_as_features": False,
        "session_training_risks": "out_of_fold_by_complete_session",
        "topic_space_crossfit_policy": "fixed_unsupervised_training_only_space",
        "risk_head_zero_topic_prior": window_topics.shape[1] == 0 and method != "cst_mil",
        "crossfit_folds": folds, "risk_classifier_fit_count": fits,
        "single_class_smoke_fallback": any(
            fold["single_class_constant_fallback"] for fold in folds
        ),
        "selected_threshold": float(model.threshold), "selected_dev_macro_f1": float(dev_macro_f1),
        "dev_metrics": binary_classification_metrics(dev_labels, probabilities,
                                                      threshold=model.threshold),
        "topic_backend": copy.deepcopy(getattr(topic_model, "info", {})),
        "risk_weighted_topics": sample_weight is not None,
        "topic_sample_weight_min": (
            float(sample_weight.min()) if sample_weight is not None else None
        ),
        "topic_sample_weight_max": (
            float(sample_weight.max()) if sample_weight is not None else None
        ),
        "timings": {
            "preparation_seconds": preparation_seconds, "topic_fit_seconds": topic_fit_seconds,
            "topic_transform_seconds": topic_transform_seconds, "risk_fit_seconds": risk_seconds,
            "session_fit_seconds": session_fit_seconds, "dev_inference_seconds": dev_seconds,
            "total_training_seconds": time.perf_counter() - started,
        },
    }
    return model
