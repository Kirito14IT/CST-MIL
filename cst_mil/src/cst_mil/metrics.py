"""Evaluation metrics for the fixed CST-MIL pilot run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _binary_array(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    result: list[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)):
            result.append(int(value))
            continue
        if isinstance(value, (int, np.integer, float, np.floating)) and float(value) in {
            0.0,
            1.0,
        }:
            result.append(int(value))
            continue
        text = str(value).strip().lower().replace("_", "-")
        if text in {"1", "cb", "cyberbullying", "cyber-bullying", "positive", "risk"}:
            result.append(1)
        elif text in {"0", "non-cb", "noncb", "negative", "normal", "safe"}:
            result.append(0)
        else:
            raise ValueError(f"Unsupported binary label: {value!r}")
    return np.asarray(result, dtype=int)


def _probability_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 1 or not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be a finite one-dimensional array")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return probabilities


def select_threshold(
    y_true: Sequence[Any] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Select the macro-F1 threshold, breaking ties toward the lower value."""

    labels = _binary_array(y_true)
    scores = _probability_array(probabilities)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("y_true and probabilities must have equal nonzero length")

    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    best_threshold = float(candidates[0])
    best_score = -1.0
    for threshold in np.sort(candidates):
        predictions = (scores >= threshold).astype(int)
        score = float(f1_score(labels, predictions, average="macro", zero_division=0))
        if score > best_score + 1e-12:
            best_score = score
            best_threshold = float(threshold)
        elif abs(score - best_score) <= 1e-12 and threshold < best_threshold:
            best_threshold = float(threshold)
    return best_threshold, best_score


def binary_classification_metrics(
    y_true: Sequence[Any] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> dict[str, Any]:
    labels = _binary_array(y_true)
    scores = _probability_array(probabilities)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("y_true and probabilities must have equal nonzero length")
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision_cb": float(precision_score(labels, predictions, zero_division=0)),
        "recall_cb": float(recall_score(labels, predictions, zero_division=0)),
        "f1_cb": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(labels, scores)),
        "confusion_matrix": matrix.astype(int).tolist(),
        "tn": int(matrix[0, 0]),
        "fp": int(matrix[0, 1]),
        "fn": int(matrix[1, 0]),
        "tp": int(matrix[1, 1]),
    }
    result["roc_auc"] = (
        float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None
    )
    return result


def stratified_bootstrap_intervals(
    y_true: Sequence[Any] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
    confidence: float = 0.95,
) -> dict[str, dict[str, float | int]]:
    """Conditional CIs for one fixed test set via class-stratified bootstrap."""

    labels = _binary_array(y_true)
    scores = _probability_array(probabilities)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("y_true and probabilities must have equal nonzero length")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    class_indices = [np.flatnonzero(labels == class_id) for class_id in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("stratified bootstrap requires both classes")

    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {"macro_f1": [], "recall_cb": [], "pr_auc": []}
    for _ in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        sampled_labels = labels[sampled]
        sampled_scores = scores[sampled]
        sampled_predictions = (sampled_scores >= threshold).astype(int)
        samples["macro_f1"].append(
            float(f1_score(sampled_labels, sampled_predictions, average="macro", zero_division=0))
        )
        samples["recall_cb"].append(
            float(recall_score(sampled_labels, sampled_predictions, zero_division=0))
        )
        samples["pr_auc"].append(float(average_precision_score(sampled_labels, sampled_scores)))

    alpha = (1.0 - confidence) / 2.0
    point = binary_classification_metrics(labels, scores, threshold=threshold)
    result: dict[str, dict[str, float | int]] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=float)
        result[name] = {
            "estimate": float(point[name]),
            "lower": float(np.quantile(array, alpha)),
            "upper": float(np.quantile(array, 1.0 - alpha)),
            "confidence": float(confidence),
            "n_bootstrap": int(n_bootstrap),
        }
    return result


def evidence_localisation_metrics(
    message_predictions: pd.DataFrame,
    message_labels: pd.DataFrame | Sequence[Mapping[str, Any]],
    *,
    top_k: int = 3,
) -> dict[str, float | int]:
    """Evaluate evidence ranking without exposing labels to model fitting.

    ``message_predictions`` must contain ``post_id``, ``comment_id``, and
    ``risk_probability``. The separate label table is merged only inside this
    evaluation function.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    required = {"post_id", "comment_id", "risk_probability"}
    if missing := required.difference(message_predictions.columns):
        raise ValueError(f"message_predictions missing columns: {sorted(missing)}")
    labels = (
        message_labels.copy()
        if isinstance(message_labels, pd.DataFrame)
        else pd.DataFrame.from_records(message_labels)
    )
    label_col = next(
        (name for name in ("comment_label", "label", "target") if name in labels.columns),
        None,
    )
    if label_col is None or not {"post_id", "comment_id"}.issubset(labels.columns):
        raise ValueError("message label table must contain post_id, comment_id, and a label column")

    truth = labels[["post_id", "comment_id", label_col]].copy()
    truth["post_id"] = truth["post_id"].astype(str)
    truth["comment_id"] = truth["comment_id"].astype(str)
    truth["comment_label"] = truth[label_col].map(lambda value: int(_binary_array([value])[0]))
    predictions = message_predictions.copy()
    predictions["post_id"] = predictions["post_id"].astype(str)
    predictions["comment_id"] = predictions["comment_id"].astype(str)
    merged = predictions.merge(
        truth[["post_id", "comment_id", "comment_label"]],
        on=["post_id", "comment_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No prediction rows matched the message label table")

    labels_array = merged["comment_label"].to_numpy(dtype=int)
    score_array = _probability_array(merged["risk_probability"].to_numpy())
    threshold_metrics = binary_classification_metrics(labels_array, score_array, threshold=0.5)

    recalls: list[float] = []
    hits: list[float] = []
    positive_sessions = 0
    for _, group in merged.groupby("post_id", sort=False):
        positives = int(group["comment_label"].sum())
        if positives == 0:
            continue
        positive_sessions += 1
        selected = group.sort_values(
            ["risk_probability", "comment_id"], ascending=[False, True], kind="mergesort"
        ).head(top_k)
        found = int(selected["comment_label"].sum())
        recalls.append(found / positives)
        hits.append(float(found > 0))

    return {
        "n_labeled_messages": int(len(merged)),
        "n_positive_sessions": int(positive_sessions),
        "comment_pr_auc": float(average_precision_score(labels_array, score_array)),
        "comment_f1_at_0_5": float(threshold_metrics["f1_cb"]),
        f"mean_recall_at_{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        f"hit_rate_at_{top_k}": float(np.mean(hits)) if hits else 0.0,
    }

