"""Comparable topic proxies, held-out risk scores and predeclared dev gates."""

from __future__ import annotations

import itertools
import math
from collections import Counter

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .common import topic_tokens


def binary_metrics(records: list[dict]) -> dict:
    if not records:
        return {"n_sessions": 0}
    y = np.asarray([r["label"] for r in records], dtype=int)
    p = np.asarray([r["risk_probability"] for r in records], dtype=float)
    pred = np.asarray([r["prediction"] for r in records], dtype=int)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid risk probabilities")
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n_sessions": len(y),
        "n_cb": int(y.sum()),
        "n_non_cb": int((y == 0).sum()),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0)),
        "recall_cb": float(recall_score(y, pred, zero_division=0)),
        "precision_cb": float(precision_score(y, pred, zero_division=0)),
        "f1_cb": float(f1_score(y, pred, zero_division=0)),
        "ap": float(average_precision_score(y, p)) if y.any() else None,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def topic_quality(words: list[list[str]], documents: list[str], predictions: list[dict]) -> dict:
    topics = [list(dict.fromkeys(w))[:10] for w in words]
    wanted = set(itertools.chain.from_iterable(topics))
    occurrences: Counter = Counter()
    pairs: Counter = Counter()
    for text in documents:
        tokens = topic_tokens(text)
        terms = set(tokens) | {a + " " + b for a, b in zip(tokens, tokens[1:], strict=False)}
        present = terms & wanted
        occurrences.update(present)
        pairs.update(itertools.combinations(sorted(present), 2))
    n = max(1, len(documents))
    scores = []
    degenerate = 0
    active = []
    for topic in topics:
        if len(topic) < 2 or sum(occurrences[w] > 0 for w in topic) < 2:
            degenerate += 1
            scores.append(-1.0)
            continue
        active.append(topic)
        local = []
        for left, right in itertools.combinations(topic, 2):
            joint = pairs[tuple(sorted((left, right)))] / n
            if joint <= 0:
                local.append(-1.0)
            elif joint >= 1:
                local.append(0.0)
            else:
                marginal = occurrences[left] * occurrences[right] / (n * n)
                local.append(math.log(joint / marginal) / -math.log(joint))
        scores.append(float(np.mean(local)))
    total_words = sum(len(t) for t in topics)
    diversity = (
        len(set(itertools.chain.from_iterable(topics))) / total_words
        if total_words and active
        else 0.0
    )
    macro_coverage = (
        float(np.mean([float(r.get("coverage", 0.0)) for r in predictions])) if predictions else 0.0
    )
    total_messages = sum(
        r.get("coverage_details", {}).get("total_messages", 0) for r in predictions
    )
    assigned_messages = sum(
        r.get("coverage_details", {}).get("assigned_messages", 0) for r in predictions
    )
    coverage = assigned_messages / total_messages if total_messages else 0.0
    return {
        "npmi": float(np.mean(scores)) if scores else -1.0,
        "topic_diversity": float(diversity),
        "coverage": coverage,
        "coverage_macro_session": macro_coverage,
        "assigned_messages": assigned_messages,
        "total_messages": total_messages,
        "n_topics": len(topics),
        "active_topics": len(active),
        "degenerate_topic_fraction": degenerate / len(topics) if topics else 1.0,
        "reference_windows": len(documents),
        "top_n": 10,
        "per_topic_npmi": scores,
        "interpretation": "unsupervised proxy; not topic-count or boundary accuracy",
    }


def evidence_metrics(predictions: list[dict], messages: list[dict], top_k: int = 3) -> dict:
    truth = {
        (m["post_id"], m["comment_id"]): int(m["comment_label"])
        for m in messages
        if m.get("comment_label") is not None
    }
    y, scores, recalls, hits = [], [], [], []
    for session in predictions:
        rows = [
            r
            for r in session.get("message_scores", [])
            if (session["post_id"], r["comment_id"]) in truth
        ]
        rows.sort(key=lambda r: int(r.get("rank", 10**9)))
        local = [truth[(session["post_id"], r["comment_id"])] for r in rows]
        y.extend(local)
        scores.extend(r["risk_probability"] for r in rows)
        if sum(local):
            found = sum(local[:top_k])
            recalls.append(found / sum(local))
            hits.append(float(found > 0))
    return {
        "n_labeled_messages": len(y),
        "n_sessions_with_positive_comments": len(hits),
        "comment_ap": float(average_precision_score(y, scores)) if y and sum(y) else None,
        "hit_rate_at_3": float(np.mean(hits)) if hits else None,
        "mean_recall_at_3": float(np.mean(recalls)) if recalls else None,
    }


def bootstrap_metrics(records: list[dict], count: int = 1000, seed: int = 42) -> dict:
    groups = [[i for i, r in enumerate(records) if r["label"] == c] for c in (0, 1)]
    if any(not group for group in groups):
        return {"status": "insufficient classes"}
    rng = np.random.default_rng(seed)
    values = {key: [] for key in ("macro_f1", "recall_cb", "ap")}
    for _ in range(count):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        metrics = binary_metrics([records[i] for i in indices])
        for key in values:
            values[key].append(metrics[key])
    point = binary_metrics(records)
    return {
        key: {
            "point": point[key],
            "low": float(np.quantile(vals, 0.025)),
            "high": float(np.quantile(vals, 0.975)),
            "bootstrap_samples": count,
            "unit": "whole session",
            "conditional_on_fixed_fit": True,
        }
        for key, vals in values.items()
    }


def paired_differences(left: list[dict], right: list[dict], count: int = 1000) -> dict:
    lookup = {r["post_id"]: r for r in right}
    if set(lookup) != {r["post_id"] for r in left}:
        raise ValueError("Paired comparison requires identical session IDs")
    right = [lookup[r["post_id"]] for r in left]
    if any(a["label"] != b["label"] for a, b in zip(left, right, strict=True)):
        raise ValueError("Label mismatch in paired comparison")
    groups = [[i for i, r in enumerate(left) if r["label"] == c] for c in (0, 1)]
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(count):
        indices = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups if g])
        a = binary_metrics([left[i] for i in indices])["macro_f1"]
        b = binary_metrics([right[i] for i in indices])["macro_f1"]
        samples.append(a - b)
    return {
        "delta_macro_f1": binary_metrics(left)["macro_f1"] - binary_metrics(right)["macro_f1"],
        "ci95": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
        "n_sessions": len(left),
        "bootstrap_samples": count,
        "unit": "paired whole session",
        "not_repeated_training": True,
    }


def stopping_gate(candidate: dict, baseline: dict) -> bool:
    return (
        candidate["risk"]["macro_f1"] >= baseline["risk"]["macro_f1"] - 0.01
        and candidate["topics"]["npmi"] >= baseline["topics"]["npmi"] - 0.02
        and candidate["topics"]["topic_diversity"] >= baseline["topics"]["topic_diversity"] - 0.05
        and candidate["topics"]["coverage"] >= baseline["topics"]["coverage"] - 0.05
        and candidate.get("functional_checks_passed", False)
    )


def replacement_gate(candidate: dict, current: dict) -> bool:
    if not candidate.get("functional_checks_passed", False):
        return False
    delta = candidate["risk"]["macro_f1"] - current["risk"]["macro_f1"]
    dc = candidate["topics"]["npmi"] - current["topics"]["npmi"]
    quality = (
        dc >= -0.02
        and candidate["topics"]["topic_diversity"] >= current["topics"]["topic_diversity"] - 0.05
        and candidate["topics"]["coverage"] >= current["topics"]["coverage"] - 0.05
    )
    return bool(quality and (delta >= 0.01 - 1e-12 or (abs(delta) < 0.01 and dc >= 0.02 - 1e-12)))
