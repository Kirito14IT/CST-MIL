"""Deterministic variable-topic evidence, independent of annotation columns."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .common import normalise_rows, topic_tokens


def _value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def message_location(message: dict[str, Any], index: int) -> dict[str, Any]:
    """Only explicitly allowed metadata is copied; annotation fields never escape."""
    return {
        "post_id": str(message.get("post_id", "")),
        "comment_id": str(message["comment_id"]),
        "parent_id": _value(message.get("parent_id")),
        "order": _value(message.get("order", index)),
        "position": index,
        "comment_time": _value(message.get("comment_time")),
        "speaker_id": _value(message.get("speaker_id", message.get("speaker"))),
        "raw_text": str(message.get("raw_text", message.get("model_text", ""))),
    }


def interpret_topics(
    messages: list[dict[str, Any]],
    topic_weights: np.ndarray,
    risks: np.ndarray,
    topic_words: list[list[str]],
    *,
    membership_threshold: float = 0.25,
    relative_threshold: float = 0.5,
    strong_threshold: float = 0.6,
    evidence_top_k: int = 3,
) -> dict[str, Any]:
    """Return zero or more supported topics, retaining overlapping membership.

    Topic weights are strengths, not calibrated probabilities. Two supporting
    messages are needed unless one reaches ``strong_threshold``. Segments use
    contiguous *input positions*, not potentially sparse source order numbers.
    """
    weights = normalise_rows(np.asarray(topic_weights, dtype=float))
    risks = np.asarray(risks, dtype=float)
    if weights.ndim != 2 or weights.shape[0] != len(messages) or risks.shape != (len(messages),):
        raise ValueError("Message, topic, and risk dimensions differ")
    if not np.all(np.isfinite(risks)) or np.any((risks < 0) | (risks > 1)):
        raise ValueError("Message risks must be finite scores between zero and one")
    if evidence_top_k < 1:
        raise ValueError("evidence_top_k must be positive")
    row_max = weights.max(axis=1, initial=0)
    membership = (weights >= membership_threshold) & (
        weights >= relative_threshold * row_max[:, None]
    ) & (weights > 0)
    locations = [message_location(message, index) for index, message in enumerate(messages)]
    topics: list[dict[str, Any]] = []
    retained = np.zeros_like(membership, dtype=bool)
    for topic_id in range(weights.shape[1]):
        indices = np.flatnonzero(membership[:, topic_id])
        if len(indices) < 2 and not (
            len(indices) == 1 and weights[indices[0], topic_id] >= strong_threshold
        ):
            continue
        retained[indices, topic_id] = True
        support = [locations[int(index)] for index in indices]
        raw_texts = [item["raw_text"] for item in support]
        global_words = topic_words[topic_id] if topic_id < len(topic_words) else []
        keywords = [word for word in global_words if any(word in raw for raw in raw_texts)]
        if not keywords:
            counts = Counter(token for raw in raw_texts for token in topic_tokens(raw))
            keywords = [word for word, _ in counts.most_common(10)
                        if any(word in raw for raw in raw_texts)]
        segments: list[dict[str, Any]] = []
        for index in indices.tolist():
            if not segments or index != segments[-1]["end_position"] + 1:
                segments.append({
                    "start_position": index,
                    "end_position": index,
                    "start_message_id": locations[index]["comment_id"],
                    "end_message_id": locations[index]["comment_id"],
                    "start_time": locations[index]["comment_time"],
                    "end_time": locations[index]["comment_time"],
                    "message_ids": [locations[index]["comment_id"]],
                })
            else:
                segments[-1].update({
                    "end_position": index,
                    "end_message_id": locations[index]["comment_id"],
                    "end_time": locations[index]["comment_time"],
                })
                segments[-1]["message_ids"].append(locations[index]["comment_id"])
        ranked = sorted(indices.tolist(), key=lambda index: (
            -weights[index, topic_id] * risks[index], -risks[index], index,
        ))[:evidence_top_k]
        evidence = [{
            **locations[index],
            "topic_strength": float(weights[index, topic_id]),
            "risk_score": float(risks[index]),
            "evidence_score": float(weights[index, topic_id] * risks[index]),
        } for index in ranked]
        topics.append({
            "topic_id": topic_id,
            "title": " / ".join(keywords[:3]) or "未命名主题",
            "keywords": keywords[:10],
            "topic_strength": float(weights[indices, topic_id].mean()),
            "risk_score": float(np.average(risks[indices], weights=weights[indices, topic_id])),
            "message_ids": [item["comment_id"] for item in support],
            "support_messages": support,
            "segments": segments,
            "evidence": evidence,
        })
    topics.sort(key=lambda topic: (-topic["topic_strength"], topic["topic_id"]))
    evidence_indices = sorted(range(len(messages)), key=lambda index: (-risks[index], index))
    rank_by_index = {index: rank + 1 for rank, index in enumerate(evidence_indices)}
    message_scores = [{
        **location,
        "risk_score": float(risks[index]),
        "risk_probability": float(risks[index]),
        "topic_strengths": weights[index].tolist(),
        "topic_ids": np.flatnonzero(retained[index]).tolist(),
        "assigned": bool(retained[index].any()),
        "rank": rank_by_index[index],
    } for index, location in enumerate(locations)]
    evidence = [{
        **locations[index], "risk_score": float(risks[index]),
        "risk_probability": float(risks[index]), "rank": rank + 1,
    } for rank, index in enumerate(evidence_indices[:evidence_top_k])]
    assigned = retained.any(axis=1)
    return {
        "topics": topics,
        "message_scores": message_scores,
        "evidence": evidence,
        "coverage": float(assigned.mean()) if len(messages) else 0.0,
        "coverage_details": {
            "assigned_messages": int(assigned.sum()),
            "total_messages": len(messages),
            "fraction": float(assigned.mean()) if len(messages) else 0.0,
            "unassigned_message_ids": [locations[index]["comment_id"]
                                       for index in np.flatnonzero(~assigned)],
            "zero_topic_messages": int((weights.sum(axis=1) == 0).sum()),
        },
    }
