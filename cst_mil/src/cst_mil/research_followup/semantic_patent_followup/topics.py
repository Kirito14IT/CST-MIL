"""Extractive discussion groups and traceable, non-cascading reply links.

This interpretation layer never changes category scores.  Risk categories may
overlap, whereas discussion membership is deliberately hard and disjoint.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from cst_mil.bench.interpretation import message_location
from cst_mil.unified_semantic_v1_1 import topics as legacy

VARIANTS = (
    "full",
    "no_reply_link",
    "no_polarity_protection",
    "frequency_naming",
    "no_counter_distance",
)
_BREAK = re.compile(r"[，,。.!！？?；;:\n\r]")
_HAN_NEGATIVE = re.compile(r"不|没|未|别|禁止|反对|拒绝|停止")
_WORD_NEGATIVE = re.compile(
    r"\b(?:no|not|never|without|não|na\u0303o|sem|nunca|cannot|can't|don['’]t|"
    r"doesn['’]t|isn['’]t|aren['’]t|won['’]t|shouldn['’]t)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"[?？]|会不会|能不能|是不是|有没有|要不要|敢不敢|该不该|可不可以")
PURE_RESPONSES = frozenset(
    "没错 对的 是的 确实 同意 赞同 支持 ok okay yes yeah agree agreed exactly sim concordo".split()
)


def _pure_response(raw: str) -> bool:
    text = raw.strip()
    while text and unicodedata.category(text[0]).startswith("P"):
        text = text[1:].lstrip()
    while text and unicodedata.category(text[-1]).startswith("P"):
        text = text[:-1].rstrip()
    return unicodedata.normalize("NFKC", text).casefold() in PURE_RESPONSES


def _key(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def _clauses(raw: str) -> list[tuple[int, int, bool]]:
    """Return exact offsets, retaining terminal punctuation in protected clauses."""
    out = []
    start = 0
    for match in list(_BREAK.finditer(raw)) + [None]:
        end = match.end() if match else len(raw)
        left, right = start, end
        while left < right and raw[left].isspace():
            left += 1
        while right > left and raw[right - 1].isspace():
            right -= 1
        text = raw[left:right]
        protected = bool(
            _HAN_NEGATIVE.search(text) or _WORD_NEGATIVE.search(text) or _QUESTION.search(text)
        )
        if text:
            out.append((left, right, protected))
        start = end
    return out


def _unprotected_candidates(raw: str) -> dict[str, tuple[int, int]]:
    """Lexical ablation: contiguous words, without negation/question scope guards."""
    tokens = legacy._tokens(raw)
    result = {}
    for i, (word, start, _) in enumerate(tokens):
        normalized = unicodedata.normalize("NFKC", word).casefold()
        if normalized in legacy.GENERIC and normalized not in (
            legacy.NEGATORS | legacy._MODAL_PREFIXES
        ):
            continue
        for size in (1, 2, 3, 4):
            group = tokens[i : i + size]
            if len(group) != size or legacy.generic_word(group[-1][0]):
                continue
            end = group[-1][2]
            text = raw[start:end]
            if _BREAK.search(text):
                continue
            if any(
                not (
                    c.isalpha()
                    or unicodedata.category(c).startswith("M")
                    or c.isspace()
                    or c in "'’"
                )
                for c in text
            ):
                continue
            key = _key(text)
            if legacy.generic_word(key) or len("".join(key.split())) < 2 or len(key) > 36:
                continue
            if legacy._HAN.fullmatch(key) and len(key) > 12:
                continue
            result.setdefault(key, (start, end))
    return result


def phrase_candidates(raw: str, *, protect_polarity: bool = True) -> dict[str, tuple[int, int]]:
    """Original spans only; a protected clause cannot become a positive fragment.

    Complete protected clauses of at most 96 characters are retained if they
    contain a content token.  Longer protected clauses abstain rather than
    truncate the scope; there is no generated paraphrase or punctuation repair.
    """
    if _pure_response(raw):
        return {}
    if not protect_polarity:
        return _unprotected_candidates(raw)
    result: dict[str, tuple[int, int]] = {}
    clauses = _clauses(raw)
    for word, (start, end) in legacy.phrase_candidates(raw).items():
        if not any(
            protected and start < right and end > left for left, right, protected in clauses
        ):
            result[word] = (start, end)
    for start, end, protected in clauses:
        if not protected:
            continue
        text = raw[start:end]
        content = any(
            not legacy.generic_word(word) and _key(word) not in legacy.NEGATORS
            for word, _, _ in legacy._tokens(text)
        )
        if content and 2 <= len("".join(text.split())) <= 96:
            result.setdefault(_key(text), (start, end))
    return result


def _keywords(
    groups: list[list[int]], candidates: list[dict], *, frequency: bool
) -> list[list[str]]:
    if not frequency:
        return legacy._keywords(groups, candidates)
    output = []
    for group in groups:
        counts = Counter(word for i in group for word in candidates[i])
        ordered = sorted(counts, key=lambda word: (-counts[word], len(word), word))
        selected: list[str] = []
        for word in ordered:
            if any(word in old or old in word for old in selected):
                continue
            selected.append(word)
            if len(selected) == 10:
                break
        output.append(selected)
    return output


def _valid_parent_indices(locations: list[dict]) -> tuple[dict[int, int], dict[int, str]]:
    lookup = {item["comment_id"]: i for i, item in enumerate(locations)}
    if len(lookup) != len(locations):
        raise ValueError("Duplicate message IDs make reply references ambiguous")
    edges, rejected = {}, {}
    for i, item in enumerate(locations):
        value = item.get("parent_id")
        if value is None or value == "":
            continue
        parent = lookup.get(str(value))
        if parent is None:
            rejected[i] = "missing_parent"
        elif parent == i:
            rejected[i] = "self_reply"
        elif locations[parent]["post_id"] != item["post_id"]:
            rejected[i] = "different_conversation"
        else:
            edges[i] = parent
    # Reject every edge whose path contains a cycle, including descendants of it.
    for i in tuple(edges):
        seen = set()
        current = i
        while current in edges:
            if current in seen:
                rejected[i] = "cyclic_reply_path"
                break
            seen.add(current)
            current = edges[current]
    for i in rejected:
        edges.pop(i, None)
    return edges, rejected


def interpret_topics(
    messages: list[dict[str, Any]],
    embeddings: np.ndarray,
    category_scores: np.ndarray,
    *,
    category_thresholds: dict[str, float] | list[float],
    variant: str = "full",
    distance_threshold: float = 0.35,
    evidence_top_k: int = 3,
) -> dict[str, Any]:
    """Preserve the legacy output contract while recording member provenance."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown interpretation variant: {variant}")
    if distance_threshold != 0.35:
        raise ValueError("This frozen follow-up uses distance_threshold=0.35 only")
    n = len(messages)
    vectors, scores = np.asarray(embeddings, dtype=float), np.asarray(category_scores, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != n or scores.shape != (n, 3):
        raise ValueError("Message, semantic and category dimensions differ")
    if not np.isfinite(vectors).all() or not np.isfinite(scores).all():
        raise ValueError("Non-finite embedding or category score")
    if np.any((scores < 0) | (scores > 1)) or evidence_top_k < 1:
        raise ValueError("Invalid category scores or evidence_top_k")
    thresholds = (
        list(category_thresholds.values())
        if isinstance(category_thresholds, dict)
        else list(category_thresholds)
    )
    if not thresholds or not all(np.isfinite(t) and 0 <= t <= 1 for t in thresholds):
        raise ValueError("Invalid category thresholds")
    attack_threshold = (
        float(category_thresholds["attack"])
        if isinstance(category_thresholds, dict)
        else float(thresholds[0])
    )
    locations = [message_location(message, i) for i, message in enumerate(messages)]
    parent_indices, rejected = _valid_parent_indices(locations)
    candidates = [
        phrase_candidates(item["raw_text"], protect_polarity=variant != "no_polarity_protection")
        for item in locations
    ]
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.divide(
        vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 1e-12
    )
    eligible = [i for i in range(n) if norms[i] > 1e-12 and candidates[i]]
    counter = [bool(legacy._COUNTER.search(item["raw_text"])) for item in locations]
    groups = []
    counter_pairs_constrained = 0
    if len(eligible) >= 2:
        distances = np.clip(1 - normalized[eligible] @ normalized[eligible].T, 0, 2)
        if variant != "no_counter_distance":
            for a, i in enumerate(eligible):
                for b in range(a):
                    j = eligible[b]
                    if (counter[i] and not counter[j] and scores[j, 0] >= attack_threshold) or (
                        counter[j] and not counter[i] and scores[i, 0] >= attack_threshold
                    ):
                        distances[a, b] = distances[b, a] = 1.0
                        counter_pairs_constrained += 1
        np.fill_diagonal(distances, 0)
        labels = AgglomerativeClustering(
            n_clusters=None,
            metric="precomputed",
            linkage="complete",
            distance_threshold=distance_threshold,
            compute_full_tree=True,
        ).fit_predict(distances)
        for label in sorted(
            set(labels), key=lambda x: min(eligible[i] for i in np.flatnonzero(labels == x))
        ):
            group = [eligible[i] for i in np.flatnonzero(labels == label)]
            if len(group) >= 2:
                groups.append(group)
    words = _keywords(groups, candidates, frequency=variant == "frequency_naming")
    groups_words = [(group, word) for group, word in zip(groups, words, strict=True) if word]
    # Freeze membership before adding contextual links: no cascading is possible.
    base_assignment = {
        i: topic_id for topic_id, (group, _) in enumerate(groups_words) for i in group
    }
    inherited: dict[int, tuple[int, int]] = {}
    if variant != "no_reply_link":
        for i, parent in parent_indices.items():
            if (
                i not in base_assignment
                and not candidates[i]
                and len("".join(locations[i]["raw_text"].split())) <= 12
                and locations[i]["raw_text"].strip()
                and parent in base_assignment
            ):
                inherited[i] = (base_assignment[parent], parent)
    strengths = np.zeros((n, len(groups_words)), dtype=float)
    risks = scores.max(axis=1) if n else np.empty(0)
    topics = []
    for topic_id, (base_indices, keywords) in enumerate(groups_words):
        centroid = normalized[base_indices].mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        similarities = np.clip(normalized[base_indices] @ centroid, 0, 1)
        strengths[base_indices, topic_id] = similarities
        linked_indices = [i for i, (assigned, _) in inherited.items() if assigned == topic_id]
        # A parent link is membership, not an invented semantic similarity score.
        indices = sorted(base_indices + linked_indices)
        supports = {
            word: [
                {
                    "message_id": locations[i]["comment_id"],
                    "start": candidates[i][word][0],
                    "end": candidates[i][word][1],
                    "excerpt": locations[i]["raw_text"][slice(*candidates[i][word])],
                }
                for i in base_indices
                if word in candidates[i]
            ]
            for word in keywords
        }
        ranked = sorted(indices, key=lambda i: (-risks[i], i))[:evidence_top_k]
        topics.append(
            {
                "topic_id": topic_id,
                "title": " / ".join(keywords[:2]),
                "keywords": keywords,
                "topic_strength": float(similarities.mean()),
                "risk_score": float(risks[indices].mean()),
                "message_ids": [locations[i]["comment_id"] for i in indices],
                "support_messages": [locations[i] for i in indices],
                "segments": legacy._segments(indices, locations),
                "evidence": [
                    {
                        **locations[i],
                        "risk_score": float(risks[i]),
                        "topic_strength": float(strengths[i, topic_id]),
                        "assignment_provenance": "direct_parent_context"
                        if i in inherited
                        else "semantic_cluster",
                    }
                    for i in ranked
                ],
                "keyword_supports": supports,
                "qualityStatus": "extractive_supported",
                "contextHint": "possible_counter_speech"
                if all(counter[i] for i in base_indices)
                else None,
                "semantic_message_ids": [locations[i]["comment_id"] for i in base_indices],
                "context_linked_message_ids": [locations[i]["comment_id"] for i in linked_indices],
                "name_statistics_message_ids": [locations[i]["comment_id"] for i in base_indices],
                "membership_mode": "hard_disjoint_with_traceable_parent_context",
            }
        )
    ordering = sorted(range(n), key=lambda i: (-risks[i], i))
    ranks = {i: rank + 1 for rank, i in enumerate(ordering)}
    assigned = np.array([i in base_assignment or i in inherited for i in range(n)], dtype=bool)
    message_scores = []
    for i, location in enumerate(locations):
        topic_ids = (
            [inherited[i][0]]
            if i in inherited
            else [base_assignment[i]]
            if i in base_assignment
            else []
        )
        message_scores.append(
            {
                **location,
                "risk_score": float(risks[i]),
                "category_scores": scores[i].tolist(),
                "topic_strengths": strengths[i].tolist(),
                "topic_ids": topic_ids,
                "assigned": bool(assigned[i]),
                "rank": ranks[i],
                "contextHint": "possible_counter_speech" if counter[i] else None,
                "assignment_provenance": (
                    "direct_parent_context"
                    if i in inherited
                    else "semantic_cluster"
                    if i in base_assignment
                    else "unassigned"
                ),
                "context_parent_message_id": (
                    locations[inherited[i][1]]["comment_id"] if i in inherited else None
                ),
                "context_link_strength": None,
                "rejected_reply_reason": rejected.get(i),
                "qualityStatus": (
                    "context_linked_original_reply"
                    if i in inherited
                    else "semantic_cluster_supported"
                    if i in base_assignment
                    else "unassigned_no_reliable_phrase"
                    if not candidates[i]
                    else "unassigned_zero_embedding"
                    if norms[i] <= 1e-12
                    else "unassigned_semantic_singleton"
                ),
                "representative_original_sentence": location["raw_text"],
            }
        )
    fraction = float(assigned.mean()) if n else 0.0
    return {
        "topics": topics,
        "message_scores": message_scores,
        "evidence": [
            {**locations[i], "risk_score": float(risks[i]), "rank": rank + 1}
            for rank, i in enumerate(ordering[:evidence_top_k])
        ],
        "coverage": fraction,
        "coverage_details": {
            "assigned_messages": int(assigned.sum()),
            "total_messages": n,
            "fraction": fraction,
            "unassigned_message_ids": [
                locations[i]["comment_id"] for i in range(n) if not assigned[i]
            ],
            "zero_topic_messages": int((~assigned).sum()),
            "semantic_assigned_messages": len(base_assignment),
            "context_linked_messages": len(inherited),
        },
        "interpretation_provenance": {
            "variant": variant,
            "distance_threshold": distance_threshold,
            "category_score_policy": "unchanged_input_scores",
            "topic_membership": "hard_disjoint",
            "risk_categories": "may_overlap",
            "name_policy": "original_contiguous_spans",
            "protected_clause_max_nonspace_characters": 96,
            "parent_link_max_nonspace_characters": 12,
            "pure_response_stoplist": sorted(PURE_RESPONSES),
            "pure_response_match_policy": "whole_text_after_outer_whitespace_and_punctuation",
            "pure_response_rationale": "standalone_acknowledgements_do_not_name_discussion_content",
            "parent_link_policy": "direct_to_frozen_semantic_group_no_cascade",
            "parent_context_semantic_strength": "unmeasured_zero_in_legacy_strength_array",
            "counter_pairs_constrained": counter_pairs_constrained,
            "rejected_reply_edges": len(rejected),
        },
    }
