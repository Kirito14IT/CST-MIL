"""Grounded, local semantic discussion groups, separate from risk categories.

Extractive names are descriptions of supported text, not generated assertions.
Counter-speech hints constrain grouping only; they never override risk scores.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any

import jieba
import jieba.posseg as posseg
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from cst_mil.bench.interpretation import message_location

GENERIC = frozenset(
    (
        "的 了 和 是 在 就 都 也 我 你 他 她 它 我们 你们 他们 一个 这个 那个 这 那 有 吗 "
        "啊 呢 吧 呀 哦 嗯 啦 哈 哈哈 呵呵 什么 怎么 为什么 还是 就是 这么 那么 "
        "但是 可是 因为 所以 一下 一些 已经 真的 真 很 太 更 最 会 可以 没有 "
        "觉得 看到 看看 说 去 来 好 关注 转发 微博 谢谢 "
        "下午 晚上 早上 大家 一起 永远 持续 越来越 看 还 比 得 能 谁 给 以为 叫 本来 时候 这种 "
        "a an the and or to of in on at is are was be been it this that you your i we they he she "
        "for with from as but if then so just very really have has do does did can could would "
        "o a os as um uma de da do das dos e em no na nos nas que por para com "
        "eu você ele ela isso "
        "url mention reply 个 些 次 场 最后 现在 以后 以前 其实 而且 though also isso muito "
        "extremely absolutely totally completely quite merely simply anyhow somehow indeed "
        "今天 昨天 明天"
    ).split()
)
_MODAL_PREFIXES = frozenset(
    "应该 应当 必须 需要 能够 可能 也许 或许 大概 好像 似乎 请问 请 "
    "should must may might shall ought".split()
)
GENERIC = GENERIC | _MODAL_PREFIXES | frozenset(
    "确实 实在 当然 然后 例如 等等".split()
)
NEGATORS = frozenset(
    (
        "不 没 未 不是 不会 没有 不能 不要 别 不可能 停止 反对 禁止 拒绝 "
        "no not never without não sem nunca"
    ).split()
)
_QUESTION_POLARITY = re.compile(r"会不会|能不能|是不是|有没有|要不要|敢不敢|该不该|可不可以")
_HAN = re.compile(r"^[\u3400-\u9fff]+$")
_SENTENCE = re.compile(r"[。！？!?；;\n]+")
_CLAUSE_PUNCT = re.compile(r"[，,。！？!?；;:\n]")
_COUNTER = re.compile(
    r"(?:不要|别|停止|反对|抵制|拒绝|禁止)[^。！？!?]{0,12}(?:骂|辱骂|围攻|人身攻击|欺负|歧视|起哄)"
    r"|(?:stop|do not|don.t)[^.!?]{0,24}(?:bully|insult|harass|attack|hate)"
    r"|(?:não|pare)[^.!?]{0,20}(?:ofend|ataqu|discrimin)",
    re.IGNORECASE,
)


_CLOSED_CLASS_PREFIXES = frozenset("rdpcuyeomqwf")


@lru_cache(maxsize=8192)
def generic_word(word: str) -> bool:
    """Reject known fillers and Chinese phrases consisting only of function words.

    POS tags provide a general guard for adverbs, pronouns and quantifiers such
    as 非常 and 一部, without blocking their presence inside content phrases.
    This lexical guard is not a semantic topic-correctness detector.
    """
    normalized = unicodedata.normalize("NFKC", word).casefold()
    if normalized in GENERIC:
        return True
    if _HAN.fullmatch(normalized):
        tokens = posseg.lcut(normalized)
        return bool(tokens) and all(
            token.flag[:1] in _CLOSED_CLASS_PREFIXES
            or token.flag == "ad"
            or unicodedata.normalize("NFKC", token.word).casefold() in GENERIC
            for token in tokens
        )
    words = normalized.split()
    return bool(words) and all(word in GENERIC or word in NEGATORS for word in words)


def _tokens(raw: str) -> list[tuple[str, int, int]]:
    """Offsets refer to original text; boundaries never match substrings."""
    result = []
    spans = []
    start = None
    for index, char in enumerate(raw + " "):
        keep = (
            char.isalpha()
            or unicodedata.category(char).startswith("M")
            or (char in "'’" and start is not None)
        )
        if keep and start is None:
            start = index
        elif not keep and start is not None:
            spans.append((start, index))
            start = None
    for base, end in spans:
        piece = raw[base:end]
        if _HAN.fullmatch(piece):
            result.extend(
                (word.casefold(), base + start, base + end)
                for word, start, end in jieba.tokenize(piece)
            )
        else:
            result.append((piece.casefold(), base, end))
    return result


def phrase_candidates(raw: str) -> dict[str, tuple[int, int]]:
    """Original contiguous short phrases, keeping negation and small compounds."""
    tokens = _tokens(raw)
    candidates: dict[str, tuple[int, int]] = {}
    negation_before = []
    scope_negative = False
    previous_end = 0
    for word, start, end in tokens:
        if _CLAUSE_PUNCT.search(raw[previous_end:start]):
            scope_negative = False
        negation_before.append(scope_negative)
        if unicodedata.normalize("NFKC", word).casefold() in NEGATORS:
            scope_negative = True
        previous_end = end
    for i, (word, start, _) in enumerate(tokens):
        # Keep content phrases with modifiers and explicit modality/negation.
        # Whole-phrase generic checks below reject purely functional candidates.
        normalized_word = unicodedata.normalize("NFKC", word).casefold()
        if normalized_word in GENERIC and normalized_word not in (NEGATORS | _MODAL_PREFIXES):
            continue
        # Conservatively abstain rather than reverse a distant, explicit negation.
        if negation_before[i]:
            continue
        if unicodedata.normalize("NFKC", word).casefold() in NEGATORS:
            boundary = _CLAUSE_PUNCT.search(raw, start)
            clause = raw[start : boundary.start() if boundary else len(raw)].rstrip()
            if (boundary and boundary.group() in ("?", "？")) or clause.endswith(("吗", "么")):
                continue
        for length in (1, 2, 3, 4):
            group = tokens[i : i + length]
            if len(group) != length or generic_word(group[-1][0]):
                continue
            end = group[-1][2]
            following = tokens[i + length] if i + length < len(tokens) else None
            # Jieba can split a three-character name into a familiar prefix + one Han.
            # Prefer the supported whole compound; do not relabel that prefix as a name.
            if (
                following is not None
                and following[1] == end
                and _HAN.fullmatch(group[-1][0])
                and len(group[-1][0]) >= 2
                and _HAN.fullmatch(following[0])
                and len(following[0]) == 1
                and not generic_word(following[0])
            ):
                continue
            span = raw[start:end]
            # "会不会" is a question, not the negative assertion "不会".
            if any(
                start < match.end()
                and end > match.start()
                and not (start <= match.start() and end >= match.end())
                for match in _QUESTION_POLARITY.finditer(raw)
            ):
                continue
            if _SENTENCE.search(span) or any(
                not (
                    c.isalpha()
                    or unicodedata.category(c).startswith("M")
                    or c.isspace()
                    or c in "'’"
                )
                for c in span
            ):
                continue
            key = unicodedata.normalize("NFC", span).casefold()
            if length == 1 and key in NEGATORS:
                continue
            if generic_word(key) or len(key.replace(" ", "")) < 2 or len(key) > 36:
                continue
            if _HAN.fullmatch(key) and len(key) > 12:
                continue
            candidates.setdefault(key, (start, end))
    return candidates


def _keywords(groups: list[list[int]], candidates: list[dict]) -> list[list[str]]:
    group_sets = [set().union(*(set(candidates[i]) for i in group)) for group in groups]
    df = Counter(word for words in group_sets for word in words)
    output = []
    for group in groups:
        counts = Counter(word for i in group for word in candidates[i])
        ordered = sorted(
            counts,
            key=lambda word: (
                -(counts[word] / len(group))
                * math.log(1 + (len(groups) + 1) / (df[word] + 1))
                * (1 + 0.12 * min(len(word.replace(" ", "")), 8)),
                -counts[word],
                len(word),
                word,
            ),
        )
        selected = []
        for word in ordered:
            # Longer near-identical phrases should not fill the title repeatedly.
            if any(word in old or old in word for old in selected):
                continue
            selected.append(word)
            if len(selected) == 10:
                break
        output.append(selected)
    return output


def _segments(indices: list[int], locations: list[dict]) -> list[dict]:
    segments = []
    for index in indices:
        item = locations[index]
        if not segments or index != segments[-1]["end_position"] + 1:
            segments.append(
                {
                    "start_position": index,
                    "end_position": index,
                    "start_message_id": item["comment_id"],
                    "end_message_id": item["comment_id"],
                    "start_time": item["comment_time"],
                    "end_time": item["comment_time"],
                    "message_ids": [item["comment_id"]],
                }
            )
        else:
            segments[-1].update(
                end_position=index, end_message_id=item["comment_id"], end_time=item["comment_time"]
            )
            segments[-1]["message_ids"].append(item["comment_id"])
    return segments


def interpret_topics(
    messages: list[dict[str, Any]],
    embeddings: np.ndarray,
    category_scores: np.ndarray,
    *,
    distance_threshold: float,
    category_thresholds: dict[str, float] | list[float],
    evidence_top_k: int = 3,
) -> dict[str, Any]:
    n = len(messages)
    vectors = np.asarray(embeddings, dtype=float)
    scores = np.asarray(category_scores, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != n or scores.shape != (n, 3):
        raise ValueError("Message, semantic and category dimensions differ")
    if (
        not np.isfinite(vectors).all()
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Non-finite embedding or invalid category scores")
    if not 0 < distance_threshold < 1 or evidence_top_k < 1:
        raise ValueError("Invalid interpretation configuration")
    thresholds = (
        list(category_thresholds.values())
        if isinstance(category_thresholds, dict)
        else list(category_thresholds)
    )
    # The attack key is explicit; do not depend on dictionary insertion order.
    attack_threshold = (
        float(category_thresholds["attack"])
        if isinstance(category_thresholds, dict)
        else float(thresholds[0])
    )
    locations = [message_location(message, i) for i, message in enumerate(messages)]
    candidates = [phrase_candidates(item["raw_text"]) for item in locations]
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.divide(
        vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 1e-12
    )
    eligible = [i for i in range(n) if norms[i] > 1e-12 and candidates[i]]
    counter = [bool(_COUNTER.search(item["raw_text"])) for item in locations]
    groups: list[list[int]] = []
    if len(eligible) >= 2:
        subset = normalized[eligible]
        distances = np.clip(1 - subset @ subset.T, 0, 2)
        for a, i in enumerate(eligible):
            for b in range(a):
                j = eligible[b]
                if (counter[i] and not counter[j] and scores[j, 0] >= attack_threshold) or (
                    counter[j] and not counter[i] and scores[i, 0] >= attack_threshold
                ):
                    distances[a, b] = distances[b, a] = 1.0
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
    words = _keywords(groups, candidates)
    groups_words = [
        (group, keyword) for group, keyword in zip(groups, words, strict=True) if keyword
    ]
    strengths = np.zeros((n, len(groups_words)), dtype=float)
    risks = scores.max(axis=1) if n else np.empty(0)
    topics = []
    for topic_id, (indices, keywords) in enumerate(groups_words):
        centroid = normalized[indices].mean(axis=0)
        norm = np.linalg.norm(centroid)
        similarities = np.clip(normalized[indices] @ (centroid / max(norm, 1e-12)), 0, 1)
        strengths[indices, topic_id] = similarities
        ranked = sorted(indices, key=lambda i: (-risks[i], i))[:evidence_top_k]
        supports = {
            word: [
                {
                    "message_id": locations[i]["comment_id"],
                    "start": candidates[i][word][0],
                    "end": candidates[i][word][1],
                    "excerpt": locations[i]["raw_text"][slice(*candidates[i][word])],
                }
                for i in indices
                if word in candidates[i]
            ]
            for word in keywords
        }
        topics.append(
            {
                "topic_id": topic_id,
                "title": " / ".join(keywords[:2]),
                "keywords": keywords,
                "topic_strength": float(similarities.mean()),
                "risk_score": float(risks[indices].mean()),
                "message_ids": [locations[i]["comment_id"] for i in indices],
                "support_messages": [locations[i] for i in indices],
                "segments": _segments(indices, locations),
                "evidence": [
                    {
                        **locations[i],
                        "risk_score": float(risks[i]),
                        "topic_strength": float(strengths[i, topic_id]),
                    }
                    for i in ranked
                ],
                "keyword_supports": supports,
                "qualityStatus": "extractive_supported",
                "contextHint": "possible_counter_speech"
                if all(counter[i] for i in indices)
                else None,
            }
        )
    ordering = sorted(range(n), key=lambda i: (-risks[i], i))
    ranks = {i: rank + 1 for rank, i in enumerate(ordering)}
    assigned = strengths.max(axis=1, initial=0) > 0
    message_scores = [
        {
            **location,
            "risk_score": float(risks[i]),
            "category_scores": scores[i].tolist(),
            "topic_strengths": strengths[i].tolist(),
            "topic_ids": np.flatnonzero(strengths[i] > 0).tolist(),
            "assigned": bool(assigned[i]),
            "rank": ranks[i],
            "contextHint": "possible_counter_speech" if counter[i] else None,
        }
        for i, location in enumerate(locations)
    ]
    evidence = [
        {**locations[i], "risk_score": float(risks[i]), "rank": rank + 1}
        for rank, i in enumerate(ordering[:evidence_top_k])
    ]
    fraction = float(assigned.mean()) if n else 0.0
    return {
        "topics": topics,
        "message_scores": message_scores,
        "evidence": evidence,
        "coverage": fraction,
        "coverage_details": {
            "assigned_messages": int(assigned.sum()),
            "total_messages": n,
            "fraction": fraction,
            "unassigned_message_ids": [
                locations[i]["comment_id"] for i in range(n) if not assigned[i]
            ],
            "zero_topic_messages": int((~assigned).sum()),
        },
    }
