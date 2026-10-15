"""One packaged supervised CST semantic model; dynamic topics are independent."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from cst_mil.bench.interpretation import message_location

from . import CATEGORY_IDS, CATEGORY_NAMES, MODEL_VERSION
from .encoder import LocalSemanticEncoder, unit_rows


def new_char_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        n_features=65536,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
    )


def message_features(
    texts: list[str], embeddings: np.ndarray, char_vectorizer: HashingVectorizer
) -> sparse.csr_matrix:
    if embeddings.shape != (len(texts), 384):
        raise ValueError("Message embedding shape mismatch")
    return sparse.hstack(
        [char_vectorizer.transform(texts), sparse.csr_matrix(unit_rows(embeddings))], format="csr"
    )


@dataclass
class ConstantHead:
    probability: float

    def predict_proba(self, matrix: Any) -> np.ndarray:
        scores = np.full(matrix.shape[0], self.probability, dtype=float)
        return np.column_stack([1 - scores, scores])


def session_features(
    messages: list[dict[str, Any]], embeddings: np.ndarray, category_scores: np.ndarray, pca: Any
) -> np.ndarray:
    """SCCD session features with observed parent links, never fabricated edges."""
    if not messages or embeddings.shape[0] != len(messages):
        raise ValueError("A session requires at least one message/fallback")
    attack = category_scores[:, 0]
    sorted_scores = np.sort(attack)[::-1]
    top_count = max(1, math.ceil(len(attack) * 0.2))
    scores = [
        float(attack.mean()),
        float(attack.max()),
        float(attack.std()),
        *[float(np.quantile(attack, q)) for q in (0.25, 0.5, 0.75, 0.9)],
        float(sorted_scores[:top_count].mean()),
        float(np.mean(attack >= 0.5)),
        float(np.log1p(len(messages))),
    ]
    reduced = np.asarray(pca.transform(embeddings), dtype=float)
    # Tiny fixtures may fit fewer dimensions; deployment always retains 64.
    reduced = np.pad(reduced, ((0, 0), (0, max(0, 64 - reduced.shape[1]))))[:, :64]
    semantic = np.concatenate([reduced.mean(axis=0), reduced.max(axis=0)])
    lookup = {str(message["comment_id"]): i for i, message in enumerate(messages)}
    cosine_differences, risk_differences = [], []
    for i, message in enumerate(messages):
        parent = lookup.get(str(message.get("parent_id")))
        if parent is not None and parent != i:
            cosine_differences.append(
                float(1 - np.clip(np.dot(embeddings[i], embeddings[parent]), -1, 1))
            )
            risk_differences.append(float(abs(attack[i] - attack[parent])))
    reply = (
        [
            float(np.mean(cosine_differences)),
            float(np.std(cosine_differences)),
            float(np.max(cosine_differences)),
            float(np.mean(risk_differences)),
            float(np.max(risk_differences)),
            len(cosine_differences) / len(messages),
        ]
        if cosine_differences
        else [0.0] * 6
    )
    return np.concatenate([np.asarray(scores), semantic, np.asarray(reply)])


@dataclass
class UnifiedSemanticModel:
    heads: list[Any]
    char_vectorizer: Any
    pca: Any
    conversation_classifier: Any
    thresholds: dict[str, float]
    conversation_threshold: float
    encoder_path: str
    encoder_fingerprint: str
    topic_distance: float = 0.35
    model_version: str = MODEL_VERSION
    training_summary: dict[str, Any] = field(default_factory=dict)

    def encoder(self) -> LocalSemanticEncoder:
        return LocalSemanticEncoder(self.encoder_path, self.encoder_fingerprint, threads=4)

    def validate_assets(self) -> dict[str, Any]:
        return self.encoder().validate_assets()

    def score_embeddings(self, texts: list[str], embeddings: np.ndarray) -> np.ndarray:
        matrix = message_features(texts, embeddings, self.char_vectorizer)
        if not texts:
            return np.zeros((0, 3), dtype=float)
        return np.column_stack([head.predict_proba(matrix)[:, 1] for head in self.heads])


def canonical_messages(
    session: dict[str, Any], messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Only original text and allowed display metadata cross inference boundary."""
    post_id = str(session.get("post_id", session.get("session_id", "session")))
    result = []
    for index, message in enumerate(messages):
        result.append(
            {
                "comment_id": str(
                    message.get("comment_id", message.get("message_id", f"m{index}"))
                ),
                "post_id": post_id,
                "raw_text": str(message.get("raw_text", message.get("text", ""))),
                "order": message.get("order", index),
                "parent_id": message.get("parent_id"),
                "speaker_id": message.get("speaker_id"),
                "comment_time": message.get("comment_time"),
            }
        )
    if not result:
        result.append(
            {
                "comment_id": f"{post_id}:root_fallback",
                "post_id": post_id,
                "raw_text": str(session.get("post_raw_text", session.get("post_text", ""))),
                "order": 0,
                "parent_id": None,
                "speaker_id": None,
                "comment_time": None,
            }
        )
    if len({m["comment_id"] for m in result}) != len(result):
        raise ValueError("Message IDs must be unique within a session")
    return result


def predict_session(
    model: UnifiedSemanticModel, session: dict[str, Any], messages: list[dict[str, Any]]
) -> dict[str, Any]:
    from .topics import interpret_topics

    current = canonical_messages(session, messages)
    texts = [message["raw_text"] for message in current]
    embeddings = model.encoder().encode(texts)
    scores = model.score_embeddings(texts, embeddings)
    vector = session_features(current, embeddings, scores, model.pca)
    conversation_score = float(model.conversation_classifier.predict_proba(vector[None, :])[0, 1])
    categories = []
    for index, (category_id, name) in enumerate(zip(CATEGORY_IDS, CATEGORY_NAMES, strict=True)):
        threshold = float(model.thresholds[category_id])
        selected = np.flatnonzero(scores[:, index] >= threshold)
        ranked = sorted(selected.tolist(), key=lambda i: (-scores[i, index], i))
        categories.append(
            {
                "id": category_id,
                "name": name,
                "score": float(scores[:, index].max()),
                "threshold": threshold,
                "detected": bool(len(selected)),
                "message_ids": [current[i]["comment_id"] for i in ranked],
                "evidence": [
                    {
                        **message_location(current[i], i),
                        "risk_score": float(scores[i, index]),
                        "score": float(scores[i, index]),
                    }
                    for i in ranked[:3]
                ],
            }
        )
    interpretation = interpret_topics(
        current,
        embeddings,
        scores,
        distance_threshold=model.topic_distance,
        category_thresholds=model.thresholds,
        evidence_top_k=3,
    )
    detected = any(category["detected"] for category in categories)
    result = {
        "schemaVersion": 2,
        "post_id": str(session.get("post_id", session.get("session_id", "session"))),
        "split": session.get("split", "input"),
        "label": None,
        "prediction": int(detected),
        "prediction_name": "发现风险线索" if detected else "未发现明确风险线索",
        "overallEvidenceScore": float(scores.max()),
        "scoreType": "max_message_category_score",
        "riskCategories": categories,
        "conversationRisk": {
            "score": conversation_score,
            "threshold": float(model.conversation_threshold),
            "prediction": int(conversation_score >= model.conversation_threshold),
            "domain": "SCCD",
            "note": "SCCD网络欺凌会话评分；校园与其他领域仅作迁移参考，非校准概率。",
        },
        "modelVersion": model.model_version,
        **interpretation,
    }
    # Explicit per-message category scores support a single message in several categories.
    by_id = {message["comment_id"]: i for i, message in enumerate(current)}
    for message in result.get("message_scores", []):
        i = by_id[str(message["comment_id"])]
        message["riskCategoryScores"] = {
            category: float(scores[i, j]) for j, category in enumerate(CATEGORY_IDS)
        }
    return result
