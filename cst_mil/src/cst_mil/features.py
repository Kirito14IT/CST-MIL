"""Feature extraction utilities for CST-MIL.

The functions in this module deliberately ignore message-level annotation fields.
Only text, ordering, and reply links are used to build model inputs.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jieba
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import MiniBatchNMF
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

jieba.setLogLevel(logging.ERROR)

_TOKEN_RE = re.compile(r"^[\u3400-\u9fffA-Za-z0-9_<>]+$")
_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "和",
        "是",
        "在",
        "就",
        "都",
        "也",
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "一个",
        "这个",
        "那个",
    }
)


RecordCollection = pd.DataFrame | Sequence[Mapping[str, Any]]


def _as_frame(records: RecordCollection, *, name: str) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes)):
        return pd.DataFrame.from_records(records)
    raise TypeError(f"{name} must be a pandas DataFrame or a sequence of mappings")


def _first_column(
    frame: pd.DataFrame,
    aliases: Sequence[str],
    *,
    required: bool = True,
) -> str | None:
    for alias in aliases:
        if alias in frame.columns:
            return alias
    if required:
        raise ValueError(f"Missing required column; expected one of: {', '.join(aliases)}")
    return None


def _normalise_id(value: Any) -> str | None:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def normalise_binary_label(value: Any) -> int:
    """Convert common SCCD binary labels to ``0`` or ``1``."""

    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value) in {0.0, 1.0}:
        return int(value)
    text = str(value).strip().lower().replace("_", "-")
    positive = {"1", "cb", "cyberbullying", "cyber-bullying", "positive", "danger", "risk"}
    negative = {
        "0",
        "non-cb",
        "noncb",
        "non-cyberbullying",
        "negative",
        "normal",
        "safe",
    }
    if text in positive:
        return 1
    if text in negative:
        return 0
    raise ValueError(f"Unsupported binary label: {value!r}")


def canonicalise_sessions(
    sessions: RecordCollection,
    *,
    require_label: bool = False,
    require_split: bool = False,
) -> pd.DataFrame:
    """Return a canonical session table without copying nonessential metadata."""

    frame = _as_frame(sessions, name="sessions")
    post_col = _first_column(frame, ("post_id", "session_id", "conversation_id"))
    label_col = _first_column(frame, ("label", "session_label", "target"), required=require_label)
    split_col = _first_column(frame, ("split", "subset", "partition"), required=require_split)
    raw_col = _first_column(
        frame,
        ("raw_text", "post_content", "text", "content"),
        required=False,
    )
    model_col = _first_column(
        frame,
        ("model_text", "normalized_text", "normalised_text", "post_content", "text"),
        required=False,
    )

    result = pd.DataFrame({"post_id": frame[post_col].map(_normalise_id)})
    if result["post_id"].isna().any():
        raise ValueError("sessions contains an empty post/session identifier")
    if result["post_id"].duplicated().any():
        duplicates = result.loc[result["post_id"].duplicated(), "post_id"].head(3).tolist()
        raise ValueError(f"sessions contains duplicate identifiers: {duplicates}")

    if label_col is not None:
        result["label"] = frame[label_col].map(normalise_binary_label).astype(int)
    else:
        result["label"] = pd.Series([pd.NA] * len(frame), dtype="Int64")

    if split_col is not None:
        result["split"] = (
            frame[split_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .replace({"validation": "dev", "valid": "dev", "val": "dev"})
        )
    else:
        result["split"] = "inference"

    result["raw_text"] = frame[raw_col].fillna("").astype(str) if raw_col else ""
    if model_col:
        result["model_text"] = frame[model_col].fillna("").astype(str)
    else:
        result["model_text"] = result["raw_text"]
    return result.reset_index(drop=True)


def canonicalise_messages(messages: RecordCollection) -> pd.DataFrame:
    """Return only inference-safe message fields in canonical form.

    In particular, ``comment_label``, ``expression``, ``sarcasm``, ``target``,
    user identifiers, and popularity metadata are intentionally not retained.
    """

    frame = _as_frame(messages, name="messages")
    post_col = _first_column(frame, ("post_id", "session_id", "conversation_id"))
    comment_col = _first_column(frame, ("comment_id", "message_id", "utterance_id"), required=False)
    raw_col = _first_column(
        frame,
        ("raw_text", "comment_content", "text", "content"),
        required=False,
    )
    model_col = _first_column(
        frame,
        ("model_text", "normalized_text", "normalised_text", "comment_content", "text"),
        required=False,
    )
    context_col = _first_column(
        frame,
        ("context_model_text", "context_text", "parent_context"),
        required=False,
    )
    parent_col = _first_column(frame, ("parent_id", "to_id", "reply_to"), required=False)
    order_col = _first_column(
        frame,
        ("order", "message_order", "comment_time", "timestamp"),
        required=False,
    )

    result = pd.DataFrame({"post_id": frame[post_col].map(_normalise_id)})
    if result["post_id"].isna().any():
        raise ValueError("messages contains an empty post/session identifier")
    if comment_col:
        result["comment_id"] = frame[comment_col].map(_normalise_id)
    else:
        result["comment_id"] = [f"message-{index}" for index in range(len(frame))]
    if result["comment_id"].isna().any():
        raise ValueError("messages contains an empty comment/message identifier")

    result["raw_text"] = frame[raw_col].fillna("").astype(str) if raw_col else ""
    result["model_text"] = (
        frame[model_col].fillna("").astype(str) if model_col else result["raw_text"]
    )
    result["context_model_text"] = (
        frame[context_col].fillna("").astype(str) if context_col else ""
    )
    result["parent_id"] = frame[parent_col].map(_normalise_id) if parent_col else None
    if order_col:
        numeric_order = pd.to_numeric(frame[order_col], errors="coerce")
        fallback = pd.Series(np.arange(len(frame), dtype=float), index=frame.index)
        result["order"] = numeric_order.fillna(fallback).astype(float)
    else:
        result["order"] = np.arange(len(frame), dtype=float)

    # Resolve context locally when the preparation stage did not precompute it.
    text_lookup = {
        (post_id, comment_id): text
        for post_id, comment_id, text in result[["post_id", "comment_id", "model_text"]].itertuples(
            index=False, name=None
        )
    }
    contexts: list[str] = []
    for row in result.itertuples(index=False):
        supplied = str(row.context_model_text).strip()
        if supplied:
            contexts.append(supplied)
            continue
        parent_text = text_lookup.get((row.post_id, row.parent_id), "") if row.parent_id else ""
        if parent_text:
            contexts.append(f"{parent_text} [REPLY] {row.model_text}")
        else:
            contexts.append(row.model_text)
    result["context_model_text"] = contexts
    return result.sort_values(["post_id", "order", "comment_id"], kind="mergesort").reset_index(
        drop=True
    )


def build_char_vectorizer(
    *,
    ngram_min: int = 2,
    ngram_max: int = 5,
    n_features: int = 65_536,
) -> HashingVectorizer:
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError("Invalid character n-gram range")
    if n_features < 2:
        raise ValueError("n_features must be at least 2")
    return HashingVectorizer(
        analyzer="char",
        ngram_range=(ngram_min, ngram_max),
        n_features=n_features,
        alternate_sign=False,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def jieba_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in jieba.lcut(str(text), cut_all=False):
        token = token.strip().lower()
        if token and token not in _STOPWORDS and _TOKEN_RE.fullmatch(token):
            tokens.append(token)
    return tokens


@dataclass
class TopicFeatureExtractor:
    """Word TF-IDF plus a fixed-width MiniBatchNMF topic representation."""

    n_components: int = 8
    max_features: int = 20_000
    min_df: int = 2
    max_df: float = 0.95
    random_state: int = 42
    top_words_count: int = 12
    vectorizer: TfidfVectorizer | None = field(default=None, init=False)
    nmf: MiniBatchNMF | None = field(default=None, init=False)
    actual_components_: int = field(default=0, init=False)
    fallback_analyzer_: str | None = field(default=None, init=False)

    def _make_word_vectorizer(self, *, min_df: int | float, max_df: int | float) -> TfidfVectorizer:
        return TfidfVectorizer(
            tokenizer=jieba_tokens,
            token_pattern=None,
            lowercase=False,
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df,
            max_features=self.max_features,
            sublinear_tf=True,
            dtype=np.float64,
        )

    def fit(self, texts: Sequence[str]) -> TopicFeatureExtractor:
        documents = [str(text) for text in texts]
        if not documents:
            raise ValueError("At least one training message is required for topic fitting")
        if self.n_components < 1:
            raise ValueError("n_components must be positive")

        effective_min_df: int | float = self.min_df
        effective_max_df: int | float = self.max_df
        if isinstance(effective_min_df, int) and effective_min_df > len(documents):
            effective_min_df = 1
        if isinstance(effective_max_df, float) and effective_max_df * len(documents) < float(
            effective_min_df
        ):
            effective_min_df = 1
            effective_max_df = 1.0

        self.vectorizer = self._make_word_vectorizer(
            min_df=effective_min_df,
            max_df=effective_max_df,
        )
        try:
            matrix = self.vectorizer.fit_transform(documents)
        except ValueError as error:
            # Degenerate smoke-test corpora can be entirely punctuation/stop words.
            # The production SCCD run remains on the requested jieba word channel.
            if "empty vocabulary" not in str(error).lower():
                raise
            self.fallback_analyzer_ = "character"
            self.vectorizer = TfidfVectorizer(
                analyzer="char",
                ngram_range=(1, 2),
                min_df=1,
                max_df=1.0,
                max_features=self.max_features,
                sublinear_tf=True,
                dtype=np.float64,
            )
            matrix = self.vectorizer.fit_transform(documents)

        self.actual_components_ = max(1, min(self.n_components, matrix.shape[0], matrix.shape[1]))
        self.nmf = MiniBatchNMF(
            n_components=self.actual_components_,
            init="nndsvda" if self.actual_components_ <= min(matrix.shape) else "random",
            random_state=self.random_state,
            batch_size=min(256, max(1, matrix.shape[0])),
            max_iter=200,
        )
        self.nmf.fit(matrix)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.vectorizer is None or self.nmf is None:
            raise RuntimeError("TopicFeatureExtractor must be fitted before transform")
        documents = [str(text) for text in texts]
        if not documents:
            return np.zeros((0, self.n_components), dtype=np.float64)
        transformed = self.nmf.transform(self.vectorizer.transform(documents))
        if transformed.shape[1] == self.n_components:
            return transformed
        padded = np.zeros((transformed.shape[0], self.n_components), dtype=np.float64)
        padded[:, : transformed.shape[1]] = transformed
        return padded

    def fit_transform(self, texts: Sequence[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    def topic_words(self, top_n: int | None = None) -> list[list[str]]:
        if self.vectorizer is None or self.nmf is None:
            raise RuntimeError("TopicFeatureExtractor must be fitted before reading topic words")
        requested = self.top_words_count if top_n is None else int(top_n)
        if requested < 1:
            raise ValueError("top_n must be positive")
        vocabulary = np.asarray(self.vectorizer.get_feature_names_out())
        topics: list[list[str]] = []
        for component in self.nmf.components_:
            order = np.argsort(component, kind="mergesort")[::-1][:requested]
            topics.append(vocabulary[order].tolist())
        return topics


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def aggregate_session_features(
    messages: pd.DataFrame,
    risk_scores: Sequence[float] | np.ndarray,
    topic_matrix: np.ndarray,
    *,
    session_ids: Sequence[str] | None = None,
    root_topics: Mapping[str, np.ndarray] | None = None,
    risk_cutoff: float = 0.5,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate message risk, topics, and reply structure to session rows."""

    canonical = canonicalise_messages(messages)
    risks = np.asarray(risk_scores, dtype=float)
    topics = np.asarray(topic_matrix, dtype=float)
    if len(canonical) != len(risks) or len(canonical) != topics.shape[0]:
        raise ValueError("messages, risk_scores, and topic_matrix must have equal row counts")
    if topics.ndim != 2:
        raise ValueError("topic_matrix must be two-dimensional")
    if not 0.0 <= risk_cutoff <= 1.0:
        raise ValueError("risk_cutoff must be in [0, 1]")

    n_topics = topics.shape[1]
    canonical = canonical.copy()
    canonical["_risk"] = risks
    canonical["_row"] = np.arange(len(canonical))
    requested_ids = (
        [str(session_id) for session_id in session_ids]
        if session_ids is not None
        else canonical["post_id"].drop_duplicates().tolist()
    )

    feature_names = [
        "risk_mean",
        "risk_max",
        "risk_top3_mean",
        "risk_std",
        "risk_fraction_ge_half",
        "risk_last",
        "log1p_message_count",
        "reply_ratio",
        "adjacent_topic_l1_mean",
        "adjacent_topic_l1_max",
        "reply_topic_l1_mean",
        "reply_topic_l1_max",
    ]
    feature_names.extend(f"topic_mean_{index}" for index in range(n_topics))
    feature_names.extend(f"topic_max_{index}" for index in range(n_topics))
    feature_names.extend(f"root_topic_{index}" for index in range(n_topics))

    rows: list[list[float]] = []
    for post_id in requested_ids:
        group = canonical.loc[canonical["post_id"] == post_id].sort_values(
            ["order", "comment_id"], kind="mergesort"
        )
        group_risks = group["_risk"].to_numpy(dtype=float)
        indices = group["_row"].to_numpy(dtype=int)
        group_topics = topics[indices] if len(indices) else np.zeros((0, n_topics), dtype=float)

        top_three = np.sort(group_risks)[-3:] if group_risks.size else np.asarray([], dtype=float)
        risk_features = [
            _safe_mean(group_risks),
            _safe_max(group_risks),
            _safe_mean(top_three),
            float(np.std(group_risks)) if group_risks.size else 0.0,
            float(np.mean(group_risks >= risk_cutoff)) if group_risks.size else 0.0,
            float(group_risks[-1]) if group_risks.size else 0.0,
            math.log1p(len(group)),
            float(group["parent_id"].notna().mean()) if len(group) else 0.0,
        ]

        if len(group_topics) > 1:
            adjacent_distances = np.abs(np.diff(group_topics, axis=0)).sum(axis=1)
        else:
            adjacent_distances = np.asarray([], dtype=float)

        local_topic_by_comment = {
            comment_id: group_topics[position]
            for position, comment_id in enumerate(group["comment_id"].tolist())
        }
        reply_distances: list[float] = []
        for position, parent_id in enumerate(group["parent_id"].tolist()):
            if parent_id in local_topic_by_comment:
                reply_distances.append(
                    float(np.abs(group_topics[position] - local_topic_by_comment[parent_id]).sum())
                )
        reply_array = np.asarray(reply_distances, dtype=float)
        structure_features = [
            _safe_mean(adjacent_distances),
            _safe_max(adjacent_distances),
            _safe_mean(reply_array),
            _safe_max(reply_array),
        ]

        topic_mean = (
            np.mean(group_topics, axis=0) if len(group_topics) else np.zeros(n_topics, dtype=float)
        )
        topic_max = (
            np.max(group_topics, axis=0) if len(group_topics) else np.zeros(n_topics, dtype=float)
        )
        if root_topics and post_id in root_topics:
            root_topic = np.asarray(root_topics[post_id], dtype=float)
            if root_topic.shape != (n_topics,):
                raise ValueError(f"root topic for {post_id!r} has an unexpected shape")
        else:
            root_topic = np.zeros(n_topics, dtype=float)
        rows.append(
            risk_features
            + structure_features
            + topic_mean.tolist()
            + topic_max.tolist()
            + root_topic.tolist()
        )

    features = pd.DataFrame(
        rows,
        columns=feature_names,
        index=pd.Index(requested_ids, name="post_id"),
    )
    return features, feature_names


def transform_char_texts(
    vectorizer: HashingVectorizer,
    messages: pd.DataFrame,
) -> sparse.csr_matrix:
    canonical = canonicalise_messages(messages)
    return vectorizer.transform(canonical["context_model_text"].tolist()).tocsr()
