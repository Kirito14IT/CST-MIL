"""Uniform masked-supervision heads and SCCD cross-fitting for seven systems."""
from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cst_mil.metrics import _binary_array, _probability_array, binary_classification_metrics
from cst_mil.unified_semantic_v1_1 import CATEGORY_IDS
from cst_mil.unified_semantic_v1_1.encoder import DEFAULT_ENCODER_PATH, LocalSemanticEncoder, unit_rows
from cst_mil.unified_semantic_v1_1.model import ConstantHead, message_features, new_char_vectorizer
from cst_mil.unified_semantic_v1_1.training import (
    balanced_source_class_weights, evaluate_messages, oof_exclusions,
)

from .topic_backend import fit_topic_backend, topic_transform

METHODS = ("lda", "btm", "gsdmm", "nmf", "bertopic", "cst_v1_1", "minilm_lr")
SEMANTIC_METHODS = frozenset({"bertopic", "cst_v1_1", "minilm_lr"})


def select_threshold_fast(y_true: Any, probabilities: Any) -> tuple[float, float]:
    """Exact old selector semantics in O(n log n), including its 1e-12 tie rule."""
    labels, scores = _binary_array(y_true), _probability_array(probabilities)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("y_true and probabilities must have equal nonzero length")
    order = np.argsort(scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    cumulative = np.concatenate(([0], np.cumsum(ordered_labels)))
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    below = np.searchsorted(ordered_scores, candidates, side="left")
    fn = cumulative[below]
    tn = below - fn
    tp = int(labels.sum()) - fn
    fp = len(labels) - int(labels.sum()) - tn
    denominators = np.column_stack((2 * tn + fp + fn, 2 * tp + fp + fn))
    numerators = np.column_stack((2 * tn, 2 * tp))
    class_f1 = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float),
                         where=denominators != 0)
    present = denominators != 0
    macro = class_f1.sum(axis=1) / present.sum(axis=1)
    best_threshold, best_score = float(candidates[0]), -1.0
    for threshold, score in zip(candidates, macro, strict=True):
        if score > best_score + 1e-12:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def _new_lr(seed: int) -> LogisticRegression:
    return LogisticRegression(C=1.0, solver="liblinear", max_iter=1000,
                              random_state=int(seed), class_weight=None)


def fit_heads(matrix: Any, records: list[dict], allowed: np.ndarray,
              seed: int) -> tuple[list[Any], list[dict]]:
    heads, audits = [], []
    for j, category in enumerate(CATEGORY_IDS):
        indices = np.asarray([int(i) for i in allowed
                              if records[int(i)]["observed_mask"][j]
                              and records[int(i)].get("fit_eligible", True)], dtype=int)
        labels = np.asarray([records[i]["labels"][j] for i in indices], dtype=int)
        sources = [records[i]["source"] for i in indices]
        weights = balanced_source_class_weights(sources, labels) if len(indices) else np.array([])
        if matrix.shape[1] == 0 or len(np.unique(labels)) < 2:
            probability = float(np.average(labels, weights=weights)) if len(labels) else 0.5
            head = ConstantHead(probability)
        else:
            head = _new_lr(seed)
            head.fit(matrix[indices], labels, sample_weight=weights)
        heads.append(head)
        audits.append({"category": category, "seed": seed, "n_observed": len(indices),
                       "class_counts": {str(c): int(np.sum(labels == c)) for c in (0, 1)},
                       "source_counts": dict(Counter(sources)),
                       "source_weight_totals": {s: float(weights[np.asarray(sources) == s].sum())
                                                for s in sorted(set(sources))},
                       "class_weight_totals": {str(c): float(weights[labels == c].sum())
                                               for c in (0, 1)},
                       "observed_record_ids": [records[i]["record_id"] for i in indices],
                       "sample_weight": weights.tolist(), "constant_head": isinstance(head, ConstantHead)})
    return heads, audits


def _scores(heads: list[Any], matrix: Any) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros((0, 3))
    return np.column_stack([head.predict_proba(matrix)[:, 1] for head in heads])


def session_feature_vectors(records: list[dict], representations: np.ndarray,
                            scores: np.ndarray, pca: Any = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Risk10 + representation mean/max + reply6, with true-edge coverage audit."""
    if not records or representations.ndim != 2 or len(representations) != len(records):
        raise ValueError("A conversation requires matching nonempty records/representations")
    if scores.shape != (len(records), 3) or not np.isfinite(scores).all():
        raise ValueError("Conversation score shape/content mismatch")
    attack = scores[:, 0]
    top_n = max(1, math.ceil(len(records) * 0.2))
    risk = np.asarray([attack.mean(), attack.max(), attack.std(),
                       *np.quantile(attack, (0.25, 0.5, 0.75, 0.9)),
                       np.sort(attack)[::-1][:top_n].mean(), np.mean(attack >= 0.5),
                       np.log1p(len(records))], dtype=float)
    if pca is not None:
        reduced = pca.transform(representations)
        reduced = np.pad(reduced, ((0, 0), (0, max(0, 64 - reduced.shape[1]))))[:, :64]
    else:
        reduced = representations
    aggregated = np.concatenate((reduced.mean(axis=0), reduced.max(axis=0)))
    ids = [str(r.get("message_id", r.get("comment_id", r["record_id"]))) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Conversation message identifiers must be unique")
    lookup = {message_id: i for i, message_id in enumerate(ids)}
    norms = np.linalg.norm(representations, axis=1)
    normalised = unit_rows(representations)
    semantic_differences, risk_differences = [], []
    true_edges = 0
    for i, record in enumerate(records):
        parent = lookup.get(str(record.get("parent_id")))
        if parent is None or parent == i:
            continue
        true_edges += 1
        risk_differences.append(float(abs(attack[i] - attack[parent])))
        if norms[i] > 1e-12 and norms[parent] > 1e-12:
            semantic_differences.append(float(1 - np.clip(
                np.dot(normalised[i], normalised[parent]), -1, 1)))
    semantic_reply = ([np.mean(semantic_differences), np.std(semantic_differences),
                       np.max(semantic_differences)] if semantic_differences else [0., 0., 0.])
    risk_reply = ([np.mean(risk_differences), np.max(risk_differences)]
                  if risk_differences else [0., 0.])
    reply = np.asarray([*semantic_reply, *risk_reply, true_edges / len(records)])
    common = np.concatenate((risk, reply))
    return np.concatenate((risk, aggregated, reply)), common, {
        "messages": len(records), "true_reply_edges": true_edges,
        "semantic_valid_reply_edges": len(semantic_differences),
        "zero_endpoint_reply_edges": true_edges - len(semantic_differences),
        "reply_edge_fraction": true_edges / len(records),
    }


def _grouping(records: list[dict]) -> dict[str, list[int]]:
    grouping: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        if record["source"] == "sccd":
            grouping[str(record["session_id"])].append(i)
    for key, indices in grouping.items():
        grouping[key] = sorted(indices, key=lambda i: (records[i].get("order", 0),
                                                       records[i].get("message_id", records[i]["record_id"])))
    return grouping


def canonical_batches(records: list[dict], batch_size: int = 64) -> list[list[int]]:
    """Frozen inference units: whole SCCD conversations, otherwise 64-ID batches."""
    batches = []
    units = sorted({(str(row.get("split", "input")), row["source"]) for row in records})
    for split, source in units:
        indices = [i for i, row in enumerate(records)
                   if row["source"] == source and str(row.get("split", "input")) == split]
        if source == "sccd":
            groups: dict[str, list[int]] = defaultdict(list)
            for i in indices:
                groups[str(records[i]["session_id"])].append(i)
            for sid in sorted(groups):
                batches.append(sorted(groups[sid], key=lambda i: (
                    records[i].get("order", 0), records[i].get("message_id", records[i]["record_id"]))))
        else:
            indices.sort(key=lambda i: records[i]["record_id"])
            batches.extend(indices[start:start + batch_size] for start in range(0, len(indices), batch_size))
    return batches


@dataclass
class ComparisonPackage:
    method: str
    seed: int
    heads: list[Any]
    thresholds: dict[str, float]
    conversation_classifier: Any
    common16_classifier: Any
    conversation_threshold: float
    common16_threshold: float
    pca: Any = None
    topic_backend: Any = None
    char_vectorizer: Any = None
    encoder_metadata: dict[str, Any] = field(default_factory=dict)
    threads: int = 4
    training_summary: dict[str, Any] = field(default_factory=dict)
    model_version: str = "v1.1-architecture-full-joint-v1"

    @property
    def eval_embeddings_needed(self) -> bool:
        return self.method in SEMANTIC_METHODS

    @property
    def parameters(self) -> dict:
        return self.training_summary.get("parameters", {})

    def encoder(self) -> LocalSemanticEncoder:
        return LocalSemanticEncoder(self.encoder_metadata.get("encoder_path", DEFAULT_ENCODER_PATH),
                                    self.encoder_metadata.get("fingerprint"), threads=self.threads)

    def _features(self, records: list[dict], embeddings: np.ndarray | None = None) -> tuple[Any, np.ndarray]:
        texts = [str(record["text"]) for record in records]
        if self.eval_embeddings_needed:
            if embeddings is None:
                embeddings = self.encoder().encode(texts)
            embeddings = unit_rows(np.asarray(embeddings))
            if embeddings.shape != (len(records), 384) or not np.isfinite(embeddings).all():
                raise ValueError("Semantic methods require finite [n_records, 384] embeddings")
        if self.method == "cst_v1_1":
            return message_features(texts, embeddings, self.char_vectorizer), embeddings
        if self.method == "minilm_lr":
            return embeddings, embeddings
        representations = topic_transform(self.topic_backend, texts, embeddings)
        return representations, representations

    def score_records(self, records: list[dict], embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        matrix, representations = self._features(records, embeddings)
        return _scores(self.heads, matrix), representations

    def _canonical_features(self, records: list[dict], embeddings: np.ndarray | None = None) -> tuple[Any, np.ndarray]:
        """Batch-sensitive UMAP/NMF transforms use the frozen inference units."""
        if self.method not in {"bertopic", "nmf"}:
            return self._features(records, embeddings)
        rows: list[Any] = [None] * len(records)
        for indices in canonical_batches(records):
            matrix, _ = self._features([records[i] for i in indices],
                                       embeddings[indices] if embeddings is not None else None)
            for i, values in zip(indices, matrix, strict=True):
                rows[i] = values
        dimension = (len(self.topic_backend.topic_ids) if self.method == "bertopic" else
                     (self.topic_backend.n_topics_ if self.topic_backend.vocabulary_ else 0))
        values = np.asarray(rows).reshape(len(records), dimension) if records else np.zeros((0, dimension))
        return values, values

    def _canonical_scores(self, records: list[dict], embeddings: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((len(records), 3))
        representations = None
        for indices in canonical_batches(records):
            current_scores, current_repr = self.score_records(
                [records[i] for i in indices], embeddings[indices] if embeddings is not None else None)
            if representations is None:
                representations = np.zeros((len(records), current_repr.shape[1]))
            scores[indices] = current_scores
            representations[indices] = current_repr
        if representations is None:
            dimensions = (384 if self.method in {"cst_v1_1", "minilm_lr"}
                          else len(self.topic_backend.topic_ids) if self.method == "bertopic"
                          else (self.topic_backend.n_topics_ if self.topic_backend.vocabulary_ else 0))
            representations = np.zeros((0, dimensions))
        return scores, representations

    def score_sessions(self, records: list[dict], sessions: list[dict], scores: np.ndarray,
                        representations: np.ndarray) -> list[dict]:
        grouping, result = _grouping(records), []
        for session in sessions:
            session_id = str(session["session_id"])
            indices = grouping.get(session_id)
            if not indices:
                raise ValueError(f"No SCCD messages for conversation {session_id}")
            main, common, coverage = session_feature_vectors(
                [records[i] for i in indices], representations[indices], scores[indices], self.pca)
            score = float(self.conversation_classifier.predict_proba(main[None, :])[0, 1])
            common_score = float(self.common16_classifier.predict_proba(common[None, :])[0, 1])
            result.append({"session_id": session_id, "split": session.get("split", "input"),
                           "label": session.get("label"), "score": score,
                           "common16_score": common_score, "threshold": self.conversation_threshold,
                           "common16_threshold": self.common16_threshold, "reply_coverage": coverage})
        return result

    def predict(self, records: list[dict], sessions: list[dict],
                embeddings: np.ndarray | None = None) -> dict[str, list[dict]]:
        scores, representations = self._canonical_scores(records, embeddings)
        message_rows = []
        for record, values in zip(records, scores, strict=True):
            row = {key: record.get(key) for key in (
                "record_id", "message_id", "source", "split", "session_id", "labels",
                "observed_mask", "duplicate_component", "parent_id", "order", "fit_eligible")}
            row.update(scores=[float(value) for value in values], text_length=len(str(record["text"])),
                       eval_group=record.get("eval_group", record.get("evaluation_group",
                                             record["duplicate_component"])))
            row["evaluation_group"] = row["eval_group"]
            message_rows.append(row)
        return {"message": message_rows,
                "session": self.score_sessions(records, sessions, scores, representations)}


def evaluate_predictions(predictions: dict[str, list[dict]], thresholds: dict[str, float]) -> dict:
    records = predictions["message"]
    scores = np.asarray([row["scores"] for row in records], dtype=float).reshape(-1, 3)
    splits = tuple(sorted({row["split"] for row in records}))
    message_metrics = evaluate_messages(records, scores, thresholds, splits=splits)
    session_metrics = {}
    for split in sorted({row["split"] for row in predictions["session"]}):
        rows = [row for row in predictions["session"] if row["split"] == split]
        session_metrics[split] = {}
        for key, threshold_key in (("score", "threshold"), ("common16_score", "common16_threshold")):
            metrics = binary_classification_metrics([row["label"] for row in rows],
                                                    [row[key] for row in rows],
                                                    threshold=rows[0][threshold_key])
            metrics.update(AP=metrics["pr_auc"], precision=metrics["precision_cb"],
                           recall=metrics["recall_cb"], n_observed=len(rows))
            session_metrics[split]["main" if key == "score" else "common16"] = metrics
    return {"messages": message_metrics, "sessions": session_metrics}


def train_method(records: list[dict], sessions: list[dict], method: str, seed: int,
                 work_dir: str | Path, embeddings: np.ndarray | None = None,
                 encoder_metadata: dict | None = None, threads: int = 4) -> tuple[ComparisonPackage, dict]:
    """Fit from train/dev only; test records are never transformed or read for fitting."""
    if method not in METHODS or not 1 <= threads <= 4:
        raise ValueError("Invalid method or CPU thread count")
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("Record IDs must be unique")
    selected = np.asarray([i for i, row in enumerate(records) if row["split"] in {"train", "dev"}], dtype=int)
    if embeddings is not None and np.asarray(embeddings).shape != (len(records), 384):
        raise ValueError("Provided embeddings must follow the complete input record order")
    current = [records[i] for i in selected]
    current_embeddings = np.asarray(embeddings)[selected] if embeddings is not None else None
    current_sessions = [row for row in sessions if row["split"] in {"train", "dev"}]
    train_indices = np.asarray([i for i, row in enumerate(current)
                                if row["split"] == "train" and row.get("fit_eligible", True)], dtype=int)
    train_sessions = [row for row in current_sessions if row["split"] == "train"]
    dev_sessions = [row for row in current_sessions if row["split"] == "dev"]
    if not len(train_indices) or not dev_sessions:
        raise ValueError("Training records and validation conversations are required")
    class_counts = Counter(row["label"] for row in train_sessions)
    if len(class_counts) != 2 or min(class_counts.values()) < 3:
        raise ValueError("Three-fold cross-fitting requires at least three conversations per class")
    grouping = _grouping(current)
    if any(str(row["session_id"]) not in grouping for row in current_sessions):
        raise ValueError("Every conversation requires its real messages or audited fallback")
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    package = ComparisonPackage(method, int(seed), [], {}, None, None, .5, .5,
                                char_vectorizer=new_char_vectorizer() if method == "cst_v1_1" else None,
                                encoder_metadata=dict(encoder_metadata or {}), threads=threads)
    stage_times, audit = {}, {"method": method, "seed": seed, "threads": threads,
                            "test_records_used_for_fit": 0, "folds": [], "head_audits": {},
                            "representation_fit_record_ids": [current[i]["record_id"] for i in train_indices],
                            "fallback_policy": "fit_eligible=False excluded from representation/PCA/head fit; retained as conversation context",
                            "fallback_record_ids": [r["record_id"] for r in current if not r.get("fit_eligible", True)]}
    started = time.perf_counter()
    with threadpool_limits(limits=threads):
        stage = time.perf_counter()
        if method in SEMANTIC_METHODS and current_embeddings is None:
            current_embeddings = package.encoder().encode([row["text"] for row in current])
        if method not in {"cst_v1_1", "minilm_lr"}:
            package.topic_backend = fit_topic_backend(
                method, [current[i]["text"] for i in train_indices], seed, threads, work_dir,
                current_embeddings[train_indices] if current_embeddings is not None else None)
        matrix, representations = package._canonical_features(current, current_embeddings)
        if method in {"cst_v1_1", "minilm_lr"}:
            package.pca = PCA(n_components=min(64, len(train_indices), representations.shape[1]),
                              svd_solver="randomized", random_state=seed)
            package.pca.fit(representations[train_indices])
        audit["representation_dimensions"] = int(representations.shape[1])
        audit["pca_seed"] = seed if package.pca is not None else None
        audit["backend"] = getattr(package.topic_backend, "info", None)
        stage_times["representation_fit_and_transform_seconds"] = time.perf_counter() - stage
        oof_scores = np.full((len(current), 3), np.nan)
        labels = np.asarray([row["label"] for row in train_sessions], dtype=int)
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        stage = time.perf_counter()
        for fold, (_, held_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            held = {str(train_sessions[i]["session_id"]) for i in held_indices}
            allowed, fold_audit = oof_exclusions(current, held)
            allowed = np.asarray([i for i in allowed if current[i].get("fit_eligible", True)], dtype=int)
            heads, head_audit = fit_heads(matrix, current, allowed, seed)
            prediction_indices = np.asarray([i for sid in sorted(held) for i in grouping[sid]], dtype=int)
            oof_scores[prediction_indices] = _scores(heads, matrix[prediction_indices])
            fold_audit.update(fold=fold, seed=seed, n_allowed=len(allowed),
                              allowed_record_ids=[current[i]["record_id"] for i in allowed])
            audit["folds"].append(fold_audit)
            audit["head_audits"][f"fold_{fold}"] = head_audit
        stage_times["crossfit_heads_seconds"] = time.perf_counter() - stage
        main_features, common_features, reply_audit = [], [], {}
        for session in train_sessions:
            indices = grouping[str(session["session_id"])]
            if not np.isfinite(oof_scores[indices]).all():
                raise ValueError("Incomplete conversation-held-out message predictions")
            main, common, coverage = session_feature_vectors(
                [current[i] for i in indices], representations[indices], oof_scores[indices], package.pca)
            main_features.append(main)
            common_features.append(common)
            reply_audit[str(session["session_id"])] = coverage
        stage = time.perf_counter()
        weights = balanced_source_class_weights(["sccd"] * len(labels), labels)
        package.conversation_classifier = make_pipeline(StandardScaler(), _new_lr(seed))
        package.common16_classifier = make_pipeline(StandardScaler(), _new_lr(seed))
        for classifier, features in ((package.conversation_classifier, main_features),
                                     (package.common16_classifier, common_features)):
            classifier.fit(np.asarray(features), labels, logisticregression__sample_weight=weights)
        package.heads, final_audit = fit_heads(matrix, current, train_indices, seed)
        audit["head_audits"]["final"] = final_audit
        audit["conversation_features"] = {"main_dimensions": len(main_features[0]),
                                           "common_dimensions": 16, "reply_coverage": reply_audit,
                                           "conversation_class_weight_totals": {
                                               str(c): float(weights[labels == c].sum()) for c in (0, 1)}}
        stage_times["final_heads_and_conversation_fit_seconds"] = time.perf_counter() - stage
        dev_indices = np.asarray([i for i, row in enumerate(current) if row["split"] == "dev"], dtype=int)
        predictions = package.predict([current[i] for i in dev_indices], dev_sessions,
                                      current_embeddings[dev_indices] if current_embeddings is not None else None)
        for j, category in enumerate(CATEGORY_IDS):
            observed = [row for row in predictions["message"] if row["observed_mask"][j]]
            if not observed:
                raise ValueError(f"No validation labels for {category}")
            package.thresholds[category] = select_threshold_fast(
                [row["labels"][j] for row in observed], [row["scores"][j] for row in observed])[0]
        dev_predictions = predictions["session"]
        package.conversation_threshold = select_threshold_fast(
            [row["label"] for row in dev_predictions], [row["score"] for row in dev_predictions])[0]
        package.common16_threshold = select_threshold_fast(
            [row["label"] for row in dev_predictions], [row["common16_score"] for row in dev_predictions])[0]
        for row in dev_predictions:
            row["threshold"] = package.conversation_threshold
            row["common16_threshold"] = package.common16_threshold
        metrics = evaluate_predictions(predictions, package.thresholds)
    stage_times["total_fit_seconds"] = time.perf_counter() - started
    parameters = {"method": method, "seed": seed, "threads": threads, "folds": 3,
                  "lr_C": 1.0, "lr_solver": "liblinear", "lr_max_iter": 1000,
                  "topic_count": 32 if method in {"lda", "btm", "gsdmm", "nmf"} else None,
                  "supervision": "same-three-masked-gold-heads", "threshold": "dev_macro_f1_lower_tie",
                  "inference_batches": "SCCD-whole-session;other-source-record_id-sorted64"}
    package.training_summary = {"parameters": parameters, "stage_times": stage_times,
                                "representation_dimensions": audit["representation_dimensions"]}
    return package, {"metrics": metrics, "dev_predictions": predictions, "fit_audit": audit,
                     "stage_times": stage_times, "parameters": parameters}
