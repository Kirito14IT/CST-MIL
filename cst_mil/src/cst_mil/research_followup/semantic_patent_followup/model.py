"""Independent, masked semantic systems and matched mechanism ablations.

This module imports immutable numerical helpers; it never changes those helpers,
loads an encoder while fitting supplied embeddings, or chooses a TEST setting.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cst_mil.research_followup.full_joint_comparison.model import (
    _new_lr, canonical_batches, evaluate_predictions, select_threshold_fast,
    session_feature_vectors,
)
from cst_mil.unified_semantic_v1_1 import CATEGORY_IDS
from cst_mil.unified_semantic_v1_1.encoder import (
    DEFAULT_ENCODER_PATH, LocalSemanticEncoder, unit_rows,
)
from cst_mil.unified_semantic_v1_1.model import ConstantHead, message_features, new_char_vectorizer
from cst_mil.unified_semantic_v1_1.training import balanced_source_class_weights, oof_exclusions

ARCHITECTURES = ("full", "context_a", "context_b")
ABLATIONS = (
    "no_char", "no_message_semantics", "no_session_semantics",
    "no_session_reply", "no_session_risk", "no_source_balance",
)


@dataclass(frozen=True)
class VariantConfig:
    architecture_base: str = "full"
    ablation: str | None = None

    def __post_init__(self) -> None:
        if self.architecture_base not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {self.architecture_base}")
        if self.ablation is not None and self.ablation not in ABLATIONS:
            raise ValueError(f"Unknown ablation: {self.ablation}")

    @property
    def variant(self) -> str:
        return self.ablation or self.architecture_base


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _embeddings(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (count, 384) or not np.isfinite(values).all():
        raise ValueError("Finite [n_records, 384] semantic embeddings required")
    return unit_rows(values)


def resolve_parents(records: list[dict]) -> tuple[np.ndarray, dict]:
    """Resolve only true same-source/split/session edges, excluding cycle edges.

    A child of a cycle node may use that node's text, but the cycle node's own
    invalid outgoing edge is removed. The remaining usable graph is acyclic.
    """
    groups: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(records):
        if row.get("session_id") is not None:
            groups[(str(row["source"]), str(row.get("split", "input")),
                    str(row["session_id"]))].append(i)
    parents = np.full(len(records), -1, dtype=np.int64)
    reasons: Counter = Counter()
    cycles: set[int] = set()
    for indices in groups.values():
        ids = [str(records[i].get("message_id", records[i]["record_id"])) for i in indices]
        if len(set(ids)) != len(ids):
            raise ValueError("Message IDs must be unique within a source/split/session")
        lookup = dict(zip(ids, indices, strict=True))
        for i in indices:
            parent_id = records[i].get("parent_id")
            if parent_id is None:
                reasons["missing"] += 1
            elif str(parent_id) not in lookup:
                reasons["absent_or_outside_partition"] += 1
            elif lookup[str(parent_id)] == i:
                reasons["self_link"] += 1
            else:
                parents[i] = lookup[str(parent_id)]
        visited: set[int] = set()
        for start in indices:
            if start in visited:
                continue
            trail: list[int] = []
            position: dict[int, int] = {}
            cursor = start
            while cursor >= 0 and cursor not in visited and cursor not in position:
                position[cursor] = len(trail)
                trail.append(cursor)
                cursor = int(parents[cursor])
            if cursor in position:
                cycles.update(trail[position[cursor]:])
            visited.update(trail)
    for i, row in enumerate(records):
        if row.get("session_id") is None:
            reasons["no_session"] += 1
    if cycles:
        parents[list(cycles)] = -1
    reasons["cycle_edge"] = len(cycles)
    valid = np.flatnonzero(parents >= 0)
    return parents, {
        "valid_edges": int(len(valid)), "invalid_reasons": dict(reasons),
        "edges": [{"child_record_id": records[i]["record_id"],
                   "parent_record_id": records[int(parents[i])]["record_id"]} for i in valid],
        "cycle_record_ids": sorted(records[i]["record_id"] for i in cycles),
        "scope": "same-source/split/session; self-links and cycle edges excluded",
    }


def message_matrix(records: list[dict], embeddings: np.ndarray, config: VariantConfig,
                   char_vectorizer: Any = None) -> tuple[Any, np.ndarray, list[dict], dict]:
    """Build labelled blocks; all added context blocks are zero without an edge."""
    representations = _embeddings(embeddings, len(records))
    texts = [str(row["text"]) for row in records]
    parents, parent_audit = resolve_parents(records)
    char_vectorizer = char_vectorizer if char_vectorizer is not None else new_char_vectorizer()
    if config == VariantConfig():
        # Preserve the existing two normalization steps and sparse dtype exactly.
        matrix = message_features(texts, representations, char_vectorizer)
        return matrix, representations, [
            {"name": "current_character", "start": 0, "stop": 65536},
            {"name": "current_semantic", "start": 65536, "stop": 65920},
        ], parent_audit
    blocks, descriptions, start = [], [], 0

    def append(name: str, values: Any) -> None:
        nonlocal start
        values = sparse.csr_matrix(values)
        blocks.append(values)
        descriptions.append({"name": name, "start": start, "stop": start + values.shape[1]})
        start += values.shape[1]

    if config.ablation != "no_char":
        append("current_character", char_vectorizer.transform(texts))
    semantics = config.ablation != "no_message_semantics"
    semantic = unit_rows(representations)
    if semantics:
        append("current_semantic", semantic)
    if config.architecture_base != "full":
        valid = parents >= 0
        parent = np.zeros_like(semantic)
        parent[valid] = semantic[parents[valid]]
        if semantics:
            append("parent_semantic", parent)
        append("valid_parent", valid.astype(np.float32)[:, None])
        if config.architecture_base == "context_b" and semantics:
            difference = np.zeros_like(semantic)
            difference[valid] = unit_rows(np.abs(semantic[valid] - parent[valid]))
            append("absolute_semantic_difference", difference)
            del difference
            product = np.zeros_like(semantic)
            product[valid] = unit_rows(semantic[valid] * parent[valid])
            append("semantic_product", product)
    matrix = sparse.hstack(blocks, format="csr") if blocks else sparse.csr_matrix((len(records), 0))
    return matrix, representations, descriptions, parent_audit


def global_class_weights(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int)
    if not len(labels):
        return np.zeros(0, dtype=float)
    classes, counts = np.unique(labels, return_counts=True)
    return np.asarray([len(labels) / (len(classes) * counts[np.searchsorted(classes, y)])
                       for y in labels], dtype=float)


def fit_heads(matrix: Any, records: list[dict], allowed: np.ndarray, seed: int,
              config: VariantConfig) -> tuple[list[Any], list[dict]]:
    heads, audits = [], []
    for j, category in enumerate(CATEGORY_IDS):
        indices = np.asarray([int(i) for i in allowed if records[int(i)]["observed_mask"][j]
                              and records[int(i)].get("fit_eligible", True)], dtype=int)
        labels = np.asarray([records[i]["labels"][j] for i in indices], dtype=int)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Observed binary labels must be zero or one")
        sources = [records[i]["source"] for i in indices]
        weights = (global_class_weights(labels) if config.ablation == "no_source_balance" else
                   balanced_source_class_weights(sources, labels) if len(labels) else np.zeros(0))
        if matrix.shape[1] == 0 or len(np.unique(labels)) < 2:
            head = ConstantHead(float(np.average(labels, weights=weights)) if len(labels) else 0.5)
        else:
            head = _new_lr(seed)
            head.fit(matrix[indices], labels, sample_weight=weights)
        heads.append(head)
        audits.append({"category": category, "seed": seed, "n_observed": len(indices),
                       "observed_record_ids": [records[i]["record_id"] for i in indices],
                       "sample_weight": weights.tolist(), "source_counts": dict(Counter(sources)),
                       "class_counts": {str(c): int(np.sum(labels == c)) for c in (0, 1)},
                       "source_weight_totals": {s: float(weights[np.asarray(sources) == s].sum())
                                                for s in sorted(set(sources))},
                       "class_weight_totals": {str(c): float(weights[labels == c].sum()) for c in (0, 1)},
                       "constant_head": isinstance(head, ConstantHead)})
    return heads, audits


def _scores(heads: list[Any], matrix: Any) -> np.ndarray:
    return (np.column_stack([head.predict_proba(matrix)[:, 1] for head in heads])
            if matrix.shape[0] else np.zeros((0, 3), dtype=float))


def _grouping(records: list[dict]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(records):
        if row["source"] == "sccd":
            groups[str(row["session_id"])].append(i)
    for sid in groups:
        groups[sid].sort(key=lambda i: (records[i].get("order", 0), records[i]["message_id"]))
    return groups


def session_vectors(records: list[dict], embeddings: np.ndarray, scores: np.ndarray,
                    pca: Any, config: VariantConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    main, common, coverage = session_feature_vectors(records, embeddings, scores, pca)
    main_indices = list(range(len(main)))
    common_indices = list(range(16))
    if config.ablation == "no_session_semantics":
        main_indices = [*range(10), *range(138, 144)]
    elif config.ablation == "no_session_reply":
        main_indices = list(range(138))
        common_indices = list(range(10))
    elif config.ablation == "no_session_risk":
        # Reply risk differences are derived risk features and must also vanish.
        main_indices = [*range(10, 138), 138, 139, 140, 143]
        common_indices = [10, 11, 12, 15]
    coverage = {**coverage, "main_feature_indices": main_indices,
                "common_feature_indices": common_indices}
    return main[main_indices], common[common_indices], coverage


@dataclass
class FollowupPackage:
    config: VariantConfig
    seed: int
    heads: list[Any] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    conversation_classifier: Any = None
    common16_classifier: Any = None
    conversation_threshold: float = 0.5
    common16_threshold: float = 0.5
    pca: Any = None
    char_vectorizer: Any = None
    encoder_metadata: dict[str, Any] = field(default_factory=dict)
    threads: int = 4
    training_summary: dict[str, Any] = field(default_factory=dict)
    method: str = "semantic_patent_followup"
    model_version: str = "semantic-patent-followup-v1"

    def encoder(self) -> LocalSemanticEncoder:
        return LocalSemanticEncoder(self.encoder_metadata.get("encoder_path", DEFAULT_ENCODER_PATH),
                                    self.encoder_metadata.get("fingerprint"), threads=self.threads)

    def encode_records(self, records: list[dict], encoder: Any = None) -> np.ndarray:
        """Canonical raw batches; orchestration owns the fixed TRAIN warmup."""
        encoder = encoder if encoder is not None else self.encoder()
        embeddings = np.zeros((len(records), 384), dtype=np.float32)
        for indices in canonical_batches(records):
            embeddings[indices] = encoder.encode([str(records[i]["text"]) for i in indices])
        return embeddings

    def predict(self, records: list[dict], embeddings: np.ndarray | None = None) -> np.ndarray:
        if not records:
            return np.zeros((0, 3), dtype=float)
        if embeddings is None:
            embeddings = self.encode_records(records)
        matrix, _, _, _ = message_matrix(records, embeddings, self.config, self.char_vectorizer)
        return _scores(self.heads, matrix)

    def score_records(self, records: list[dict], embeddings: np.ndarray | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
        if embeddings is None:
            embeddings = self.encode_records(records)
        representations = _embeddings(embeddings, len(records))
        return self.predict(records, embeddings), representations

    def score_sessions(self, records: list[dict], scores: np.ndarray, embeddings: np.ndarray,
                       sessions: list[dict] | None = None) -> list[dict]:
        grouping, result = _grouping(records), []
        representations = _embeddings(embeddings, len(records))
        if sessions is None:
            sessions = [{"session_id": sid, "split": records[indices[0]].get("split", "input")}
                        for sid, indices in sorted(grouping.items())]
        for session in sessions:
            sid = str(session["session_id"])
            indices = grouping.get(sid)
            if not indices:
                raise ValueError(f"No matching SCCD messages for {sid}")
            main, common, coverage = session_vectors(
                [records[i] for i in indices], representations[indices], scores[indices], self.pca, self.config)
            result.append({"session_id": sid, "split": session.get("split", "input"),
                           "label": session.get("label"),
                           "score": float(self.conversation_classifier.predict_proba(main[None])[0, 1]),
                           "common16_score": float(self.common16_classifier.predict_proba(common[None])[0, 1]),
                           "threshold": self.conversation_threshold,
                           "common16_threshold": self.common16_threshold, "reply_coverage": coverage})
        return result

    def predict_payload(self, records: list[dict], sessions: list[dict],
                        embeddings: np.ndarray | None = None) -> dict[str, list[dict]]:
        if embeddings is None:
            embeddings = self.encode_records(records)
        scores = self.predict(records, embeddings)
        messages = []
        for record, values in zip(records, scores, strict=True):
            row = {key: record.get(key) for key in (
                "record_id", "message_id", "source", "split", "session_id", "labels", "observed_mask",
                "duplicate_component", "parent_id", "order", "fit_eligible")}
            row.update(scores=values.tolist(), text_length=len(str(record["text"])),
                       eval_group=record.get("eval_group", record.get("evaluation_group",
                                             record.get("duplicate_component"))))
            row["evaluation_group"] = row["eval_group"]
            messages.append(row)
        return {"message": messages, "session": self.score_sessions(records, scores, embeddings, sessions)}


def fit_variant(records: list[dict], embeddings: np.ndarray, sessions: list[dict], *,
                variant: str = "full", seed: int = 42, base_package: Any = None,
                config: VariantConfig | None = None, encoder_metadata: dict | None = None,
                threads: int = 4) -> tuple[FollowupPackage, dict, list[dict]]:
    """Fit TRAIN only; DEV selects thresholds, never architectures. No disk writes."""
    if not 1 <= threads <= 4:
        raise ValueError("CPU thread cap must be between 1 and 4")
    if config is None:
        architecture = (getattr(base_package, "config", VariantConfig()).architecture_base
                        if variant in ABLATIONS else variant)
        config = VariantConfig(architecture, variant if variant in ABLATIONS else None)
    if len({row["record_id"] for row in records}) != len(records):
        raise ValueError("Record IDs must be unique")
    if np.asarray(embeddings).shape != (len(records), 384):
        raise ValueError("Embeddings must follow complete input record order")
    selected = np.asarray([i for i, row in enumerate(records) if row["split"] in {"train", "dev"}], dtype=int)
    current = [records[i] for i in selected]
    current_embeddings = np.asarray(embeddings)[selected]
    train_indices = np.asarray([i for i, row in enumerate(current) if row["split"] == "train"
                                and row.get("fit_eligible", True)], dtype=int)
    train_sessions = [row for row in sessions if row["split"] == "train"]
    dev_sessions = [row for row in sessions if row["split"] == "dev"]
    labels = np.asarray([row["label"] for row in train_sessions], dtype=int)
    counts = Counter(labels.tolist())
    if not len(train_indices) or not dev_sessions or set(counts) != {0, 1} or min(counts.values()) < 3:
        raise ValueError("TRAIN and DEV required, with at least three TRAIN sessions of each class")
    grouping = _grouping(current)
    if any(str(row["session_id"]) not in grouping for row in [*train_sessions, *dev_sessions]):
        raise ValueError("Each TRAIN/DEV session requires its real messages or audited fallback")
    package = FollowupPackage(config, int(seed), char_vectorizer=new_char_vectorizer(),
                              encoder_metadata=dict(encoder_metadata or getattr(base_package, "encoder_metadata", {})),
                              threads=threads)
    audit = {"config": asdict(config), "seed": seed, "threads": threads,
             "test_records_used_for_fit": 0, "folds": [], "head_audits": {},
             "representation_fit_record_ids": [current[i]["record_id"] for i in train_indices],
             "fallback_record_ids": [r["record_id"] for r in current if not r.get("fit_eligible", True)],
             "fallback_policy": "excluded from heads/PCA; retained as conversation context"}
    stage_times, started = {}, time.perf_counter()
    with threadpool_limits(limits=threads):
        stage = time.perf_counter()
        matrix, representations, blocks, parents = message_matrix(
            current, current_embeddings, config, package.char_vectorizer)
        package.pca = PCA(n_components=min(64, len(train_indices), 384),
                          svd_solver="randomized", random_state=seed)
        package.pca.fit(representations[train_indices])
        audit.update(feature_blocks=blocks, parent_context=parents, pca_seed=seed,
                     message_feature_dimensions=matrix.shape[1],
                     training_embedding_sha256=hashlib.sha256(
                         np.ascontiguousarray(representations[train_indices]).tobytes()).hexdigest())
        stage_times["representation_fit_and_transform_seconds"] = time.perf_counter() - stage
        oof = np.full((len(current), 3), np.nan)
        stage = time.perf_counter()
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for fold, (_, held_indices) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            held = {str(train_sessions[i]["session_id"]) for i in held_indices}
            allowed, fold_audit = oof_exclusions(current, held)
            allowed = np.asarray([i for i in allowed if current[i].get("fit_eligible", True)], dtype=int)
            heads, head_audit = fit_heads(matrix, current, allowed, seed, config)
            prediction_indices = np.asarray([i for sid in sorted(held) for i in grouping[sid]], dtype=int)
            oof[prediction_indices] = _scores(heads, matrix[prediction_indices])
            fold_audit.update(fold=fold, seed=seed, n_allowed=len(allowed),
                              allowed_record_ids=[current[i]["record_id"] for i in allowed])
            audit["folds"].append(fold_audit)
            audit["head_audits"][f"fold_{fold}"] = head_audit
        stage_times["crossfit_heads_seconds"] = time.perf_counter() - stage
        main_features, common_features, replies = [], [], {}
        for session in train_sessions:
            indices = grouping[str(session["session_id"])]
            if not np.isfinite(oof[indices]).all():
                raise ValueError("Incomplete held-session OOF predictions")
            main, common, coverage = session_vectors(
                [current[i] for i in indices], representations[indices], oof[indices], package.pca, config)
            main_features.append(main)
            common_features.append(common)
            replies[str(session["session_id"])] = coverage
        stage = time.perf_counter()
        weights = balanced_source_class_weights(["sccd"] * len(labels), labels)
        package.conversation_classifier = make_pipeline(StandardScaler(), _new_lr(seed))
        package.common16_classifier = make_pipeline(StandardScaler(), _new_lr(seed))
        for classifier, features in ((package.conversation_classifier, main_features),
                                     (package.common16_classifier, common_features)):
            classifier.fit(np.asarray(features), labels, logisticregression__sample_weight=weights)
        package.heads, final_audit = fit_heads(matrix, current, train_indices, seed, config)
        audit["head_audits"]["final"] = final_audit
        audit["conversation_features"] = {
            "main_dimensions": len(main_features[0]), "common_dimensions": len(common_features[0]),
            "reply_coverage": replies, "session_ids": [row["session_id"] for row in train_sessions],
            "sample_weight": weights.tolist(),
            "class_weight_totals": {str(c): float(weights[labels == c].sum()) for c in (0, 1)}}
        stage_times["final_heads_and_conversation_fit_seconds"] = time.perf_counter() - stage
        dev_indices = np.asarray([i for i, row in enumerate(current) if row["split"] == "dev"], dtype=int)
        stage = time.perf_counter()
        predictions = package.predict_payload([current[i] for i in dev_indices], dev_sessions,
                                              current_embeddings[dev_indices])
        for j, category in enumerate(CATEGORY_IDS):
            observed = [row for row in predictions["message"] if row["observed_mask"][j]]
            if not observed:
                raise ValueError(f"No DEV labels for {category}")
            package.thresholds[category] = select_threshold_fast(
                [row["labels"][j] for row in observed], [row["scores"][j] for row in observed])[0]
        for attr, field_name in (("conversation_threshold", "score"), ("common16_threshold", "common16_score")):
            setattr(package, attr, select_threshold_fast(
                [row["label"] for row in predictions["session"]],
                [row[field_name] for row in predictions["session"]])[0])
        for row in predictions["session"]:
            row["threshold"] = package.conversation_threshold
            row["common16_threshold"] = package.common16_threshold
        metrics = evaluate_predictions(predictions, package.thresholds)
        stage_times["dev_prediction_and_threshold_seconds"] = time.perf_counter() - stage
    stage_times["total_fit_seconds"] = time.perf_counter() - started
    parameters = {"config": asdict(config), "seed": seed, "threads": threads, "folds": 3,
                  "lr_C": 1.0, "lr_solver": "liblinear", "lr_max_iter": 1000,
                  "threshold": "dev_macro_f1_lower_tie", "pca_dimensions": 64,
                  "message_feature_blocks": blocks,
                  "source_class_weights": "global_class_mean_one" if config.ablation == "no_source_balance"
                  else "equal_source_then_equal_within_source_class_mean_one",
                  "parent_missing_policy": "all_added_context_blocks_zero",
                  "no_message_semantics_policy": "remove_all_semantic_blocks_keep_structural_valid_parent",
                  "no_session_risk_policy": "remove_risk10_and_both_reply_risk_differences"}
    package.training_summary = {"parameters": parameters, "stage_times": stage_times,
                                "training_record_ids_digest": _digest(audit["representation_fit_record_ids"]),
                                "training_embedding_sha256": audit["training_embedding_sha256"]}
    oof_rows = [{"record_id": row["record_id"], "session_id": row["session_id"],
                 "message_id": row["message_id"], "scores": oof[i].tolist(),
                 "split": "train"} for i, row in enumerate(current)
                if row["source"] == "sccd" and row["split"] == "train"]
    return package, {"metrics": metrics, "dev_predictions": predictions, "fit_audit": audit,
                     "stage_times": stage_times, "parameters": parameters}, oof_rows
