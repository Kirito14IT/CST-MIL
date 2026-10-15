"""Reproducible masked supervision, genuine SCCD message cross fitting, evaluation."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cst_mil.metrics import binary_classification_metrics, select_threshold

from . import CATEGORY_IDS, MODEL_VERSION
from .encoder import DEFAULT_ENCODER_PATH, LocalSemanticEncoder, digest_json, file_sha256
from .model import (
    ConstantHead,
    UnifiedSemanticModel,
    message_features,
    new_char_vectorizer,
    session_features,
)

PARAMETERS = {
    "model_version": MODEL_VERSION,
    "seed": 42,
    "threads": 4,
    "folds": 3,
    "lr_C": 1.0,
    "lr_solver": "liblinear",
    "lr_max_iter": 1000,
    "char_ngram_range": [2, 5],
    "char_n_features": 65536,
    "pca_dimensions": 64,
    "encoder_content_tokens": 112,
    "encoder_overlap": 16,
    "source_class_weights": "equal-source_then_equal-class-within-source_mean-one",
    "conversation_features": "attack10+pca64_mean_max+real_reply6-v1",
    "threshold_selection": "dev_macro_f1_lower_tie",
    "topic_distance_initial": 0.35,
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def balanced_source_class_weights(sources: list[str], labels: np.ndarray) -> np.ndarray:
    """Each source has equal weight, split equally between its observed classes.

    Unknowns must already be removed by the caller. Mean-one normalization keeps
    the fixed C comparable across folds; class_weight is never additionally set.
    """
    labels = np.asarray(labels, dtype=int)
    if len(sources) != len(labels) or not len(labels):
        raise ValueError("Source/label arrays must have equal nonzero length")
    counts = Counter(zip(sources, labels.tolist(), strict=True))
    classes = {source: {label for s, label in counts if s == source} for source in set(sources)}
    weights = np.asarray(
        [
            1 / (len(classes[source]) * counts[(source, int(label))])
            for source, label in zip(sources, labels, strict=True)
        ]
    )
    return weights * (len(weights) / weights.sum())


def _new_lr() -> LogisticRegression:
    return LogisticRegression(
        C=1.0, solver="liblinear", max_iter=1000, random_state=42, class_weight=None
    )


def fit_heads(
    matrix: Any, records: list[dict[str, Any]], allowed_indices: np.ndarray
) -> tuple[list[Any], list[dict[str, Any]]]:
    heads, audit = [], []
    for category_index, category in enumerate(CATEGORY_IDS):
        indices = np.asarray(
            [i for i in allowed_indices if records[int(i)]["observed_mask"][category_index]],
            dtype=int,
        )
        labels = np.asarray([records[i]["labels"][category_index] for i in indices], dtype=int)
        sources = [records[i]["source"] for i in indices]
        weights = balanced_source_class_weights(sources, labels) if len(labels) else np.array([])
        if len(np.unique(labels)) < 2:
            head = ConstantHead(float(labels[0]) if len(labels) else 0.5)
        else:
            head = _new_lr()
            head.fit(matrix[indices], labels, sample_weight=weights)
        heads.append(head)
        audit.append(
            {
                "category": category,
                "n_observed": len(indices),
                "class_counts": {str(c): int(np.sum(labels == c)) for c in (0, 1)},
                "source_counts": dict(Counter(sources)),
                "source_weight_totals": {
                    source: float(weights[np.asarray(sources) == source].sum())
                    for source in sorted(set(sources))
                },
                "class_weight_totals": {str(c): float(weights[labels == c].sum()) for c in (0, 1)},
                "degenerate": isinstance(head, ConstantHead),
                "observed_record_ids": [records[i]["record_id"] for i in indices],
                "sample_weight": weights.tolist(),
            }
        )
    return heads, audit


def _scores(heads: list[Any], matrix: Any) -> np.ndarray:
    return np.column_stack([head.predict_proba(matrix)[:, 1] for head in heads])


def oof_exclusions(
    records: list[dict[str, Any]], held_sessions: set[str]
) -> tuple[np.ndarray, dict]:
    held_indices = [
        i
        for i, record in enumerate(records)
        if record["source"] == "sccd" and record["session_id"] in held_sessions
    ]
    components = {records[i]["duplicate_component"] for i in held_indices}
    allowed = np.asarray(
        [
            i
            for i, record in enumerate(records)
            if record["split"] == "train"
            and not (record["source"] == "sccd" and record["session_id"] in held_sessions)
            and record["duplicate_component"] not in components
        ],
        dtype=int,
    )
    return allowed, {
        "held_session_ids": sorted(held_sessions),
        "held_record_ids": [records[i]["record_id"] for i in held_indices],
        "held_duplicate_components": sorted(components),
        "allowed_record_ids": [records[i]["record_id"] for i in allowed],
        "excluded_cross_source_record_ids": [
            r["record_id"]
            for r in records
            if r["split"] == "train"
            and r["source"] != "sccd"
            and r["duplicate_component"] in components
        ],
    }


def _messages(records: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "comment_id": records[i]["message_id"],
            "raw_text": records[i]["text"],
            "order": records[i]["order"],
            "parent_id": records[i]["parent_id"],
        }
        for i in indices
    ]


def _metrics(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    if not labels:
        return {"n_observed": 0, "status": "not_annotated", "AP": None, "roc_auc": None}
    metrics = binary_classification_metrics(labels, scores, threshold=threshold)
    metrics.update(
        {
            "AP": metrics["pr_auc"],
            "precision": metrics["precision_cb"],
            "recall": metrics["recall_cb"],
            "n_observed": len(labels),
            "status": "both_classes" if len(set(labels)) == 2 else "one_class",
        }
    )
    return metrics


def evaluate_messages(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    thresholds: dict[str, float],
    *,
    splits: tuple[str, ...] = ("dev", "test"),
) -> dict[str, Any]:
    result = {}
    for split in splits:
        result[split] = {}
        for source in ["pooled", *sorted({r["source"] for r in records})]:
            source_metrics = {}
            for j, category in enumerate(CATEGORY_IDS):
                indices = [
                    i
                    for i, r in enumerate(records)
                    if r["split"] == split
                    and (source == "pooled" or r["source"] == source)
                    and r["observed_mask"][j]
                ]
                metric = _metrics(
                    [records[i]["labels"][j] for i in indices],
                    [float(scores[i, j]) for i in indices],
                    thresholds[category],
                )
                total = sum(
                    r["split"] == split and (source == "pooled" or r["source"] == source)
                    for r in records
                )
                metric.update(
                    {
                        "n_total": total,
                        "annotation_coverage": len(indices) / total if total else 0.0,
                    }
                )
                source_metrics[category] = metric
            result[split][source] = source_metrics
        result[split]["source_macro"] = {}
        for category in CATEGORY_IDS:
            observed = [
                result[split][source][category]
                for source in sorted({r["source"] for r in records})
                if result[split][source][category]["n_observed"]
            ]
            result[split]["source_macro"][category] = {
                name: float(np.mean([m[name] for m in observed if m.get(name) is not None]))
                if any(m.get(name) is not None for m in observed)
                else None
                for name in ("AP", "roc_auc", "macro_f1", "precision", "recall")
            }
            result[split]["source_macro"][category]["n_sources"] = len(observed)
    return result


def _code_fingerprints() -> dict[str, str]:
    root = Path(__file__).parent
    paths = [
        root / name for name in ("__init__.py", "encoder.py", "model.py", "training.py", "data.py")
    ]
    return {p.name: file_sha256(p) for p in paths if p.is_file()}


def _identity(prepared_dir: Path, encoder: LocalSemanticEncoder) -> dict[str, Any]:
    return {
        "parameters": PARAMETERS,
        "code": _code_fingerprints(),
        "prepared": {
            name: file_sha256(prepared_dir / name)
            for name in ("records.jsonl", "sessions.jsonl", "manifest.json")
        },
        "encoder_fingerprint": encoder.fingerprint,
    }


class _PeakMemory:
    def __init__(self):
        self.peak_bytes = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        import psutil

        process = psutil.Process()
        while not self.stop.is_set():
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
            self.stop.wait(0.2)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def train(
    output_dir: str | Path,
    prepared_dir: str | Path,
    resume: bool = False,
    *,
    defer_test: bool = True,
) -> dict[str, Any]:
    """Fit one frozen-protocol candidate. Validation gates are applied by caller.

    Returns model_path, metrics_path, summary_path and run fingerprint. Existing
    output is never silently replaced; --resume requires exact input identity.
    """
    output_dir, prepared_dir = Path(output_dir).resolve(), Path(prepared_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = LocalSemanticEncoder(DEFAULT_ENCODER_PATH, threads=4)
    encoder.validate_assets()
    identity = _identity(prepared_dir, encoder)
    identity["defer_test"] = defer_test
    fingerprint = digest_json(identity)
    state_path = output_dir / "run_state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not resume:
            raise FileExistsError("Run already exists; use resume with matching fingerprints")
        if state.get("fingerprint") != fingerprint:
            raise ValueError("Resume code/data/encoder/parameter fingerprint mismatch")
        if state.get("stage") == "complete":
            for name, expected in state["output_fingerprints"].items():
                if file_sha256(output_dir / name) != expected:
                    raise ValueError(f"Completed run output changed: {name}")
            return state["result"]
    _write_json(state_path, {"fingerprint": fingerprint, "identity": identity, "stage": "starting"})
    records = _read_jsonl(prepared_dir / "records.jsonl")
    sessions = _read_jsonl(prepared_dir / "sessions.jsonl")
    if not records or len({r["record_id"] for r in records}) != len(records):
        raise ValueError("Prepared records must have unique IDs and be nonempty")
    if any(not r["duplicate_component"] for r in records):
        raise ValueError("All records require a nonempty duplicate component")
    train_indices = np.asarray(
        [i for i, r in enumerate(records) if r["split"] == "train"], dtype=int
    )
    train_sessions = [s for s in sessions if s["split"] == "train"]
    counts = Counter(s["label"] for s in train_sessions)
    if len(counts) != 2 or min(counts.values()) < 3:
        raise ValueError("SCCD cross fitting requires at least 3 sessions in each class")
    started = time.perf_counter()
    stage_times = {}
    with _PeakMemory() as memory, threadpool_limits(limits=4):
        stage = time.perf_counter()
        embeddings = encoder.encode(
            [r["text"] for r in records], cache_path=output_dir / "embeddings.npz"
        )
        stage_times["embedding_seconds"] = time.perf_counter() - stage
        _write_json(
            state_path, {"fingerprint": fingerprint, "identity": identity, "stage": "encoded"}
        )
        vectorizer = new_char_vectorizer()
        matrix = message_features([r["text"] for r in records], embeddings, vectorizer)
        n_components = min(64, len(train_indices), embeddings.shape[1])
        pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
        pca.fit(embeddings[train_indices])
        grouping: dict[str, list[int]] = defaultdict(list)
        for i, record in enumerate(records):
            if record["source"] == "sccd":
                grouping[record["session_id"]].append(i)
        for session_id, indices in grouping.items():
            grouping[session_id] = sorted(
                indices, key=lambda i: (records[i]["order"], records[i]["message_id"])
            )
        oof_scores = np.full((len(records), 3), np.nan)
        folds, head_audits = [], {}
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        labels = np.asarray([s["label"] for s in train_sessions], dtype=int)
        stage = time.perf_counter()
        for fold, (_, held_session_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            held = {train_sessions[i]["session_id"] for i in held_session_indices}
            allowed, fold_audit = oof_exclusions(records, held)
            heads, fitting_audit = fit_heads(matrix, records, allowed)
            held_record_indices = np.asarray(
                [i for session_id in held for i in grouping[session_id]], dtype=int
            )
            oof_scores[held_record_indices] = _scores(heads, matrix[held_record_indices])
            fold_audit.update(
                {
                    "fold": fold,
                    "seed": 42,
                    "n_allowed": len(allowed),
                    "head_counts": [
                        {
                            k: v
                            for k, v in a.items()
                            if k not in {"observed_record_ids", "sample_weight"}
                        }
                        for a in fitting_audit
                    ],
                }
            )
            folds.append(fold_audit)
            head_audits[f"fold_{fold}"] = fitting_audit
            _write_json(output_dir / "oof_folds.json", folds)
            _write_json(
                state_path,
                {
                    "fingerprint": fingerprint,
                    "identity": identity,
                    "stage": f"crossfit_fold_{fold + 1}_complete",
                },
            )
        stage_times["crossfit_heads_seconds"] = time.perf_counter() - stage
        train_session_features = []
        for session in train_sessions:
            indices = grouping[session["session_id"]]
            if not np.isfinite(oof_scores[indices]).all():
                raise ValueError("Incomplete OOF predictions for SCCD training session")
            train_session_features.append(
                session_features(
                    _messages(records, indices), embeddings[indices], oof_scores[indices], pca
                )
            )
        conversation_classifier = make_pipeline(StandardScaler(), _new_lr())
        conversation_weights = balanced_source_class_weights(["sccd"] * len(labels), labels)
        conversation_classifier.fit(
            np.asarray(train_session_features),
            labels,
            logisticregression__sample_weight=conversation_weights,
        )
        stage = time.perf_counter()
        final_heads, final_audit = fit_heads(matrix, records, train_indices)
        head_audits["final"] = final_audit
        scores = _scores(final_heads, matrix)
        stage_times["final_heads_seconds"] = time.perf_counter() - stage
        thresholds = {}
        for j, category in enumerate(CATEGORY_IDS):
            indices = [
                i for i, r in enumerate(records) if r["split"] == "dev" and r["observed_mask"][j]
            ]
            if not indices:
                raise ValueError(f"No validation supervision for category {category}")
            thresholds[category] = select_threshold(
                [records[i]["labels"][j] for i in indices], scores[indices, j]
            )[0]
        conversation_predictions = []
        for session in sessions:
            if defer_test and session["split"] == "test":
                continue
            indices = grouping[session["session_id"]]
            current_scores = oof_scores[indices] if session["split"] == "train" else scores[indices]
            vector = session_features(
                _messages(records, indices), embeddings[indices], current_scores, pca
            )
            value = float(conversation_classifier.predict_proba(vector[None, :])[0, 1])
            conversation_predictions.append(
                {
                    "session_id": session["session_id"],
                    "split": session["split"],
                    "label": session["label"],
                    "score": value,
                    "message_score_mode": "crossfit" if session["split"] == "train" else "final",
                }
            )
        dev = [p for p in conversation_predictions if p["split"] == "dev"]
        conversation_threshold = select_threshold(
            [p["label"] for p in dev], [p["score"] for p in dev]
        )[0]
        conversation_metrics = {}
        evaluation_splits = ("dev",) if defer_test else ("dev", "test")
        for split in evaluation_splits:
            values = [p for p in conversation_predictions if p["split"] == split]
            conversation_metrics[split] = _metrics(
                [p["label"] for p in values], [p["score"] for p in values], conversation_threshold
            )
        for prediction in conversation_predictions:
            prediction["threshold"] = conversation_threshold
            prediction["prediction"] = int(prediction["score"] >= conversation_threshold)
        metrics = {
            "messages": evaluate_messages(records, scores, thresholds, splits=evaluation_splits),
            "conversations": conversation_metrics,
            "thresholds": thresholds,
            "conversation_threshold": conversation_threshold,
            "test_not_used_for_selection": True,
            "test_deferred": defer_test,
            "threshold_selection_weighting": "unweighted_observed_dev_pooled_macro_f1",
        }
        elapsed = time.perf_counter() - started
        summary = {
            "model_version": MODEL_VERSION,
            "fingerprint": fingerprint,
            "identity": identity,
            "parameters": PARAMETERS,
            "encoder_path": encoder.model_path,
            "encoder_fingerprint": encoder.fingerprint,
            "encoder_info": encoder.info,
            "records_by_source_split": {
                f"{source}/{split}": n
                for (source, split), n in Counter(
                    (r["source"], r["split"]) for r in records
                ).items()
            },
            "session_counts": dict(Counter(s["split"] for s in sessions)),
            "thresholds": thresholds,
            "conversation_threshold": conversation_threshold,
            "training_seconds": elapsed,
            "stage_seconds": stage_times,
            "peak_process_rss_bytes": memory.peak_bytes,
            "peak_memory_sampling_seconds": 0.2,
            "pca_fitted_dimensions": n_components,
            "pca_fit_record_ids": [records[i]["record_id"] for i in train_indices],
            "weights_mean_one": True,
            "class_weight_extra": None,
            "supervision": "comment_gold_masked + SCCD_session_gold; no bag-to-message copying",
            "conversation_feature_dimensions": int(np.asarray(train_session_features).shape[1]),
            "deployment_dependencies": {
                "python": "3.10-3.12",
                "torch": "cpu",
                "sentence_transformers": True,
                "GPU_required": False,
            },
        }
        model = UnifiedSemanticModel(
            final_heads,
            vectorizer,
            pca,
            conversation_classifier,
            thresholds,
            conversation_threshold,
            encoder.model_path,
            str(encoder.fingerprint),
            training_summary=summary,
        )
        final_identity = _identity(prepared_dir, encoder)
        final_identity["defer_test"] = defer_test
        if digest_json(final_identity) != fingerprint:
            raise ValueError("Source/data identity changed during training; refusing checkpoint")
        checkpoint = output_dir / "model.joblib"
        joblib.dump(model, checkpoint, compress=3)
        summary["model_bytes"] = checkpoint.stat().st_size
        encoder_manifest = encoder.validate_assets()
        summary["encoder_asset_bytes"] = sum(f["bytes"] for f in encoder_manifest["files"].values())
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "training_summary.json", summary)
        _write_json(output_dir / "head_fitting_audit.json", head_audits)
        _write_jsonl(output_dir / "conversation_predictions.jsonl", conversation_predictions)
        prediction_rows = []
        for i, record in enumerate(records):
            if defer_test and record["split"] == "test":
                continue
            prediction_rows.append(
                {
                    "record_id": record["record_id"],
                    "source": record["source"],
                    "split": record["split"],
                    "session_id": record["session_id"],
                    "message_id": record["message_id"],
                    "labels": record["labels"],
                    "observed_mask": record["observed_mask"],
                    "scores": {
                        category: float(scores[i, j]) for j, category in enumerate(CATEGORY_IDS)
                    },
                    "oof_scores": {
                        category: float(oof_scores[i, j]) if np.isfinite(oof_scores[i, j]) else None
                        for j, category in enumerate(CATEGORY_IDS)
                    },
                }
            )
        _write_jsonl(output_dir / "message_predictions.jsonl", prediction_rows)
    result = {
        "model_path": str(output_dir / "model.joblib"),
        "metrics_path": str(output_dir / "metrics.json"),
        "summary_path": str(output_dir / "training_summary.json"),
        "fingerprint": fingerprint,
    }
    output_names = [
        "model.joblib",
        "metrics.json",
        "training_summary.json",
        "head_fitting_audit.json",
        "oof_folds.json",
        "message_predictions.jsonl",
        "conversation_predictions.jsonl",
    ]
    _write_json(
        state_path,
        {
            "fingerprint": fingerprint,
            "identity": identity,
            "stage": "complete",
            "result": result,
            "output_fingerprints": {name: file_sha256(output_dir / name) for name in output_names},
        },
    )
    return result


def evaluate_test(
    output_dir: str | Path, prepared_dir: str | Path, freeze_path: str | Path
) -> dict[str, Any]:
    """Produce test metrics once risk thresholds and topic configuration are frozen.

    The caller writes a concrete freeze record with model_sha256, thresholds,
    conversation_threshold, and topic_distance before calling this function.
    """
    output_dir, prepared_dir = Path(output_dir).resolve(), Path(prepared_dir).resolve()
    frozen = json.loads(Path(freeze_path).read_text(encoding="utf-8"))
    freeze_hash = file_sha256(freeze_path)
    test_state_path = output_dir / "test_evaluation_state.json"
    if test_state_path.is_file():
        state = json.loads(test_state_path.read_text(encoding="utf-8"))
        if state.get("freeze_sha256") != freeze_hash:
            raise ValueError("Test has already been opened under a different frozen configuration")
    model_path = output_dir / "model.joblib"
    if frozen.get("model_sha256") != file_sha256(model_path):
        raise ValueError("Test evaluation requires the exact frozen model SHA256")
    model: UnifiedSemanticModel = joblib.load(model_path)
    if (
        frozen.get("thresholds") != model.thresholds
        or frozen.get("conversation_threshold") != model.conversation_threshold
        or frozen.get("topic_distance") != model.topic_distance
    ):
        raise ValueError("Test freeze thresholds/topic configuration mismatch")
    original = model.training_summary["identity"]
    encoder = model.encoder()
    encoder.validate_assets()
    current = _identity(prepared_dir, encoder)
    current["defer_test"] = original.get("defer_test", True)
    if digest_json(current) != model.training_summary["fingerprint"]:
        raise ValueError("Test evaluation code/data/encoder fingerprint changed")
    if test_state_path.is_file() and state.get("stage") == "complete":
        for name, expected in state["output_fingerprints"].items():
            if file_sha256(output_dir / name) != expected:
                raise ValueError(f"Test output changed: {name}")
        return json.loads((output_dir / "metrics_test.json").read_text(encoding="utf-8"))
    _write_json(test_state_path, {"freeze_sha256": freeze_hash, "stage": "starting"})
    records = _read_jsonl(prepared_dir / "records.jsonl")
    sessions = _read_jsonl(prepared_dir / "sessions.jsonl")
    embeddings = encoder.encode(
        [r["text"] for r in records], cache_path=output_dir / "embeddings.npz"
    )
    scores = model.score_embeddings([r["text"] for r in records], embeddings)
    grouping: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        if record["source"] == "sccd":
            grouping[record["session_id"]].append(i)
    predictions = []
    for session in sessions:
        if session["split"] != "test":
            continue
        indices = sorted(
            grouping[session["session_id"]],
            key=lambda i: (records[i]["order"], records[i]["message_id"]),
        )
        vector = session_features(
            _messages(records, indices), embeddings[indices], scores[indices], model.pca
        )
        score = float(model.conversation_classifier.predict_proba(vector[None, :])[0, 1])
        predictions.append(
            {
                "session_id": session["session_id"],
                "split": "test",
                "label": session["label"],
                "score": score,
                "threshold": model.conversation_threshold,
                "prediction": int(score >= model.conversation_threshold),
            }
        )
    metrics = {
        "messages": evaluate_messages(records, scores, model.thresholds, splits=("test",)),
        "conversations": {
            "test": _metrics(
                [p["label"] for p in predictions],
                [p["score"] for p in predictions],
                model.conversation_threshold,
            )
        },
        "freeze_sha256": freeze_hash,
        "model_sha256": file_sha256(model_path),
        "test_not_used_for_selection": True,
    }
    rows = [
        {
            "record_id": record["record_id"],
            "source": record["source"],
            "split": "test",
            "session_id": record["session_id"],
            "message_id": record["message_id"],
            "labels": record["labels"],
            "observed_mask": record["observed_mask"],
            "scores": {category: float(scores[i, j]) for j, category in enumerate(CATEGORY_IDS)},
        }
        for i, record in enumerate(records)
        if record["split"] == "test"
    ]
    _write_json(output_dir / "metrics_test.json", metrics)
    _write_jsonl(output_dir / "message_test_predictions.jsonl", rows)
    _write_jsonl(output_dir / "conversation_test_predictions.jsonl", predictions)
    output_names = (
        "metrics_test.json",
        "message_test_predictions.jsonl",
        "conversation_test_predictions.jsonl",
    )
    _write_json(
        test_state_path,
        {
            "freeze_sha256": freeze_hash,
            "stage": "complete",
            "output_fingerprints": {name: file_sha256(output_dir / name) for name in output_names},
        },
    )
    return metrics
