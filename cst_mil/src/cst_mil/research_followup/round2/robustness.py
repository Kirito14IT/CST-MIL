"""Equal-weight, fixed-topic head controls and paired complete-session inference.

Only cached unsupervised topics are read from the first round. Supervised heads
are refitted within each complete-session fold and all five seeds are retained.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import vstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from cst_mil.bench.dataset import load_suite
from cst_mil.bench.evaluation import binary_metrics
from cst_mil.bench.model import (
    ConstantRisk,
    RiskTextFeatures,
    _bag_features,
    _bag_weights,
    _risk_texts,
    _window_average,
)
from cst_mil.bench.resources import configure_cpu
from cst_mil.bench.runner import ResourceMonitor
from cst_mil.metrics import select_threshold

from .core import (
    METHODS,
    OLD,
    OUT,
    PROJECT,
    RUNS,
    code_sources,
    jsonl_read,
    jsonl_write,
    read,
    save,
    source_hashes,
    stable,
    verify_sources,
)

SEEDS = (42, 43, 44, 45, 46)
FAMILY_SIZE = 22
MARGIN = 0.01


def schedule(seeds=None):
    seeds = list(SEEDS if seeds is None else seeds)
    if not seeds or len(seeds) != len(set(seeds)) or not set(seeds) <= set(SEEDS):
        raise ValueError("Seeds must be a nonempty unique subset of 42..46")
    return [
        dict(method=method, seed=int(seed), instance_head=ih, session_head=sh)
        for method in METHODS
        for seed in seeds
        for ih in (["lr"] if method == "cst_mil_r0" else ["lr", "rf"])
        for sh in ["lr", "rf"]
    ]


def contrasts():
    return [
        dict(method=m, factor="session", control="lr_lr", treatment="lr_rf") for m in METHODS
    ] + [
        dict(method=m, factor="instance", control="lr_lr", treatment="rf_lr")
        for m in METHODS
        if m != "cst_mil_r0"
    ]


def forest(seed, *, session=False):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced" if session else None,
        random_state=int(seed),
        n_jobs=4,
    )


def weight_summary(bags, labels, weights, indices):
    return dict(
        n_instances=len(indices),
        n_bags=len(np.unique(bags)),
        minimum=float(weights.min()),
        maximum=float(weights.max()),
        sum=float(weights.sum()),
        class_weight_sum={str(c): float(weights[labels == c].sum()) for c in [0, 1]},
        bag_weight_sum={str(int(b)): float(weights[bags == b].sum()) for b in np.unique(bags)},
        selected_instance_indices=np.asarray(indices).tolist(),
        selected_instance_indices_sha256=hashlib.sha256(
            np.asarray(indices, dtype="<i8").tobytes()
        ).hexdigest(),
        explicit_weight_sha256=hashlib.sha256(
            np.asarray(weights, dtype="<f8").tobytes()
        ).hexdigest(),
    )


def fit_instance(x, labels, bags, head, config, mil=False):
    """Mirror original LR/MIL fits, recording the actual weights at every fit."""
    labels, bags = np.asarray(labels, dtype=int), np.asarray(bags, dtype=int)
    if head not in {"lr", "rf"} or (mil and head != "lr"):
        raise ValueError("Only LR supports the frozen MIL mechanism")
    if len(labels) != x.shape[0] or len(bags) != len(labels):
        raise ValueError("Instance, bag and label dimensions differ")
    if any(len(np.unique(labels[bags == b])) != 1 for b in np.unique(bags)):
        raise ValueError("Every complete session must have a single inherited label")
    if len(np.unique(labels)) < 2 or x.shape[1] == 0:
        prior = float(labels[0]) if len(np.unique(labels)) == 1 else 0.5
        return ConstantRisk(prior), []
    rounds = int(config.get("mil_rounds", 3)) if mil else 0
    fraction = float(config.get("positive_top_fraction", 0.2))
    if rounds < 0 or not 0 < fraction <= 1:
        raise ValueError("Invalid MIL rounds/fraction")
    selected = np.arange(len(labels))
    history = []
    for round_index in range(rounds + 1):
        weights = _bag_weights(bags[selected], labels[selected])
        history.append(weight_summary(bags[selected], labels[selected], weights, selected))
        if head == "lr":
            classifier = LogisticRegression(
                C=float(config.get("instance_C", config.get("instance_c", 4.0 if mil else 1.0))),
                solver="liblinear",
                max_iter=1000,
                random_state=int(config.get("seed", 42)),
            )
        else:
            classifier = forest(config.get("seed", 42))
        classifier.fit(x[selected], labels[selected], sample_weight=weights)
        if round_index < rounds:
            scores = classifier.predict_proba(x)[:, 1]
            indices = np.flatnonzero(labels == 0).tolist()
            for bag in np.unique(bags[labels == 1]):
                candidates = np.flatnonzero(bags == bag)
                k = max(1, math.ceil(len(candidates) * fraction))
                order = np.argsort(-scores[candidates], kind="stable")[:k]
                indices.extend(candidates[order].tolist())
            selected = np.asarray(indices, dtype=int)
    return classifier, history


def partitions(labels, seed):
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=2)
    if set(np.unique(labels)) != {0, 1} or min(counts) < 2:
        raise ValueError("Need at least two complete training sessions per class")
    return list(
        StratifiedKFold(
            n_splits=min(3, int(min(counts))),
            shuffle=True,
            random_state=int(seed),
        ).split(np.arange(len(labels)), labels)
    )


def crossfit(x, bags, labels_by_bag, *, head, config, mil=False):
    bags, labels_by_bag = np.asarray(bags, dtype=int), np.asarray(labels_by_bag, dtype=int)
    if set(bags) != set(range(len(labels_by_bag))) or x.shape[0] != len(bags):
        raise ValueError("Every complete session must be represented in the instance matrix")
    labels = labels_by_bag[bags]
    scores = np.full(len(bags), np.nan)
    history = []
    for fold, (train_bags, held_bags) in enumerate(partitions(labels_by_bag, config["seed"])):
        train = np.flatnonzero(np.isin(bags, train_bags))
        held = np.flatnonzero(np.isin(bags, held_bags))
        classifier, weights = fit_instance(
            x[train], labels[train], bags[train], head, config, mil=mil
        )
        scores[held] = classifier.predict_proba(x[held])[:, 1]
        history.append(
            dict(
                fold=fold,
                train_bag_indices=train_bags.tolist(),
                held_bag_indices=held_bags.tolist(),
                train_instance_indices=train.tolist(),
                held_instance_indices=held.tolist(),
                fitted_instances=len(train),
                held_instances=len(held),
                weight_rounds=weights,
                constant_fallback=isinstance(classifier, ConstantRisk),
            )
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("Incomplete out-of-fold predictions")
    classifier, full_weights = fit_instance(x, labels, bags, head, config, mil=mil)
    return scores, classifier, history, full_weights


def session_estimator(head, config):
    if head == "rf":
        return forest(config["seed"], session=True)
    if head != "lr":
        raise ValueError("Unknown session head")
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config.get("session_C", config.get("session_c", 1.0))),
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=int(config["seed"]),
        ),
    )


def load_frozen_cache(path, binding, sessions):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required immutable topic cache missing: {path}")
    payload = joblib.load(path)
    if payload.get("binding") != binding:
        raise ValueError("Frozen topic cache binding mismatch")
    if payload.get("sessions") != sessions or not payload.get("saved_message_topic_parity"):
        raise ValueError("Frozen topic cache corpus/order/parity mismatch")
    for name in ["local", "windows", "window_topics", "message_topics"]:
        if len(payload[name]) != len(sessions):
            raise ValueError(f"Frozen topic cache length mismatch: {name}")
    config = payload["config"]
    if any(config.get(k, False) for k in ["word_features", "multiscale", "risk_weighted_topics"]):
        raise ValueError(
            "Frozen protocol requires stateless hashing and unweighted single-scale topics"
        )
    return payload


def weighted_metrics(labels, scores, weights):
    """AP and AUC for many resamples, with exact weighted tie grouping.

    Each row of weights is a bootstrap session multiplicity vector. Sorting the
    original scores once is equivalent to materializing every repeated session.
    """
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    if weights.shape[1] != len(labels) or len(scores) != len(labels):
        raise ValueError("Bootstrap dimensions differ")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Bootstrap scores/weights must be finite and weights nonnegative")
    order = np.argsort(-scores, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1]
    positive = np.add.reduceat(weights[:, order] * labels[order], cuts, axis=1)
    negative = np.add.reduceat(weights[:, order] * (1 - labels[order]), cuts, axis=1)
    tp, fp = positive.cumsum(axis=1), negative.cumsum(axis=1)
    n_pos, n_neg = tp[:, -1], fp[:, -1]
    if (n_pos <= 0).any() or (n_neg <= 0).any():
        raise ValueError("Every bootstrap resample must include both classes")
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    ap = (positive * precision).sum(axis=1) / n_pos
    auc = (negative * (tp - 0.5 * positive)).sum(axis=1) / (n_pos * n_neg)
    return ap, auc


def bootstrap_counts(labels, bootstrap=10000, seed=42):
    labels = np.asarray(labels, dtype=int)
    if set(np.unique(labels)) != {0, 1} or bootstrap < 1:
        raise ValueError("Stratified bootstrap requires two classes and positive repetitions")
    rng = np.random.default_rng(seed)
    counts = np.zeros((bootstrap, len(labels)), dtype=np.int32)
    row = np.arange(bootstrap)[:, None]
    for label in [0, 1]:
        indices = np.flatnonzero(labels == label)
        draws = rng.choice(indices, size=(bootstrap, len(indices)), replace=True)
        np.add.at(counts, (np.broadcast_to(row, draws.shape), draws), 1)
    return counts


def mean_seed_bootstrap(labels, predictions, counts):
    predictions = np.asarray(predictions, dtype=float)
    output = {name: np.zeros(len(counts)) for name in ["ap", "roc_auc"]}
    for scores in predictions:
        for start in range(0, len(counts), 1024):
            ap, auc = weighted_metrics(labels, scores, counts[start : start + 1024])
            output["ap"][start : start + 1024] += ap / len(predictions)
            output["roc_auc"][start : start + 1024] += auc / len(predictions)
    return output


def choose_configuration(candidates):
    return min(
        candidates,
        key=lambda r: (
            -r["dev_ap_mean"],
            -r["dev_roc_auc_mean"],
            (r["instance_head"] == "rf") + (r["session_head"] == "rf"),
            r["instance_head"] == "rf",
            r["session_head"] == "rf",
        ),
    )


def effect_label(interval, *, rankings_stable):
    low, high = interval
    if low > MARGIN:
        return "meaningful_positive"
    if high < -MARGIN:
        return "meaningful_negative"
    if low >= -MARGIN and high <= MARGIN and rankings_stable:
        return "small_exploratory"
    return "inconclusive"


def _job_name(job):
    return f"{job['method']}__{job['instance_head']}_{job['session_head']}__seed{job['seed']}"


def _runtime():
    return dict(
        python=sys.version.split()[0],
        packages={
            k: importlib.metadata.version(k)
            for k in ["numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl"]
        },
    )


def _prepare_risks(payload, train, seed, head):
    sessions, local = payload["sessions"], payload["local"]
    config = copy.deepcopy(payload["config"])
    config["seed"] = int(seed)
    is_cst = payload["method"] == "cst_mil_r0"
    if is_cst:
        # This transformer is stateless. Fitting a vocabulary globally is forbidden.
        if config.get("word_features", False):
            raise ValueError("Supervised folds require the frozen stateless hashing features")
        texts = [_risk_texts(ms) for ms in local]
        features = RiskTextFeatures(config)
        features.fit_transform([text for i in train for text in texts[i]])
        if features.word_vectorizer is not None:
            raise ValueError("Unexpected fitted word vocabulary")
        matrices = [features.transform(text) for text in texts]
        x = vstack([matrices[i] for i in train], format="csr")
    else:
        matrices = payload["window_topics"]
        x = np.concatenate([matrices[i] for i in train])
    labels = np.asarray([sessions[i]["label"] for i in train], dtype=int)
    bags = np.concatenate(
        [np.full(matrices[i].shape[0], j, dtype=int) for j, i in enumerate(train)]
    )
    oof, classifier, folds, weights = crossfit(
        x, bags, labels, head=head, config=config, mil=is_cst
    )
    train_position = {i: j for j, i in enumerate(train)}
    risks = []
    for i in range(len(sessions)):
        scores = (
            oof[bags == train_position[i]]
            if i in train_position
            else classifier.predict_proba(matrices[i])[:, 1]
        )
        risks.append(
            scores if is_cst else _window_average(local[i], payload["windows"][i], scores)[:, 0]
        )
    vectors = np.asarray(
        [
            _bag_features(m, t, r, structure=is_cst)
            for m, t, r in zip(local, payload["message_topics"], risks, strict=True)
        ]
    )
    fold_for_bag = {bag: fold["fold"] for fold in folds for bag in fold["held_bag_indices"]}
    within_bag = np.concatenate([np.arange(matrices[i].shape[0]) for i in train])
    provenance = [
        dict(
            post_id=sessions[train[int(bag)]]["post_id"],
            split="train",
            label=int(labels[bag]),
            bag_index=int(bag),
            instance_index=int(within_bag[index]),
            global_instance_index=index,
            instance_unit="message" if is_cst else "window",
            oof_fold=fold_for_bag[int(bag)],
            risk_probability=float(oof[index]),
        )
        for index, bag in enumerate(bags)
    ]
    for fold in folds:
        fold["train_post_ids"] = [sessions[train[i]]["post_id"] for i in fold["train_bag_indices"]]
        fold["held_post_ids"] = [sessions[train[i]]["post_id"] for i in fold["held_bag_indices"]]
    return vectors, provenance, dict(folds=folds, full_fit_weight_rounds=weights), config


def _predictions(sessions, indices, probabilities, threshold):
    return [
        dict(
            post_id=sessions[i]["post_id"],
            split=sessions[i]["split"],
            label=int(sessions[i]["label"]),
            risk_probability=float(probabilities[i]),
            prediction=int(probabilities[i] >= threshold),
            threshold=float(threshold),
        )
        for i in indices
    ]


def _check_resume(destination, binding):
    marker = destination / "binding.json"
    if marker.exists() and read(marker) != binding:
        raise ValueError(
            f"Changed participating implementation/input; use revision directory: {destination}"
        )
    result_path = destination / "result.json"
    if not result_path.exists():
        return None
    prior = read(result_path)
    if prior["binding"] != binding:
        raise ValueError(f"Changed completed run binding: {destination}")
    verify_sources(prior["artifact_sha256"])
    return prior


def run(out=OUT, seeds=None):
    """Execute serially; orchestration must release its timing/CPU scheduling gate."""
    out = Path(out)
    jobs = schedule(seeds)
    configure_cpu(4)
    limits = threadpool_limits(limits=4)
    manifest, sessions, _ = load_suite("full677")
    data_folder = PROJECT / "benchmark/data/full677"
    common = code_sources(
        Path(__file__),
        PROJECT / "src/cst_mil/data.py",
        PROJECT / "pyproject.toml",
        PROJECT / "uv.lock",
    )
    common.update(source_hashes(list(data_folder.glob("*"))))
    train = [i for i, s in enumerate(sessions) if s["split"] == "train"]
    dev = [i for i, s in enumerate(sessions) if s["split"] == "dev"]
    test = [i for i, s in enumerate(sessions) if s["split"] == "test"]
    labels = np.asarray([sessions[i]["label"] for i in train], dtype=int)
    completed = []
    runtime = _runtime()
    try:
        for method in METHODS:
            cache = OLD / "cache" / f"{method}.joblib"
            model = RUNS / method / "model.joblib"
            original_path = RUNS / method / "predictions.jsonl"
            config_path = RUNS / method / "config.json"
            # Missing caches fail here; no legacy cache writer is ever called.
            sources = {**common, **source_hashes([cache, model, original_path, config_path])}
            cache_binding = dict(
                schema=1,
                corpus=manifest["content_fingerprint"],
                model=sources[str(model.resolve())],
            )
            payload = load_frozen_cache(cache, cache_binding, sessions)
            if payload["config"] != read(config_path):
                raise ValueError(f"Frozen cache/model configuration mismatch: {method}")
            payload["method"] = method
            original = None
            risk_key, prepared = None, None
            for job in [j for j in jobs if j["method"] == method]:
                destination = out / "robustness/runs" / _job_name(job)
                binding = dict(
                    schema=1,
                    job=job,
                    corpus=manifest["content_fingerprint"],
                    cache_binding=cache_binding,
                    source_sha256=sources,
                    runtime=runtime,
                    instance_weight="explicit_bag_weight_only",
                    session_weight="balanced_without_sample_weight",
                )
                verify_sources(sources)
                prior = _check_resume(destination, binding)
                if prior is not None:
                    completed.append(prior)
                    continue
                save(destination / "binding.json", binding)
                print(f"robustness fit {_job_name(job)}", flush=True)
                started = time.perf_counter()
                with ResourceMonitor() as monitor:
                    key = job["seed"], job["instance_head"]
                    if key != risk_key:
                        prepared = _prepare_risks(payload, train, *key)
                        risk_key = key
                    vectors, provenance, folds, config = prepared
                    classifier = session_estimator(job["session_head"], config)
                    classifier.fit(vectors[train], labels)
                    probabilities = classifier.predict_proba(vectors)[:, 1]
                    threshold, dev_f1 = select_threshold(
                        [sessions[i]["label"] for i in dev], probabilities[dev]
                    )
                    dev_predictions = _predictions(sessions, dev, probabilities, threshold)
                    test_predictions = _predictions(sessions, test, probabilities, threshold)
                    parity = None
                    if job["seed"] == 42 and job["instance_head"] == job["session_head"] == "lr":
                        if original is None:
                            original = {r["post_id"]: r for r in jsonl_read(original_path)}
                        parity = max(
                            abs(r["risk_probability"] - original[r["post_id"]]["risk_probability"])
                            for r in dev_predictions + test_predictions
                        )
                        if parity > 1e-7:
                            raise ValueError(
                                f"Original seed42 LR/LR parity failed: {method}, {parity}"
                            )
                # A result is not committed if any participating code/input changed during fitting.
                verify_sources(sources)
                jsonl_write(destination / "train_oof.jsonl", provenance)
                save(destination / "folds.json", folds)
                jsonl_write(destination / "dev_predictions.jsonl", dev_predictions)
                jsonl_write(destination / "test_predictions.jsonl", test_predictions)
                artifacts = source_hashes(
                    [
                        destination / name
                        for name in [
                            "train_oof.jsonl",
                            "folds.json",
                            "dev_predictions.jsonl",
                            "test_predictions.jsonl",
                        ]
                    ]
                )
                result = dict(
                    **job,
                    binding=binding,
                    artifact_sha256=artifacts,
                    threshold=float(threshold),
                    threshold_selection="dev_macro_f1_only",
                    dev_macro_f1=float(dev_f1),
                    dev_metrics=binary_metrics(dev_predictions),
                    test_metrics=binary_metrics(test_predictions),
                    training_sessions=len(train),
                    dev_sessions=len(dev),
                    test_sessions=len(test),
                    feature_dimensions=int(vectors.shape[1]),
                    seed42_original_lr_lr_max_absolute_difference=parity,
                    crossfit_unit="complete_session",
                    crossfit_membership_sha256=stable(
                        json.dumps(
                            [[f["train_post_ids"], f["held_post_ids"]] for f in folds["folds"]],
                            ensure_ascii=False,
                        )
                    ),
                    instance_weight_policy="explicit_bag_weight_only; RF class_weight=None",
                    session_weight_policy="class_weight=balanced; no explicit sample_weight",
                    session_balanced_weight_summary=weight_summary(
                        np.arange(len(labels)),
                        labels,
                        len(labels) / (2 * np.bincount(labels)[labels]),
                        np.arange(len(labels)),
                    ),
                    stateless_risk_features="HashingVectorizer; word_features=False",
                    seconds=time.perf_counter() - started,
                    peak_rss_mib=monitor.peak / 2**20,
                    cpu_affinity_width=monitor.peak_affinity_width,
                    cost_note=(
                        "Includes reusable risk fits only on their first session-head run; "
                        "not end-to-end timing"
                    ),
                )
                verify_sources(sources)
                save(destination / "result.json", result)
                completed.append(result)
                save(
                    out / "robustness/progress.json",
                    dict(
                        completed=len(list((out / "robustness/runs").glob("*/result.json"))),
                        full_expected=110,
                        latest=_job_name(job),
                    ),
                )
                print(
                    f"robustness finished {_job_name(job)} AP={result['test_metrics']['ap']:.4f}",
                    flush=True,
                )
    finally:
        limits.restore_original_limits()
    return dict(completed=len(completed), requested=len(jobs), full_expected=110)


def _rankings_stable(rows, factor, metric):
    methods = METHODS if factor == "session" else [m for m in METHODS if m != "cst_mil_r0"]
    treatment = "lr_rf" if factor == "session" else "rf_lr"
    by_key = {
        (r["method"], r["instance_head"] + "_" + r["session_head"], r["seed"]): r for r in rows
    }
    rankings = []
    for seeds in [list(SEEDS)] + [[seed] for seed in SEEDS]:
        pair = []
        for variant in ["lr_lr", treatment]:
            pair.append(
                sorted(
                    methods,
                    key=lambda method: (
                        -float(
                            np.mean(
                                [
                                    by_key[(method, variant, seed)]["test_metrics"][metric]
                                    for seed in seeds
                                ]
                            )
                        ),
                        method,
                    ),
                )
            )
        rankings.append(
            dict(seeds=seeds, control=pair[0], treatment=pair[1], stable=pair[0] == pair[1])
        )
    return all(r["stable"] for r in rankings), rankings


def summarize(out=OUT, bootstrap=10000):
    """Require the full design; choose heads only by mean development metrics."""
    out = Path(out)
    rows, sources = [], {}
    for job in schedule():
        destination = out / "robustness/runs" / _job_name(job)
        row = read(destination / "result.json")
        if any(row[k] != v for k, v in job.items()):
            raise ValueError("Run identity differs from the frozen schedule")
        for path, sha in row["binding"]["source_sha256"].items():
            if path in sources and sources[path] != sha:
                raise ValueError("Cannot combine different participating source revisions")
            sources[path] = sha
        verify_sources(row["artifact_sha256"])
        sources.update(row["artifact_sha256"])
        sources.update(source_hashes([destination / "result.json"]))
        rows.append(row)
    verify_sources(sources)
    sources.update(code_sources(Path(__file__)))
    # Equal folds across every topic method and instance/session head for a seed.
    for seed in SEEDS:
        if len({r["crossfit_membership_sha256"] for r in rows if r["seed"] == seed}) != 1:
            raise ValueError("Methods do not share identical complete-session folds")
    variants, prediction_sets = [], {}
    reference = None
    for method in METHODS:
        for ih in ["lr"] if method == "cst_mil_r0" else ["lr", "rf"]:
            for sh in ["lr", "rf"]:
                chosen = [
                    r
                    for r in rows
                    if (r["method"], r["instance_head"], r["session_head"]) == (method, ih, sh)
                ]
                chosen.sort(key=lambda r: r["seed"])
                item = dict(method=method, instance_head=ih, session_head=sh, seeds=list(SEEDS))
                for split in ["dev", "test"]:
                    for metric in [
                        "ap",
                        "roc_auc",
                        "macro_f1",
                        "accuracy",
                        "recall_cb",
                        "precision_cb",
                        "f1_cb",
                    ]:
                        values = [r[split + "_metrics"][metric] for r in chosen]
                        item[f"{split}_{metric}_mean"] = float(np.mean(values))
                        item[f"{split}_{metric}_sd"] = float(np.std(values, ddof=1))
                        item[f"{split}_{metric}_seeds"] = values
                variants.append(item)
                scores = []
                for row in chosen:
                    predictions = jsonl_read(
                        out / "robustness/runs" / _job_name(row) / "test_predictions.jsonl"
                    )
                    ids_labels = [(p["post_id"], p["label"]) for p in predictions]
                    if reference is None:
                        reference = ids_labels
                    if ids_labels != reference or any(p["split"] != "test" for p in predictions):
                        raise ValueError("Paired test sessions/order/labels differ")
                    scores.append([p["risk_probability"] for p in predictions])
                prediction_sets[(method, ih + "_" + sh)] = np.asarray(scores)
    selected = [
        choose_configuration([r for r in variants if r["method"] == method]) for method in METHODS
    ]
    labels = np.array([label for _, label in reference], dtype=int)
    counts = bootstrap_counts(labels, bootstrap, seed=42)
    distributions = {
        key: mean_seed_bootstrap(labels, scores, counts) for key, scores in prediction_sets.items()
    }
    results = []
    for contrast in contrasts():
        method, control, treatment = (contrast[k] for k in ["method", "control", "treatment"])
        for metric in ["ap", "roc_auc"]:
            delta = (
                distributions[(method, treatment)][metric]
                - distributions[(method, control)][metric]
            )
            ordinary = np.quantile(delta, [0.025, 0.975]).tolist()
            adjusted = np.quantile(delta, [0.025 / FAMILY_SIZE, 1 - 0.025 / FAMILY_SIZE]).tolist()
            stable_rank, rankings = _rankings_stable(rows, contrast["factor"], metric)
            estimates = {}
            for variant in [control, treatment]:
                estimates[variant] = float(
                    np.mean(
                        [
                            r["test_metrics"][metric]
                            for r in rows
                            if r["method"] == method
                            and r["instance_head"] + "_" + r["session_head"] == variant
                        ]
                    )
                )
            results.append(
                dict(
                    **contrast,
                    metric=metric,
                    estimate=estimates[treatment] - estimates[control],
                    percentile_ci95=ordinary,
                    bonferroni_family_ci95=adjusted,
                    family_size=FAMILY_SIZE,
                    exploratory_equivalence_margin=MARGIN,
                    rankings_stable=stable_rank,
                    rankings=rankings,
                    interpretation=effect_label(adjusted, rankings_stable=stable_rank),
                    bootstrap_repetitions=int(bootstrap),
                )
            )
    summary = dict(
        completed=110,
        seeds=list(SEEDS),
        configurations=variants,
        selected_configurations=selected,
        selection_rule="mean DEV AP over all five seeds, then mean DEV AUC, then fewer RF heads",
        selected_seed=None,
        reported_variability="sample SD across all five seeds (ddof=1)",
        bootstrap=dict(
            repetitions=int(bootstrap),
            seed=42,
            unit="complete_session",
            stratified_by="binary_label",
            paired_across=(
                "all methods, head configurations and seeds; identical multiplicity matrix"
            ),
            estimand="difference of mean per-seed metrics; not metric of averaged probabilities",
            session_ids=[post_id for post_id, _ in reference],
            multiplicity_sha256=hashlib.sha256(counts.astype("<i4").tobytes()).hexdigest(),
            ordinary_coverage=0.95,
            family_coverage=0.95,
            family_size=FAMILY_SIZE,
            family_interval_alpha=0.05 / FAMILY_SIZE,
        ),
        contrasts=results,
        source_sha256=sources,
        component_controls_note=(
            "CST component removals are historical and were not rerun. The historical no_reply "
            "control jointly removed parent-reply text context and all six structural statistics, "
            "including adjacency and reply statistics; it does not isolate reply structure."
        ),
        interpretation_note=(
            "The +/-0.01 margin is exploratory. A small-effect interpretation requires the entire "
            "Bonferroni interval within the margin and unchanged method rankings for the mean "
            "and every seed; otherwise report meaningful effect or inconclusive."
        ),
    )
    verify_sources(sources)
    save(out / "robustness/contrasts.json", results)
    save(out / "robustness/summary.json", summary)
    return summary
