"""Fixed-topic, grouped cross-fit head and component controls.

Caches are bound to the immutable corpus and fitted topic model. Original models,
predictions and source modules are never rewritten.
"""
from __future__ import annotations

import copy
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cst_mil.bench.common import build_windows, canonical_messages
from cst_mil.bench.evaluation import binary_metrics
from cst_mil.bench.model import (
    ConstantRisk, RiskTextFeatures, _bag_features, _bag_weights, _fit_risk,
    _message_topics, _risk_texts, _window_average,
)
from cst_mil.bench.runner import ResourceMonitor
from cst_mil.metrics import select_threshold
from cst_mil.preprocess import normalize_text

from .io import METHODS, OUT, RUNS, digest, inputs, read, rows, save, table


def forest(seed=42):
    return RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                  class_weight='balanced', random_state=seed, n_jobs=4)


def fit_instance(x, labels, bags, head: str, config: dict, mil=False):
    if head == 'lr':
        return _fit_risk(x, labels, bags, mil=mil, config=config)[0]
    if head != 'rf' or mil:
        raise ValueError('RF is a non-MIL baseline instance head')
    if x.shape[1] == 0 or len(np.unique(labels)) < 2:
        return ConstantRisk(float(np.mean(labels)))
    clf = forest(config.get('seed', 42))
    clf.fit(x, labels, sample_weight=_bag_weights(bags, labels))
    return clf


def crossfit(x, bags, labels_by_bag, *, head, config, mil=False):
    bags, labels_by_bag = np.asarray(bags), np.asarray(labels_by_bag)
    if set(bags.tolist()) != set(range(len(labels_by_bag))):
        raise ValueError('Every training bag must be represented exactly in the bag index domain')
    if x.shape[0] != len(bags):
        raise ValueError('Instance and bag dimensions differ')
    counts = np.bincount(labels_by_bag, minlength=2)
    if min(counts) < 2:
        raise ValueError('Need at least two complete sessions per class')
    folds = StratifiedKFold(n_splits=min(3, int(min(counts))), shuffle=True,
                           random_state=config.get('seed', 42))
    scores = np.full(len(bags), np.nan)
    history = []
    labels = labels_by_bag[bags]
    for train_bags, held_bags in folds.split(np.arange(len(labels_by_bag)), labels_by_bag):
        train = np.flatnonzero(np.isin(bags, train_bags))
        held = np.flatnonzero(np.isin(bags, held_bags))
        clf = fit_instance(x[train], labels[train], bags[train], head, config, mil)
        scores[held] = clf.predict_proba(x[held])[:, 1]
        history.append(dict(train_bags=train_bags.tolist(), held_bags=held_bags.tolist(),
                            train_instances=len(train), held_instances=len(held)))
    if not np.isfinite(scores).all():
        raise ValueError('Incomplete out-of-fold scores')
    return scores, fit_instance(x, labels, bags, head, config, mil), history


def make_features(messages, topics, risks, *, structure=False, use_topics=True, direct=False):
    if direct:
        return np.concatenate([topics.mean(axis=0), topics.max(axis=0)])
    full = _bag_features(messages, topics, risks, structure=structure)
    if use_topics:
        return full
    # Ten risk/length summaries and, if enabled, ONLY the two non-topic reply summaries.
    return np.concatenate([full[:10], full[-2:]]) if structure else full[:10]


def frozen_features(method: str):
    manifest, sessions, _, grouped = inputs()
    cache = OUT / 'cache' / f'{method}.joblib'
    binding = dict(schema=1, corpus=manifest['content_fingerprint'],
                   model=digest(RUNS / method / 'model.joblib'))
    if cache.exists():
        payload = joblib.load(cache)
        if payload['binding'] != binding:
            raise ValueError('Frozen feature cache binding mismatch')
        return payload
    model = joblib.load(RUNS / method / 'model.joblib')
    # Save/load of old BERTopic only; use its existing fully populated read-only cache.
    local = [canonical_messages(s, grouped[s['post_id']]) for s in sessions]
    windows = [build_windows(s, m, size=5, stride=2)
               for s, m in zip(sessions, local, strict=True)]
    old = {r['post_id']: r for r in rows(RUNS / method / 'predictions.jsonl')}
    started = time.perf_counter()
    parts = []
    # Native inference is documentwise; verify this optimization against every saved message.
    if method != 'bertopic':
        combined = model.topic_model.transform([w['text'] for ws in windows for w in ws])
        cuts = np.cumsum([len(ws) for ws in windows])[:-1]
        parts = list(np.split(combined, cuts))
    else:
        for i, ws in enumerate(windows):
            parts.append(model.topic_model.transform([w['text'] for w in ws]))
            if i % 50 == 0:
                print(f'topic cache {method}: {i}/{len(windows)}', flush=True)
    message_topics = []
    for i, (s, m, ws, part) in enumerate(zip(sessions, local, windows, parts, strict=True)):
        mt = _message_topics(model.topic_model, m, ws, part)
        prior = old[s['post_id']]['message_scores']
        if [x['comment_id'] for x in prior] != [x['comment_id'] for x in m]:
            raise ValueError('Message ordering mismatch')
        expected = np.asarray([x['topic_strengths'] for x in prior])
        if not np.allclose(mt, expected, atol=1e-8, rtol=1e-7):
            # Preserve exact original session batching if a backend proves context-sensitive.
            parts[i] = model.topic_model.transform([w['text'] for w in ws])
            mt = _message_topics(model.topic_model, m, ws, parts[i])
            if not np.allclose(mt, expected, atol=1e-8, rtol=1e-7):
                raise ValueError(f'Frozen topic parity failed: {method} {s["post_id"]}')
        message_topics.append(mt)
    payload = dict(binding=binding, sessions=sessions, local=local, windows=windows,
                   window_topics=parts, message_topics=message_topics, config=model.config,
                   cache_build_seconds=time.perf_counter() - started,
                   saved_message_topic_parity=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temp = cache.with_suffix('.tmp')
    joblib.dump(payload, temp)
    temp.replace(cache)
    return payload


def variants(method):
    if method == 'cst_mil_r0':
        return [('lr', 'lr', 'full'), ('lr', 'rf', 'full'),
                ('lr', 'lr', 'no_mil'), ('lr', 'lr', 'no_reply'), ('lr', 'lr', 'no_topics')]
    return [(ih, sh, 'full') for ih in ['lr', 'rf'] for sh in ['lr', 'rf']] + [
        ('none', 'lr', 'direct_topics')]


def run_method(method):
    payload = frozen_features(method)
    sessions, local = payload['sessions'], payload['local']
    windows, parts = payload['windows'], payload['window_topics']
    topics, config = payload['message_topics'], payload['config']
    train = [i for i, s in enumerate(sessions) if s['split'] == 'train']
    dev = [i for i, s in enumerate(sessions) if s['split'] == 'dev']
    test = [i for i, s in enumerate(sessions) if s['split'] == 'test']
    train_labels = np.array([sessions[i]['label'] for i in train])
    old = {r['post_id']: r for r in rows(RUNS / method / 'predictions.jsonl')}
    risk_cache = {}
    completed = []
    for ih, sh, mode in variants(method):
        name = f'{method}__{ih}_{sh}__{mode}'
        dest = OUT / 'ablations' / name
        binding = dict(**payload['binding'], implementation=digest(__file_path()),
                       variant=name)
        if (dest / 'result.json').exists():
            prior = read(dest / 'result.json')
            if prior['binding'] != binding:
                raise ValueError(f'Changed implementation for completed {name}; use revision directory')
            completed.append(prior)
            continue
        started = time.perf_counter()
        print(f'fit {name}', flush=True)
        with ResourceMonitor() as monitor:
            key = (ih, mode if mode in ['no_mil', 'no_reply'] else 'full')
            history = []
            if mode == 'direct_topics':
                risks = [np.zeros(len(m)) for m in local]
            elif key in risk_cache:
                risks, history = risk_cache[key]
            else:
                cfg = copy.deepcopy(config)
                is_cst = method == 'cst_mil_r0'
                if is_cst:
                    texts = [[normalize_text(m['raw_text']) for m in ms] if mode == 'no_reply'
                             else _risk_texts(ms) for ms in local]
                    rf = RiskTextFeatures(cfg)
                    rf.fit_transform([text for i in train for text in texts[i]])
                    matrices = [rf.transform(text) for text in texts]
                    from scipy.sparse import vstack
                    x = vstack([matrices[i] for i in train], format='csr')
                    if mode == 'no_mil':
                        cfg['mil_rounds'] = 0
                else:
                    matrices = parts
                    x = np.concatenate([parts[i] for i in train])
                bags = np.concatenate([np.full(matrices[i].shape[0], j)
                                       for j, i in enumerate(train)])
                oof, classifier, history = crossfit(x, bags, train_labels, head=ih,
                                                    config=cfg, mil=is_cst)
                train_position = {i: j for j, i in enumerate(train)}
                risks = []
                for i in range(len(sessions)):
                    instance_scores = (oof[bags == train_position[i]] if i in train_position
                                       else classifier.predict_proba(matrices[i])[:, 1])
                    risks.append(instance_scores if is_cst else
                                 _window_average(local[i], windows[i], instance_scores)[:, 0])
                risk_cache[key] = (risks, history)
            vectors = np.asarray([make_features(
                m, t, r, structure=method == 'cst_mil_r0' and mode != 'no_reply',
                use_topics=mode != 'no_topics', direct=mode == 'direct_topics')
                for m, t, r in zip(local, topics, risks, strict=True)])
            if sh == 'lr':
                clf = make_pipeline(StandardScaler(), LogisticRegression(
                    C=config.get('session_C', config.get('session_c', 1.0)),
                    solver='liblinear', max_iter=1000, class_weight='balanced', random_state=42))
            else:
                clf = forest()
            if vectors.shape[1] == 0:
                clf = ConstantRisk(.5)
            else:
                clf.fit(vectors[train], train_labels)
            probabilities = clf.predict_proba(vectors)[:, 1]
            threshold, dev_f1 = select_threshold([sessions[i]['label'] for i in dev], probabilities[dev])
            predictions = [dict(post_id=sessions[i]['post_id'], split='test',
                                label=sessions[i]['label'], risk_probability=float(probabilities[i]),
                                prediction=int(probabilities[i] >= threshold)) for i in test]
            if ih == sh == 'lr' and mode == 'full':
                diffs = [abs(r['risk_probability'] - old[r['post_id']]['risk_probability'])
                         for r in predictions]
                if max(diffs) > 1e-7:
                    raise ValueError(f'Original full-model risk parity failed: {name} max={max(diffs)}')
            metrics = binary_metrics(predictions)
        result = dict(binding=binding, method=method, instance_head=ih, session_head=sh,
                      mode=mode, feature_dimensions=vectors.shape[1], threshold=threshold,
                      dev_macro_f1=dev_f1, **metrics, seconds=time.perf_counter() - started,
                      peak_rss_mib=monitor.peak / 2**20, cpu_affinity_width=monitor.peak_affinity_width,
                      fit_cost_note='cache reuse makes timings diagnostic, not end-to-end costs',
                      crossfit_unit='complete_session', training_sessions=406, test_sessions=136)
        save(dest / 'predictions.json', predictions)
        save(dest / 'crossfit.json', history)
        save(dest / 'result.json', result)
        completed.append(result)
        print(f'finished {name}: AP={metrics["ap"]:.4f} F1={metrics["macro_f1"]:.4f}', flush=True)
        consolidate()
    return completed


def __file_path():
    from pathlib import Path
    return Path(__file__)


def consolidate():
    results = [read(p) for p in sorted((OUT / 'ablations').glob('*/result.json'))]
    table(OUT / 'ablations.csv', [{k: v for k, v in r.items() if k != 'binding'} for r in results])
    save(OUT / 'ablations.json', results)
    return results


def run():
    for method in METHODS:
        run_method(method)
    return {'completed': len(consolidate())}
