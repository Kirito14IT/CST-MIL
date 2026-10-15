"""Protocol tests use tiny synthetic sessions, never the formal experiment corpus."""

import importlib
import importlib.util

from cst_mil.bench.resources import configure_cpu

configure_cpu(4)

import numpy as np  # noqa: E402  CPU budget is enforced before numerical imports.
import pytest  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402


def implementation():
    name = "cst_mil.research_followup.round2.robustness"
    assert importlib.util.find_spec(name) is not None, "Round-two robustness is not implemented"
    return importlib.import_module(name)


def test_complete_schedule_and_prespecified_contrasts():
    r = implementation()
    jobs = r.schedule()
    assert len(jobs) == 110
    assert (
        len(set((x["method"], x["seed"], x["instance_head"], x["session_head"]) for x in jobs))
        == 110
    )
    assert len([x for x in jobs if x["method"] == "cst_mil_r0"]) == 10
    assert all(x["instance_head"] == "lr" for x in jobs if x["method"] == "cst_mil_r0")
    assert len(r.contrasts()) == 11


def test_equal_bag_weights_and_rf_has_no_second_class_balance():
    r = implementation()
    bags = np.array([0, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    x = np.arange(20).reshape(10, 2)
    lr, lr_history = r.fit_instance(x, labels, bags, "lr", {"seed": 44})
    rf, rf_history = r.fit_instance(x, labels, bags, "rf", {"seed": 44})
    assert lr_history == rf_history
    assert lr_history[0]["class_weight_sum"] == pytest.approx({"0": 5, "1": 5})
    assert lr_history[0]["bag_weight_sum"] == pytest.approx({"0": 2.5, "1": 2.5, "2": 5})
    assert rf.class_weight is None
    assert rf.random_state == lr.random_state == 44
    assert rf.n_estimators == 200 and rf.max_depth == 10 and rf.min_samples_leaf == 5
    assert r.session_estimator("rf", {"seed": 46}).class_weight == "balanced"
    assert r.session_estimator("lr", {"seed": 46})[-1].random_state == 46


def test_crossfit_holds_out_complete_sessions_and_matches_old_lr():
    from cst_mil.bench.model import _crossfit_risk

    r = implementation()
    labels = np.array([0, 1] * 6)
    bags = np.repeat(np.arange(12), np.arange(12) % 3 + 1)
    x = np.random.default_rng(8).random((len(bags), 4))
    cfg = {"seed": 42, "instance_c": 2}
    scores, _, history, _ = r.crossfit(x, bags, labels, head="lr", config=cfg)
    old_scores = _crossfit_risk(None, x, bags, labels, cfg)[0]
    np.testing.assert_allclose(scores, old_scores, atol=1e-14)
    seen = []
    for fold in history:
        assert not set(fold["train_bag_indices"]) & set(fold["held_bag_indices"])
        seen.extend(fold["held_bag_indices"])
    assert sorted(seen) == list(range(12))
    changed = r.crossfit(x, bags, labels, head="lr", config={"seed": 43})[2]
    assert [h["held_bag_indices"] for h in history] != [h["held_bag_indices"] for h in changed]


@pytest.mark.parametrize("ties", [False, True])
def test_weighted_bootstrap_ap_auc_matches_sklearn_with_ties(ties):
    r = implementation()
    rng = np.random.default_rng(16)
    y = np.array([0, 1] * 17)
    p = rng.random(len(y))
    if ties:
        p = np.round(p, 1)
    weights = rng.integers(0, 5, size=(13, len(y)))
    ap, auc = r.weighted_metrics(y, p, weights)
    for i, w in enumerate(weights):
        assert ap[i] == pytest.approx(average_precision_score(y, p, sample_weight=w), abs=1e-14)
        assert auc[i] == pytest.approx(roc_auc_score(y, p, sample_weight=w), abs=1e-14)


def test_stratified_bootstrap_and_identical_pairs_have_zero_difference():
    r = implementation()
    y = np.array([0] * 7 + [1] * 5)
    counts = r.bootstrap_counts(y, 80, seed=42)
    assert np.all(counts[:, :7].sum(axis=1) == 7)
    assert np.all(counts[:, 7:].sum(axis=1) == 5)
    p = np.linspace(0.1, 0.9, len(y))
    values = r.mean_seed_bootstrap(y, np.tile(p, (5, 1)), counts)
    assert np.all(values["ap"] == 1)
    assert np.all(values["roc_auc"] == 1)


def test_bootstrap_averages_seed_metrics_not_probabilities():
    r = implementation()
    labels = np.array([0, 1, 0, 1])
    scores = np.array([[0.1, 0.3, 0.2, 0.4], [0.1, 0.3, 0.2, 0]])
    values = r.mean_seed_bootstrap(labels, scores, np.ones((1, 4), dtype=int))
    assert values["ap"][0] == pytest.approx(0.875)
    assert values["roc_auc"][0] == pytest.approx(0.75)
    assert values["roc_auc"][0] != roc_auc_score(labels, scores.mean(axis=0))


def test_selection_uses_all_seed_dev_metrics_then_auc_then_lr():
    r = implementation()
    candidates = [
        {"instance_head": "rf", "session_head": "rf", "dev_ap_mean": 0.8, "dev_roc_auc_mean": 0.9},
        {"instance_head": "lr", "session_head": "lr", "dev_ap_mean": 0.8, "dev_roc_auc_mean": 0.9},
        {"instance_head": "lr", "session_head": "rf", "dev_ap_mean": 0.81, "dev_roc_auc_mean": 0.1},
    ]
    assert r.choose_configuration(candidates)["session_head"] == "rf"
    assert r.choose_configuration(candidates[:2])["instance_head"] == "lr"
    assert r.effect_label([-0.009, 0.009], rankings_stable=True) == "small_exploratory"
    assert r.effect_label([-0.009, 0.009], rankings_stable=False) == "inconclusive"
    assert r.effect_label([-0.02, 0.02], rankings_stable=True) == "inconclusive"
    assert r.effect_label([0.011, 0.04], rankings_stable=True) == "meaningful_positive"


def test_missing_cache_and_changed_binding_are_rejected(tmp_path):
    import joblib

    r = implementation()
    cache = tmp_path / "cache.joblib"
    with pytest.raises(FileNotFoundError):
        r.load_frozen_cache(cache, {"schema": 1, "corpus": "a", "model": "b"}, [])
    joblib.dump({"binding": {"schema": 1, "corpus": "x", "model": "b"}}, cache)
    with pytest.raises(ValueError, match="binding"):
        r.load_frozen_cache(cache, {"schema": 1, "corpus": "a", "model": "b"}, [])


def test_cst_mil_matches_old_lr_across_all_mil_rounds():
    from cst_mil.bench.model import _fit_risk

    r = implementation()
    y = np.array([0, 1] * 6)
    bags = np.repeat(np.arange(12), 4)
    x = np.random.default_rng(66).random((len(bags), 5))
    cfg = {"seed": 42, "mil_rounds": 3, "positive_top_fraction": 0.3}
    clf, history = r.fit_instance(x, y[bags], bags, "lr", cfg, mil=True)
    expected = _fit_risk(x, y[bags], bags, mil=True, config=cfg)[0]
    np.testing.assert_allclose(clf.predict_proba(x), expected.predict_proba(x), atol=1e-14)
    assert len(history) == 4
    assert history[0]["n_instances"] == 48
    assert all(h["n_instances"] == 36 for h in history[1:])


def tiny_corpus(tmp_path, monkeypatch):
    import joblib

    r = implementation()
    sessions, local, windows, topics = [], [], [], []
    for split, n in [("train", 12), ("dev", 6), ("test", 6)]:
        for j in range(n):
            key = f"{split}{j}"
            sessions.append({"post_id": key, "split": split, "label": j % 2})
            local.append([{"comment_id": key, "raw_text": key}])
            windows.append([{"message_ids": [key]}])
            topics.append(np.array([[0.1 + 0.7 * (j % 2), 0.8 - 0.7 * (j % 2)]]))
    config = {"seed": 42, "word_features": False}
    old, runs = tmp_path / "history", tmp_path / "original_runs"
    (runs / "lda").mkdir(parents=True)
    (runs / "lda/model.joblib").write_bytes(b"fixture-model-fingerprint-only")
    (runs / "lda/predictions.jsonl").write_text("", encoding="utf-8")
    r.save(runs / "lda/config.json", config)
    model_hash = r.source_hashes([runs / "lda/model.joblib"])
    binding = dict(schema=1, corpus="fixture-corpus", model=next(iter(model_hash.values())))
    (old / "cache").mkdir(parents=True)
    joblib.dump(
        dict(
            binding=binding,
            sessions=sessions,
            local=local,
            windows=windows,
            window_topics=topics,
            message_topics=topics,
            config=config,
            saved_message_topic_parity=True,
        ),
        old / "cache/lda.joblib",
    )
    data = tmp_path / "benchmark/data/full677"
    data.mkdir(parents=True)
    r.save(data / "manifest.json", {"content_fingerprint": "fixture-corpus"})
    source = tmp_path / "source.py"
    source.write_text("revision = 1", encoding="utf-8")
    monkeypatch.setattr(r, "PROJECT", tmp_path)
    monkeypatch.setattr(r, "OLD", old)
    monkeypatch.setattr(r, "RUNS", runs)
    monkeypatch.setattr(r, "METHODS", ["lda"])
    monkeypatch.setattr(r, "code_sources", lambda *args: r.source_hashes([source]))
    monkeypatch.setattr(
        r, "load_suite", lambda name: ({"content_fingerprint": "fixture-corpus"}, sessions, [])
    )
    return r, source, sessions


def test_tiny_run_saves_oof_dev_thresholds_and_rejects_changed_resume(tmp_path, monkeypatch):
    r, source, sessions = tiny_corpus(tmp_path, monkeypatch)
    out = tmp_path / "output"
    assert r.run(out, seeds=[43])["completed"] == 4
    result_paths = sorted((out / "robustness/runs").glob("*/result.json"))
    rows = [r.read(path) for path in result_paths]
    assert len({x["crossfit_membership_sha256"] for x in rows}) == 1
    for path, row in zip(result_paths, rows, strict=True):
        dev = r.jsonl_read(path.parent / "dev_predictions.jsonl")
        from cst_mil.metrics import select_threshold

        threshold, _ = select_threshold(
            [x["label"] for x in dev], [x["risk_probability"] for x in dev]
        )
        assert row["threshold"] == threshold
        assert row["session_balanced_weight_summary"]["class_weight_sum"] == {"0": 6, "1": 6}
        assert len(r.jsonl_read(path.parent / "train_oof.jsonl")) == 12
        for fold in r.read(path.parent / "folds.json")["folds"]:
            assert not set(fold["train_post_ids"]) & set(fold["held_post_ids"])
            assert all(key.startswith("train") for key in fold["train_post_ids"])
    before = [path.read_bytes() for path in result_paths]
    assert r.run(out, seeds=[43])["completed"] == 4
    assert before == [path.read_bytes() for path in result_paths]
    source.write_text("revision = 2", encoding="utf-8")
    with pytest.raises(ValueError, match="Changed participating"):
        r.run(out, seeds=[43])


def test_source_edit_during_fit_prevents_result_commit(tmp_path, monkeypatch):
    r, source, _ = tiny_corpus(tmp_path, monkeypatch)
    original = r.session_estimator

    def estimator(head, config):
        clf = original(head, config)
        fit = clf.fit

        def altered_fit(*args, **kwargs):
            result = fit(*args, **kwargs)
            source.write_text("revision = 2", encoding="utf-8")
            return result

        clf.fit = altered_fit
        return clf

    monkeypatch.setattr(r, "session_estimator", estimator)
    out = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Participating source"):
        r.run(out, seeds=[43])
    assert not list((out / "robustness/runs").glob("*/result.json"))


def test_summary_requires_all_seeds_uses_dev_and_pairs_same_sessions(tmp_path, monkeypatch):
    r = implementation()
    source = tmp_path / "source.py"
    source.write_text("fixture", encoding="utf-8")
    sources = r.source_hashes([source])
    monkeypatch.setattr(r, "code_sources", lambda *args: sources)
    for job in r.schedule():
        path = tmp_path / "robustness/runs" / r._job_name(job)
        # RF is stronger on dev and worse on test; test must not select LR.
        is_rf = job["session_head"] == "rf"
        scores = [0.1, 0.8, 0.7, 0.9] if is_rf else [0.1, 0.7, 0.2, 0.9]
        predictions = [
            dict(
                post_id=str(i),
                split="test",
                label=i % 2,
                risk_probability=p,
                prediction=int(p >= 0.5),
            )
            for i, p in enumerate(scores)
        ]
        r.jsonl_write(path / "test_predictions.jsonl", predictions)
        metrics = r.binary_metrics(predictions)
        r.save(
            path / "result.json",
            dict(
                **job,
                binding={"source_sha256": sources},
                artifact_sha256=r.source_hashes([path / "test_predictions.jsonl"]),
                crossfit_membership_sha256=f"folds{job['seed']}",
                dev_metrics={**metrics, "ap": 0.9 if is_rf else 0.8},
                test_metrics=metrics,
            ),
        )
    summary = r.summarize(tmp_path, bootstrap=31)
    assert summary["completed"] == 110
    assert summary["selected_seed"] is None
    assert all(row["session_head"] == "rf" for row in summary["selected_configurations"])
    assert all(row["instance_head"] == "lr" for row in summary["selected_configurations"])
    assert len(summary["contrasts"]) == 22
    assert all(row["family_size"] == 22 for row in summary["contrasts"])
    (tmp_path / "robustness/runs/lda__lr_lr__seed42/result.json").unlink()
    with pytest.raises(FileNotFoundError):
        r.summarize(tmp_path, bootstrap=31)


def test_rankings_stable_rejects_single_seed_reversal_despite_stable_mean():
    r = implementation()
    scores = {"lda": 0.9, "btm": 0.8, "gsdmm": 0.7, "nmf": 0.6,
              "bertopic": 0.5, "cst_mil_r0": 0.4}
    rows = []
    for method, score in scores.items():
        for seed in [42, 43, 44, 45, 46]:
            for head in ["lr", "rf"]:
                value = 0.75 if (method, seed, head) == ("lda", 42, "rf") else score
                rows.append(dict(method=method, seed=seed, instance_head="lr", session_head=head,
                                 test_metrics={"ap": value}))

    stable, rankings = r._rankings_stable(rows, "session", "ap")
    mean = next(row for row in rankings if row["seeds"] == [42, 43, 44, 45, 46])
    seed42 = next(row for row in rankings if row["seeds"] == [42])
    expected = ["lda", "btm", "gsdmm", "nmf", "bertopic", "cst_mil_r0"]
    assert mean["control"] == mean["treatment"] == expected
    assert seed42["control"] == expected
    assert seed42["treatment"] == ["btm", "lda", "gsdmm", "nmf", "bertopic", "cst_mil_r0"]
    assert stable is False
    assert r.effect_label([-0.009, 0.009], rankings_stable=stable) == "inconclusive"
