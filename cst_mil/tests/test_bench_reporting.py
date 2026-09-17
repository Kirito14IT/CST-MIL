"""Comparison generation from isolated synthetic checkpoint artifacts."""

import copy
import csv
import shutil
from pathlib import Path

import pytest

from cst_mil.bench import reporting
from cst_mil.bench.common import file_hash, fingerprint, read_json, write_json, write_jsonl
from cst_mil.bench.evaluation import binary_metrics, paired_differences


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    config = {
        "seed": 42, "n_topics": 32, "threads": 4, "window_size": 5, "stride": 2,
        "membership_threshold": .25, "relative_threshold": .5, "strong_threshold": .6,
        "evidence_top_k": 3, "session_c": 1., "word_features": False,
    }
    sessions, messages = [], []
    for index in range(6):
        session = {"post_id": f"p{index}", "split": ["train", "dev", "test"][index // 2],
                   "label": index % 2, "pilot_member": True, "fresh_test": False}
        sessions.append(session)
        for position in range(2):
            messages.append({"post_id": session["post_id"], "comment_id": f"p{index}m{position}",
                             "raw_text": "这是测试原文，出现旅行和争吵。", "speaker_id": "anon-1"})
    manifest = {"suite": "pilot300", "content_fingerprint": "synthetic-cohort",
                "n_sessions": 6, "n_comments": 12, "split_caveat": "synthetic test only"}
    encoder_dir = benchmark / "models" / "tiny"
    encoder_dir.mkdir(parents=True)
    weights = encoder_dir / "model.safetensors"
    weights.write_bytes(b"synthetic-not-an-actual-model")
    encoder_manifest = encoder_dir / "benchmark_model_manifest.json"
    write_json(encoder_manifest, {"model_fingerprint": "encoder-test-fingerprint",
                                 "files": {weights.name: {"bytes": weights.stat().st_size,
                                                          "sha256": file_hash(weights)}}})
    names = [*reporting.BASELINES, "cst_mil_r0", "cst_mil_r1"]
    for method_index, name in enumerate(names):
        path = benchmark / "runs" / "pilot300" / name
        path.mkdir(parents=True)
        method = "cst_mil" if name.startswith("cst_mil_") else name
        revision = name.rsplit("_", 1)[-1] if method == "cst_mil" else None
        local_config = {**config, "word_features": revision == "r1"}
        binding = {"suite": "pilot300", "cohort": manifest["content_fingerprint"],
                   "method": method, "revision": revision, "config": local_config,
                   "runtime": {"packages": {"test": "1"}, "source_sha256": {"model.py": "x"}}}
        binding_hash = fingerprint(binding)
        write_json(path / "run_manifest.json", {"binding": binding, "binding_sha256": binding_hash})
        model = path / "model.joblib"
        model.write_bytes(name.encode())
        model_hash = file_hash(model)
        backend = {}
        if method == "bertopic":
            backend = {"encoder_manifest": str(encoder_manifest),
                       "encoder_fingerprint": "encoder-test-fingerprint",
                       "encoding_cold_seconds": 1.5, "encoding_cached_seconds": .1,
                       "encoder_load_seconds": .5, "training_noise_fraction": .2}
        write_json(path / "TRAINED.json", {
            "model_sha256": model_hash, "model_bytes": model.stat().st_size,
            "training_seconds": method_index + 1., "training_peak_rss_bytes": 10485760,
            "formal_fit_attempts": 1, "backend_info": backend,
        })
        write_json(path / "COMPLETED.json", {"status": "complete", "completed": 6,
                                              "model_sha256": model_hash})
        predictions = []
        for session in sessions:
            score = .8 if session["label"] else .2
            if name == "cst_mil_r0" and session["split"] == "test":
                score = .2
            evidence = [dict(message, risk_probability=score)
                        for message in messages if message["post_id"] == session["post_id"]]
            row = {**session, "risk_probability": score, "prediction": int(score >= .5),
                   "threshold": .5, "topics": [{"title": "测试原文"}], "evidence": evidence,
                   "message_scores": evidence, "coverage": 1.}
            predictions.append(row)
            write_json(path / "sessions" / f"{session['post_id']}.json", {
                "prediction": row, "prediction_sha256": fingerprint(row),
                "binding": binding_hash, "model": model_hash, "inference_seconds": .1,
                "peak_rss_bytes": 1048576,
                "encoding_counters": {"encoding_cold_seconds": .01, "cached_texts": 2},
            })
        write_jsonl(path / "predictions.jsonl", predictions)
        risk = binary_metrics([row for row in predictions if row["split"] == "test"])
        quality = {"npmi": .5 if name == "nmf" else .1, "topic_diversity": .8,
                   "coverage": 1., "n_topics": 1, "degenerate_topic_fraction": 0.}
        write_json(path / "metrics.json", {
            "test": risk, "topics": quality,
            "evidence": {"comment_ap": .5, "hit_rate_at_3": 1., "mean_recall_at_3": 1.,
                         "n_labeled_messages": 4, "n_sessions_with_positive_comments": 1},
            "test_ci": {"macro_f1": {"low": max(0, risk["macro_f1"] - .1),
                                       "high": min(1, risk["macro_f1"] + .1)}},
        })
        write_json(path / "dev_metrics.json", {
            "risk": binary_metrics([row for row in predictions if row["split"] == "dev"]),
            "topics": quality,
        })
        write_json(path / "training_summary.json", {
            "message_labels_used_for_training": False, "training_session_ids": ["p0", "p1"],
            "risk_classifier_fit_count": 4,
            "timings": {"dev_inference_seconds": 1., "preparation_seconds": .1,
                        "topic_fit_seconds": 1., "risk_fit_seconds": .5},
        })
    write_json(benchmark / "selection.json", {
        "revision": "r0", "config": config, "test_used_for_selection": False,
        "baseline_reference": "nmf", "stop_target_met": False,
        "history": [{"revision": "r0", "accepted": True},
                    {"revision": "r1", "base_revision": "r0", "change": "word_features",
                     "accepted": False}],
    })
    write_json(benchmark / "runs/pilot300/cst_mil_r2/error_test.json",
               {"type": "ExampleError", "message": "synthetic failure"})
    monkeypatch.setattr(reporting, "BENCHMARK", benchmark)
    monkeypatch.setattr(reporting, "load_suite", lambda suite: (manifest, sessions, messages))
    monkeypatch.setitem(reporting.EXPECTED_COUNTS, "pilot300", (6, 12, 2, 2, 2))
    monkeypatch.setitem(reporting.EXPECTED_TEST_SUBSETS, "pilot300", (2, 0))
    monkeypatch.setattr(reporting, "paired_differences",
                        lambda left, right: paired_differences(left, right, count=20))
    return benchmark, manifest, sessions, messages


def test_complete_comparison_writes_real_figures_and_bound_numbers(artifacts):
    benchmark, _, _, _ = artifacts
    result = reporting.create_comparison("pilot300")
    assert result["selected"] == "cst_mil_r0"
    assert result["baseline_reference"] == "nmf"
    assert result["incomplete_revisions"] == 1
    assert result["delta_macro_f1"] < 0
    directory = Path(result["output_dir"])
    assert len(result["figures"]) == 6
    for name in result["figures"]:
        asset = directory / "figures" / name
        assert asset.stat().st_size > 1000
        signature = asset.read_bytes()[:8]
        assert (signature.startswith(b"\x89PNG") if asset.suffix == ".png"
                else signature.startswith(b"%PDF"))
    stats = read_json(directory / "statistics.json")
    topic_counts = stats["selected_topic_count_distribution"]
    assert topic_counts["all_test"]["bins"]["1"] == 2
    assert topic_counts["non_cb_test"]["n_sessions"] == 1
    assert topic_counts["cb_test"]["n_sessions"] == 1
    bertopic = next(row for row in stats["rows"] if row["method"] == "bertopic")
    assert bertopic["external_encoder_bytes"] == len(b"synthetic-not-an-actual-model")
    assert bertopic["known_deployed_disk_bytes"] > bertopic["model_bytes"]
    assert bertopic["evidence_n_sessions_with_positive_comments"] == 1
    assert bertopic["inference_encoding_cold_seconds"] == pytest.approx(.06)
    with (directory / "comparisons.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    text = (directory / "analysis-report.md").read_text(encoding="utf-8")
    assert "cst_mil_r1" in text and "cst_mil_r2" in text
    assert "这是测试原文" in text
    assert "扩大到 677 个会话" in text
    assert "冻结时是否达到预定停止目标：`False`" in text
    assert "当前表格来自修复后统一 CPU 亲和性限制的替代运行" in text
    assert "不是 full677 的再次迭代" not in text
    assert stats["full_test_subsets"] == {}
    assert "未经同时覆盖校正" in (directory / "stats-appendix.md").read_text(encoding="utf-8")
    assert benchmark / "comparison/pilot300" == directory


@pytest.mark.parametrize("kind", ["partial", "model", "prediction", "metric", "training_ids"])
def test_corrupt_or_incomplete_input_is_blocked_before_outputs(artifacts, kind):
    benchmark, _, _, _ = artifacts
    run = benchmark / "runs/pilot300/lda"
    if kind == "partial":
        (run / "COMPLETED.json").unlink()
    elif kind == "model":
        (run / "model.joblib").write_bytes(b"changed")
    elif kind == "prediction":
        path = run / "sessions/p0.json"
        wrapper = read_json(path)
        wrapper["prediction"]["risk_probability"] = .3
        write_json(path, wrapper)
    elif kind == "metric":
        path = run / "metrics.json"
        value = read_json(path)
        value["test"]["macro_f1"] = .01
        write_json(path, value)
    else:
        path = run / "training_summary.json"
        value = read_json(path)
        value["training_session_ids"] = ["p4", "p5"]
        write_json(path, value)
    with pytest.raises(ValueError):
        reporting.create_comparison("pilot300")
    assert not (benchmark / "comparison").exists()


def test_changed_common_config_or_cohort_rejected(artifacts):
    benchmark, manifest, sessions, _ = artifacts
    run = benchmark / "runs/pilot300/nmf"
    modified = copy.deepcopy(manifest)
    modified["content_fingerprint"] = "another-cohort"
    with pytest.raises(ValueError, match="different cohort"):
        reporting._load_run(run, modified, sessions)
    selection = read_json(benchmark / "selection.json")
    selection["test_used_for_selection"] = True
    write_json(benchmark / "selection.json", selection)
    with pytest.raises(ValueError, match="dev-only"):
        reporting.create_comparison("pilot300")


def test_changed_encoder_is_not_silently_excluded_from_footprint(artifacts):
    benchmark, _, _, _ = artifacts
    (benchmark / "models/tiny/model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="External encoder"):
        reporting.create_comparison("pilot300")
    assert not (benchmark / "comparison").exists()


def test_topic_count_distribution_is_descriptive_not_accuracy():
    records = [{"topics": [{}] * number} for number in [0, 1, 2, 3, 4, 7]]
    summary = reporting._topic_count_distribution(records)
    assert summary["bins"] == {"0": 1, "1": 1, "2": 1, "3": 1, ">3": 2}
    assert summary["maximum"] == 7
    assert summary["mean"] == pytest.approx(17 / 6)
    assert summary["has_semantic_count_ground_truth"] is False
    assert reporting._topic_count_distribution([])["mean"] is None


def test_full_comparison_separates_old_and_fresh_tests(artifacts, monkeypatch):
    benchmark, manifest, sessions, _ = artifacts
    manifest.update(suite="full677", content_fingerprint="synthetic-full-cohort")
    sessions[-1].update(pilot_member=False, fresh_test=True)
    full = benchmark / "runs/full677"
    for name in [*reporting.BASELINES, "cst_mil_r0"]:
        shutil.copytree(benchmark / "runs/pilot300" / name, full / name)
    for run in full.iterdir():
        if not (run / "COMPLETED.json").exists():
            continue
        saved = read_json(run / "run_manifest.json")
        saved["binding"].update(suite="full677", cohort=manifest["content_fingerprint"])
        saved["binding_sha256"] = fingerprint(saved["binding"])
        write_json(run / "run_manifest.json", saved)
        predictions = []
        for path in sorted((run / "sessions").glob("*.json")):
            wrapper = read_json(path)
            if wrapper["prediction"]["post_id"] == sessions[-1]["post_id"]:
                wrapper["prediction"].update(pilot_member=False, fresh_test=True)
            wrapper["prediction_sha256"] = fingerprint(wrapper["prediction"])
            wrapper["binding"] = saved["binding_sha256"]
            write_json(path, wrapper)
            predictions.append(wrapper["prediction"])
        write_jsonl(run / "predictions.jsonl", predictions)
    errors = [
        ("btm", "training", "RuntimeError",
         "Observed process CPU affinity exceeds frozen resource cap"),
        ("btm", "inference", "RuntimeError",
         "Observed inference process CPU affinity exceeds resource cap"),
        ("bertopic", "interrupt", "KeyboardInterrupt", ""),
    ]
    for method, name, error_type, message in errors:
        write_json(full / method / f"error_{name}.json", {
            "type": error_type, "message": message, "traceback": f"synthetic {name} traceback",
        })
    monkeypatch.setitem(reporting.EXPECTED_COUNTS, "full677", (6, 12, 2, 2, 2))
    monkeypatch.setitem(reporting.EXPECTED_TEST_SUBSETS, "full677", (1, 1))
    monkeypatch.setattr(reporting, "_figures", lambda rows, path: [])
    result = reporting.create_comparison("full677")
    directory = Path(result["output_dir"])
    stats = read_json(directory / "statistics.json")
    for name in ["fresh_test", "previously_seen_test"]:
        subset = stats["full_test_subsets"][name]
        assert subset["selected"]["n_sessions"] == 1
        assert subset["baseline"]["n_sessions"] == 1
        assert "ci95" in subset["paired_difference"]
        assert set(subset["methods"]) == {*reporting.BASELINES, "cst_mil_r0"}
        assert all(row["n_sessions"] == 1 for row in subset["methods"].values())
        assert subset["methods"]["cst_mil_r0"] == subset["selected"]
        assert subset["methods"]["nmf"] == subset["baseline"]
    assert stats["selection_source_suite"] == "pilot300"
    assert stats["trained_cst_revisions"] == ["cst_mil_r0"]
    recorded = stats["completed_run_errors"]
    assert len(recorded) == 3
    assert {row["stage"] for row in recorded if row["method"] == "btm"} == {"training", "inference"}
    assert all(row["final_status"] == "complete" for row in recorded)
    bertopic_error = next(row for row in recorded if row["method"] == "bertopic")
    assert bertopic_error["category"] == "user_keyboard_interrupt"
    assert all(Path(row["path"]).is_file() and row["error"]["traceback"] for row in recorded)
    text = (directory / "analysis-report.md").read_text(encoding="utf-8")
    assert "新增测试 fresh1：各方法风险指标" in text
    assert "已查看测试 seen1：各方法风险指标" in text
    assert text.count("| 方法 | 会话数 | macro-F1 | CB Recall | CB Precision | AP | ROC-AUC |") == 2
    assert "来自 pilot300 验证集选版，不是 full677 的再次迭代" in text
    assert "已重训的 CST 配置为 `cst_mil_r0`" in text
    assert "pilot300 冻结选版时的 `stop_target_met=False`" in text
    assert "扩大到 677" not in text
    assert "多随机种子实验、人工主题及边界标注" in text
    assert "冻结阈值的影响与漏检案例" in text
    assert "不是本轮 full677 的失败运行" in text
    assert "CPU affinity 资源限制拒绝" in text
    assert "用户中断（KeyboardInterrupt）" in text
    assert "这些运行随后均已成功完成" in text
    assert "不能将以下记录误称为算法最终失败" in text


def test_full_subset_leaders_follow_metrics_not_the_selected_method():
    rows = {
        "bertopic": {"n_sessions": 76, "macro_f1": .85, "recall_cb": 1.,
                     "precision_cb": .8, "ap": .90, "roc_auc": .92},
        "cst_mil_r0": {"n_sessions": 76, "macro_f1": .81, "recall_cb": .85,
                       "precision_cb": .83, "ap": .95, "roc_auc": .96},
    }
    values = {"methods": rows, "selected": rows["cst_mil_r0"],
              "paired_difference": {"delta_macro_f1": .04, "ci95": [-.03, .10]}}
    text = reporting._full_subset_section({"fresh_test": values, "previously_seen_test": values})
    assert "macro-F1 点值最高为 `bertopic`；AP 点值最高为 `cst_mil_r0`" in text
    assert "AP 衡量分数排序，macro-F1 还受已冻结阈值影响" in text
    rows["cst_mil_r0"]["macro_f1"] = .85
    text = reporting._full_subset_section({"fresh_test": values, "previously_seen_test": values})
    assert "macro-F1 点值最高为 `bertopic`、`cst_mil_r0`（并列）" in text
