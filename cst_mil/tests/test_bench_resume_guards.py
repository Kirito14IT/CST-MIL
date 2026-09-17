"""Exercise durable runner guards with isolated artifacts and no real fitting."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from cst_mil.bench import model as bench_model
from cst_mil.bench import native_topics, runner
from cst_mil.bench.common import file_hash, fingerprint, read_json, read_jsonl, write_json


class _Topics:
    info = {"backend": "test-only-no-fitting"}

    def topic_words(self, top_n=10):
        return [["test", "topic"]][:top_n]


class _Monitor:
    peak_affinity_width = 1
    peak = 123

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None


def _prediction(session):
    return {
        "post_id": session["post_id"], "split": session["split"],
        "label": session["label"], "risk_probability": 0.75,
        "prediction": 1, "topics": [], "evidence": [],
        "message_scores": [], "coverage": 0.0,
    }


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        (project / name).write_text("test fixture only\n", encoding="utf-8")
    benchmark = project / "benchmark"
    sessions = [
        {"post_id": f"p{i}", "split": split, "label": i % 2,
         "pilot_member": True, "fresh_test": False}
        for i, split in enumerate(("train", "train", "dev", "test", "test"))
    ]
    messages = [
        {"post_id": s["post_id"], "comment_id": s["post_id"] + "m0",
         "split": s["split"], "raw_text": "test topic", "model_text": "test topic",
         "order": 0, "comment_label": 999 if s["split"] == "test" else None}
        for s in sessions
    ]
    state = {
        "cohort": "fixture-cohort-a", "fit_calls": [], "predict_calls": [],
        "config": {"threads": 1, "seed": 42},
        "runtime": {
            "source_sha256": {name: file_hash(project / name)
                              for name in ("pyproject.toml", "uv.lock")},
            "native_source_sha256": {}, "native_tool_sha256": {},
            "packages": {"numpy": "fixture-version"}, "python": "fixture-python",
        },
    }

    def load(suite):
        return ({"suite": suite, "content_fingerprint": state["cohort"]},
                copy.deepcopy(sessions), copy.deepcopy(messages))

    def fit(fitting, fitting_messages, method, config, work_dir):
        state["fit_calls"].append({"sessions": fitting, "messages": fitting_messages})
        assert all(s["split"] != "test" for s in fitting)
        assert all(m["comment_label"] != 999 for m in fitting_messages)
        return SimpleNamespace(
            topic_model=_Topics(), training_summary={"fake": True},
            dev_predictions=[_prediction(s) for s in fitting if s["split"] == "dev"],
        )

    def predict(model, session, rows):
        state["predict_calls"].append(session["post_id"])
        return _prediction(session)

    monkeypatch.setattr(runner, "PROJECT", project)
    monkeypatch.setattr(runner, "BENCHMARK", benchmark)
    monkeypatch.setattr(runner, "load_suite", load)
    monkeypatch.setattr(runner, "resolve_config", lambda *a: (copy.deepcopy(state["config"]), None))
    monkeypatch.setattr(runner, "runtime_fingerprint", lambda: copy.deepcopy(state["runtime"]))
    monkeypatch.setattr(runner, "functional_checks_passed", lambda: True)
    monkeypatch.setattr(runner, "ResourceMonitor", _Monitor)
    monkeypatch.setattr(runner, "binary_metrics", lambda rows: {
        "n_sessions": len(rows), "macro_f1": 0.5,
    })
    monkeypatch.setattr(runner, "topic_quality", lambda *a: {"npmi": 0.0})
    monkeypatch.setattr(runner, "bootstrap_metrics", lambda *a: {})
    monkeypatch.setattr(runner, "evidence_metrics", lambda *a: {})
    monkeypatch.setattr(native_topics, "prepare_native_tools", lambda method: None)
    monkeypatch.setattr(bench_model, "train_model", fit)
    monkeypatch.setattr(bench_model, "predict_session", predict)
    return SimpleNamespace(project=project, benchmark=benchmark, state=state, sessions=sessions)


def test_limit_is_cumulative_and_resume_preserves_completed_records(isolated_runner):
    state = isolated_runner.state
    first = runner.run_method("smoke5", "nmf", limit="2")
    path = runner.run_path("smoke5", "nmf")
    saved = {p.name: file_hash(p) for p in (path / "sessions").glob("*.json")}
    assert first["completed"] == 2 and first["status"] == "paused"
    assert runner.run_method("smoke5", "nmf", limit="3", resume=True)["completed"] == 3
    assert runner.run_method("smoke5", "nmf", limit="1", resume=True)["completed"] == 3
    assert runner.run_method("smoke5", "nmf", stage="train", resume=True)["completed"] == 3
    final = runner.run_method("smoke5", "nmf", limit="all", resume=True)
    assert final["completed"] == 5 and final["status"] == "complete"
    assert state["predict_calls"] == ["p0", "p1", "p2", "p3", "p4"]
    assert len(state["fit_calls"]) == 1
    assert read_json(path / "run_manifest.json")["formal_fit_attempts"] == 1
    assert len(read_jsonl(path / "predictions.jsonl")) == 5
    assert all(file_hash(path / "sessions" / name) == value for name, value in saved.items())
    assert (path / "COMPLETED.json").exists()


def test_cpu_cap_violation_prevents_training_commit(isolated_runner, monkeypatch):
    monkeypatch.setattr(_Monitor, "peak_affinity_width", 8)
    with pytest.raises(RuntimeError, match="CPU affinity exceeds"):
        runner.run_method("smoke5", "nmf", limit="1")
    path = runner.run_path("smoke5", "nmf")
    assert not (path / "TRAINED.json").exists()
    assert not (path / "model.joblib").exists()


def test_tampered_model_is_rejected_before_loading_or_predicting(isolated_runner):
    runner.run_method("smoke5", "nmf", limit="1")
    path = runner.run_path("smoke5", "nmf")
    model_path = path / "model.joblib"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Model fingerprint mismatch"):
        runner.run_method("smoke5", "nmf", limit="all", resume=True)
    assert isolated_runner.state["predict_calls"] == ["p0"]
    assert len(isolated_runner.state["fit_calls"]) == 1
    assert not (path / ".lock").exists()
    assert list(path.glob("error_*.json"))


@pytest.mark.parametrize("change", ["content", "binding", "model", "duplicate", "unknown_id"])
def test_invalid_session_artifacts_are_rejected(isolated_runner, change):
    runner.run_method("smoke5", "nmf", limit="1")
    path = runner.run_path("smoke5", "nmf")
    source = next((path / "sessions").glob("*.json"))
    wrapper = read_json(source)
    destination = source
    if change == "content":
        wrapper["prediction"]["risk_probability"] = 0.1
    elif change in {"binding", "model"}:
        wrapper[change] = "wrong-fingerprint"
    elif change == "duplicate":
        destination = source.with_name("duplicate.json")
    else:
        wrapper["prediction"]["post_id"] = "outside-frozen-cohort"
        wrapper["prediction_sha256"] = fingerprint(wrapper["prediction"])
    write_json(destination, wrapper)
    expected = ("Prediction content was modified" if change == "content" else
                "Prediction binding mismatch" if change in {"binding", "model"} else
                "Invalid/duplicate completed session IDs")
    with pytest.raises(ValueError, match=expected):
        runner.run_method("smoke5", "nmf", limit="all", resume=True)
    assert isolated_runner.state["predict_calls"] == ["p0"]
    assert len(isolated_runner.state["fit_calls"]) == 1


@pytest.mark.parametrize("change", ["source", "environment", "data", "config"])
def test_run_binding_rejects_changed_inputs(isolated_runner, change):
    runner.run_method("smoke5", "nmf", limit="1")
    state = isolated_runner.state
    if change == "source":
        state["runtime"]["source_sha256"]["pyproject.toml"] = "changed-source"
    elif change == "environment":
        state["runtime"]["packages"]["numpy"] = "changed-version"
    elif change == "data":
        state["cohort"] = "changed-cohort"
    else:
        state["config"]["seed"] = 7
    with pytest.raises(ValueError, match="config/source/environment/data changed"):
        runner.run_method("smoke5", "nmf", limit="all", resume=True)
    assert state["predict_calls"] == ["p0"] and len(state["fit_calls"]) == 1


def test_train_stage_excludes_test_and_test_unlock_is_required(isolated_runner):
    first = runner.run_method("pilot300", "nmf", stage="train")
    path = runner.run_path("pilot300", "nmf")
    assert first["status"] == "trained" and first["completed"] == 0
    assert not list((path / "sessions").glob("*.json"))
    assert not isolated_runner.state["predict_calls"]
    trained_hash = file_hash(path / "TRAINED.json")
    with pytest.raises(ValueError, match="test inference locked"):
        runner.run_method("pilot300", "nmf", limit="all", resume=True)
    assert file_hash(path / "TRAINED.json") == trained_hash
    assert not isolated_runner.state["predict_calls"]
    assert runner.run_method("pilot300", "nmf", limit="3", resume=True)["test_completed"] == 0
    write_json(isolated_runner.benchmark / "selection.json", {"test_only_frozen_choice": True})
    assert runner.run_method("pilot300", "nmf", resume=True)["test_completed"] == 2
    assert len(isolated_runner.state["fit_calls"]) == 1


def test_full_suite_requires_selection_before_any_training(isolated_runner):
    with pytest.raises(ValueError, match="requires frozen pilot selection"):
        runner.run_method("full677", "nmf", stage="train")
    assert not isolated_runner.state["fit_calls"]
    assert not isolated_runner.benchmark.exists()


def test_existing_run_requires_explicit_resume(isolated_runner):
    runner.run_method("smoke5", "nmf", limit="1")
    with pytest.raises(FileExistsError, match="pass --resume"):
        runner.run_method("smoke5", "nmf", limit="2")
    assert len(isolated_runner.state["fit_calls"]) == 1
    assert isolated_runner.state["predict_calls"] == ["p0"]


@pytest.mark.parametrize("limit", ["-1", "6"])
def test_out_of_range_limit_writes_nothing(isolated_runner, limit):
    with pytest.raises(ValueError, match="limit must be between"):
        runner.run_method("smoke5", "nmf", limit=limit)
    assert not isolated_runner.state["fit_calls"]
    assert not isolated_runner.benchmark.exists()
