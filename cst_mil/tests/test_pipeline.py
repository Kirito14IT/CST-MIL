from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from cst_mil.pipeline import (
    _release_run_lock,
    _start_run,
    _validate_prepared,
    evaluate_command,
    train_command,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_validate_prepared_rejects_session_leakage(tmp_path: Path) -> None:
    sessions = [
        {"post_id": "a", "label": 0, "split": "train", "model_text": "safe"},
        {"post_id": "a", "label": 0, "split": "test", "model_text": "safe"},
    ]
    messages = [
        {
            "post_id": "a",
            "comment_id": "m1",
            "split": "train",
            "model_text": "safe",
            "context_model_text": "safe",
        }
    ]
    _write_jsonl(tmp_path / "sessions.jsonl", sessions)
    _write_jsonl(tmp_path / "messages.jsonl", messages)
    with pytest.raises(ValueError, match="leakage"):
        _validate_prepared(tmp_path)


def test_validate_prepared_accepts_disjoint_sessions(tmp_path: Path) -> None:
    sessions = [
        {"post_id": "a", "label": 0, "split": "train", "model_text": "safe"},
        {"post_id": "b", "label": 1, "split": "dev", "model_text": "risk"},
        {"post_id": "c", "label": 1, "split": "test", "model_text": "risk"},
    ]
    messages = [
        {
            "post_id": post_id,
            "comment_id": f"m-{post_id}",
            "split": split,
            "model_text": "text",
            "context_model_text": "text",
        }
        for post_id, split in (("a", "train"), ("b", "dev"), ("c", "test"))
    ]
    _write_jsonl(tmp_path / "sessions.jsonl", sessions)
    _write_jsonl(tmp_path / "messages.jsonl", messages)
    session_frame, message_frame = _validate_prepared(tmp_path)
    assert isinstance(session_frame, pd.DataFrame)
    assert len(message_frame) == 3


def test_formal_run_lock_is_exclusive(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    lock = _start_run(run_dir)
    with pytest.raises(FileExistsError, match="owns the formal run lock"):
        _start_run(run_dir)
    _release_run_lock(lock)


def test_synthetic_train_evaluate_smoke(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    session_specs = [
        ("train-safe-1", 0, "train", "今天一起吃饭"),
        ("train-safe-2", 0, "train", "天气很好散步"),
        ("train-risk-1", 1, "train", "你真是个废物"),
        ("train-risk-2", 1, "train", "大家一起辱骂他"),
        ("dev-safe", 0, "dev", "正常聊天内容"),
        ("dev-risk", 1, "dev", "恶意攻击辱骂"),
        ("test-safe", 0, "test", "友好讨论学习"),
        ("test-risk", 1, "test", "继续攻击这个人"),
    ]
    sessions = [
        {
            "post_id": post_id,
            "label": label,
            "split": split,
            "raw_text": text,
            "model_text": text,
        }
        for post_id, label, split, text in session_specs
    ]
    messages: list[dict[str, object]] = []
    for post_id, label, split, text in session_specs:
        for order in range(2):
            comment_id = f"{post_id}-m{order}"
            messages.append(
                {
                    "post_id": post_id,
                    "comment_id": comment_id,
                    "split": split,
                    "raw_text": f"{text}{order}",
                    "model_text": f"{text}{order}",
                    "context_model_text": f"{text}{order}",
                    "parent_id": None,
                    "order": order,
                    "comment_label": label if split == "test" else None,
                }
            )
    _write_jsonl(prepared / "sessions.jsonl", sessions)
    _write_jsonl(prepared / "messages.jsonl", messages)
    pd.DataFrame(sessions)[["post_id", "label", "split"]].to_csv(
        prepared / "splits.csv", index=False
    )
    (prepared / "cleaning_report.json").write_text("{}", encoding="utf-8")
    config = {
        "seed": 42,
        "n_sessions": 8,
        "positive_sessions": 4,
        "negative_sessions": 4,
        "split": {"train": 4, "dev": 2, "test": 2},
        "char_features": {
            "ngram_min": 2,
            "ngram_max": 3,
            "n_features": 256,
            "mil_rounds": 1,
            "positive_top_fraction": 0.5,
        },
        "topics": {
            "n_components": 2,
            "max_features": 100,
            "min_df": 1,
            "max_df": 1.0,
            "top_words": 3,
        },
        "evaluation": {
            "evidence_top_k": 2,
            "bootstrap_samples": 10,
            "threshold_metric": "macro_f1",
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    (prepared / "cleaning_report.json").write_text(
        json.dumps(
            {
                "source_commit": "cf4015b802cabd651885211dc382152c1a270c31",
                "source_manifest_verified": True,
                "selection": {"seed": 42, "requested_sessions": 8},
            }
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    train_command(config_path, prepared, run_dir)
    sessions_path = prepared / "sessions.jsonl"
    original_sessions = sessions_path.read_text(encoding="utf-8")
    sessions_path.write_text(original_sessions + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        evaluate_command(run_dir, prepared)
    sessions_path.write_text(original_sessions, encoding="utf-8")
    metrics = evaluate_command(run_dir, prepared)
    assert metrics["formal_training_runs"] == 1
    assert metrics["session_metrics"]["n_sessions"] == 2
    assert (run_dir / "COMPLETED.json").is_file()
    assert (run_dir / "artifact_manifest.json").is_file()
    assert (run_dir / "analysis" / "figures" / "figure-01-confusion-matrix.pdf").is_file()
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        train_command(config_path, prepared, run_dir)
    with pytest.raises(FileExistsError, match="already complete"):
        evaluate_command(run_dir, prepared)
