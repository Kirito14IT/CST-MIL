from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cst_mil.research_followup.full_joint_comparison.errors import _rows, build_error_review


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(value, ensure_ascii=False) for value in values), encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(root: Path, *, no_errors: bool = False) -> list[dict]:
    records = []
    for source in ("sccd", "conda", "hatebr", "toxicn", "scamspot"):
        for i in range(5):
            label = i % 2
            masks = [source != "scamspot", source == "toxicn", source == "scamspot"] if i < 4 else [False] * 3
            records.append({"record_id": f"{source}-{i}", "message_id": str(i),
                            "source": source, "session_id": f"s{i // 3}" if source == "sccd" else f"{source}-s",
                            "split": "test", "labels": [label] * 3, "observed_mask": masks,
                            "order": i, "text": f"原始正文 {source} {i} " * 30, "parent_id": None,
                            "is_fallback": i == 4 and source == "sccd"})
    sessions = [{"session_id": "s0", "split": "test", "label": 0},
                {"session_id": "s1", "split": "test", "label": 1}]
    prepared = root / "prepared"
    _jsonl(prepared / "records.jsonl", records)
    _jsonl(prepared / "sessions.jsonl", sessions)
    _json(prepared / "manifest.json", {"artifacts": {name: _hash(prepared / name) for name in ("records.jsonl", "sessions.jsonl")}})
    methods, seeds = ["lda", "cst_v1_1", "minilm_lr"], [42]
    selected = {task: {"method": "lda"} for task in ("sccd.attack", "conda.attack", "hatebr.attack", "toxicn.attack", "toxicn.hate", "scamspot.spam_scam", "sccd.conversation")}
    _json(root / "analysis/analysis.json", {"status": "complete", "methods": methods, "seeds": seeds})
    _json(root / "analysis/selected_baselines.json", selected)
    hashes = {}
    for method in methods:
        folder = root / "runs" / method / "seed_42"
        thresholds = dict.fromkeys(("attack", "hate", "spam_scam", "conversation", "conversation_common16"), .5)
        _json(folder / "run_manifest.json", {"method": method, "seed": 42, "thresholds": thresholds})
        rows = []
        for raw in records:
            label = raw["labels"][0]
            correct = method == "minilm_lr" or no_errors
            score = (.9 if label else .1) if correct else (.1 if label else .9)
            rows.append({**raw, "scores": dict.fromkeys(("attack", "hate", "spam_scam"), score)})
        predicted_sessions = [{**raw, "score": (.9 if raw["label"] else .1) if method == "minilm_lr" or no_errors else (.1 if raw["label"] else .9),
                               "common16_score": .6, "threshold": .5, "common16_threshold": .5} for raw in sessions]
        _jsonl(folder / "predictions_message_test.jsonl", rows)
        _jsonl(folder / "predictions_session_test.jsonl", predicted_sessions)
        for name in ("run_manifest.json", "predictions_message_test.jsonl", "predictions_session_test.jsonl"):
            path = folder / name
            hashes[str(path.relative_to(root))] = _hash(path)
    _json(root / "analysis/verification.json", {"status": "passed", "prediction_and_manifest_hashes": hashes})
    return records


def test_frozen_errors_have_real_text_gold_comparators_and_no_unknown_gold(tmp_path: Path) -> None:
    records = _prepare(tmp_path)
    result = build_error_review(tmp_path)
    task = result["cases"]["toxicn.hate"]
    assert task["FP"]["total_error_count"] == 2
    assert task["FN"]["total_error_count"] == 2
    # Equal scores resolve deterministically by record ID.
    assert [row["record_id"] for row in task["FP"]["cases"]] == ["toxicn-0", "toxicn-2"]
    first = task["FN"]["cases"][0]
    assert first["record_id"] == "toxicn-1"
    assert first["full_text"] == next(row["text"] for row in records if row["record_id"] == "toxicn-1")
    assert first["comparators"]["minilm_lr"]["prediction"] == 1
    assert first["comparators"]["lda"]["prediction"] == 0
    counts = list(csv.DictReader((tmp_path / "errors/confusion_counts.csv").open(encoding="utf-8-sig")))
    row = next(row for row in counts if row["method"] == "cst_v1_1" and row["task"] == "toxicn.hate")
    assert (row["n_observed"], row["n_unknown"], row["FP"], row["FN"]) == ("4", "1", "2", "2")
    session = result["cases"]["sccd.conversation"]["FN"]["cases"][0]
    assert all(row["record_id"] != "sccd-4" for row in session["top3_comment_evidence"])
    assert all("gold_comment_label" in row for row in session["top3_comment_evidence"])
    assert (tmp_path / "errors/cst_seed42_cases.md").exists()
    assert (tmp_path / "errors/cst_seed42_cases.csv").exists()


def test_no_errors_is_explicit_na(tmp_path: Path) -> None:
    _prepare(tmp_path, no_errors=True)
    result = build_error_review(tmp_path)
    assert result["cases"]["conda.attack"]["FP"]["status"] == "N/A"
    assert result["cases"]["conda.attack"]["FN"]["reason"] == "no_FN"


def test_prediction_drift_fails_before_review_is_written(tmp_path: Path) -> None:
    _prepare(tmp_path)
    path = tmp_path / "runs/lda/seed_42/predictions_message_test.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="inputs changed"):
        build_error_review(tmp_path)
    assert not (tmp_path / "errors").exists()


def test_jsonl_preserves_unicode_line_separators_inside_multilingual_text(tmp_path: Path) -> None:
    originals = [{"record_id": "first", "text": "中文\u2028English\u2029Português\u0085日本語"},
                 {"record_id": "second", "text": "العربية\u0085한국어\u2028ไทย\u2029"}]
    path = tmp_path / "unicode.jsonl"
    # ensure_ascii=False leaves these legal Unicode characters inside JSON
    # strings rather than hiding the boundary bug with ASCII escape sequences.
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in originals) + "\n", encoding="utf-8-sig")
    with path.open(encoding="utf-8-sig") as handle:
        assert len(list(handle)) == 2
    assert _rows(path) == originals
