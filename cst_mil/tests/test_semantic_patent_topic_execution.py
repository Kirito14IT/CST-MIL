import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from cst_mil.research_followup.semantic_patent_followup import topic_execution as execution
from cst_mil.research_followup.semantic_patent_followup.common import save


class FakePackage:
    seed = 42
    thresholds = {"attack": 0.5, "hate": 0.5, "spam_scam": 0.5}
    config = SimpleNamespace(architecture_base="full")

    def score_records(self, records, embeddings):
        scores = np.full((len(records), 3), 0.1)
        if len(records) > 2:
            scores[2] = [0.08, 0.02, 0.03]
        return scores, embeddings


class FakeEncoder:
    def encode(self, texts, batch_size=32):
        return np.tile(np.r_[1.0, np.zeros(383)], (len(texts), 1)).astype(np.float32)


def record(case_id, i, *, source="sccd", text="校园比赛需要认真训练"):
    return {
        "record_id": f"{case_id}:{i}",
        "source": source,
        "split": "development_challenge" if source == "authored_challenge" else "dev",
        "session_id": case_id,
        "message_id": f"m{i}",
        "text": text,
        "order": i,
        "parent_id": "m0" if i == 2 else None,
        "labels": [1, 0, 0],
        "observed_mask": [True, False, False],
    }


def fixture_campaign(tmp_path, monkeypatch):
    package = tmp_path / "runs/full/seed_42/model.joblib"
    package.parent.mkdir(parents=True)
    joblib.dump(FakePackage(), package)
    save(tmp_path / "selection_freeze.json", {"selected_architecture": "full"})
    cases = []
    for index in range(72):
        source = "authored_challenge" if index >= 60 else "sccd"
        case_id = f"case{index}"
        cases.append(
            {
                "id": case_id,
                "kind": "new_dev_assisted_review"
                if index < 30
                else "authored_challenge"
                if source == "authored_challenge"
                else "real_dev",
                "eval_groups": [f"eval:{index}"],
                "messages": [
                    record(case_id, 0, source=source),
                    record(case_id, 1, source=source, text="校园比赛专注训练"),
                    record(case_id, 2, source=source, text="没错"),
                ],
            }
        )
    save(tmp_path / "prepared/cases.json", {"old_cases": cases[30:], "new_cases": cases[:30]})
    save(tmp_path / "raw_embeddings/dev/complete.json", {"dummy": True})
    monkeypatch.setattr(execution, "guard", lambda _: {"identity": "test_protocol"})
    calls = {"bank": 0, "raw": 0, "warmup": 0}

    def embeddings_for(records, output):
        assert all(row["split"] in ("train", "dev") for row in records)
        calls["bank"] += 1
        return FakeEncoder().encode([row["text"] for row in records])

    def new_encoder(output):
        calls["raw"] += 1
        return FakeEncoder()

    def warmup(encoder, records):
        calls["warmup"] += 1
        return {"split": "train", "dummy": True}

    monkeypatch.setattr(execution, "embeddings_for", embeddings_for)
    monkeypatch.setattr(execution, "new_encoder", new_encoder)
    monkeypatch.setattr(execution, "fixed_warmup", warmup)
    monkeypatch.setattr(execution, "load_data", lambda _: ([], []))
    return calls


def test_explicit_schema_conversion_excludes_labels_and_upstream_filenames():
    item = record("p", 0)
    item["source_ref"] = {"path": "secret_label_file", "comment_time": "3", "speaker_id": "s"}
    converted = execution.convert_messages([item])[0]
    assert converted["comment_id"] == "m0" and converted["post_id"] == "p"
    assert converted["raw_text"] == item["text"] and converted["comment_time"] == "3"
    assert set(converted).isdisjoint({"labels", "observed_mask", "source", "source_ref"})


def test_full_72_case_execution_early_review_and_checked_resume(tmp_path, monkeypatch):
    calls = fixture_campaign(tmp_path, monkeypatch)
    summary = execution.run_topics(tmp_path)
    assert summary["status"] == "passed"
    assert summary["case_count"] == 72
    assert calls == {"bank": 1, "raw": 1, "warmup": 1}
    assert summary["configurations"]["full"]["linked"] == 72
    assert summary["configurations"]["no_reply_link"]["linked"] == 0
    assert summary["review_packets"]["case_count"] == 30
    assert summary["review_packets"]["configuration_count"] == 5
    indices = summary["review_packets"]["review_packet_indices"]
    for path in indices:
        packet = json.loads(Path(path).read_text(encoding="utf-8"))
        assert len(packet["cases"]) == 30
    # Completed cache must not initialize a bank or encoder again.
    again = execution.run_topics(tmp_path)
    assert again == summary and calls == {"bank": 1, "raw": 1, "warmup": 1}
    first = tmp_path / "topics/cases/case_001.json"
    first.write_text(first.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(RuntimeError):
        execution.run_topics(tmp_path)


def test_partial_resume_checks_case_receipt(tmp_path, monkeypatch):
    fixture_campaign(tmp_path, monkeypatch)
    execution.run_topics(tmp_path)
    (tmp_path / "topics/complete.json").unlink()
    first = tmp_path / "topics/cases/case_001.json"
    value = json.loads(first.read_text(encoding="utf-8"))
    value["variants"]["full"]["coverage"] = 0.0
    save(first, value)
    with pytest.raises(RuntimeError, match="fingerprint"):
        execution.run_topics(tmp_path)


def test_test_record_and_selection_change_are_rejected(tmp_path, monkeypatch):
    fixture_campaign(tmp_path, monkeypatch)
    cases_path = tmp_path / "prepared/cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    cases["new_cases"][0]["messages"][0]["split"] = "test"
    save(cases_path, cases)
    with pytest.raises(ValueError, match="TEST"):
        execution.run_topics(tmp_path)
    cases["new_cases"][0]["messages"][0]["split"] = "dev"
    save(cases_path, cases)
    execution.run_topics(tmp_path)
    save(tmp_path / "selection_freeze.json", {"selected_architecture": "full", "changed": True})
    with pytest.raises(RuntimeError, match="identity changed"):
        execution.run_topics(tmp_path)
