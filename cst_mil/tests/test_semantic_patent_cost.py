"""Scientific reference/resume integrity, using fake encoders and tiny local artifacts."""

import numpy as np
import pytest

from cst_mil.research_followup.semantic_patent_followup import cost
from cst_mil.research_followup.semantic_patent_followup.common import (
    digest,
    read,
    save,
    source_hashes,
    write_jsonl,
)


class Encoder:
    calls = 0

    def encode(self, texts, **kwargs):
        self.calls += 1
        return np.zeros((len(texts), 384), dtype=np.float32)


class Package:
    thresholds = {"attack": .5, "hate": .5, "spam_scam": .5}

    def __init__(self):
        self.encoder_instance = Encoder()

    def encoder(self):
        return self.encoder_instance

    def predict_payload(self, records, sessions, embeddings):
        return {"message": [{"record_id": r["record_id"], "scores": [.2, .2, .2]}
                            for r in records],
                "session": [{"session_id": s["session_id"], "score": .3,
                             "common16_score": .4, "threshold": .5, "common16_threshold": .5}
                            for s in sessions]}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    output = tmp_path / "cost_fixture"
    folder = output / "runs/full/seed_42"
    folder.mkdir(parents=True)
    model = folder / "model.joblib"
    model.write_bytes(b"fixture_model_not_a_real_pickle")
    records = [{"record_id": f"{source}:{i}", "message_id": str(i), "source": source,
                "split": "test", "session_id": "session" if source == "sccd" else None,
                "order": i, "text": "fixture"}
               for source in ("sccd", "conda") for i in range(2)]
    sessions = [{"session_id": "session"}]
    package = Package()
    payload = package.predict_payload(records, sessions, None)
    paths = []
    for level, rows in payload.items():
        path = folder / f"predictions_{level}_test.jsonl"
        write_jsonl(path, rows)
        paths.append(path)
    save(folder / "complete.json", {"artifacts": source_hashes([model])})
    save(folder / "test_complete.json", {"binding": {"model_sha256": digest(model),
        "selection_identity": "fixture-selection"}, "artifacts": source_hashes(paths)})
    save(output / "protocol.json", {"encoder": {"asset_bytes": 1}})
    monkeypatch.setattr(cost, "selected", lambda output: {"selected_architecture": "full",
                        "identity": "fixture-selection"})
    monkeypatch.setattr(cost, "guard", lambda output: None)
    monkeypatch.setattr(cost, "load_data", lambda *args: (records, sessions))
    monkeypatch.setattr(cost.joblib, "load", lambda path: package)
    monkeypatch.setattr(cost, "fixed_warmup", lambda *args: {"record_ids": ["TRAIN-fixture"]})
    return output, folder, package


def test_three_raw_passes_then_resume_without_reencoding(setup):
    output, _, package = setup
    result = cost.measure(output)
    assert len(result["raw_repeats"]) == 3
    assert all(r["status"] == "passed" for r in result["raw_repeats"])
    calls = package.encoder_instance.calls
    assert calls == 6  # Two canonical units, each actually encoded in three passes.
    cost.measure(output)
    assert package.encoder_instance.calls == calls


def test_reference_tampering_rejected_even_after_all_passes(setup):
    output, folder, _ = setup
    cost.measure(output)
    target = folder / "predictions_message_test.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Frozen files changed"):
        cost.measure(output)


def test_reference_for_another_selection_rejected_before_encoding(setup):
    output, folder, package = setup
    target = folder / "test_complete.json"
    seal = read(target)
    seal["binding"]["selection_identity"] = "other-selection"
    save(target, seal)
    with pytest.raises(RuntimeError, match="different model or selection"):
        cost.measure(output)
    assert package.encoder_instance.calls == 0


def test_failed_pass_cannot_silently_resume(setup):
    output, _, package = setup
    cost.measure(output)
    target = output / "cost/repeat_2.json"
    previous = read(target)
    previous["status"] = "failed"
    save(target, previous)
    calls = package.encoder_instance.calls
    with pytest.raises(RuntimeError, match="Earlier failed raw pass"):
        cost.measure(output)
    assert package.encoder_instance.calls == calls
