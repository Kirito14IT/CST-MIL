from pathlib import Path

import pytest

from cst_mil.unified_semantic_v1_1.cli import freeze, seal_caches, select_topic_distance, sha, write


def prepare_review_inputs(output: Path):
    cases = [{"id": f"c{i}"} for i in range(42)]
    write(output / "qa/cases.json", {"cases": cases})
    digests = {}
    for d in ("0.25", "0.35", "0.45"):
        write(output / f"qa/distance_{d}.json", [{"id": c["id"], "result": {}} for c in cases])
        digests[d] = sha(output / f"qa/distance_{d}.json")
    write(output / "qa/automatic_checks.json", {"0.35": {"hard_checks_passed": True}})
    review = {
        "case_manifest_sha256": sha(output / "qa/cases.json"),
        "candidate_sha256": digests,
        "cases": [{"id": c["id"], "ratings": {d: {} for d in digests}} for c in cases],
    }
    for name in ("reviewer_a", "reviewer_b"):
        write(output / f"qa/{name}.json", review)
    return review


def test_review_must_cover_unique_fixed_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cst_mil.unified_semantic_v1_1.cli.performance_gate", lambda _: {"passed": True}
    )
    review = prepare_review_inputs(tmp_path)
    review["cases"][-1]["id"] = review["cases"][0]["id"]
    write(tmp_path / "qa/reviewer_a.json", review)
    with pytest.raises(ValueError, match="42 fixed cases"):
        freeze(tmp_path, 0.35)


def test_stale_review_candidate_rejected_before_model_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cst_mil.unified_semantic_v1_1.cli.performance_gate", lambda _: {"passed": True}
    )
    prepare_review_inputs(tmp_path)
    write(tmp_path / "qa/distance_0.35.json", [{"changed": True}])
    with pytest.raises(ValueError, match="outputs changed"):
        freeze(tmp_path, 0.35)


def test_modified_embedding_bytes_rejected(tmp_path):
    (tmp_path / "embeddings.npz").write_bytes(b"original_cache_bytes")
    sealed = seal_caches(tmp_path)
    assert sealed and seal_caches(tmp_path, verify=True) == sealed
    (tmp_path / "embeddings.npz").write_bytes(b"changed_cache_bytes")
    with pytest.raises(ValueError, match="content changed"):
        seal_caches(tmp_path, verify=True)


def test_refreezing_after_test_opened_refused(tmp_path):
    write(tmp_path / "test_evaluation_state.json", {"stage": "starting"})
    with pytest.raises(ValueError, match="Test already opened"):
        freeze(tmp_path, 0.35)


def test_showcase_preferences_cannot_override_real_development_selection(tmp_path):
    review = prepare_review_inputs(tmp_path)
    cases = [
        {"id": f"c{i}", "kind": "real_dev" if i < 30 else "authored_challenge"} for i in range(42)
    ]
    write(tmp_path / "qa/cases.json", {"cases": cases})
    review["case_manifest_sha256"] = sha(tmp_path / "qa/cases.json")
    for i, case in enumerate(review["cases"]):
        for distance in ("0.25", "0.35", "0.45"):
            score = (4 if distance == "0.35" else 3) if i < 30 else (5 if distance == "0.45" else 1)
            case["ratings"][distance] = {
                k: score for k in ("coherence", "name_support", "assignment")
            }
    for name in ("reviewer_a", "reviewer_b"):
        write(tmp_path / f"qa/{name}.json", review)
    write(
        tmp_path / "qa/automatic_checks.json",
        {d: {"hard_checks_passed": True} for d in ("0.25", "0.35", "0.45")},
    )
    assert select_topic_distance(tmp_path)["distance"] == 0.35
    review["cases"][0]["ratings"]["0.35"]["coherence"] = 6
    write(tmp_path / "qa/reviewer_a.json", review)
    with pytest.raises(ValueError, match="from 1 to 5"):
        select_topic_distance(tmp_path)


def test_failed_deployment_candidate_can_be_frozen_for_honest_evaluation(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import joblib

    prepare_review_inputs(tmp_path)
    write(tmp_path / "performance_gate.json", {"passed": False})
    monkeypatch.setattr(
        "cst_mil.unified_semantic_v1_1.cli.performance_gate", lambda _: {"passed": False}
    )
    monkeypatch.setattr(
        "cst_mil.unified_semantic_v1_1.cli.select_topic_distance", lambda _: {"distance": 0.35}
    )
    write(tmp_path / "qa/selection.json", {"distance": 0.35})
    (tmp_path / "embeddings.npz").write_bytes(b"cache")
    seal_caches(tmp_path)
    joblib.dump(
        SimpleNamespace(
            topic_distance=0.35, thresholds={"attack": 0.5}, conversation_threshold=0.5
        ),
        tmp_path / "model.joblib",
    )
    result = freeze(tmp_path, 0.35)
    assert result["deployment_eligible"] is False
    assert (tmp_path / "model_before_topic_freeze.joblib").is_file()
