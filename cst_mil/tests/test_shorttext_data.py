"""Isolation and provenance checks for the independently prepared short-text cohorts."""

from __future__ import annotations

import hashlib
import importlib
import io
import itertools
import json
import random

import pytest


def test_module_exposes_public_preparation_api():
    try:
        module = importlib.import_module("cst_mil.research_followup.shorttext.data")
    except ModuleNotFoundError:
        pytest.fail("The independent shorttext data preparation module is not implemented")
    assert callable(module.prepare_all)
    assert callable(module.load_dataset)


@pytest.fixture
def data():
    return importlib.import_module("cst_mil.research_followup.shorttext.data")


def record(source_id, text, label=0, **kwargs):
    return dict(source_id=source_id, text=text, label=label, source_partition="all", **kwargs)


def unique_records(count=180):
    rng = random.Random(7)
    return [
        record(str(i), "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=38)), i % 2)
        for i in range(count)
    ]


def test_normalization_preserves_content_and_short_strings(data):
    assert data.normalized("Ａ B\tC\n") == "abc"
    assert data.normalized("https://A.test/x") != data.normalized("https://B.test/x")
    rows = [record("a", "hi"), record("b", "hiya"), record("c", "HI")]
    kept, audit, components, _ = data.deduplicate(rows)
    assert len(kept) == 2
    assert len(components) == 2
    assert len(audit) == 1


def test_exact_conflict_quarantines_every_copy(data):
    rows = [record("a", "same words", 0), record("b", "same\twords", 1)]
    kept, audit, components, stats = data.deduplicate(rows)
    assert kept == []
    assert len(audit) == 2
    assert {r["reason"] for r in audit} == {"conflicting_component_labels"}
    assert stats["conflict_components"] == 1
    assert components[0]["conflicting_labels"]


def test_transitive_near_duplicate_conflict_not_only_pairwise(data):
    # One disjoint character substitution adds/removes five grams. Adjacent
    # variants clear .85; two substitutions do not, so only transitivity links ends.
    text = "".join(chr(0x5000 + i) for i in range(90))
    b = text[:20] + "X" + text[21:]
    c = b[:60] + "Y" + b[61:]
    assert data.jaccard(text, b) >= 0.85
    assert data.jaccard(b, c) >= 0.85
    assert data.jaccard(text, c) < 0.85
    kept, audit, components, _ = data.deduplicate(
        [record("a", text, 0), record("b", b, 0), record("c", c, 1)]
    )
    assert kept == []
    assert len(audit) == 3
    assert len(components) == 1


def test_prefix_search_matches_exhaustive_oracle(data):
    rng = random.Random(11)
    rows = []
    for i in range(10):
        text = "".join(rng.choices("abcdefghi", k=120))
        rows.extend(record(f"{i}-{j}", text[:j] + "Z" + text[j + 1 :]) for j in (10, 11, 60))
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    for a, b in itertools.combinations(range(len(rows)), 2):
        if data.jaccard(rows[a]["text"], rows[b]["text"]) >= 0.85:
            parents[root(a)] = root(b)
    oracle = {
        frozenset(r["source_id"] for i, r in enumerate(rows) if root(i) == key)
        for key in {root(i) for i in range(len(rows))}
    }
    _, _, components, _ = data.deduplicate(rows)
    actual = {frozenset(c["source_ids"]) for c in components}
    assert actual == oracle


def test_toxicn_cross_official_partition_component_quarantined(data):
    rows = [
        record("a", "a sufficiently long identical comment", source_split="train"),
        record("b", "a sufficiently long identical comment", source_split="test"),
    ]
    kept, audit, _, _ = data.deduplicate(rows, preserve_official_test=True)
    assert kept == []
    assert {r["reason"] for r in audit} == {"component_crosses_official_train_test"}


def test_component_and_match_assignment_are_deterministic(data):
    rows = unique_records()
    for i, r in enumerate(rows):
        r["component_id"] = r["source_id"]
        r["match_id"] = str(i // 6)
    assigned = data.assign_splits(rows, "conda", 42)
    assert all(
        len({r["split"] for r in assigned if r["match_id"] == group}) == 1
        for group in {r["match_id"] for r in assigned}
    )
    reversed_assigned = data.assign_splits(list(reversed(rows)), "conda", 42)
    assert {r["source_id"]: r["split"] for r in assigned} == {
        r["source_id"]: r["split"] for r in reversed_assigned
    }


def test_official_toxicn_test_never_migrates(data):
    rows = unique_records(40)
    for i, r in enumerate(rows):
        r.update(component_id=r["source_id"], source_split="test" if i < 10 else "train")
    assigned = data.assign_splits(rows, "toxicn", 42)
    assert all(r["split"] == "test" for r in assigned if r["source_split"] == "test")
    assert all(r["split"] != "test" for r in assigned if r["source_split"] == "train")


def test_hatebr_real_post_group_not_random_comments(data):
    rows = unique_records(40)
    for i, r in enumerate(rows):
        r.update(component_id=r["source_id"], source_post=f"https://instagram.test/p/{i // 4}")
    assigned = data.assign_splits(rows, "hatebr", 42)
    assert all(
        len({r["split"] for r in assigned if r["source_post"] == post}) == 1
        for post in {r["source_post"] for r in assigned}
    )


def test_binary_parsers_reject_bad_labels_and_metadata_stays_separate(data):
    payload = [{"content": "真实正文", "toxic": 1, "platform": "SECRET_META", "topic": "risk"}]
    rows = data.parse_toxicn(payload, "train")
    assert rows[0]["text"] == "真实正文"
    assert "SECRET_META" not in rows[0]["text"]
    with pytest.raises(ValueError, match="label"):
        data.parse_toxicn([{"content": "x", "toxic": 7}], "train")
    scam = io.StringIO("number,scam,msg_length,msg\n1,1,8,正文只有这些\n")
    rows = data.parse_scamspot(scam)
    assert rows[0]["label"] == 1
    assert rows[0]["text"] == "正文只有这些"


def test_conda_unannotated_test_not_assigned_negative(data):
    raw = io.StringIO(
        "Id,matchId,conversationId,utterance,chatTime,playerSlot,playerId\n1,8,9,hello,0,1,u\n"
    )
    assert data.parse_conda(raw, "test") == []
    raw = io.StringIO("Id,matchId,conversationId,utterance,intentClass\n1,8,9,hello,I\n")
    rows = data.parse_conda(raw, "train")
    assert rows[0]["label"] == 1
    assert rows[0]["match_id"] == "8"


def test_canonical_unit_is_text_only_and_no_message_gold(data):
    r = record("a", "язык 中文 áé 🤖", 1, split="train", username="SECRET_USER", component_id="c")
    session, messages = data.canonical_unit(r, "fixture")
    assert session["comments"] == []
    assert session["n_messages"] == 1
    assert messages[0]["parent_id"] is None
    assert messages[0]["speaker_id"] is None
    assert messages[0]["comment_label"] is None
    assert messages[0]["raw_text"] == messages[0]["model_text"] == r["text"]
    assert "SECRET_USER" not in json.dumps(messages)


def test_balanced_shrink_and_minimum_class_stops(data):
    rows = unique_records(30)
    for i, r in enumerate(rows):
        r["split"] = ("train", "dev", "test")[i % 3]
    selected, counts, shortages = data.select_balanced(
        rows, {"train": 20, "dev": 10, "test": 10}, 42
    )
    assert all(c["0"] == c["1"] for c in counts.values())
    assert shortages["train"]
    assert len(selected) == 30
    assert not data.formal_eligible(counts, minimum=50)


def test_prepared_fingerprint_guards_resume_and_complete_prediction_units(data, tmp_path):
    rows = unique_records()
    source = tmp_path / "source.csv"
    source.write_text("original", encoding="utf-8")
    sources = [{"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}]
    manifest = data.prepare_records(
        rows,
        "scamspot",
        tmp_path,
        seed=42,
        targets={"train": 12, "dev": 6, "test": 6},
        minimum_eval=1,
        sources=sources,
    )
    assert manifest["status"] == "ready"
    assert manifest["n_sessions"] == manifest["n_messages"] == 24
    loaded, sessions, messages = data.load_dataset("scamspot", tmp_path)
    assert loaded["cohort_fingerprint"] == manifest["cohort_fingerprint"]
    assert len(sessions) == len(messages) == 24
    for split in ("train", "dev", "test"):
        assert (tmp_path / "prepared" / "scamspot" / split / "sessions.jsonl").exists()
    resumed = data.prepare_records(
        rows,
        "scamspot",
        tmp_path,
        seed=42,
        targets={"train": 12, "dev": 6, "test": 6},
        minimum_eval=1,
        sources=sources,
    )
    assert resumed == manifest
    with pytest.raises(ValueError, match="configuration"):
        data.prepare_records(
            rows,
            "scamspot",
            tmp_path,
            seed=43,
            targets={"train": 12, "dev": 6, "test": 6},
            minimum_eval=1,
            sources=sources,
        )
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        data.load_dataset("scamspot", tmp_path)


def test_download_checks_frozen_hash_before_replacing(data, tmp_path):
    source = tmp_path / "input.txt"
    source.write_bytes(b"payload")
    dest = tmp_path / "download.txt"
    with pytest.raises(ValueError, match="hash"):
        data.fetch(source.as_uri(), dest, sha256="0" * 64)
    assert not dest.exists()
    expected = hashlib.sha256(b"payload").hexdigest()
    info = data.fetch(source.as_uri(), dest, sha256=expected)
    assert info["sha256"] == expected
    dest.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        data.fetch(source.as_uri(), dest, sha256=expected)


def test_statistics_keep_two_length_definitions_distinct(data):
    stats = data.text_statistics([record("a", "Ａ B\n"), record("b", "中 文")])
    assert stats["raw_chars"]["mean"] == 3.5
    assert stats["nfkc_no_whitespace_chars"]["mean"] == 2


def test_primary_evidence_preserves_blocked_access_and_cached_hash(data, tmp_path, monkeypatch):
    from io import BytesIO
    from urllib.error import HTTPError

    def blocked(request, timeout):
        raise HTTPError(request.full_url, 403, "Forbidden", {}, BytesIO(b"blocked"))

    monkeypatch.setattr(data, "urlopen", blocked)
    receipt = data._primary_attempt("https://example.org/paper", tmp_path, "publisher")
    assert receipt["http_status"] == 403
    assert receipt["status"] == "http_error"
    assert (tmp_path / "publisher.body").read_bytes() == b"blocked"
    assert data._primary_attempt("https://example.org/paper", tmp_path, "publisher") == receipt
    with pytest.raises(ValueError, match="URL"):
        data._primary_attempt("https://example.org/different", tmp_path, "publisher")
    (tmp_path / "publisher.body").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        data._primary_attempt("https://example.org/paper", tmp_path, "publisher")
