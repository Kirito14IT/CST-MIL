"""Full-candidate, masked-label and isolation checks without training models."""

import copy

import pytest

from cst_mil.research_followup.full_joint_comparison import data


def record(identifier, text, *, source="sccd", split="train", label=0, group=None):
    group = group or source + ":session:" + identifier
    return {
        "record_id": identifier,
        "source": source,
        "split": split,
        "group_id": group,
        "base_eval_group": group,
        "eval_group": group,
        "session_id": identifier,
        "message_id": identifier + ":m",
        "order": 0,
        "parent_id": None,
        "text": text,
        "labels": [label, 0, 0],
        "observed_mask": [True, False, False],
        "original_label": {"label": label},
        "source_ref": {"source_row": 2},
        "duplicate_component": "",
        "is_fallback": False,
        "fit_eligible": True,
    }


@pytest.mark.parametrize(
    "source,original,labels,mask",
    [
        ("conda", {"intentClass": "E"}, [1, 0, 0], [True, False, False]),
        ("hatebr", {"label_final": "1"}, [1, 0, 0], [True, False, False]),
        ("scamspot", {"scam": "1"}, [0, 0, 1], [False, False, True]),
        (
            "toxicn",
            {"toxic": 1, "toxic_type": 2, "expression": 3},
            [0, 0, 0],
            [False, False, False],
        ),
    ],
)
def test_labels_keep_unknown_head_masks(source, original, labels, mask):
    assert data.map_labels(source, original) == (labels, mask)


def source_fixture(tmp_path, monkeypatch, *, source="conda", count=3):
    monkeypatch.setattr(data, "SHORTTEXT", tmp_path)
    base = tmp_path / "prepared" / source
    base.mkdir(parents=True)
    candidates, originals = [], {}
    for index in range(count):
        identifier = f"{source}:train:{index}"
        text = f"Original text {index} keeps all words and punctuation!"
        row = {
            "source_id": identifier,
            "source_partition": "train",
            "source_row": index + 2,
            "label": index % 2,
            "source_label": "E" if index % 2 else "O",
            "match_id": "real-match-7",
            "conversation_id": str(index),
            "text_sha256": data._hash(data.normalized(text)),
            "component_id": str(index),
            "split": "train",
            "split_group": "real-match-7",
        }
        candidates.append(row)
        originals[identifier] = {
            "utterance": text,
            "intentClass": row["source_label"],
            "matchId": "real-match-7",
            "_source_row": index + 2,
            "_source_partition": "train",
            "_source_path": "raw.csv",
            "metadata_secret": "DO_NOT_USE_IN_FEATURES",
        }
    for name, rows in [
        ("candidates.jsonl", candidates),
        ("source_catalog.jsonl", candidates),
        (
            "exclusions.jsonl",
            [{"reason": "not_selected_fixed_budget"}, {"reason": "empty_normalized_text"}],
        ),
    ]:
        data._write_jsonl(base / name, rows)
    monkeypatch.setattr(data, "_originals", lambda source: originals)
    return base, candidates, originals


def test_full_loader_uses_all_candidates_and_preserves_true_group(tmp_path, monkeypatch):
    _, candidates, originals = source_fixture(tmp_path, monkeypatch, count=9)
    rows, catalog, exclusions = data._load_shorttext("conda")
    assert len(rows) == len(candidates) == 9
    assert len(catalog) == 9
    assert {r["group_id"] for r in rows} == {"conda:match:real-match-7"}
    assert all(r["parent_id"] is None and r["order"] == 0 for r in rows)
    assert rows[0]["text"] == originals[candidates[0]["source_id"]]["utterance"]
    assert all("DO_NOT_USE_IN_FEATURES" not in r["text"] for r in rows)
    assert [row["reason"] for row in exclusions] == ["empty_normalized_text"]


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda row: row.update(text_sha256="wrong"), "full-text fingerprint"),
        (lambda row: row.update(source_row=999), "row/partition"),
        (lambda row: row.update(label=1), "binary label"),
        (lambda row: row.update(match_id="invented"), "true match"),
    ],
)
def test_full_loader_rejects_join_or_label_drift(tmp_path, monkeypatch, mutation, error):
    base, candidates, _ = source_fixture(tmp_path, monkeypatch)
    mutation(candidates[0])
    data._write_jsonl(base / "candidates.jsonl", candidates)
    with pytest.raises(ValueError, match=error):
        data._load_shorttext("conda")


def test_eval_groups_union_true_groups_through_duplicate_components():
    rows = [
        record("a", "one", source="conda", group="match-A"),
        record("b", "two", source="conda", group="match-A"),
        record("c", "two", source="conda", group="match-B"),
        record("d", "three", source="conda", group="match-C"),
    ]
    for row, component in zip(rows, ["one", "two", "two", "three"], strict=True):
        row["duplicate_component"] = component
    audit = data._assign_eval_groups(rows)
    assert len({r["eval_group"] for r in rows[:3]}) == 1
    assert rows[3]["eval_group"] != rows[0]["eval_group"]
    assert audit["base_groups"] == 3 and audit["merged_groups"] == 2


def fixture_inputs(monkeypatch):
    row = record("s1", "train unique comment")
    session = {
        "session_id": "s1",
        "split": "train",
        "label": 1,
        "message_ids": ["s1:m"],
        "post_text": "Training post",
        "original_message_count": 1,
    }
    monkeypatch.setattr(data, "_input_fingerprints", lambda: {"raw": "fixed"})
    monkeypatch.setattr(
        data, "_load_sccd", lambda: ([copy.deepcopy(row)], [copy.deepcopy(session)])
    )
    monkeypatch.setattr(data, "_sccd_raw_audit", lambda records: ([], []))
    monkeypatch.setattr(data, "_load_shorttext", lambda source: ([], [], []))


def test_prepare_and_resume_validate_all_hashes_and_no_bag_label_copy(tmp_path, monkeypatch):
    fixture_inputs(monkeypatch)
    manifest = data.prepare(tmp_path)
    rows, sessions, resumed = data.load_prepared(tmp_path)
    assert resumed == manifest == data.prepare(tmp_path)
    assert rows[0]["labels"] == [0, 0, 0] and sessions[0]["label"] == 1
    assert rows[0]["eval_group"]
    assert manifest["candidate_counts"]["sccd"] == 1
    monkeypatch.setattr(data, "_input_fingerprints", lambda: {"raw": "changed"})
    with pytest.raises(ValueError, match="resume fingerprint"):
        data.load_prepared(tmp_path)


def test_resume_rejects_tampered_output_and_incomplete_directory(tmp_path, monkeypatch):
    fixture_inputs(monkeypatch)
    data.prepare(tmp_path)
    path = tmp_path / "prepared/records.jsonl"
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact fingerprint"):
        data.load_prepared(tmp_path)
    other = tmp_path / "partial"
    (other / "prepared").mkdir(parents=True)
    (other / "prepared/records.jsonl").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        data.prepare(other)


def test_cross_source_eval_duplicates_remove_train_and_fallback_stays_unknown(
    tmp_path, monkeypatch
):
    fixture_inputs(monkeypatch)
    duplicate = record("eval", "train unique comment", source="conda", split="test", label=1)
    monkeypatch.setattr(
        data,
        "_load_shorttext",
        lambda source: ([duplicate], [], []) if source == "conda" else ([], [], []),
    )
    data.prepare(tmp_path)
    rows, sessions, manifest = data.load_prepared(tmp_path)
    fallback = next(r for r in rows if r["source"] == "sccd")
    assert fallback["is_fallback"] and not fallback["fit_eligible"]
    assert fallback["observed_mask"] == [False, False, False]
    assert sessions[0]["retained_message_count"] == 0
    assert sessions[0]["excluded_message_count"] == 1
    assert manifest["record_count"] == 2
    excluded = data._read_jsonl(tmp_path / "prepared/exclusions.jsonl")
    assert excluded[0]["reason"] == "train_component_overlaps_dev_or_test"


def test_true_group_split_overlap_is_rejected():
    rows = [
        record("a", "one", source="conda", split="train", group="actual-match"),
        record("b", "two", source="conda", split="test", group="actual-match"),
    ]
    for row in rows:
        row["duplicate_component"] = row["record_id"]
    with pytest.raises(ValueError, match="True match/post"):
        data._validate(rows, [])


def test_frozen_full_candidate_counts_without_running_prepare():
    """A cheap read-only guard against accidentally reading old sampled messages."""
    for source, expected in data.EXPECTED_CANDIDATES.items():
        path = data.SHORTTEXT / "prepared" / source / "candidates.jsonl"
        if not path.exists():
            pytest.skip("Local acquisition is unavailable")
        with path.open(encoding="utf-8") as handle:
            assert sum(bool(line.strip()) for line in handle) == expected
