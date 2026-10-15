"""Isolation and label-mask invariants for the joint supervised data protocol."""

import copy
import itertools
import json
import random
from pathlib import Path

import pytest

from cst_mil.unified_semantic import data


def record(identifier, text, *, source="sccd", split="train", labels=None, mask=None):
    return {
        "record_id": identifier,
        "source": source,
        "split": split,
        "group_id": identifier,
        "session_id": identifier,
        "message_id": identifier + ":m",
        "order": 0,
        "parent_id": None,
        "text": text,
        "labels": labels or [0, 0, 0],
        "observed_mask": mask or [True, False, False],
        "original_label": {},
        "source_ref": {},
        "duplicate_component": "",
    }


@pytest.mark.parametrize(
    "source,original,expected,mask",
    [
        ("sccd", {"label": "CB"}, [1, 0, 0], [True, False, False]),
        ("sccd", {"label": "Non-CB"}, [0, 0, 0], [True, False, False]),
        ("sccd", {"fallback": True}, [0, 0, 0], [False, False, False]),
        ("conda", {"intentClass": "I"}, [1, 0, 0], [True, False, False]),
        ("conda", {"intentClass": "A"}, [0, 0, 0], [True, False, False]),
        ("hatebr", {"label_final": "1"}, [1, 0, 0], [True, False, False]),
        ("scamspot", {"scam": "1"}, [0, 0, 1], [False, False, True]),
        ("toxicn", {"toxic": 0, "toxic_type": 0, "expression": 0}, [0, 0, 0], [True, True, False]),
        ("toxicn", {"toxic": 1, "toxic_type": 1, "expression": 0}, [1, 0, 0], [True, True, False]),
        ("toxicn", {"toxic": 1, "toxic_type": 2, "expression": 2}, [1, 1, 0], [True, True, False]),
        (
            "toxicn",
            {"toxic": 1, "toxic_type": 2, "expression": 3},
            [0, 0, 0],
            [False, False, False],
        ),
    ],
)
def test_source_mapping_never_invents_unknown_negatives(source, original, expected, mask):
    assert data.map_labels(source, original) == (expected, mask)


def test_mapping_rejects_unsupported_or_inconsistent_original_labels():
    with pytest.raises(ValueError, match="CONDA"):
        data.map_labels("conda", {"intentClass": "unlabeled"})
    with pytest.raises(ValueError, match="Inconsistent"):
        data.map_labels("toxicn", {"toxic": 0, "toxic_type": 2, "expression": 0})


def test_exact_near_rule_keeps_full_content_and_distinguishes_short_texts():
    assert data.normalized("Ａ b\nC。") == "abc。"
    assert data.jaccard("看比赛", "看比") == 0
    assert data.jaccard("Ａ B", "ab") == 1
    assert data.normalized("https://site/x") != data.normalized("https://other/y")


def brute_components(rows):
    parents = list(range(len(rows)))

    def root(i):
        while parents[i] != i:
            i = parents[i]
        return i

    for left, right in itertools.combinations(range(len(rows)), 2):
        if data.jaccard(rows[left]["text"], rows[right]["text"]) >= 0.85:
            parents[root(left)] = root(right)
    return {
        frozenset(rows[i]["record_id"] for i in range(len(rows)) if root(i) == r)
        for r in {root(i) for i in range(len(rows))}
    }


def test_lossless_prefix_candidates_match_exhaustive_search_and_are_order_invariant():
    rng = random.Random(42)
    texts = ["", "x", "X", "abc", "abd", "abcdef", "abcdefgh"]
    for _ in range(35):
        base = "".join(rng.choices("abcde你好危险文本", k=rng.randint(25, 100)))
        texts.extend([base, base + "!", base[1:], base.replace("a", "z", 1)])
    rows = [record(str(i), text) for i, text in enumerate(texts)]
    mapping, components, _ = data.duplicate_components(rows)
    assert {frozenset(c["record_ids"]) for c in components} == brute_components(rows)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    assert data.duplicate_components(shuffled)[0] == mapping
    assert all(mapping.values())


def test_eval_membership_preserved_train_duplicates_removed_without_label_inheritance():
    rows = [
        record("training", "不要直接骂队员", labels=[0, 0, 0]),
        record("validation", "不要直接骂队员", split="dev", labels=[1, 0, 0]),
        record(
            "testing",
            "不要直接骂队员",
            source="toxicn",
            split="test",
            labels=[1, 1, 0],
            mask=[True, True, False],
        ),
        record("unique", "努力认真训练，多讨论战术", labels=[0, 0, 0]),
    ]
    original = copy.deepcopy(rows)
    kept, excluded, _, audit = data.clean_records(rows)
    assert {r["record_id"] for r in kept} == {"validation", "testing", "unique"}
    assert excluded[0]["reason"] == "train_component_overlaps_dev_or_test"
    assert rows == original
    assert [r["labels"] for r in kept if r["split"] == "dev"] == [[1, 0, 0]]
    assert len(audit["dev_test_dependencies"]) == 1


def test_conflicts_mask_only_affected_train_category_and_keep_other_labels():
    rows = [
        record(
            "one",
            "这一段是重复的中文评论",
            source="toxicn",
            labels=[1, 1, 0],
            mask=[True, True, False],
        ),
        record("two", "这一段是重复的中文评论", source="sccd", labels=[0, 0, 0]),
    ]
    kept, excluded, _, audit = data.clean_records(rows)
    assert not excluded
    assert kept[0]["observed_mask"] == [False, True, False]
    assert kept[1]["observed_mask"] == [False, False, False]
    assert audit["conflicting_components"][0]["categories"] == ["attack"]


def test_root_fallback_has_unknown_message_labels_even_for_positive_session():
    session = {"session_id": "p1", "split": "train", "label": 1, "post_text": "校园比赛"}
    fallback = data._fallback(session, "original_empty_session")
    assert fallback["text"] == "校园比赛"
    assert fallback["labels"] == [0, 0, 0]
    assert fallback["observed_mask"] == [False, False, False]
    assert fallback["parent_id"] is None
    assert fallback["message_id"] == "p1:root_fallback"


def fixture_inputs(monkeypatch):
    row = record("r1", "努力训练讨论战术", labels=[0, 0, 0])
    session = {
        "session_id": "r1",
        "split": "train",
        "label": 0,
        "message_ids": ["r1:m"],
        "post_text": "校园比赛",
        "original_message_count": 1,
    }
    monkeypatch.setattr(data, "_input_fingerprints", lambda: {"raw": "frozen"})
    monkeypatch.setattr(
        data, "_load_sccd", lambda: ([copy.deepcopy(row)], [copy.deepcopy(session)])
    )
    monkeypatch.setattr(data, "_load_shorttext", lambda source: [])


def test_preparation_recovers_only_with_matching_inputs_code_and_output_hashes(
    tmp_path, monkeypatch
):
    fixture_inputs(monkeypatch)
    first = data.prepare_dataset(tmp_path)
    assert data.prepare_dataset(tmp_path) == first
    assert data.read_records(tmp_path / "prepared")[0]["text"] == "努力训练讨论战术"
    assert data.read_sessions(tmp_path / "prepared")[0]["message_ids"] == ["r1:m"]
    monkeypatch.setattr(data, "_input_fingerprints", lambda: {"raw": "changed"})
    with pytest.raises(ValueError, match="resume fingerprint"):
        data.prepare_dataset(tmp_path)


def test_prepared_output_tampering_and_partial_directory_fail_closed(tmp_path, monkeypatch):
    fixture_inputs(monkeypatch)
    data.prepare_dataset(tmp_path)
    path = tmp_path / "prepared/records.jsonl"
    path.write_text(path.read_text("utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact fingerprint"):
        data.prepare_dataset(tmp_path)
    second = tmp_path / "partial"
    (second / "prepared").mkdir(parents=True)
    (second / "prepared/records.jsonl").write_text("partial", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete"):
        data.prepare_dataset(second)


def test_all_comments_excluded_session_stays_with_unknown_fallback(tmp_path, monkeypatch):
    fixture_inputs(monkeypatch)
    duplicate = record("eval", "努力训练讨论战术", split="test", source="conda")
    monkeypatch.setattr(
        data, "_load_shorttext", lambda source: [duplicate] if source == "conda" else []
    )
    data.prepare_dataset(tmp_path)
    records = data.read_records(tmp_path / "prepared")
    fallback = next(r for r in records if r["source"] == "sccd")
    assert fallback["is_fallback"]
    assert fallback["observed_mask"] == [False, False, False]
    assert fallback["duplicate_component"]
    session = data.read_sessions(tmp_path / "prepared")[0]
    assert session["excluded_message_count"] == 1
    assert session["retained_message_count"] == 0
    assert session["message_ids"] == [fallback["message_id"]]
    audit = json.loads((tmp_path / "prepared/audit.json").read_text("utf-8"))
    assert audit["sccd_fallback_sessions"][0]["reason"] == "all_comments_excluded"


def test_metadata_never_enters_text_by_label_join(tmp_path, monkeypatch):
    base = tmp_path / "toxicn"
    base.mkdir()
    message = {"post_id": "p", "comment_id": "m", "raw_text": "讨论比赛", "order": 0}
    session = {"post_id": "p", "source_id": "toxicn:train:0", "split": "train", "label": 1}
    provenance = {
        "post_id": "p",
        "source_id": "toxicn:train:0",
        "split": "train",
        "label": 1,
        "source_topic": "should_not_enter_feature",
        "split_group": "group",
    }
    for filename, payload in [
        ("messages.jsonl", message),
        ("sessions.jsonl", session),
        ("provenance.jsonl", provenance),
    ]:
        (base / filename).write_text(json.dumps(payload) + "\n", encoding="utf-8")
    monkeypatch.setattr(data, "SHORTTEXT", tmp_path.parent)
    # Select only the fixture path; no old source acquisition or source mutation.
    real_read = data._read_jsonl
    monkeypatch.setattr(data, "_read_jsonl", lambda path: real_read(base / Path(path).name))
    monkeypatch.setattr(
        data,
        "_originals",
        lambda source: {
            "toxicn:train:0": {
                "content": "讨论比赛",
                "toxic": 1,
                "toxic_type": 2,
                "expression": 3,
                "topic": "should_not_enter_feature",
            }
        },
    )
    output = data._load_shorttext("toxicn")[0]
    assert output["text"] == "讨论比赛"
    assert output["observed_mask"] == [False, False, False]
    assert output["source_ref"]["source_topic"] == "should_not_enter_feature"
