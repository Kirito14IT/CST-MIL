from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from cst_mil import data
from cst_mil.data import download_sccd, prepare_dataset, validate_raw_schema
from cst_mil.preprocess import normalize_text
from cst_mil.schemas import SCCD_COMMIT, SCCDSchemaError

POST_FIELDS = [
    "post_id",
    "user_id",
    "publish_time",
    "like_num",
    "comment_num",
    "repost_num",
    "post_content",
    "label",
    "cyberbullying_severity",
]
COMMENT_FIELDS = [
    "comment_id",
    "label",
    "expression",
    "sarcasm",
    "target",
    "group category",
    "post_id",
    "user_id",
    "comment_time",
    "like_num",
    "comment_content",
    "to_id",
]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_raw_fixture(raw_dir: Path) -> None:
    raw_dir.mkdir()
    posts: list[dict[str, str]] = []
    comments: list[dict[str, str]] = []
    for label_name, prefix in (("CB", "p"), ("Non-CB", "n")):
        for index in range(6):
            post_id = f"{prefix}{index}"
            posts.append(
                {
                    "post_id": post_id,
                    "user_id": f"u-{post_id}",
                    "publish_time": "2024-01-01",
                    "like_num": "99",
                    "comment_num": "2",
                    "repost_num": "0",
                    "post_content": f"原帖 {post_id}",
                    "label": label_name,
                    "cyberbullying_severity": "high" if label_name == "CB" else "none",
                }
            )
            comments.extend(
                [
                    {
                        "comment_id": f"{post_id}-c1",
                        "label": label_name,
                        "expression": "explicit",
                        "sarcasm": "no",
                        "target": "individual",
                        "group category": "",
                        "post_id": post_id,
                        "user_id": "secret-user-id",
                        "comment_time": "1",
                        "like_num": "12",
                        "comment_content": f"评论 {post_id} A",
                        "to_id": "",
                    },
                    {
                        "comment_id": f"{post_id}-c2",
                        "label": label_name,
                        "expression": "explicit",
                        "sarcasm": "no",
                        "target": "individual",
                        "group category": "",
                        "post_id": post_id,
                        "user_id": "another-secret-id",
                        "comment_time": "2",
                        "like_num": "7",
                        "comment_content": f"回复 {post_id} B",
                        "to_id": f"{post_id}-c1",
                    },
                ]
            )

    exact_duplicate = dict(comments[0])
    comments.append(exact_duplicate)
    comments.extend(
        [
            {
                **comments[1],
                "comment_id": "conflict-id",
                "comment_content": "第一种内容",
                "label": "CB",
            },
            {
                **comments[1],
                "comment_id": "conflict-id",
                "comment_content": "第二种内容",
                "label": "Non-CB",
            },
            {
                **comments[1],
                "comment_id": "orphan-id",
                "post_id": "not-a-post",
            },
        ]
    )
    _write_csv(raw_dir / "posts.csv", POST_FIELDS, posts)
    _write_csv(raw_dir / "comments.csv", COMMENT_FIELDS, comments)
    manifest = {
        "commit": SCCD_COMMIT,
        "files": [
            {
                "name": name,
                "sha256": hashlib.sha256((raw_dir / name).read_bytes()).hexdigest(),
            }
            for name in ("posts.csv", "comments.csv")
        ],
    }
    (raw_dir / "download_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_normalize_text_is_conservative() -> None:
    actual = normalize_text("ＡＢＣ  @某人\nHTTP://EXAMPLE.COM/X  哈哈哈!!!")
    assert actual == "abc <mention> <url> 哈哈哈!!!"


def test_schema_validation_reports_missing_column(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_raw_fixture(raw_dir)
    comments = (raw_dir / "comments.csv").read_text(encoding="utf-8")
    (raw_dir / "comments.csv").write_text(
        comments.replace("comment_content,", "missing_content,"), encoding="utf-8"
    )
    with pytest.raises(SCCDSchemaError, match="comment_content"):
        validate_raw_schema(raw_dir)


def test_prepare_dataset_cleans_and_splits_without_leakage(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _make_raw_fixture(raw_dir)
    output_one = tmp_path / "prepared-one"
    output_two = tmp_path / "prepared-two"

    report = prepare_dataset(raw_dir, output_one, seed=42, n_sessions=10)
    prepare_dataset(raw_dir, output_two, seed=42, n_sessions=10)

    assert (output_one / "splits.csv").read_bytes() == (output_two / "splits.csv").read_bytes()
    sessions = _read_jsonl(output_one / "sessions.jsonl")
    messages = _read_jsonl(output_one / "messages.jsonl")
    assert len(sessions) == 10
    assert {session["split"] for session in sessions} == {"train", "dev", "test"}

    split_label_counts: dict[tuple[str, int], int] = {}
    for session in sessions:
        key = (str(session["split"]), int(session["label"]))
        split_label_counts[key] = split_label_counts.get(key, 0) + 1
    assert split_label_counts == {
        ("train", 0): 3,
        ("train", 1): 3,
        ("dev", 0): 1,
        ("dev", 1): 1,
        ("test", 0): 1,
        ("test", 1): 1,
    }

    split_by_post = {str(session["post_id"]): str(session["split"]) for session in sessions}
    assert all(message["split"] == split_by_post[str(message["post_id"])] for message in messages)
    assert all(
        message["comment_label"] is None
        for message in messages
        if message["split"] in {"train", "dev"}
    )
    assert all(
        message["comment_label"] in {0, 1}
        for message in messages
        if message["split"] == "test"
    )
    assert all("secret-user-id" not in json.dumps(message) for message in messages)
    assert any(message["parent_resolved"] for message in messages)

    cleaning = report["comment_cleaning"]
    assert cleaning["exact_duplicate_rows_removed"] == 1
    assert cleaning["conflicting_comment_ids"] == ["conflict-id"]
    assert cleaning["conflicting_rows_removed"] == 2
    assert cleaning["orphan_rows_removed"] == 1
    assert report["records"]["comment_labels_visible_by_split"]["train"] == 0
    assert report["records"]["comment_labels_visible_by_split"]["dev"] == 0
    profile = report["descriptive_profile"]
    assert profile["sessions"] == 12
    assert profile["session_labels"] == {"CB": 6, "Non-CB": 6}
    assert profile["comments_per_session"]["sessions_without_comments"] == 0


def test_download_is_pinned_hashed_and_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "posts.csv": b"post_id,post_content,label,cyberbullying_severity\np1,x,CB,high\n",
        "comments.csv": (
            b"comment_id,label,post_id,comment_time,comment_content,to_id\n"
            b"c1,CB,p1,1,x,\n"
        ),
    }
    calls: list[str] = []

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        return payloads[url.rsplit("/", 1)[-1]]

    monkeypatch.setattr(data, "_fetch_bytes", fake_fetch)
    manifest = download_sccd(tmp_path)
    cached_manifest = download_sccd(tmp_path)

    assert manifest["commit"] == SCCD_COMMIT
    assert all(SCCD_COMMIT in url for url in calls)
    assert len(calls) == 2
    assert cached_manifest["cached"] is True
    for entry in manifest["files"]:
        expected = hashlib.sha256(payloads[entry["name"]]).hexdigest()
        assert entry["sha256"] == expected
