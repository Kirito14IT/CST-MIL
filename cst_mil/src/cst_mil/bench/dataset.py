"""Nested immutable SCCD cohorts and isolated five-session engineering fixture."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from cst_mil.data import (
    _build_records,
    _cached_download_is_valid,
    _clean_comments,
    validate_raw_schema,
)
from cst_mil.preprocess import canonical_binary_label, normalize_text

from .common import file_hash, fingerprint, read_json, read_jsonl, write_json, write_jsonl

PROJECT = Path(__file__).resolve().parents[3]
BENCHMARK = PROJECT / "benchmark"
SPLIT_ORDER = {"train": 0, "dev": 1, "test": 2}


def stable_key(post_id: str) -> str:
    return fingerprint({"protocol": "sccd-full-v1", "seed": 42, "post_id": str(post_id)})


def _save_suite(name: str, sessions: list[dict], messages: list[dict], extra: dict) -> dict:
    destination = BENCHMARK / "data" / name
    sessions = sorted(sessions, key=lambda r: (SPLIT_ORDER[r["split"]], stable_key(r["post_id"])))
    for index, session in enumerate(sessions):
        session["inference_index"] = index
    lookup = {s["post_id"]: s for s in sessions}
    messages = sorted(messages, key=lambda m: (lookup[m["post_id"]]["inference_index"], m["order"]))
    manifest = {
        "schema_version": 1,
        "suite": name,
        "unit": "complete_post_id_session",
        "seed": 42,
        "n_sessions": len(sessions),
        "n_comments": len(messages),
        "split_counts": dict(Counter(s["split"] for s in sessions)),
        "class_counts": {
            split: dict(Counter(str(s["label"]) for s in sessions if s["split"] == split))
            for split in SPLIT_ORDER
        },
        "order": [s["post_id"] for s in sessions],
        "training_supervision": "session labels only",
        "comment_labels_available_in": ["test"],
        **extra,
    }
    manifest["content_fingerprint"] = fingerprint(
        {"sessions": sessions, "messages": messages, "manifest": manifest}
    )
    if destination.exists():
        existing, _, _ = load_suite(name)
        if existing["content_fingerprint"] != manifest["content_fingerprint"]:
            raise ValueError(f"Immutable cohort differs: {destination}; use a new suite name")
        return existing
    destination.mkdir(parents=True)
    write_jsonl(destination / "sessions.jsonl", sessions)
    write_jsonl(destination / "messages.jsonl", messages)
    manifest["files_sha256"] = {
        name: file_hash(destination / name) for name in ("sessions.jsonl", "messages.jsonl")
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


def load_suite(name: str) -> tuple[dict, list[dict], list[dict]]:
    if name not in {"pilot300", "full677", "smoke5"}:
        raise ValueError(f"Unknown suite {name}")
    destination = BENCHMARK / "data" / name
    manifest = read_json(destination / "manifest.json")
    for filename, expected in manifest["files_sha256"].items():
        if file_hash(destination / filename) != expected:
            raise ValueError(f"Dataset fingerprint mismatch: {destination / filename}")
    sessions = read_jsonl(destination / "sessions.jsonl")
    messages = read_jsonl(destination / "messages.jsonl")
    payload = {
        key: value
        for key, value in manifest.items()
        if key not in {"files_sha256", "content_fingerprint"}
    }
    if (
        fingerprint({"sessions": sessions, "messages": messages, "manifest": payload})
        != manifest["content_fingerprint"]
    ):
        raise ValueError("Cohort manifest/content fingerprint mismatch")
    for filename, expected in manifest["source_hashes"].items():
        if file_hash(PROJECT / "data/raw" / filename) != expected:
            raise ValueError(f"Raw source fingerprint mismatch: {filename}")
    if file_hash(PROJECT / "data/prepared/splits.csv") != manifest["original_split_sha256"]:
        raise ValueError("Original pilot split fingerprint mismatch")
    ids = {s["post_id"] for s in sessions}
    if len(ids) != len(sessions) or len(ids) != manifest["n_sessions"]:
        raise ValueError("Duplicate or missing sessions")
    lookup = {s["post_id"]: s for s in sessions}
    mids = set()
    for message in messages:
        key = message["comment_id"]
        if key in mids or message["post_id"] not in ids:
            raise ValueError("Duplicate message or unknown parent session")
        mids.add(key)
        if message["split"] != lookup[message["post_id"]]["split"]:
            raise ValueError("Message/session split mismatch")
        if message["split"] != "test" and message.get("comment_label") is not None:
            raise ValueError("Message labels must not be exposed outside test")
    return manifest, sessions, messages


def prepare_suites() -> dict:
    raw = PROJECT / "data" / "raw"
    source_manifest = read_json(raw / "download_manifest.json")
    if not _cached_download_is_valid(raw, source_manifest):
        raise ValueError("Pinned raw SCCD hashes do not match")
    posts, comments = validate_raw_schema(raw)
    old_sessions = read_jsonl(PROJECT / "data" / "prepared" / "sessions.jsonl")
    old = {s["post_id"]: s for s in old_sessions}
    if len(old) != 300 or Counter(s["split"] for s in old.values()) != {
        "train": 180,
        "dev": 60,
        "test": 60,
    }:
        raise ValueError("Expected unchanged original 300-session split")
    posts["_post_id"] = posts["post_id"].astype(str)
    mapped = posts["label"].map(canonical_binary_label)
    posts["_label"] = mapped.map(lambda x: x[0])
    posts["_label_name"] = mapped.map(lambda x: x[1])
    if len(posts) != 677:
        raise ValueError("Expected pinned SCCD 677 posts")
    cleaned, cleaning = _clean_comments(comments, set(posts["_post_id"]))
    if len(cleaned) != 38872:
        raise ValueError("Unexpected cleaned comment count")
    assignment = {pid: s["split"] for pid, s in old.items()}
    target = {"train": {1: 212, 0: 194}, "dev": {1: 71, 0: 64}, "test": {1: 71, 0: 65}}
    # Place exact duplicate roots beside their existing anchored original session.
    by_text: dict[str, list[str]] = defaultdict(list)
    for row in posts.to_dict("records"):
        value = normalize_text(row["post_content"])
        if value:
            by_text[value].append(row["_post_id"])
    duplicate_groups = []
    for group in by_text.values():
        if len(group) < 2:
            continue
        duplicate_groups.append(sorted(group))
        existing_splits = {assignment[p] for p in group if p in assignment}
        if len(existing_splits) > 1:
            raise ValueError("Existing split has exact duplicate roots across partitions")
        if existing_splits:
            split = next(iter(existing_splits))
            for pid in group:
                assignment[pid] = split
    labels = dict(zip(posts["_post_id"], posts["_label"], strict=True))
    for label in (0, 1):
        remaining = sorted(
            [p for p in labels if labels[p] == label and p not in assignment], key=stable_key
        )
        for split in ("train", "dev", "test"):
            need = target[split][label] - sum(
                labels[p] == label and v == split for p, v in assignment.items()
            )
            if need < 0 or need > len(remaining):
                raise ValueError("Cannot satisfy nested full split quotas")
            for pid in remaining[:need]:
                assignment[pid] = split
            remaining = remaining[need:]
        if remaining:
            raise ValueError("Unassigned SCCD rows")
    posts["_split"] = posts["_post_id"].map(assignment)
    sessions_raw, messages_raw, _ = _build_records(posts, cleaned)
    sessions = [s.as_dict() for s in sessions_raw]
    messages = [m.as_dict() for m in messages_raw]
    user_by_post = dict(zip(posts["_post_id"], posts["user_id"], strict=True))
    user_by_message = dict(zip(cleaned["_comment_id"], cleaned["user_id"], strict=True))
    for session in sessions:
        session["speaker_id"] = user_by_post.get(session["post_id"]) or None
        session["pilot_member"] = session["post_id"] in old
        session["fresh_test"] = session["split"] == "test" and not session["pilot_member"]
    for message in messages:
        message["speaker_id"] = user_by_message.get(message["comment_id"]) or None
    extra = {
        "source_commit": source_manifest["commit"],
        "source_hashes": {e["name"]: e["sha256"] for e in source_manifest["files"]},
        "original_split_sha256": file_hash(PROJECT / "data/prepared/splits.csv"),
        "cleaning": cleaning,
        "exact_duplicate_root_groups": duplicate_groups,
        "split_caveat": (
            "Existing pilot has shared users/repeated comments; "
            "post separation is not a blanket no-leakage claim."
        ),
    }
    full = _save_suite("full677", sessions, messages, extra)
    pilot = _save_suite(
        "pilot300",
        [dict(s) for s in sessions if s["post_id"] in old],
        [dict(m) for m in messages if m["post_id"] in old],
        extra,
    )
    if pilot["n_comments"] != 17312:
        raise ValueError("Pilot comment membership differs from legacy")
    # All five smoke source conversations are from the formal training region.
    pool = sorted(
        [s for s in sessions if s["split"] == "train" and 25 <= s["message_count"] <= 80],
        key=lambda s: stable_key(s["post_id"]),
    )
    selected = []
    taken = set()
    for split, label in [("train", 0), ("train", 1), ("dev", 0), ("dev", 1), ("test", 1)]:
        source = next(s for s in pool if s["label"] == label and s["post_id"] not in taken)
        row = dict(source)
        row.update(split=split, source_full_split="train", fresh_test=False)
        selected.append(row)
        taken.add(source["post_id"])
    smoke_split = {s["post_id"]: s["split"] for s in selected}
    truths = dict(zip(cleaned["_comment_id"], cleaned["label"], strict=True))
    smoke_messages = []
    for message in messages:
        if message["post_id"] in taken:
            row = dict(message)
            row["split"] = smoke_split[row["post_id"]]
            label, name = canonical_binary_label(truths[row["comment_id"]])
            row["comment_label"] = label if row["split"] == "test" else None
            row["comment_label_name"] = name if row["split"] == "test" else None
            smoke_messages.append(row)
    smoke = _save_suite(
        "smoke5",
        selected,
        smoke_messages,
        {**extra, "engineering_only": True, "source_full_split": "train"},
    )
    return {"pilot300": pilot, "full677": full, "smoke5": smoke}
