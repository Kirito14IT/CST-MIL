"""Pinned SCCD download, validation, cleaning and leakage-safe preparation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import urllib.request
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocess import (
    as_clean_string,
    build_context_text,
    canonical_binary_label,
    comment_sort_key,
    normalise_identifier,
    normalize_text,
)
from .schemas import (
    COMMENT_REQUIRED_COLUMNS,
    POST_REQUIRED_COLUMNS,
    SCCD_COMMIT,
    SCCD_RAW_BASE_URL,
    SCCD_REPOSITORY,
    SCCD_SOURCE_FILES,
    DownloadedFile,
    MessageRecord,
    SCCDSchemaError,
    SessionRecord,
    source_paths,
)

DOWNLOAD_MANIFEST = "download_manifest.json"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch_bytes(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "cst-mil/0.1 SCCD research downloader"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _cached_download_is_valid(data_dir: Path, manifest: dict[str, Any] | None) -> bool:
    if not manifest or manifest.get("commit") != SCCD_COMMIT:
        return False
    entries = manifest.get("files")
    if not isinstance(entries, list):
        return False
    by_name = {entry.get("name"): entry for entry in entries if isinstance(entry, dict)}
    for name, path in source_paths(data_dir).items():
        entry = by_name.get(name)
        if not path.is_file() or not entry:
            return False
        if entry.get("sha256") != _sha256_file(path):
            return False
    return True


def download_sccd(data_dir: Path) -> dict[str, Any]:
    """Download the two SCCD CSV files from the one approved immutable commit.

    The function is idempotent.  Existing files are reused only when their current
    SHA256 hashes agree with a manifest for the pinned commit.
    """

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / DOWNLOAD_MANIFEST
    manifest = _read_json(manifest_path)
    if _cached_download_is_valid(data_dir, manifest):
        result = dict(manifest or {})
        result["cached"] = True
        return result

    downloaded: list[DownloadedFile] = []
    for name in SCCD_SOURCE_FILES:
        url = f"{SCCD_RAW_BASE_URL}/{name}"
        payload = _fetch_bytes(url)
        if not payload:
            raise RuntimeError(f"Downloaded empty SCCD source file: {url}")
        destination = data_dir / name
        _atomic_write_bytes(destination, payload)
        downloaded.append(
            DownloadedFile(
                name=name,
                url=url,
                path=str(destination.resolve()),
                sha256=_sha256_bytes(payload),
                size_bytes=len(payload),
            )
        )

    manifest = {
        "dataset": "SCCD",
        "repository": SCCD_REPOSITORY,
        "commit": SCCD_COMMIT,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "license_notice": (
            "The upstream repository did not expose a clear standalone LICENSE at "
            "the pinned revision. Use these files locally for research and do not "
            "redistribute them without confirming permission."
        ),
        "files": [item.as_dict() for item in downloaded],
        "cached": False,
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required SCCD file is missing: {path}")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.ParserError) as error:
        raise SCCDSchemaError(f"Could not parse {path.name}: {error}") from error
    frame.columns = [as_clean_string(column) for column in frame.columns]
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise SCCDSchemaError(f"{path.name} is missing required columns: {missing}")
    return frame


def validate_raw_schema(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SCCD source CSVs as strings and validate their required columns."""

    paths = source_paths(Path(raw_dir))
    posts = _read_csv(paths["posts.csv"], POST_REQUIRED_COLUMNS)
    comments = _read_csv(paths["comments.csv"], COMMENT_REQUIRED_COLUMNS)

    post_ids = posts["post_id"].map(normalise_identifier)
    if post_ids.isna().any():
        raise SCCDSchemaError("posts.csv contains an empty post_id")
    if post_ids.duplicated().any():
        duplicate_ids = sorted(post_ids[post_ids.duplicated(keep=False)].unique().tolist())
        raise SCCDSchemaError(f"posts.csv contains duplicate post_id values: {duplicate_ids[:10]}")
    for label in posts["label"]:
        try:
            canonical_binary_label(label)
        except ValueError as error:
            raise SCCDSchemaError(str(error)) from error
    return posts, comments


def _clean_comments(
    comments: pd.DataFrame,
    valid_post_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = comments.copy()
    input_rows = len(frame)

    exact_mask = frame.duplicated(keep="first")
    exact_rows_removed = int(exact_mask.sum())
    frame = frame.loc[~exact_mask].copy()

    frame["_comment_id"] = frame["comment_id"].map(normalise_identifier)
    invalid_id_mask = frame["_comment_id"].isna()
    invalid_id_rows_removed = int(invalid_id_mask.sum())
    frame = frame.loc[~invalid_id_mask].copy()

    duplicate_id_mask = frame["_comment_id"].duplicated(keep=False)
    conflicting_ids = sorted(frame.loc[duplicate_id_mask, "_comment_id"].unique().tolist())
    conflicting_rows_removed = int(duplicate_id_mask.sum())
    frame = frame.loc[~duplicate_id_mask].copy()

    frame["_post_id"] = frame["post_id"].map(normalise_identifier)
    orphan_mask = ~frame["_post_id"].isin(valid_post_ids)
    orphan_post_ids = sorted(
        value for value in frame.loc[orphan_mask, "_post_id"].dropna().unique().tolist()
    )
    orphan_rows_removed = int(orphan_mask.sum())
    frame = frame.loc[~orphan_mask].copy()

    report = {
        "input_rows": input_rows,
        "exact_duplicate_rows_removed": exact_rows_removed,
        "invalid_comment_id_rows_removed": invalid_id_rows_removed,
        "conflicting_comment_ids": conflicting_ids,
        "conflicting_comment_id_count": len(conflicting_ids),
        "conflicting_rows_removed": conflicting_rows_removed,
        "orphan_post_ids": orphan_post_ids,
        "orphan_post_id_count": len(orphan_post_ids),
        "orphan_rows_removed": orphan_rows_removed,
        "output_rows": len(frame),
    }
    return frame, report


def _balanced_sample_and_split(
    posts: pd.DataFrame,
    n_sessions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if n_sessions < 10 or n_sessions % 2:
        raise ValueError("n_sessions must be an even integer of at least 10")
    per_class = n_sessions // 2
    train_per_class = int(round(per_class * 0.60))
    dev_per_class = int(round(per_class * 0.20))
    test_per_class = per_class - train_per_class - dev_per_class
    if min(train_per_class, dev_per_class, test_per_class) < 1:
        raise ValueError("n_sessions is too small for a non-empty 60/20/20 stratified split")

    working = posts.copy()
    working["_post_id"] = working["post_id"].map(normalise_identifier)
    mapped = working["label"].map(canonical_binary_label)
    working["_label"] = mapped.map(lambda pair: pair[0])
    working["_label_name"] = mapped.map(lambda pair: pair[1])

    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []
    available: dict[str, int] = {}
    for label_value, label_name in ((0, "Non-CB"), (1, "CB")):
        candidates = working.loc[working["_label"] == label_value].sort_values("_post_id")
        available[label_name] = len(candidates)
        if len(candidates) < per_class:
            raise ValueError(
                f"SCCD has only {len(candidates)} {label_name} sessions, "
                f"but {per_class} were requested"
            )
        chosen_indices = rng.choice(candidates.index.to_numpy(), size=per_class, replace=False)
        chosen = candidates.loc[chosen_indices].copy()
        permuted_indices = rng.permutation(chosen.index.to_numpy())
        chosen = chosen.loc[permuted_indices].copy()
        chosen["_split"] = (
            ["train"] * train_per_class
            + ["dev"] * dev_per_class
            + ["test"] * test_per_class
        )
        selected_parts.append(chosen)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected = selected.sort_values(["_split", "_label", "_post_id"]).reset_index(drop=True)
    split_counts = {
        split: {
            name: int(
                ((selected["_split"] == split) & (selected["_label_name"] == name)).sum()
            )
            for name in ("CB", "Non-CB")
        }
        for split in ("train", "dev", "test")
    }
    return selected, {
        "seed": seed,
        "requested_sessions": n_sessions,
        "available_sessions": available,
        "selected_per_class": per_class,
        "split_counts": split_counts,
    }


def _build_records(
    selected_posts: pd.DataFrame,
    clean_comments: pd.DataFrame,
) -> tuple[list[SessionRecord], list[MessageRecord], dict[str, Any]]:
    selected_post_ids = set(selected_posts["_post_id"].tolist())
    selected_comments = clean_comments.loc[
        clean_comments["_post_id"].isin(selected_post_ids)
    ].copy()
    comment_groups = {
        post_id: group.copy()
        for post_id, group in selected_comments.groupby("_post_id", sort=False)
    }

    session_records: list[SessionRecord] = []
    message_records: list[MessageRecord] = []
    unresolved_parent_count = 0
    invalid_comment_labels = Counter()

    for post_data in selected_posts.to_dict(orient="records"):
        post_id = post_data["_post_id"]
        split = post_data["_split"]
        label = int(post_data["_label"])
        label_name = post_data["_label_name"]
        post_raw_text = as_clean_string(post_data["post_content"])
        post_model_text = normalize_text(post_raw_text)

        group = comment_groups.get(post_id, selected_comments.iloc[0:0].copy())
        if not group.empty:
            group["_sort_key"] = group.apply(
                lambda row: comment_sort_key(row["comment_time"], row["_comment_id"]),
                axis=1,
            )
            group = group.sort_values("_sort_key", kind="stable").copy()

        text_by_id = {
            row["_comment_id"]: normalize_text(row["comment_content"])
            for _, row in group.iterrows()
        }
        message_ids: list[str] = []
        session_raw_parts = [post_raw_text] if post_raw_text else []
        session_model_parts = [post_model_text] if post_model_text else []
        reply_count = 0

        for order, (_, row) in enumerate(group.iterrows()):
            comment_id = row["_comment_id"]
            parent_id = normalise_identifier(row["to_id"])
            parent_resolved = parent_id is not None and parent_id in text_by_id
            if parent_id is not None:
                reply_count += 1
                if not parent_resolved:
                    unresolved_parent_count += 1

            raw_text = as_clean_string(row["comment_content"])
            model_text = normalize_text(raw_text)
            context_text = build_context_text(
                model_text,
                text_by_id[parent_id] if parent_resolved and parent_id else None,
            )
            comment_label: int | None = None
            comment_label_name: str | None = None
            if split == "test":
                try:
                    comment_label, comment_label_name = canonical_binary_label(row["label"])
                except ValueError:
                    invalid_comment_labels[as_clean_string(row["label"])] += 1

            message_ids.append(comment_id)
            if raw_text:
                session_raw_parts.append(raw_text)
            if model_text:
                session_model_parts.append(model_text)
            message_records.append(
                MessageRecord(
                    comment_id=comment_id,
                    post_id=post_id,
                    split=split,
                    order=order,
                    comment_time=as_clean_string(row["comment_time"]),
                    parent_id=parent_id,
                    parent_resolved=parent_resolved,
                    raw_text=raw_text,
                    model_text=model_text,
                    context_model_text=context_text,
                    comment_label=comment_label,
                    comment_label_name=comment_label_name,
                )
            )

        session_records.append(
            SessionRecord(
                post_id=post_id,
                label=label,
                label_name=label_name,
                split=split,
                severity=as_clean_string(post_data["cyberbullying_severity"]),
                post_raw_text=post_raw_text,
                post_model_text=post_model_text,
                raw_text="\n".join(session_raw_parts),
                model_text="\n".join(session_model_parts),
                message_ids=message_ids,
                message_count=len(message_ids),
                reply_count=reply_count,
            )
        )

    build_report = {
        "selected_comment_rows": len(message_records),
        "selected_sessions_without_comments": sum(
            record.message_count == 0 for record in session_records
        ),
        "resolved_parent_count": sum(record.parent_resolved for record in message_records),
        "unresolved_parent_count": unresolved_parent_count,
        "invalid_test_comment_labels": dict(sorted(invalid_comment_labels.items())),
        "comment_labels_visible_by_split": {
            split: sum(
                record.comment_label is not None
                for record in message_records
                if record.split == split
            )
            for split in ("train", "dev", "test")
        },
    }
    return session_records, message_records, build_report


def _descriptive_profile(
    posts: pd.DataFrame,
    clean_comments: pd.DataFrame,
) -> dict[str, Any]:
    """Summarise observable sparsity and structure without using it as a feature."""

    lengths = (
        clean_comments["comment_content"]
        .fillna("")
        .astype(str)
        .str.replace(r"\s+", "", regex=True)
        .str.len()
        .astype(float)
    )
    counts = (
        clean_comments.groupby("_post_id")
        .size()
        .reindex(posts["_post_id"], fill_value=0)
        .astype(float)
    )
    post_labels = posts["label"].map(canonical_binary_label).map(lambda pair: pair[1])
    nonempty_parent = clean_comments["to_id"].map(normalise_identifier).notna()
    return {
        "sessions": int(len(posts)),
        "session_labels": {
            str(name): int(count) for name, count in post_labels.value_counts().sort_index().items()
        },
        "clean_comments": int(len(clean_comments)),
        "comment_length_characters_without_whitespace": {
            "mean": float(lengths.mean()),
            "median": float(lengths.median()),
            "q25": float(lengths.quantile(0.25)),
            "q75": float(lengths.quantile(0.75)),
            "between_5_and_20_count": int(lengths.between(5, 20, inclusive="both").sum()),
            "between_5_and_20_fraction": float(
                lengths.between(5, 20, inclusive="both").mean()
            ),
            "at_most_20_fraction": float((lengths <= 20).mean()),
            "over_50_fraction": float((lengths > 50).mean()),
        },
        "comments_per_session": {
            "mean": float(counts.mean()),
            "median": float(counts.median()),
            "q25": float(counts.quantile(0.25)),
            "q75": float(counts.quantile(0.75)),
            "sessions_without_comments": int((counts == 0).sum()),
        },
        "nonempty_reply_pointer_rows": int(nonempty_parent.sum()),
    }


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    _atomic_write_text(path, payload)


def prepare_dataset(
    raw_dir: Path,
    prepared_dir: Path,
    seed: int = 42,
    n_sessions: int = 300,
) -> dict[str, Any]:
    """Validate, clean and prepare one deterministic balanced SCCD pilot split."""

    raw_dir = Path(raw_dir)
    prepared_dir = Path(prepared_dir)
    raw_manifest = _read_json(raw_dir / DOWNLOAD_MANIFEST)
    if not _cached_download_is_valid(raw_dir, raw_manifest):
        raise ValueError(
            "SCCD raw files are not verified against the pinned download manifest; "
            "run download-data first"
        )
    prepared_dir.mkdir(parents=True, exist_ok=True)
    posts, comments = validate_raw_schema(raw_dir)

    posts = posts.copy()
    posts["_post_id"] = posts["post_id"].map(normalise_identifier)
    valid_post_ids = set(posts["_post_id"].tolist())
    clean_comments, comment_cleaning = _clean_comments(comments, valid_post_ids)
    selected_posts, selection_report = _balanced_sample_and_split(posts, n_sessions, seed)
    sessions, messages, build_report = _build_records(selected_posts, clean_comments)

    session_dicts = [record.as_dict() for record in sessions]
    message_dicts = [record.as_dict() for record in messages]
    _write_jsonl(prepared_dir / "sessions.jsonl", session_dicts)
    _write_jsonl(prepared_dir / "messages.jsonl", message_dicts)

    splits = pd.DataFrame(
        [
            {
                "post_id": record.post_id,
                "label": record.label,
                "label_name": record.label_name,
                "split": record.split,
            }
            for record in sessions
        ]
    ).sort_values(["split", "label", "post_id"])
    split_path = prepared_dir / "splits.csv"
    temporary_split = split_path.with_name(f".{split_path.name}.part")
    splits.to_csv(temporary_split, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(temporary_split, split_path)

    source_hashes = {
        name: _sha256_file(raw_dir / name)
        for name in SCCD_SOURCE_FILES
        if (raw_dir / name).is_file()
    }
    report = {
        "dataset": "SCCD",
        "source_commit": SCCD_COMMIT,
        "source_hashes": source_hashes,
        "source_manifest_present": raw_manifest is not None,
        "source_manifest_verified": True,
        "raw_posts": len(posts),
        "raw_comments": len(comments),
        "comment_cleaning": comment_cleaning,
        "descriptive_profile": _descriptive_profile(posts, clean_comments),
        "selection": selection_report,
        "records": build_report,
        "outputs": {
            "sessions": str((prepared_dir / "sessions.jsonl").resolve()),
            "messages": str((prepared_dir / "messages.jsonl").resolve()),
            "splits": str(split_path.resolve()),
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "leakage_guard": {
            "split_unit": "post_id",
            "training_supervision": "session label only",
            "comment_labels_available_in": ["test"],
            "excluded_feature_columns": [
                "comment.label",
                "comment.expression",
                "comment.sarcasm",
                "comment.target",
                "comment.group category",
                "user_id",
                "like_num",
                "comment_num",
                "repost_num",
            ],
        },
    }
    report_path = prepared_dir / "cleaning_report.json"
    report["outputs"]["cleaning_report"] = str(report_path.resolve())
    _atomic_write_text(
        report_path,
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return report
