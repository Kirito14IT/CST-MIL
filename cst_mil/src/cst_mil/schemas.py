"""Schemas and serialisable records used by the SCCD data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCCD_COMMIT = "cf4015b802cabd651885211dc382152c1a270c31"
SCCD_REPOSITORY = "https://github.com/STAIR-BUPT/SCCD"
SCCD_RAW_BASE_URL = (
    "https://raw.githubusercontent.com/STAIR-BUPT/SCCD/"
    f"{SCCD_COMMIT}"
)
SCCD_SOURCE_FILES = ("posts.csv", "comments.csv")

POST_REQUIRED_COLUMNS = (
    "post_id",
    "post_content",
    "label",
    "cyberbullying_severity",
)
COMMENT_REQUIRED_COLUMNS = (
    "comment_id",
    "label",
    "post_id",
    "comment_time",
    "comment_content",
    "to_id",
)


class SCCDSchemaError(ValueError):
    """Raised when SCCD input files do not satisfy the expected schema."""


@dataclass(frozen=True)
class DownloadedFile:
    """Integrity metadata for one downloaded source file."""

    name: str
    url: str
    path: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SessionRecord:
    """One selected SCCD post and its chronological comment conversation."""

    post_id: str
    label: int
    label_name: str
    split: str
    severity: str
    post_raw_text: str
    post_model_text: str
    raw_text: str
    model_text: str
    message_ids: list[str]
    message_count: int
    reply_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MessageRecord:
    """One comment instance used by MIL and, on test only, evidence evaluation."""

    comment_id: str
    post_id: str
    split: str
    order: int
    comment_time: str
    parent_id: str | None
    parent_resolved: bool
    raw_text: str
    model_text: str
    context_model_text: str
    comment_label: int | None
    comment_label_name: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def source_paths(data_dir: Path) -> dict[str, Path]:
    """Return the expected local paths for the two pinned SCCD files."""

    return {name: data_dir / name for name in SCCD_SOURCE_FILES}
