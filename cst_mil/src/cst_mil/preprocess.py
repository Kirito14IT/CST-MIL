"""Text normalisation and deterministic helpers for SCCD preparation."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

_URL_RE = re.compile(r"(?i)(?:https?://|www\.)\S+")
_MENTION_RE = re.compile(r"@[\w\-\u3400-\u9fff]+")
_WHITESPACE_RE = re.compile(r"\s+")
_NULL_IDS = {"", "nan", "none", "null", "nat", "<na>"}


def as_clean_string(value: Any) -> str:
    """Convert a CSV scalar to a stable string without leaking pandas NaNs."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalise_identifier(value: Any) -> str | None:
    """Normalise optional identifiers while preserving their lexical identity."""

    text = as_clean_string(value)
    if text.casefold() in _NULL_IDS:
        return None
    return text


def normalize_text(value: Any) -> str:
    """Create model text while retaining punctuation, emoji and repeated characters.

    SCCD already removes part of the original social-media noise.  This function
    intentionally performs only conservative normalisation so that the character
    n-gram channel can still observe misspellings, repetitions and punctuation.
    """

    text = unicodedata.normalize("NFKC", as_clean_string(value)).lower()
    text = _URL_RE.sub("<url>", text)
    text = _MENTION_RE.sub("<mention>", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def canonical_binary_label(value: Any) -> tuple[int, str]:
    """Map SCCD's binary label spelling to a numeric and display label."""

    key = re.sub(r"[\s_]", "-", as_clean_string(value).casefold())
    key = re.sub(r"-+", "-", key)
    if key in {"cb", "1", "true", "yes"}:
        return 1, "CB"
    if key in {"non-cb", "noncb", "0", "false", "no"}:
        return 0, "Non-CB"
    raise ValueError(f"Unsupported SCCD binary label: {value!r}")


def comment_sort_key(comment_time: Any, comment_id: Any) -> tuple[float, str]:
    """Sort comments by relative time and then by stable lexical comment ID."""

    raw_time = as_clean_string(comment_time)
    try:
        numeric_time = float(raw_time)
        if math.isnan(numeric_time):
            numeric_time = math.inf
    except ValueError:
        numeric_time = math.inf
    return numeric_time, as_clean_string(comment_id)


def build_context_text(current_text: str, parent_text: str | None) -> str:
    """Combine a resolved parent comment and current comment for character features."""

    if parent_text:
        return f"{parent_text} <reply> {current_text}".strip()
    return current_text
