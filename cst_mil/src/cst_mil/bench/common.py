"""Shared immutable inputs and serialization for the benchmark protocol."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import jieba
import numpy as np

from cst_mil.preprocess import normalize_text

STOPWORDS = frozenset(
    (
        "的 了 和 是 在 就 都 也 我 你 他 她 它 我们 你们 他们 一个 这个 那个 这 那 有 不 吗 "
        "啊 呢 吧 呀 哦 嗯 啦 呀呀 啊啊啊 哈 哈哈 哈哈哈 哈哈哈哈 哈哈哈哈哈 呵呵 嘻嘻 嘿嘿 "
        "什么 怎么 为什么 "
        "还是 就是 不是 这么 那么 但是 可是 因为 所以 一下 一些 已经 真的 真 很 太 更 最 会 可以 "
        "没有 觉得 看到 看看 说 去 来 好 关注 共同 资讯 热点 支持 博主 转发 微博 谢谢 下午 晚上 "
        "早上 大家 一起 永远 持续 越来越 mention url reply"
    ).split()
)
_TOKEN = re.compile(r"^[\u3400-\u9fffA-Za-z0-9_]+$")
_LAUGH = re.compile(r"^(?:哈|呵|嘿|嘻|啊|呀|哦|嗯|啦){2,}$")


def topic_tokens(text: str) -> list[str]:
    text = re.sub(r"<[^>]+>|\[[^\]]{1,12}\]", " ", str(text).lower())
    result = []
    for token in jieba.lcut(text, cut_all=False):
        token = token.strip()
        if (
            token
            and token not in STOPWORDS
            and _TOKEN.fullmatch(token)
            and not _LAUGH.fullmatch(token)
            and not token.isdecimal()
        ):
            result.append(token)
    return result


def normalise_rows(values: Any) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("Topic matrix must be two dimensional")
    if not np.isfinite(matrix).all() or np.any(matrix < -1e-10):
        raise ValueError("Non-finite or negative topic values")
    matrix = np.maximum(matrix, 0)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 1e-15)


def canonical_messages(session: dict, messages: list[dict]) -> list[dict]:
    post_id = str(session["post_id"])
    rows = [dict(m) for m in messages if str(m.get("post_id", post_id)) == post_id]
    rows.sort(key=lambda m: (float(m.get("order", 0)), str(m.get("comment_id", ""))))
    if not rows:
        raw = str(session.get("post_raw_text", session.get("raw_text", "")))
        rows = [
            {
                "post_id": post_id,
                "comment_id": post_id + ":root",
                "order": 0,
                "raw_text": raw,
                "model_text": normalize_text(raw),
                "parent_id": None,
                "speaker_id": session.get("speaker_id"),
                "comment_time": session.get("publish_time"),
                "is_root": True,
            }
        ]
    for i, row in enumerate(rows):
        row.setdefault("post_id", post_id)
        row.setdefault("comment_id", f"{post_id}:message:{i}")
        row.setdefault("raw_text", str(row.get("text", "")))
        row.setdefault("model_text", normalize_text(row["raw_text"]))
        row.setdefault("speaker_id", row.get("speaker"))
        row.setdefault("order", i)
        row.setdefault("parent_id", None)
        row.setdefault("comment_time", None)
    return rows


def build_windows(
    session: dict, messages: list[dict], size: int = 5, stride: int = 2
) -> list[dict]:
    if size < 1 or stride < 1 or stride > size:
        raise ValueError("Require 1 <= stride <= window size")
    rows = canonical_messages(session, messages)
    starts = list(range(0, max(1, len(rows) - size + 1), stride))
    tail = max(0, len(rows) - size)
    if tail not in starts:
        starts.append(tail)
    windows = []
    for start in starts:
        group = rows[start : start + size]
        # Deduplicate repeated text inside the representation, never its evidence IDs.
        texts = list(dict.fromkeys(str(m["model_text"]) for m in group))
        windows.append(
            {
                "window_id": f"{session['post_id']}:w{size}:{start}",
                "post_id": str(session["post_id"]),
                "text": "\n".join(texts),
                "message_ids": [str(m["comment_id"]) for m in group],
                "start": start,
                "end": start + len(group) - 1,
                "size": size,
            }
        )
    return windows


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                json_safe(value), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )
        handle.flush()
        import os

        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
