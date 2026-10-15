from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from cst_mil.bench.common import json_safe
from cst_mil.bench.dataset import PROJECT, load_suite

OUT = PROJECT / "benchmark/research_followup/20260922"
RUNS = PROJECT / "benchmark/runs/full677"
METHODS = ["lda", "btm", "gsdmm", "nmf", "bertopic", "cst_mil_r0"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def table(path: Path, data: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in data for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)


def inputs():
    manifest, sessions, messages = load_suite("full677")
    grouped = {s["post_id"]: [] for s in sessions}
    for message in messages:
        grouped[message["post_id"]].append(message)
    return manifest, sessions, messages, grouped


def protect(verify: bool = False) -> dict:
    destination = OUT / "preservation.json"
    if destination.exists():
        previous = read(destination)
        changed = [
            name
            for name, sha in previous.items()
            if not Path(name).exists() or digest(Path(name)) != sha
        ]
        if changed:
            raise RuntimeError(f"Protected files changed: {changed}")
        return {"protected_files": len(previous), "unchanged": True}
    if verify:
        raise FileNotFoundError("Preservation manifest missing")
    files = list((PROJECT / "src/cst_mil/bench").glob("*.py"))
    files += list((PROJECT / "benchmark/data/full677").glob("*"))
    files += list((PROJECT / "benchmark/comparison/full677").glob("*.*"))
    for method in METHODS:
        files += list((RUNS / method).glob("*.*"))
    files += [
        PROJECT / "pyproject.toml",
        PROJECT / "uv.lock",
        PROJECT.parent / "待办事项.md",
        PROJECT.parent / ".gitignore",
        PROJECT.parent / "任务要求.docx",
    ]
    record = {str(p.resolve()): digest(p) for p in files if p.is_file()}
    save(destination, record)
    return {"protected_files": len(record), "unchanged": True}
