"""Evidence bindings and additive output protection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
OUT = PROJECT / "benchmark/research_followup/20261004/shorttext_expansion"
METHODS = ("lda", "btm", "gsdmm", "nmf", "bertopic", "cst_mil_r0")


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def identity(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


def read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".pending")
    tmp.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    tmp.replace(path)


def source_hashes(paths):
    return {str(Path(p).resolve()): digest(p) for p in sorted(set(map(Path, paths))) if p.is_file()}


def verify_sources(mapping):
    changed = [p for p, sha in mapping.items() if not Path(p).is_file() or digest(p) != sha]
    if changed:
        raise RuntimeError(f"Participating files changed: {changed[:5]}")


def bind_or_verify(path, expected):
    path = Path(path)
    if path.exists():
        if read(path) != expected:
            raise ValueError("Source, cohort, configuration or runtime fingerprint changed")
    else:
        save(path, expected)


def guard_output(path):
    path = Path(path).resolve()
    protected = [
        PROJECT / name
        for name in (
            "data",
            "benchmark/data",
            "benchmark/runs",
            "benchmark/comparison",
            "benchmark/research_followup/20260922",
            "benchmark/research_followup/20260929",
            "benchmark/research_followup/20260930",
        )
    ]
    if any(path == p.resolve() or p.resolve() in path.parents for p in protected):
        raise ValueError("Output would overwrite frozen evidence")
    return path


def preservation():
    path = OUT.parent / "preservation_all.json"
    if path.exists():
        mapping = read(path)
        verify_sources(mapping)
        return {"files": len(mapping), "unchanged": True}
    roots = [
        PROJECT / name
        for name in (
            "data",
            "benchmark/data",
            "benchmark/runs/full677",
            "benchmark/comparison/full677",
            "benchmark/research_followup/20260922",
            "benchmark/research_followup/20260929",
            "benchmark/research_followup/20260930",
        )
    ]
    paths = [
        p
        for root in roots
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".lock", ".tmp"}
    ]
    paths += [
        p
        for p in (PROJECT / "src").rglob("*.py")
        if not any(n in p.parts for n in ("shorttext", "multimodel"))
    ]
    paths += [
        p
        for p in (PROJECT / "reports").rglob("*")
        if p.is_file() and not any(n in p.parts for n in ("完整框架_20261004", "20261004"))
    ]
    paths += [
        PROJECT / "pyproject.toml",
        PROJECT / "uv.lock",
        PROJECT.parent / "专利交底书模板.docx",
        PROJECT.parent / "任务要求.docx",
    ]
    mapping = source_hashes(paths)
    save(path, mapping)
    return {"files": len(mapping), "unchanged": True, "initial": True}
