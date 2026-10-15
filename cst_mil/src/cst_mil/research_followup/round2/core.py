"""Immutable protocol and additive artifact boundaries for round two."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cst_mil.research_followup.io import METHODS, PROJECT, digest, read
from cst_mil.research_followup.io import save as _save

OUT = PROJECT / "benchmark/research_followup/20260929"
OLD = PROJECT / "benchmark/research_followup/20260922"
RUNS = PROJECT / "benchmark/runs/full677"


def stable(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def guard_output(path: Path) -> Path:
    path = Path(path).resolve()
    protected = [
        OLD,
        PROJECT / "benchmark/runs",
        PROJECT / "benchmark/data",
        PROJECT / "benchmark/comparison",
        PROJECT / "data",
    ]
    if any(path == p.resolve() or p.resolve() in path.parents for p in protected):
        raise ValueError(f"Output points to protected history: {path}")
    return path


def save(path: Path, value: object) -> None:
    _save(guard_output(path), value)


def jsonl_read(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_write(path: Path, rows: list[dict]) -> None:
    path = guard_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def source_hashes(paths) -> dict:
    return {str(Path(p).resolve()): digest(Path(p)) for p in sorted(set(map(Path, paths)))}


def verify_sources(mapping: dict) -> None:
    changed = [p for p, sha in mapping.items() if not Path(p).is_file() or digest(Path(p)) != sha]
    if changed:
        raise RuntimeError(f"Participating source or artifact changed: {changed[:5]}")


def protocol() -> dict:
    return dict(
        protocol_id="cst-round2-20260929-v1",
        seed=42,
        seeds=[42, 43, 44, 45, 46],
        methods=METHODS,
        cpu_threads=4,
        gemini=dict(
            model="gemini-2.5-flash",
            n_dev=6,
            n_test=30,
            budget_usd=2,
            max_attempts=2,
            max_tokens=2048,
            threshold=0.5,
            salt="20260929",
            sampling="label x training text-length tertile; hash order",
            max_consecutive_infrastructure_failures=3,
        ),
        robustness=dict(
            n_configs=110,
            bootstrap=10000,
            margin=0.01,
            contrast_family_size=22,
            instance_weight="explicit_bag_weight_only",
            session_weight="balanced",
            selection="mean dev AP, then mean dev AUC, then simpler heads",
        ),
    )


def initialize(out: Path = OUT, *, preserve=True) -> dict:
    out = guard_output(out)
    expected = protocol()
    path = out / "protocol.json"
    if path.exists() and read(path) != expected:
        raise ValueError("Frozen protocol differs; use a revision directory")
    if not path.exists():
        save(path, expected)
    if preserve:
        check_preservation(out)
    return expected


def check_preservation(out: Path = OUT) -> dict:
    path = out / "preservation.json"
    if path.exists():
        records = read(path)
        verify_sources(records)
    else:
        files = [
            p
            for p in OLD.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and not any(x.startswith("test_tmp") for x in p.parts)
        ]
        for folder in [RUNS, PROJECT / "benchmark/data/full677", PROJECT / "src/cst_mil/bench"]:
            files += [p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        files += list((PROJECT / "src/cst_mil/research_followup").glob("*.py"))
        files += [
            PROJECT / "pyproject.toml",
            PROJECT / "uv.lock",
            PROJECT.parent / ".gitignore",
            PROJECT.parent / "待办事项.md",
        ]
        files += [p for p in (PROJECT / "reports/专题研究报告").glob("*") if p.is_file()]
        records = source_hashes(files)
        save(path, records)
    result = dict(protected_files=len(records), unchanged=True)
    save(out / "preservation_check.json", result)
    return result


def code_sources(*extra: Path) -> dict:
    paths = list((PROJECT / "src/cst_mil/bench").glob("*.py"))
    paths += [
        PROJECT / "src/cst_mil/preprocess.py",
        PROJECT / "src/cst_mil/metrics.py",
        Path(__file__),
        PROJECT / "src/cst_mil/research_followup/io.py",
        *extra,
    ]
    return source_hashes(paths)
