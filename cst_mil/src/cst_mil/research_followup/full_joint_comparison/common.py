"""Atomic evidence I/O, immutable identities and process-tree CPU/RSS evidence."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = PROJECT / "benchmark/research_followup/20261014/full_joint_comparison"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.pending")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.pending")
    with temp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def verify_hashes(mapping: dict[str, str]) -> None:
    changed = [p for p, sha in mapping.items() if not Path(p).is_file() or digest(Path(p)) != sha]
    if changed:
        raise RuntimeError(f"Frozen files changed; stop affected stage: {changed[:10]}")


def source_hashes(paths) -> dict[str, str]:
    return {str(p.resolve()): digest(p) for p in sorted(set(map(Path, paths))) if p.is_file()}


def participating_code() -> dict[str, str]:
    roots = [
        PROJECT / "src/cst_mil/unified_semantic_v1_1",
        PROJECT / "src/cst_mil/research_followup/shorttext_nmf_stable",
        PROJECT / "src/cst_mil/research_followup/shorttext",
        PROJECT / "src/cst_mil/bench",
    ]
    paths = [p for root in roots for p in root.glob("*.py")]
    paths += [
        Path(__file__),
        Path(__file__).with_name("__init__.py"),
        Path(__file__).with_name("execution.py"),
        Path(__file__).with_name("__main__.py"),
        Path(__file__).with_name("cost.py"),
    ]
    paths += [Path(__file__).with_name(n) for n in ("data.py", "model.py", "topic_backend.py")]
    paths += [
        PROJECT / "src/cst_mil/metrics.py",
        PROJECT / "src/cst_mil/preprocess.py",
        PROJECT / "pyproject.toml",
        PROJECT / "uv.lock",
    ]
    for root in (
        PROJECT.parent / "baseline/LDA",
        PROJECT.parent / "baseline/BTM",
        PROJECT.parent / "baseline/GSDMM",
    ):
        paths += [p for p in root.rglob("*") if p.suffix in {".cpp", ".c", ".h", ".java", ".jar"}]
    native = PROJECT / "benchmark/tools/native"
    paths += [
        p
        for p in native.rglob("*")
        if p.is_file() and (p.suffix in {".exe", ".class"} or p.name == "build.json")
    ]
    for runtime in (PROJECT / "benchmark/tools").glob("jdk-*"):
        paths += [
            runtime / name
            for name in ("bin/java.exe", "bin/server/jvm.dll", "lib/modules", "release")
        ]
    return source_hashes(paths)


def guard_output(path: Path) -> Path:
    path = Path(path).resolve()
    protected = [
        PROJECT / "data",
        PROJECT / "benchmark/data",
        PROJECT / "benchmark/runs",
        PROJECT / "benchmark/comparison",
        PROJECT / "reports",
    ]
    protected += [
        PROJECT / f"benchmark/research_followup/{day}"
        for day in ("20260922", "20260929", "20260930", "20261004", "20261005")
    ]
    if any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in protected):
        raise ValueError("Output must not overwrite frozen data, experiments or manuscripts")
    return path


def preservation(output: Path, *, verify: bool = False) -> dict:
    """Hash pre-existing scientific evidence, excluding caches and test fixtures."""
    manifest = output / "preservation_before.json"
    if manifest.exists():
        frozen = read(manifest)
        if verify:
            verify_hashes(frozen)
            result = {"checked_files": len(frozen), "unchanged": True}
            save(output / "preservation_after.json", result)
            return result
        return {"checked_files": len(frozen), "existing_snapshot": True}
    if verify:
        raise ValueError("No original preservation snapshot")
    roots = [
        PROJECT / "data",
        PROJECT / "benchmark/data",
        PROJECT / "benchmark/runs/full677",
        PROJECT / "benchmark/comparison/full677",
        PROJECT / "reports",
    ]
    roots += [
        PROJECT / f"benchmark/research_followup/{day}"
        for day in ("20260922", "20260929", "20260930", "20261004", "20261005")
    ]

    def eligible(p: Path) -> bool:
        return (
            p.is_file()
            and not any(
                part.startswith(("_pytest", "pytest-", "test_tmp", "pytest_"))
                or part in {"__pycache__", ".venv", "node_modules", ".cache", "tools"}
                for part in p.parts
            )
            and p.suffix
            not in {".pyc", ".lock", ".tmp", ".pending", ".log", ".sqlite-shm", ".sqlite-wal"}
        )

    paths = [p for root in roots if root.exists() for p in root.rglob("*") if eligible(p)]
    paths += [
        p
        for p in (PROJECT / "src").rglob("*.py")
        if "full_joint_comparison" not in p.parts and "__pycache__" not in p.parts
    ]
    paths += [PROJECT.parent / "计算机项目综合开发创新实践/app/config/deployment.json"]
    mapping = source_hashes(paths)
    save(manifest, mapping)
    return {"checked_files": len(mapping), "snapshot_created": True}


@contextmanager
def campaign_lock(output: Path):
    """Refuse concurrent training/cost runners; safely recover a dead process lock."""
    import psutil

    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.lock"
    active_child = output / "active_child.json"
    if active_child.exists():
        child = read(active_child)
        try:
            process = psutil.Process(child["pid"])
            alive = abs(process.create_time() - child["created"]) < 0.1
        except psutil.Error:
            alive = False
        if alive:
            raise RuntimeError(f"An earlier stage still runs in process {child['pid']}")
    if path.exists():
        old = read(path)
        try:
            proc = psutil.Process(old["pid"])
            active = abs(proc.create_time() - old["created"]) < 0.1
        except psutil.Error:
            active = False
        if active:
            raise RuntimeError(f"Campaign already running in process {old['pid']}")
        path.unlink()
    proc = psutil.Process()
    with path.open("x", encoding="utf8") as handle:
        json.dump({"pid": proc.pid, "created": proc.create_time()}, handle)
    try:
        yield
    finally:
        if path.exists() and read(path).get("pid") == proc.pid:
            path.unlink()


class TreeMonitor:
    """Sample the complete benchmark process tree including transient JVMs."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak = 0
        self.max_affinity = 0
        self.max_children = 0
        self.samples = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self):
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        rss = 0
        self.max_children = max(self.max_children, len(processes) - 1)
        for process in processes:
            try:
                rss += process.memory_info().rss
                self.max_affinity = max(self.max_affinity, len(process.cpu_affinity()))
            except (psutil.Error, AttributeError):
                continue
        self.peak = max(self.peak, rss)
        self.samples += 1

    def _run(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self._sample()
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.thread.join()
        self._sample()
        self.seconds = time.perf_counter() - self.started

    def result(self) -> dict:
        return {
            "peak_tree_rss_bytes": self.peak,
            "peak_affinity_width": self.max_affinity,
            "peak_child_processes": self.max_children,
            "samples": self.samples,
            "sample_interval_seconds": self.interval,
        }
