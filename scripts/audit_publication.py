"""Check publication paths, English documentation and deployment fingerprints.

Uses only the standard library. The default audit reads tracked working-tree
files. --include-untracked also checks non-ignored candidate files before staging.
--history checks result paths across all locally reachable commits; it does not
require old filenames or documentation to be English. --purge-manifest additionally
checks audited old blob IDs, including result-bearing versions of documentation.

This is a publication guard, not a content classifier. It cannot establish that
arbitrary text, media, archives, release assets or remote caches contain no results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CATALOGS = (
    "models/semantic-demo-2026.10.15.json",
    "cst_mil/src/cst_mil/deployment/catalog.json",
)
BENCHMARK_SOURCES = {
    "cst_mil/benchmark/finalize_delivery.py",
    "cst_mil/benchmark/preserve_legacy.py",
    "cst_mil/benchmark/validate_functional.py",
}
PRIVATE_PREFIXES = (
    "cst_mil/reports/", "cst_mil/releases/", "cst_mil/analysis/",
    "cst_mil/artifacts/", "benchmark/", ".runtime/",
)
RESULT_DIRECTORIES = {
    "runs", "results", "predictions", "per_case", "per-case", "raw_results",
    "raw-results", "experiment_results", "experiment-results",
    "benchmark_results", "benchmark-results", "evaluation_results",
    "evaluation-results", "experiment_outputs", "experiment-outputs",
}
PRIVATE_RECORDS = {"notes.md", "task_plan.md", "report_evidence.md"}
SOURCE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".mjs", ".mts", ".c", ".cpp", ".h", ".java", ".sh"}
# This reviewed fixture contains authored messages and display expectations.
# New fixture files do not automatically inherit this exception.
AUTHORED_FIXTURES = {"WebApp/tests/fixtures/latest-display-cases.json"}
RESULT_NAME = re.compile(
    r"(?:^|[_.-])(?:predictions?|per[_-]?case|metrics?|results?|scores?|"
    r"statistics|performance|comparisons?|costs?|timings?)(?:[_.-]|$)", re.I,
)
HAN = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U000323af]")
SHA256 = re.compile(r"[0-9a-f]{64}")


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def result_path_reason(name: str) -> str | None:
    path = PurePosixPath(name)
    if name.startswith("cst_mil/benchmark/"):
        return None if name in BENCHMARK_SOURCES else "private benchmark artifact"
    if name.startswith(PRIVATE_PREFIXES):
        return "private research or runtime directory"
    if path.name.lower() in PRIVATE_RECORDS:
        return "private process record"
    if any(part.lower() in RESULT_DIRECTORIES for part in path.parts[:-1]):
        return "generated-result directory"
    if name in AUTHORED_FIXTURES or path.suffix.lower() in SOURCE_SUFFIXES:
        return None
    if RESULT_NAME.search(path.stem):
        return "generated-result filename"
    return None


def tracked_files(include_untracked: bool) -> list[str]:
    args = ["ls-files", "-z", "--cached"]
    if include_untracked:
        args += ["--others", "--exclude-standard"]
    names = git(*args).decode("utf-8").split("\0")
    # Deleted files are omitted from a working-tree audit; CI checks a clean tree.
    return sorted({name for name in names if name and (ROOT / name).is_file()})


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def audit_catalogs(files: set[str], issues: list[str]) -> int:
    try:
        for name in CATALOGS:
            if name not in files:
                raise ValueError(f"Missing tracked catalog: {name}")
        public, packaged = (read_json(ROOT / name) for name in CATALOGS)
        if public != packaged:
            issues.append("Public and packaged deployment catalogs differ")
        fingerprints = public.get("codeFingerprints")
        if not isinstance(fingerprints, dict) or not fingerprints:
            raise ValueError("Catalog codeFingerprints must be a nonempty object")
        for name, expected in fingerprints.items():
            if (not isinstance(name, str) or "\\" in name or ":" in name
                    or PurePosixPath(name).is_absolute()
                    or any(part in {"", ".", ".."} for part in name.split("/"))):
                issues.append(f"Unsafe catalog source path: {name!r}")
                continue
            file = (ROOT / name).resolve()
            if name not in files or not file.is_relative_to(ROOT):
                issues.append(f"Catalog source is not a tracked repository file: {name}")
            elif not isinstance(expected, str) or not SHA256.fullmatch(expected):
                issues.append(f"Invalid catalog source SHA-256: {name}")
            elif hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                issues.append(f"Catalog source SHA-256 mismatch: {name}")
        return len(fingerprints)
    except (OSError, ValueError, TypeError) as error:
        issues.append(f"Catalog check failed: {error}")
        return 0


def audit_history(issues: list[str], purge_manifest: Path | None) -> int:
    if git("rev-parse", "--is-shallow-repository").strip() == b"true":
        issues.append("History audit requires a full clone (fetch all branches and tags)")
        return 0
    trees = set(git("log", "--all", "--format=%T").decode("ascii").splitlines())
    historical_paths: set[str] = set()
    for tree in trees:
        historical_paths.update(git("ls-tree", "-rz", "--name-only", tree).decode("utf-8").split("\0"))
    for name in sorted(historical_paths - {""}):
        if reason := result_path_reason(name):
            issues.append(f"History: {name}: {reason}")
    if purge_manifest is not None:
        manifest = read_json(purge_manifest)
        reachable = set(git("rev-list", "--objects", "--all", "--no-object-names").decode("ascii").splitlines())
        for item in manifest["purge"]:
            for blob in item["versions"]:
                if blob in reachable:
                    issues.append(f"History retains audited blob {blob}: {item['path']}")
    return len(trees)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-untracked", action="store_true")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--purge-manifest", type=Path,
                        help="Local audit manifest with purge[].path and purge[].versions blob IDs; requires --history")
    args = parser.parse_args()
    if args.purge_manifest and not args.history:
        parser.error("--purge-manifest requires --history")
    issues: list[str] = []
    files = tracked_files(args.include_untracked)
    readmes = 0
    for name in files:
        if HAN.search(name):
            issues.append(f"Chinese filename: {name}")
        if reason := result_path_reason(name):
            issues.append(f"{name}: {reason}")
        if PurePosixPath(name).name.startswith("README") and name.endswith(".md"):
            readmes += 1
            if HAN.search((ROOT / name).read_text(encoding="utf-8-sig")):
                issues.append(f"README contains Chinese text: {name}")
    if (ROOT / "README.md").read_bytes() != (ROOT / "README.en.md").read_bytes():
        issues.append("README.md and README.en.md differ")
    fingerprints = audit_catalogs(set(files), issues)
    trees = audit_history(issues, args.purge_manifest) if args.history else None
    print(json.dumps({"files": len(files), "readmes": readmes,
                      "source_fingerprints": fingerprints,
                      "historical_trees": trees, "issues": issues}, indent=2, ensure_ascii=True))
    return 1 if issues else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Publication audit failed: {error}", file=sys.stderr)
        sys.exit(1)
