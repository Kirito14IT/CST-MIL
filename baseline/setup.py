"""Install optional pinned author implementations; the default demo does not use this.

Only explicitly requested sources are fetched. Archives and project edits are
checked before an installation is made visible. No compiler or downloaded code
is executed by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

BASELINE_ROOT = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def confined(root: Path, relative: str) -> Path:
    name = PurePosixPath(relative)
    if not relative or "\\" in relative or name.is_absolute() or ":" in relative:
        raise ValueError(f"Unsafe source path: {relative!r}")
    if ".." in name.parts:
        raise ValueError(f"Unsafe source path: {relative!r}")
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError(f"Source path escaped its installation: {relative!r}")
    return target


def patched_text(original: bytes, change: dict) -> bytes:
    text = original.decode("utf-8-sig").replace("\r\n", "\n")
    if sha256(text.encode("utf-8")) != change["before_sha256"]:
        raise ValueError(f"Unexpected upstream source: {change['path']}")
    lines = text.splitlines(keepends=True)
    previous_start = len(lines) + 1
    for edit in reversed(change["edits"]):
        start, stop = edit["start"], edit["stop"]
        if not 0 <= start <= stop <= len(lines) or stop > previous_start:
            raise ValueError("Overlapping or invalid adapter edit")
        if sha256("".join(lines[start:stop]).encode("utf-8")) != edit["before_sha256"]:
            raise ValueError(f"Adapter context mismatch: {change['path']}")
        lines[start:stop] = edit["after"]
        previous_start = start
    result = "".join(lines).encode("utf-8")
    if sha256(result) != change["after_sha256"]:
        raise ValueError(f"Adapted source fingerprint mismatch: {change['path']}")
    return result


def extract_sources(archive: Path, destination: Path, include: list[str]) -> None:
    wanted = set(include)
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if len(members) > 10_000 or sum(m.file_size for m in members) > 256 * 1024**2:
            raise ValueError("Source archive exceeds extraction limits")
        roots: set[str] = set()
        indexed: dict[str, zipfile.ZipInfo] = {}
        for member in members:
            confined(destination, member.orig_filename)
            confined(destination, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError("Symbolic links are not accepted in source archives")
            parts = PurePosixPath(member.filename).parts
            if not parts:
                raise ValueError("Empty source archive path")
            roots.add(parts[0])
            if len(parts) > 1 and not member.is_dir():
                relative = PurePosixPath(*parts[1:]).as_posix()
                if relative in indexed:
                    raise ValueError("Duplicate source archive path")
                indexed[relative] = member
        if len(roots) != 1 or not wanted.issubset(indexed):
            raise ValueError("Unexpected source archive layout or missing pinned files")
        for relative in sorted(wanted):
            target = confined(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(indexed[relative]))


def fetch_archive(source: dict, cache: Path) -> Path:
    target = cache / f"{source['destination'].lower()}-{source['commit']}.zip"
    if target.exists():
        if sha256(target.read_bytes()) == source["archive_sha256"]:
            return target
        raise ValueError("Cached source archive failed SHA-256 validation")
    if not source["url"].startswith("https://codeload.github.com/"):
        raise ValueError("Source URL must be a pinned GitHub HTTPS archive")
    cache.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".zip.pending")
    request = urllib.request.Request(source["url"], headers={"User-Agent": "CST-MIL-release/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as stream, temporary.open("wb") as out:
            total = 0
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(block)
                if total > source["archive_bytes"]:
                    raise ValueError("Source archive exceeded pinned byte length")
                out.write(block)
        if total != source["archive_bytes"] or sha256(temporary.read_bytes()) != source["archive_sha256"]:
            raise ValueError("Downloaded source archive failed pinned size/SHA-256")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def verify_installation(target: Path, source: dict, adapter_hash: str) -> dict:
    marker = target / ".cst-source.json"
    if not marker.is_file():
        raise ValueError(f"Existing {target.name} has no trusted installation identity")
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("commit") != source["commit"] or data.get("adapter_sha256") != adapter_hash:
        raise ValueError("Existing source installation has a different identity")
    expected = set(source["include"]) | set(source["bridge_files"])
    if set(data.get("files", {})) != expected:
        raise ValueError("Existing source installation has an incomplete file manifest")
    if data["files"] != source["installed_files"]:
        raise ValueError("Existing source manifest differs from the pinned catalog")
    for relative, fingerprint in data["files"].items():
        file = confined(target, relative)
        if not file.is_file() or sha256(file.read_bytes()) != fingerprint:
            raise ValueError(f"Modified installed source refused: {relative}")
    actual = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    if actual != expected | {".cst-source.json"}:
        raise ValueError("Unexpected files in installed source tree")
    return data


def install(method: str, source: dict, *, root: Path, archive: Path) -> dict:
    root = root.resolve()
    destination = confined(root, source["destination"])
    adapter = confined(root, f"adapters/{method}/edits.json")
    adapter_hash = sha256(adapter.read_bytes())
    if adapter_hash != source["adapter_sha256"]:
        raise ValueError("Project adapter failed SHA-256 validation")
    if destination.exists():
        return verify_installation(destination, source, adapter_hash)
    if sha256(archive.read_bytes()) != source["archive_sha256"]:
        raise ValueError("Source archive failed SHA-256 validation")
    stage = Path(tempfile.mkdtemp(prefix=f".{method}-install-", dir=root))
    try:
        extract_sources(archive, stage, source["include"])
        for change in json.loads(adapter.read_text(encoding="utf-8")):
            target = confined(stage, change["path"])
            target.write_bytes(patched_text(target.read_bytes(), change))
        for relative, bridge in source["bridge_files"].items():
            data = confined(root, bridge["source"]).read_bytes()
            if sha256(data) != bridge["sha256"]:
                raise ValueError("Project bridge failed SHA-256 validation")
            target = confined(stage, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        result = {"commit": source["commit"], "url": source["url"],
                  "archive_sha256": source["archive_sha256"], "adapter_sha256": adapter_hash,
                  "files": {p.relative_to(stage).as_posix(): sha256(p.read_bytes())
                            for p in sorted(stage.rglob("*")) if p.is_file()}}
        if result["files"] != source["installed_files"]:
            raise ValueError("Installed files differ from the pinned catalog")
        (stage / ".cst-source.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError("Source installation appeared while preparing; retry safely")
        stage.replace(destination)
        return result
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", required=True, choices=("lda", "gsdmm"))
    parser.add_argument("--accept-upstream-terms", action="store_true",
                        help="Acknowledge reviewing upstream terms; no additional license is granted")
    parser.add_argument("--archives-dir", type=Path, help="Use already obtained pinned ZIP archives")
    args = parser.parse_args(argv)
    if not args.accept_upstream_terms:
        parser.error("Read THIRD_PARTY_NOTICES.md and supply --accept-upstream-terms explicitly")
    catalog = json.loads((BASELINE_ROOT / "sources.json").read_text(encoding="utf-8"))["sources"]
    for method in dict.fromkeys(args.methods):
        source = catalog[method]
        name = f"{method}-{source['commit']}.zip"
        archive = (args.archives_dir / name) if args.archives_dir else fetch_archive(source, BASELINE_ROOT / ".downloads")
        identity = install(method, source, root=BASELINE_ROOT, archive=archive)
        print(f"{method}: verified commit {identity['commit']} ({len(identity['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
