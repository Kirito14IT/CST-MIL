"""Small standard-library asset installer with pinned hashes and safe resumption."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

CATALOG_PATH = Path(__file__).with_name("catalog.json")
MODEL_HASH = "39e842dbffe10356c6dbef4f17bb320868ef7b007c7b1972a38c315d4af87a59"
ENCODER_HASH = "84508dc5c543fddd28e637cf0786ce79dd0fd3e67cce8614ef11c608ec01da1e"
TAG = "demo-2026.10.15"
HASH = re.compile(r"^[a-f0-9]{64}$")


def sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(file: Path) -> dict[str, Any]:
    value = json.loads(file.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {file.name}")
    return value


def atomic_json(file: Path, value: dict[str, Any]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_name(file.name + ".pending")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, file)


def repo_root(root: Path | None = None) -> Path:
    result = root.resolve() if root else Path(__file__).resolve().parents[4]
    if not (result / "cst_mil/src/cst_mil").is_dir():
        raise ValueError("Cannot find repository; pass --root with the cloned repository path")
    return result


def safe_path(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative or "\0" in relative
            or ":" in relative or PurePosixPath(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError(f"Unsafe asset path: {relative!r}")
    target = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Asset path escapes its directory: {relative}")
    return target


def verify_file(file: Path, expected: dict[str, Any]) -> None:
    if (type(expected.get("bytes")) is not int or expected["bytes"] < 0
            or not HASH.fullmatch(str(expected.get("sha256", "")))):
        raise ValueError("Invalid published asset size/hash")
    if not file.is_file() or file.stat().st_size != expected["bytes"] or sha256(file) != expected["sha256"]:
        raise ValueError(f"SHA256 or size mismatch: {file.name}")


def catalog() -> dict[str, Any]:
    value = load_json(CATALOG_PATH)
    encoder = value.get("encoder", {})
    if (value.get("catalogVersion") != 1 or value.get("release") != TAG
            or value.get("model", {}).get("sha256") != MODEL_HASH
            or encoder.get("fingerprint") != ENCODER_HASH
            or encoder.get("revision") != "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"):
        raise ValueError("Unsupported public model catalog")
    files = encoder.get("manifest", {}).get("files", {})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != ENCODER_HASH or not files or not value.get("codeFingerprints"):
        raise ValueError("Published encoder inventory/source fingerprint is inconsistent")
    return value


def verify_code(root: Path, published: dict[str, Any]) -> None:
    for relative, expected in published["codeFingerprints"].items():
        if not HASH.fullmatch(str(expected)) or sha256(safe_path(root, relative)) != expected:
            raise ValueError(f"Published source fingerprint mismatch: {relative}")


def download(url: str, target: Path, expected: dict[str, Any], *, offline: bool = False) -> None:
    """Resume only an acknowledged HTTP byte range, then hash the complete file."""
    if target.is_file():
        verify_file(target, expected)
        return
    if offline:
        raise ValueError(f"Offline asset is missing: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    start = part.stat().st_size if part.exists() else 0
    if start > expected["bytes"]:
        part.unlink()
        start = 0
    if start == expected["bytes"] and part.is_file():
        verify_file(part, expected)
        os.replace(part, target)
        return
    headers = {"User-Agent": "CST-MIL-asset-installer/1", "Accept-Encoding": "identity"}
    if start:
        headers["Range"] = f"bytes={start}-"
    try:
        response = urlopen(Request(url, headers=headers), timeout=60)
    except HTTPError as error:
        raise RuntimeError(f"Download returned HTTP {error.code}: {url}") from error
    with response:
        status = response.status
        if start and status == 206:
            if response.headers.get("Content-Range") != f"bytes {start}-{expected['bytes'] - 1}/{expected['bytes']}":
                raise ValueError("Server returned an unverified resumed byte range")
            mode = "ab"
        elif status == 200:
            mode = "wb"  # Server ignored Range: do not append duplicate bytes.
        else:
            raise RuntimeError(f"Unexpected HTTP status {status}")
        with part.open(mode) as stream:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(block)
                if stream.tell() > expected["bytes"]:
                    raise ValueError("Download exceeds the published asset size")
    try:
        verify_file(part, expected)
    except ValueError:
        # A complete but corrupt file cannot be repaired by adding a suffix.
        if part.exists() and part.stat().st_size >= expected["bytes"]:
            part.unlink()
        raise
    os.replace(part, target)


def extract_model(archive: Path, destination: Path, files: dict[str, Any]) -> None:
    """Extract exactly the pinned ZIP inventory; reject traversal and symlinks."""
    with zipfile.ZipFile(archive) as opened:
        entries = opened.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)) or set(names) != set(files):
            raise ValueError("Model ZIP inventory differs from the pinned release")
        for entry in entries:
            target = safe_path(destination, entry.filename)
            if entry.is_dir() or stat.S_ISLNK(entry.external_attr >> 16) or entry.file_size != files[entry.filename]["bytes"]:
                raise ValueError(f"Unsafe model ZIP member: {entry.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".pending")
            with opened.open(entry) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
            verify_file(temporary, files[entry.filename])
            os.replace(temporary, target)


def _relative(root: Path, file: Path) -> str:
    return file.resolve().relative_to(root.resolve()).as_posix()


def _bind_manifest(root: Path, published: dict[str, Any]) -> dict[str, Any]:
    runtime = root / ".runtime"
    bundle = safe_path(root, f".runtime/models/{TAG}")
    encoder = safe_path(root, ".runtime/encoder/" + published["encoder"]["directory"])
    bound = load_json(bundle / "model_manifest.json")
    bound["encoder"]["encoder_path"] = str(encoder.resolve())
    bound["encoder"]["asset_hashes"] = {
        str(safe_path(encoder, name)): entry["sha256"]
        for name, entry in published["encoder"]["manifest"]["files"].items()
    }
    bound_file = bundle / "runtime_model_manifest.json"
    atomic_json(bound_file, bound)
    paths = {"manifestPath": _relative(root, bound_file),
             "explanationPath": _relative(root, bundle / "explanation_config.json"),
             "seed": 42, "architecture": "full"}
    encoder_manifest = encoder / "benchmark_model_manifest.json"
    proof_paths = [bound_file, bundle / "explanation_config.json", encoder_manifest,
                   bundle / "model_manifest.json"]
    manifest = {
        "manifestVersion": 1, "modelVersion": "semantic-patent-followup-v1", "schemaVersion": 2,
        "taskName": "五来源全量短文本风险线索与隐主题分析",
        "modelPath": _relative(root, bundle / "model.joblib"), "modelSha256": MODEL_HASH,
        "predictorModule": "cst_mil.deployment.adapter", "modelClass": "FollowupPackage",
        "predictFunction": "predict_session", "codeFingerprints": published["codeFingerprints"],
        "assetFingerprints": {_relative(root, file): sha256(file) for file in proof_paths},
        "bundle": paths,
        "encoder": {"path": _relative(root, encoder), "fingerprint": ENCODER_HASH,
                    "manifestPath": _relative(root, encoder_manifest)},
        "thresholds": {key: bound["thresholds"][key] for key in ("attack", "hate", "spam_scam")},
        "conversationThreshold": bound["thresholds"]["conversation"],
        "trainingSources": bound["training_sources"],
        "portable": {"release": TAG, "catalogSha256": sha256(CATALOG_PATH),
                     "publishedManifestPath": _relative(root, bundle / "model_manifest.json"),
                     "archiveSha256": published["archive"]["sha256"]},
    }
    return manifest


def setup(root: Path | None = None, *, model_archive: Path | None = None,
          encoder_source: Path | None = None, offline: bool = False) -> dict[str, Any]:
    root, published = repo_root(root), catalog()
    verify_code(root, published)
    runtime = safe_path(root, ".runtime")
    if model_archive is None:
        model_archive = safe_path(root, ".runtime/downloads/" + published["archive"]["name"])
        download(published["archive"]["url"], model_archive, published["archive"], offline=offline)
    else:
        model_archive = model_archive.resolve()
        verify_file(model_archive, published["archive"])
    bundle = safe_path(root, f".runtime/models/{TAG}")
    extract_model(model_archive, bundle, published["archive_files"])
    encoder = safe_path(root, ".runtime/encoder/" + published["encoder"]["directory"])
    encoder.mkdir(parents=True, exist_ok=True)
    for name, entry in published["encoder"]["manifest"]["files"].items():
        target = safe_path(encoder, name)
        if encoder_source is not None:
            source = safe_path(encoder_source.resolve(), name)
            verify_file(source, entry)
            if not target.is_file() or sha256(target) != entry["sha256"]:
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".pending")
                shutil.copyfile(source, temporary)
                verify_file(temporary, entry)
                os.replace(temporary, target)
        else:
            url = published["encoder"]["base_url"] + "/" + name
            download(url, target, entry, offline=offline)
    atomic_json(encoder / "benchmark_model_manifest.json", published["encoder"]["manifest"])
    manifest = _bind_manifest(root, published)
    # Installation becomes visible to Node only after every asset was validated.
    validate_manifest(root, manifest, published)
    atomic_json(runtime / "deployment.json", manifest)
    return {"status": "installed", "release": TAG, "modelSha256": MODEL_HASH,
            "manifest": str(runtime / "deployment.json"), "encoderFingerprint": ENCODER_HASH}


def validate_manifest(root: Path, manifest: dict[str, Any], published: dict[str, Any] | None = None) -> None:
    published = published or catalog()
    verify_code(root, published)
    if (manifest.get("codeFingerprints") != published["codeFingerprints"]
            or manifest.get("portable", {}).get("catalogSha256") != sha256(CATALOG_PATH)
            or manifest.get("portable", {}).get("release") != TAG
            or manifest.get("portable", {}).get("archiveSha256") != published["archive"]["sha256"]
            or manifest.get("predictorModule") != "cst_mil.deployment.adapter"
            or manifest.get("modelClass") != "FollowupPackage"
            or manifest.get("modelSha256") != MODEL_HASH
            or manifest.get("schemaVersion") != 2 or manifest.get("manifestVersion") != 1
            or manifest.get("modelVersion") != "semantic-patent-followup-v1"
            or manifest.get("predictFunction") != "predict_session"):
        raise ValueError("Runtime deployment does not match the public release")
    bundle_path = safe_path(root, manifest["portable"]["publishedManifestPath"])
    verify_file(bundle_path, published["archive_files"]["model_manifest.json"])
    public_bundle = load_json(bundle_path)
    if (manifest.get("thresholds") != {key: public_bundle["thresholds"][key] for key in ("attack", "hate", "spam_scam")}
            or manifest.get("conversationThreshold") != public_bundle["thresholds"]["conversation"]
            or manifest.get("trainingSources") != public_bundle["training_sources"]):
        raise ValueError("Frozen category thresholds or sources were changed")
    for name, entry in published["archive_files"].items():
        verify_file(safe_path(bundle_path.parent, name), entry)
    verify_file(safe_path(root, manifest["modelPath"]), published["model"])
    required_proofs = {manifest["bundle"]["manifestPath"], manifest["bundle"]["explanationPath"],
                       manifest["encoder"]["manifestPath"], manifest["portable"]["publishedManifestPath"]}
    if set(manifest.get("assetFingerprints", {})) != required_proofs:
        raise ValueError("Runtime deployment is missing exact model/encoder/config proofs")
    for relative, expected in manifest["assetFingerprints"].items():
        if sha256(safe_path(root, relative)) != expected:
            raise ValueError(f"Runtime asset fingerprint mismatch: {relative}")
    encoder = safe_path(root, manifest["encoder"]["path"])
    if manifest["encoder"]["fingerprint"] != ENCODER_HASH:
        raise ValueError("Encoder fingerprint was changed")
    encoder_manifest = load_json(safe_path(root, manifest["encoder"]["manifestPath"]))
    if encoder_manifest != published["encoder"]["manifest"]:
        raise ValueError("Encoder manifest was changed")
    expected_names = set(encoder_manifest["files"]) | {"benchmark_model_manifest.json"}
    actual_names = {file.relative_to(encoder).as_posix() for file in encoder.rglob("*") if file.is_file()}
    if actual_names != expected_names:
        raise ValueError("Encoder snapshot inventory differs from the public release")
    for name, entry in encoder_manifest["files"].items():
        verify_file(safe_path(encoder, name), entry)
    bound = load_json(safe_path(root, manifest["bundle"]["manifestPath"]))
    expected_bound = copy.deepcopy(public_bundle)
    expected_bound["encoder"]["encoder_path"] = str(encoder)
    expected_bound["encoder"]["asset_hashes"] = {
        str(safe_path(encoder, name)): entry["sha256"] for name, entry in encoder_manifest["files"].items()
    }
    if (bound != expected_bound or manifest["bundle"].get("seed") != 42
            or manifest["bundle"].get("architecture") != "full"
            or load_json(safe_path(root, manifest["bundle"]["explanationPath"])) != public_bundle["explanation_config"]):
        raise ValueError("Runtime model/explanation binding was changed")


def verify_installation(root: Path | None = None) -> dict[str, Any]:
    root = repo_root(root)
    file = root / ".runtime/deployment.json"
    if not file.is_file():
        raise ValueError("Assets are not installed; run python -m cst_mil.deployment setup")
    manifest = load_json(file)
    validate_manifest(root, manifest)
    return {"status": "verified", "release": TAG, "modelSha256": MODEL_HASH,
            "manifest": str(file), "deploymentSha256": sha256(file)}
