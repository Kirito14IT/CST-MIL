"""Asset contract tests need no model, torch, network account or private data."""
from __future__ import annotations

import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import zipfile

import pytest

from cst_mil.deployment import assets


def expected(data: bytes) -> dict:
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture()
def installation(tmp_path: Path, monkeypatch):
    root = tmp_path / "fresh-repository"
    (root / "cst_mil/src/cst_mil").mkdir(parents=True)
    code = root / "cst_mil/src/cst_mil/source.py"
    code.write_text("# a pinned test source\n", encoding="utf-8")
    encoder_source = tmp_path / "encoder-source"
    encoder_source.mkdir()
    encoder_bytes = b"test-only synthetic encoder"
    (encoder_source / "model.safetensors").write_bytes(encoder_bytes)
    encoder_files = {"model.safetensors": expected(encoder_bytes)}
    fingerprint = hashlib.sha256(json.dumps(encoder_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    model_bytes = b"test-only synthetic model: not deserialized"
    monkeypatch.setattr(assets, "MODEL_HASH", expected(model_bytes)["sha256"])
    monkeypatch.setattr(assets, "ENCODER_HASH", fingerprint)
    technical = {
        "seed": 42, "architecture": "full", "model_sha256": assets.MODEL_HASH,
        "thresholds": {"attack": .4, "hate": .5, "spam_scam": .6, "conversation": .7},
        "training_sources": ["source-a"],
        "encoder": {"encoder_path": "encoder", "fingerprint": fingerprint,
                    "asset_hashes": {"model.safetensors": expected(encoder_bytes)["sha256"]}},
        "explanation_config": {"distance_threshold": .35},
    }
    entries = {"model.joblib": model_bytes,
               "model_manifest.json": json.dumps(technical).encode(),
               "explanation_config.json": json.dumps(technical["explanation_config"]).encode(),
               "LICENSE-model.txt": b"test fixture only"}
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        for name, data in entries.items():
            opened.writestr(name, data)
    published = {
        "archive": {**expected(archive.read_bytes()), "name": "model.zip", "url": "https://example.invalid/model.zip"},
        "archive_files": {name: expected(data) for name, data in entries.items()},
        "model": expected(model_bytes),
        "codeFingerprints": {"cst_mil/src/cst_mil/source.py": assets.sha256(code)},
        "encoder": {"directory": "encoder", "fingerprint": fingerprint,
                    "manifest": {"model_fingerprint": fingerprint, "files": encoder_files}},
    }
    monkeypatch.setattr(assets, "catalog", lambda: published)
    return root, archive, encoder_source, published


def test_repeat_installation_is_identical_and_verifiable(installation):
    root, archive, encoder, _ = installation
    for _ in range(2):
        result = assets.setup(root, model_archive=archive, encoder_source=encoder, offline=True)
        assert result["status"] == "installed"
        assert assets.verify_installation(root)["status"] == "verified"
        current = (root / ".runtime/deployment.json").read_bytes()
        if "previous" in locals():
            assert current == previous
        previous = current
    assert (root / ".runtime/models" / assets.TAG / "model.joblib").read_bytes() == b"test-only synthetic model: not deserialized"


def test_wrong_archive_hash_never_publishes_installation(installation):
    root, archive, encoder, _ = installation
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="mismatch"):
        assets.setup(root, model_archive=archive, encoder_source=encoder, offline=True)
    assert not (root / ".runtime/deployment.json").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:/outside", "folder/../../outside", "folder\\outside"])
def test_zip_path_traversal_is_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr(name, b"x")
    with pytest.raises(ValueError, match="Unsafe|inventory"):
        assets.extract_model(archive, tmp_path / "output", {name: expected(b"x")})
    assert not (tmp_path / "outside").exists()


def test_zip_symlink_and_unexpected_inventory_are_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    entry = zipfile.ZipInfo("model.joblib")
    entry.create_system = 3
    entry.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as opened:
        opened.writestr(entry, b"../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        assets.extract_model(archive, tmp_path / "out", {"model.joblib": expected(b"../outside")})
    with pytest.raises(ValueError, match="inventory"):
        assets.extract_model(archive, tmp_path / "out", {"other": expected(b"x")})


def test_modified_threshold_or_binding_is_rejected(installation):
    root, archive, encoder, _ = installation
    assets.setup(root, model_archive=archive, encoder_source=encoder, offline=True)
    original = assets.load_json(root / ".runtime/deployment.json")
    for change in (lambda m: m["thresholds"].update(attack=.1),
                   lambda m: m.update(codeFingerprints={}),
                   lambda m: m.update(assetFingerprints={}),
                   lambda m: m["portable"].update(archiveSha256="0" * 64),
                   lambda m: m["bundle"].update(architecture="context_a")):
        mutated = copy.deepcopy(original)
        change(mutated)
        with pytest.raises(ValueError):
            assets.validate_manifest(root, mutated)


def test_encoder_corruption_and_extra_file_are_rejected(installation):
    root, archive, encoder, _ = installation
    assets.setup(root, model_archive=archive, encoder_source=encoder, offline=True)
    installed = root / ".runtime/encoder/encoder"
    (installed / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="inventory"):
        assets.verify_installation(root)
    (installed / "extra.json").unlink()
    (installed / "model.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="mismatch"):
        assets.verify_installation(root)


def test_partial_offline_install_does_not_publish_manifest(installation):
    root, archive, encoder, _ = installation
    (encoder / "model.safetensors").unlink()
    with pytest.raises(ValueError):
        assets.setup(root, model_archive=archive, encoder_source=encoder, offline=True)
    assert not (root / ".runtime/deployment.json").exists()


@pytest.mark.parametrize("ignore_range", [False, True])
def test_http_resume_and_ignored_range_are_both_verified(tmp_path, ignore_range):
    data = b"0123456789abcdef" * 1024
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            observed.append(self.headers.get("Range"))
            start = 0 if ignore_range else int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(data) - start))
            if start:
                self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
            self.end_headers()
            self.wfile.write(data[start:])

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "asset.bin"
    target.with_suffix(".bin.part").write_bytes(data[:123])
    try:
        assets.download(f"http://127.0.0.1:{server.server_port}/asset", target, expected(data))
        assert target.read_bytes() == data
        assert observed == ["bytes=123-"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_offline_download_and_wrong_sha_rejected(tmp_path):
    with pytest.raises(ValueError, match="Offline"):
        assets.download("https://example.invalid/x", tmp_path / "x", expected(b"x"), offline=True)
    target = tmp_path / "wrong"
    target.write_bytes(b"y")
    with pytest.raises(ValueError, match="mismatch"):
        assets.download("https://example.invalid/x", target, expected(b"x"), offline=True)


@pytest.mark.parametrize("tamper", ["model", "catalog"])
def test_direct_python_worker_authenticates_before_deserialization(installation, monkeypatch, tamper):
    """Even a directly launched worker must never deserialize an unpinned pickle."""
    root, archive, encoder, _ = installation
    assets.setup(root, model_archive=archive, encoder_source=encoder, offline=True)
    file = root / ".runtime/deployment.json"
    manifest = assets.load_json(file)
    if tamper == "model":
        model_path = assets.safe_path(root, manifest["modelPath"])
        model_path.write_bytes(b"attacker-controlled serialized object")
        # The supplied SHA matches the supplied file, but is not the release SHA.
        manifest["modelSha256"] = assets.sha256(model_path)
    else:
        manifest["portable"]["catalogSha256"] = "0" * 64
    assets.atomic_json(file, manifest)
    worker_path = Path(__file__).resolve().parents[2] / "计算机项目综合开发创新实践/app/python/worker.py"
    spec = importlib.util.spec_from_file_location("cst_portable_worker_test", worker_path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    worker.REPO_ROOT = root
    worker.RESEARCH_ROOT = root / "cst_mil"
    calls = []
    fake_joblib = ModuleType("joblib")
    fake_joblib.load = lambda path: calls.append(path)
    fake_resources = ModuleType("cst_mil.bench.resources")
    fake_resources.configure_cpu = lambda _: {}
    monkeypatch.setitem(sys.modules, "joblib", fake_joblib)
    monkeypatch.setitem(sys.modules, "cst_mil.bench.resources", fake_resources)
    worker.importlib = SimpleNamespace(import_module=lambda _: SimpleNamespace())
    with pytest.raises(ValueError, match="public release"):
        worker.load(file)
    assert calls == [], "joblib.load must not run before authenticating the pinned release"
