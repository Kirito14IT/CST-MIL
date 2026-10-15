"""Pinned baseline installer checks with synthetic source, without network or compiler."""
from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "baseline/setup.py"
SPEC = importlib.util.spec_from_file_location("cst_baseline_setup", SCRIPT)
setup = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(setup)


def fixture(tmp_path):
    root = tmp_path / "baseline"
    adapter = root / "adapters/lda/edits.json"
    adapter.parent.mkdir(parents=True)
    changes = [{"path": "code.c", "before_sha256": setup.sha256(b"before\n"),
                "after_sha256": setup.sha256(b"after\n"),
                "edits": [{"start": 0, "stop": 1, "before_sha256": setup.sha256(b"before\n"),
                           "after": ["after\n"]}]}]
    adapter.write_text(json.dumps(changes), encoding="utf-8")
    archive = tmp_path / "upstream.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("upstream/code.c", "before\r\n")
        out.writestr("upstream/license.txt", "fixture-only license\n")
        out.writestr("upstream/data.csv", "not installed\n")
    source = {"destination": "LDA", "commit": "a" * 40, "url": "https://example.test/source",
              "include": ["code.c", "license.txt"], "bridge_files": {},
              "adapter_sha256": setup.sha256(adapter.read_bytes()),
              "archive_sha256": setup.sha256(archive.read_bytes()),
              "installed_files": {"code.c": setup.sha256(b"after\n"),
                                  "license.txt": setup.sha256(b"fixture-only license\n")}}
    return root, archive, source


def test_install_exact_edits_license_and_idempotent_verified_resume(tmp_path):
    root, archive, source = fixture(tmp_path)
    first = setup.install("lda", source, root=root, archive=archive)
    assert (root / "LDA/code.c").read_bytes() == b"after\n"
    assert (root / "LDA/license.txt").is_file()
    assert not (root / "LDA/data.csv").exists()
    assert setup.install("lda", source, root=root, archive=archive) == first
    (root / "LDA/code.c").write_text("user change", encoding="utf-8")
    with pytest.raises(ValueError, match="Modified"):
        setup.install("lda", source, root=root, archive=archive)


def test_reject_bad_archive_and_modified_adapter_without_installing(tmp_path):
    root, archive, source = fixture(tmp_path)
    bad = {**source, "archive_sha256": "0" * 64}
    with pytest.raises(ValueError, match="archive"):
        setup.install("lda", bad, root=root, archive=archive)
    assert not (root / "LDA").exists()
    with (root / "adapters/lda/edits.json").open("a") as out:
        out.write(" ")
    with pytest.raises(ValueError, match="adapter"):
        setup.install("lda", source, root=root, archive=archive)


@pytest.mark.parametrize("bad", ["../escape", "/absolute", "C:/drive", "root/../../escape", "root\\escape"])
def test_reject_unsafe_archive_members_even_if_not_selected(tmp_path, bad):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("root/code.c", "safe")
        out.writestr(bad, "unsafe")
    # zipfile on Windows rewrites backslashes on creation; model a malicious
    # external archive by changing both equal-length local/central file names.
    if "\\" in bad:
        archive.write_bytes(archive.read_bytes().replace(bad.replace("\\", "/").encode(), bad.encode()))
    with pytest.raises(ValueError, match="Unsafe|escaped"):
        setup.extract_sources(archive, tmp_path / "stage", ["code.c"])
    assert not (tmp_path / "escape").exists()


def test_failed_adapter_keeps_destination_absent(tmp_path):
    root, archive, source = fixture(tmp_path)
    adapter = root / "adapters/lda/edits.json"
    data = json.loads(adapter.read_text())
    data[0]["before_sha256"] = "0" * 64
    adapter.write_text(json.dumps(data), encoding="utf-8")
    source["adapter_sha256"] = setup.sha256(adapter.read_bytes())
    with pytest.raises(ValueError, match="upstream"):
        setup.install("lda", source, root=root, archive=archive)
    assert not (root / "LDA").exists()
    assert not list(root.glob(".lda-install-*"))


def test_refuse_modified_file_even_when_local_identity_is_rewritten(tmp_path):
    root, archive, source = fixture(tmp_path)
    setup.install("lda", source, root=root, archive=archive)
    (root / "LDA/code.c").write_bytes(b"changed\n")
    marker = root / "LDA/.cst-source.json"
    identity = json.loads(marker.read_text(encoding="utf-8"))
    identity["files"]["code.c"] = setup.sha256(b"changed\n")
    marker.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned catalog"):
        setup.install("lda", source, root=root, archive=archive)
