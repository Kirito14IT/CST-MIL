"""Public inference packages must exclude measured experiment metadata."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cst_mil.deployment import adapter


@pytest.fixture()
def package_contract(tmp_path: Path, monkeypatch):
    """Use technical manifests and a stub model; no model assets are loaded."""
    public_encoder = {
        "encoder_path": "encoder",
        "fingerprint": "a" * 64,
        "encoding_version": "synthetic-fixture",
        "asset_hashes": {"model.safetensors": "b" * 64},
    }
    frozen_metadata = {
        **public_encoder,
        "asset_hashes": {"encoder/model.safetensors": "b" * 64},
    }
    bound_encoder = {
        **public_encoder,
        "encoder_path": str(tmp_path / "installed-encoder"),
        "asset_hashes": {str(tmp_path / "installed-encoder/model.safetensors"): "b" * 64},
    }
    (tmp_path / "public.json").write_text(
        json.dumps({"encoder": public_encoder}), encoding="utf-8"
    )
    (tmp_path / "bound.json").write_text(
        json.dumps({"encoder": bound_encoder}), encoding="utf-8"
    )
    manifest = {
        "portable": {"publishedManifestPath": "public.json"},
        "bundle": {"manifestPath": "bound.json"},
    }
    model = SimpleNamespace(encoder_metadata=frozen_metadata, training_summary={})
    published = {"synthetic": True}
    authenticate = Mock()
    validate_original = Mock()
    monkeypatch.setattr(adapter, "catalog", lambda: published)
    monkeypatch.setattr(adapter, "validate_manifest", authenticate)
    monkeypatch.setattr(adapter.original, "validate_package", validate_original)
    return model, manifest, tmp_path, bound_encoder, published, authenticate, validate_original


def test_clean_package_binds_local_paths_before_original_validation(package_contract):
    model, manifest, root, bound, published, authenticate, validate_original = package_contract
    original = copy.deepcopy(model.encoder_metadata)
    adapter.validate_package(model, manifest, root)
    authenticate.assert_called_once_with(root, manifest, published)
    validate_original.assert_called_once_with(model, manifest, root)
    assert model.encoder_metadata == bound
    assert original["encoder_path"] == "encoder"
    assert model.training_summary == {}


def test_public_package_rejects_training_run_summary_before_binding(package_contract):
    model, manifest, root, _, _, _, validate_original = package_contract
    model.training_summary = {"stage_times": {"total_fit_seconds": 1.25}}
    original = model.encoder_metadata
    with pytest.raises(ValueError, match="training run summaries"):
        adapter.validate_package(model, manifest, root)
    assert model.encoder_metadata is original
    validate_original.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("encoding_seconds_by_split", {"train": 1.25}),
        ("encoding_peak_tree_rss_bytes", 1024),
        ("unexpected_measurement", 0.5),
    ],
)
def test_public_package_rejects_extra_encoder_metadata(package_contract, field, value):
    model, manifest, root, _, _, _, validate_original = package_contract
    model.encoder_metadata[field] = value
    original = model.encoder_metadata
    with pytest.raises(ValueError, match="encoder metadata"):
        adapter.validate_package(model, manifest, root)
    assert model.encoder_metadata is original
    validate_original.assert_not_called()


def test_failed_original_validation_restores_frozen_metadata(package_contract):
    model, manifest, root, _, _, _, validate_original = package_contract
    original = model.encoder_metadata
    validate_original.side_effect = ValueError("synthetic identity mismatch")
    with pytest.raises(ValueError, match="synthetic identity mismatch"):
        adapter.validate_package(model, manifest, root)
    assert model.encoder_metadata is original
    validate_original.assert_called_once_with(model, manifest, root)
