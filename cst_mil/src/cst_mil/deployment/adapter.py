"""Bind verified local encoder paths in memory, without rewriting model weights."""
from __future__ import annotations

import copy
from pathlib import Path, PureWindowsPath
from typing import Any

from cst_mil import demo_semantic_adapter as original
from cst_mil.research_followup.semantic_patent_followup.model import FollowupPackage
from cst_mil.research_followup.semantic_patent_followup import topics

from .assets import catalog, load_json, safe_path, validate_manifest

predict_session = original.predict_session


def validate_package(model: FollowupPackage, manifest: dict[str, Any], repo_root: Path) -> None:
    """Accept only the pinned artifact and exact inventory, then use the old validator."""
    published = catalog()
    validate_manifest(repo_root, manifest, published)
    public = load_json(safe_path(repo_root, manifest["portable"]["publishedManifestPath"]))
    metadata = model.encoder_metadata
    wanted = public["encoder"]
    path_fields = {"encoder_path", "asset_hashes"}
    if getattr(model, "training_summary", {}):
        raise ValueError("Public deployment weights must not contain training run summaries")
    if (any(metadata.get(key) != value for key, value in wanted.items() if key not in path_fields)
            or set(metadata) - set(wanted)):
        raise ValueError("Frozen model encoder metadata disagrees with the public technical manifest")
    old_root = PureWindowsPath(metadata["encoder_path"])
    try:
        hashes = {PureWindowsPath(name).relative_to(old_root).as_posix(): value
                  for name, value in metadata["asset_hashes"].items()}
    except ValueError as error:
        raise ValueError("Frozen model contains an encoder asset outside its original root") from error
    if hashes != wanted["asset_hashes"]:
        raise ValueError("Frozen model asset SHA inventory differs from the published encoder")
    bound = load_json(safe_path(repo_root, manifest["bundle"]["manifestPath"]))
    replacement = copy.deepcopy(metadata)
    replacement["encoder_path"] = bound["encoder"]["encoder_path"]
    replacement["asset_hashes"] = bound["encoder"]["asset_hashes"]
    model.encoder_metadata = replacement  # Only paths change; the serialized file remains untouched.
    try:
        original.validate_package(model, manifest, repo_root)
    except Exception:
        model.encoder_metadata = metadata
        raise
