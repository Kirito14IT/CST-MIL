"""Archive one predetermined seed42 package, with explanation and asset bindings."""

from __future__ import annotations

import shutil
from pathlib import Path

from .common import (
    MODULE,
    OUTPUT,
    digest,
    guard,
    identity,
    read,
    save,
    source_hashes,
    verify_hashes,
)
from .execution import folder_for, selected
from .topics import PURE_RESPONSES


def create_bundle(output: Path = OUTPUT) -> dict:
    guard(output)
    selection = selected(output)
    folder = folder_for(output, selection["selected_architecture"], 42)
    verify_hashes(read(folder / "complete.json")["artifacts"])
    verify_hashes(read(folder / "test_complete.json")["artifacts"])
    cost = read(output / "cost/summary.json")
    if len(cost["raw_repeats"]) != 3 or any(r["status"] != "passed" for r in cost["raw_repeats"]):
        raise RuntimeError("Bundle requires three accepted raw passes")
    topic = read(output / "topics/summary.json")
    if topic["status"] != "passed":
        raise RuntimeError("Bundle requires accepted grounding/coverage checks")
    protocol = read(output / "protocol.json")
    config = {
        "distance_threshold": 0.35,
        "link_short_reply_max_nonspace": 12,
        "protected_clause_max_nonspace": 96,
        "pure_responses": sorted(PURE_RESPONSES),
        "topic_membership": (
            "one basic semantic group or direct-parent association, no overlapping claims"
        ),
        "inherit_parent_risk": False,
        "association_contributes_to_names": False,
        "span_index_unit": "Unicode code point, half-open original string interval",
    }
    value = {
        "technical_name": "会话风险筛查与主题证据定位",
        "seed": 42,
        "seed_selection": "predetermined seed42, not best-test seed",
        "architecture": selection["selected_architecture"],
        "categories": ["attack", "hate", "spam_scam"],
        "category_names": ["攻击／冒犯", "仇恨／歧视", "垃圾／诈骗推广"],
        "training_sources": ["sccd", "conda", "toxicn", "scamspot", "hatebr"],
        "thresholds": read(folder / "manifest.json")["thresholds"],
        "encoder": protocol["encoder"],
        "model_source": str(folder / "model.joblib"),
        "model_sha256": digest(folder / "model.joblib"),
        "prediction_entry": (
            "cst_mil.research_followup.semantic_patent_followup.model."
            "FollowupPackage.predict_payload"
        ),
        "explanation_entry": (
            "cst_mil.research_followup.semantic_patent_followup.topics.interpret_topics"
        ),
        "explanation_config": config,
        "code_hashes": source_hashes(MODULE.glob("*.py")),
        "evidence_hashes": source_hashes(
            [
                output / "selection_freeze.json",
                output / "analysis/summary.json",
                output / "topics/summary.json",
                output / "topics/assisted_review_summary.json",
                output / "qa/final_crossfit_reuse_audit.json",
                output / "cost/summary.json",
            ]
        ),
        "existing_demo_auto_switched": False,
        "limits": [
            "SCCD conversation classifier; "
            "other single-comment tasks evaluate message categories only.",
            "Explanation judgments are model-assisted, not human theme gold.",
            "Encoder assets stay at the audited shared local path; "
            "they are required, not omitted from asset size.",
        ],
    }
    value["identity"] = identity(value)
    target = output / "final_model_bundle"
    target.mkdir(parents=True, exist_ok=True)
    manifest = target / "manifest.json"
    if manifest.exists():
        old = read(manifest)
        if old != value:
            raise RuntimeError("Existing final bundle binding differs")
        if digest(target / "model.joblib") != value["model_sha256"]:
            raise RuntimeError("Final model copy was altered")
        if read(target / "explanation_config.json") != old["explanation_config"]:
            raise RuntimeError("Final explanation configuration was altered")
        verify_hashes(old["code_hashes"])
        verify_hashes(old["evidence_hashes"])
        return old
    shutil.copy2(folder / "model.joblib", target / "model.joblib")
    if digest(target / "model.joblib") != value["model_sha256"]:
        raise RuntimeError("Model copy differs")
    save(target / "explanation_config.json", config)
    save(manifest, value)
    return value
