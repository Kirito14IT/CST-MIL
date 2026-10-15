"""Full, frozen five-source cohort for the joint comparison.

Candidate membership is inherited from the audited short-text acquisition,
before its old balanced-budget selection. Only original message bodies become
features. Labels and grouping metadata are retained for supervision and audit.
"""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

from cst_mil.unified_semantic_v1_1 import data as frozen

PROTOCOL = "cst-full-joint-data-20261014-v1"
SOURCES = frozen.SOURCES
CATEGORIES = frozen.CATEGORIES
SHORTTEXT = frozen.SHORTTEXT
EXPECTED_CANDIDATES = {"conda": 23629, "toxicn": 12001, "scamspot": 3061, "hatebr": 7000}
file_hash = frozen.file_hash
map_labels = frozen.map_labels
normalized = frozen.normalized
clean_records = frozen.clean_records
_read_json = frozen._read_json
_read_jsonl = frozen._read_jsonl
_write_json = frozen._write_json
_write_jsonl = frozen._write_jsonl
_hash = frozen._hash


def _input_fingerprints() -> dict[str, str]:
    """Check the old pinned inputs as well as all full-candidate audit files."""
    result = frozen._input_fingerprints()
    for source in SOURCES[1:]:
        base = SHORTTEXT / "prepared" / source
        manifest = _read_json(base / "manifest.json")
        for name in (
            "candidates.jsonl",
            "source_catalog.jsonl",
            "components.jsonl",
            "exclusions.jsonl",
        ):
            path = base / name
            expected = manifest["artifacts"].get(str(path.resolve()))
            if expected is None:
                expected = manifest["artifacts"].get(name)
            if expected is None or file_hash(path) != expected:
                raise ValueError(f"Frozen full-candidate fingerprint mismatch: {path}")
            result[str(path.resolve())] = expected
    return dict(sorted(result.items()))


def _originals(source: str) -> dict[str, dict]:
    """Joinable raw objects with their physical parsed-record ordinal."""
    base = SHORTTEXT / "sources" / source
    result = {}
    if source == "toxicn":
        for partition in ("train", "test"):
            path = base / f"ToxiCN_ex/ToxiCN/data/{partition}.json"
            for index, item in enumerate(_read_json(path)):
                result[f"toxicn:{partition}:{index}"] = {
                    **item,
                    "_source_row": index,
                    "_source_partition": partition,
                    "_source_path": str(path),
                }
    else:
        partitions = ("train", "valid") if source == "conda" else ("all",)
        for partition in partitions:
            relative = (
                f"data/CONDA_{partition}.csv"
                if source == "conda"
                else "data/comments-rated.csv"
                if source == "scamspot"
                else "dataset/HateBR.csv"
            )
            path = base / relative
            key = "Id" if source == "conda" else "number" if source == "scamspot" else "id"
            for ordinal, item in enumerate(frozen._csv_rows(path), 2):
                identifier = (
                    f"conda:{partition}:{item[key]}"
                    if source == "conda"
                    else f"{source}:{ordinal}:{item[key]}"
                )
                if identifier in result:
                    raise ValueError(f"Duplicate original source identifier: {identifier}")
                result[identifier] = {
                    **item,
                    "_source_row": ordinal,
                    "_source_partition": partition,
                    "_source_path": str(path),
                }
    return result


def _load_shorttext(source: str) -> tuple[list[dict], list[dict], list[dict]]:
    base = SHORTTEXT / "prepared" / source
    candidates = _read_jsonl(base / "candidates.jsonl")
    originals = _originals(source)
    text_key = {
        "conda": "utterance",
        "toxicn": "content",
        "scamspot": "msg",
        "hatebr": "comentario",
    }[source]
    label_keys = {
        "conda": ("intentClass",),
        "toxicn": ("toxic", "toxic_type", "expression", "target"),
        "scamspot": ("scam",),
        "hatebr": ("label_final", "anotator1", "anotator2", "anotator3"),
    }[source]
    records = []
    for ref in candidates:
        identifier = ref["source_id"]
        original = originals[identifier]
        if (
            int(ref["source_row"]) != original["_source_row"]
            or ref["source_partition"] != original["_source_partition"]
        ):
            raise ValueError(f"Source row/partition mismatch: {identifier}")
        text = original[text_key]
        if not isinstance(text, str) or _hash(normalized(text)) != ref["text_sha256"]:
            raise ValueError(f"Original full-text fingerprint mismatch: {identifier}")
        labels, mask = map_labels(source, original)
        binary = (
            int(original["toxic"])
            if source == "toxicn"
            else labels[2 if source == "scamspot" else 0]
        )
        if binary != int(ref["label"]):
            raise ValueError(f"Candidate binary label mismatch: {identifier}")
        if source == "conda":
            if str(ref["match_id"]) != str(original["matchId"]):
                raise ValueError(f"CONDA true match ID mismatch: {identifier}")
            group = f"conda:match:{original['matchId']}"
            if str(ref["split_group"]) != str(original["matchId"]):
                raise ValueError(f"CONDA frozen group mismatch: {identifier}")
        elif source == "hatebr":
            post = original.get("links_post") or None
            if not post or ref.get("source_post") != post or ref["split_group"] != post:
                raise ValueError(f"HateBR true post ID mismatch: {identifier}")
            group = "hatebr:post:" + post
        else:
            group = source + ":component:" + ref["component_id"]
        if source == "toxicn" and ((ref["source_partition"] == "test") != (ref["split"] == "test")):
            raise ValueError(f"ToxiCN official test boundary mismatch: {identifier}")
        records.append(
            {
                "record_id": identifier,
                "source": source,
                "split": ref["split"],
                "group_id": group,
                "base_eval_group": group,
                "eval_group": group,
                "session_id": identifier,
                "message_id": identifier,
                "order": 0,
                "parent_id": None,
                "text": text,
                "labels": labels,
                "observed_mask": mask,
                "original_label": {key: original.get(key) for key in label_keys},
                "source_ref": {**ref, "path": original["_source_path"]},
                "duplicate_component": "",
                "is_fallback": False,
                "fit_eligible": True,
            }
        )
    if len({r["record_id"] for r in records}) != len(candidates):
        raise ValueError(f"Full candidates contain repeated IDs: {source}")
    original_catalog = [
        {"source": source, **row} for row in _read_jsonl(base / "source_catalog.jsonl")
    ]
    # Old budget omissions are now included, so they are not exclusions in this campaign.
    source_exclusions = [
        {"source": source, **row}
        for row in _read_jsonl(base / "exclusions.jsonl")
        if row["reason"] != "not_selected_fixed_budget"
    ]
    return records, original_catalog, source_exclusions


def _load_sccd() -> tuple[list[dict], list[dict]]:
    records, sessions = frozen._load_sccd()
    for row in records:
        row["is_fallback"] = bool(row.get("is_fallback"))
        row["fit_eligible"] = not row["is_fallback"]
        row["base_eval_group"] = "sccd:session:" + row["session_id"]
        row["eval_group"] = row["base_eval_group"]
    return records, sessions


def _sccd_raw_audit(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Catalog every raw comment row, including the upstream frozen removals."""
    preferred = {row["source_ref"]["source_row"] for row in records if not row["is_fallback"]}
    catalog, excluded = [], []
    for ordinal, raw in enumerate(frozen._csv_rows(frozen.RAW_COMMENTS), 2):
        text = raw["comment_content"]
        item = {
            "source": "sccd",
            "source_id": f"sccd:raw:{ordinal}",
            "source_row": ordinal,
            "post_id": raw["post_id"],
            "comment_id": raw["comment_id"],
            "original_label": raw["label"],
            "raw_text_sha256": _hash(text),
            "text_sha256": _hash(normalized(text)),
            "retained_in_frozen_full677": ordinal in preferred,
        }
        catalog.append(item)
        if ordinal not in preferred:
            excluded.append(
                {
                    **item,
                    "reason": "upstream_frozen_full677_cleaning_exclusion",
                    "detail": (
                        "original full677 manifest contains the exact upstream cleaning "
                        "policy and counts"
                    ),
                }
            )
    return catalog, excluded


def _assign_eval_groups(records: list[dict]) -> dict:
    """Merge real sampling groups connected by duplicate components."""
    parent: dict[str, str] = {}

    def root(group: str) -> str:
        parent.setdefault(group, group)
        if parent[group] != group:
            parent[group] = root(parent[group])
        return parent[group]

    component_groups: dict[str, set[str]] = defaultdict(set)
    for row in records:
        group = row["base_eval_group"]
        root(group)
        component_groups[row["duplicate_component"]].add(group)
    for groups in component_groups.values():
        ordered = sorted(groups)
        for group in ordered[1:]:
            left, right = root(ordered[0]), root(group)
            if left != right:
                parent[max(left, right)] = min(left, right)
    members: dict[str, list[str]] = defaultdict(list)
    for group in parent:
        members[root(group)].append(group)
    identifiers = {
        group: "eval:" + _hash("|".join(sorted(groups))) for group, groups in members.items()
    }
    for row in records:
        row["eval_group"] = identifiers[root(row["base_eval_group"])]
    return {
        "rule": (
            "true session/match/post groups unioned by all normalized exact/near "
            "duplicate components"
        ),
        "base_groups": len(parent),
        "merged_groups": len(members),
        "dependent_groups": [
            {"eval_group": identifiers[key], "base_groups": sorted(groups)}
            for key, groups in sorted(members.items())
            if len(groups) > 1
        ],
    }


def _validate(records: list[dict], sessions: list[dict]) -> None:
    frozen._validate(records, sessions)
    by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in records:
        if not row.get("eval_group") or not row.get("base_eval_group"):
            raise ValueError("Missing actual evaluation group")
        if row["source"] in {"conda", "hatebr"}:
            by_group[row["source"], row["base_eval_group"]].add(row["split"])
        if row["source"] != "sccd" and (row["parent_id"] is not None or row["order"] != 0):
            raise ValueError("Single-comment sources must not acquire synthetic replies")
        if row.get("is_fallback") and (any(row["observed_mask"]) or row.get("fit_eligible")):
            raise ValueError(
                "Fallback context must not become message supervision or fitted representation"
            )
    if any(len(splits) > 1 for splits in by_group.values()):
        raise ValueError("True match/post group overlaps train/dev/test")


def _configuration() -> dict:
    return {
        "protocol": PROTOCOL,
        "sources": list(SOURCES),
        "categories": list(CATEGORIES),
        "near_threshold": frozen.NEAR_THRESHOLD,
        "sampling": "all frozen source candidates; no budget subsampling",
        "data_code_sha256": file_hash(Path(__file__)),
        "frozen_helper_code_sha256": file_hash(Path(frozen.__file__)),
        "input_hashes": _input_fingerprints(),
    }


def load_prepared(output: Path) -> tuple[list[dict], list[dict], dict]:
    """Resume only when raw files, code and every prepared artifact still match."""
    folder = Path(output) / "prepared"
    manifest = _read_json(folder / "manifest.json")
    configuration = _configuration()
    if manifest["fingerprint"] != _hash(
        json.dumps(configuration, sort_keys=True, ensure_ascii=False)
    ):
        raise ValueError("Prepared resume fingerprint mismatch; use a new output directory")
    for name, expected in manifest["artifacts"].items():
        if file_hash(folder / name) != expected:
            raise ValueError(f"Prepared artifact fingerprint mismatch: {name}")
    records, sessions = (
        _read_jsonl(folder / "records.jsonl"),
        _read_jsonl(folder / "sessions.jsonl"),
    )
    _validate(records, sessions)
    return records, sessions, manifest


def prepare(output: Path) -> dict:
    """Prepare the full cohort once without changing old cohorts or their splits."""
    folder = Path(output) / "prepared"
    if (folder / "manifest.json").exists():
        return load_prepared(output)[2]
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(
            "Incomplete prepared directory; preserve it and use a new output directory"
        )
    configuration = _configuration()
    records, sessions = _load_sccd()
    original_catalog, source_exclusions = _sccd_raw_audit(records)
    candidate_counts = {"sccd": len(records)}
    for source in SOURCES[1:]:
        rows, catalog, exclusions = _load_shorttext(source)
        candidate_counts[source] = len(rows)
        records.extend(rows)
        original_catalog.extend(catalog)
        source_exclusions.extend(exclusions)
    candidates = copy.deepcopy(records)
    original_counts = frozen._counts(records)
    kept, exclusions, components, duplicate_audit = clean_records(records)
    session_rows: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        if row["source"] == "sccd":
            session_rows[row["session_id"]].append(row)
    additional_fallbacks = []
    for session in sessions:
        rows = sorted(session_rows[session["session_id"]], key=lambda row: row["order"])
        if not rows:
            row = frozen._fallback(session, "all_comments_excluded")
            row.update(
                is_fallback=True,
                fit_eligible=False,
                base_eval_group="sccd:session:" + session["session_id"],
                eval_group="sccd:session:" + session["session_id"],
            )
            additional_fallbacks.append(row)
            rows = [row]
            session["fallback_reason"] = "all_comments_excluded"
        session["message_ids"] = [row["message_id"] for row in rows]
        session["retained_message_count"] = sum(not row["is_fallback"] for row in rows)
        session["excluded_message_count"] = (
            session["original_message_count"] - session["retained_message_count"]
        )
    if additional_fallbacks:
        kept.extend(additional_fallbacks)
        mapping, components, fallback_audit = frozen.duplicate_components(kept + exclusions)
        for row in kept + exclusions:
            row["duplicate_component"] = mapping[row["record_id"]]
        duplicate_audit["fallback_component_audit"] = fallback_audit
    eval_groups = _assign_eval_groups(kept)
    _validate(kept, sessions)
    kept.sort(
        key=lambda row: (
            SOURCES.index(row["source"]),
            row["split"],
            row["session_id"],
            row["order"],
            row["record_id"],
        )
    )
    sessions.sort(key=lambda row: (row["split"], row["session_id"]))
    audit = {
        "protocol": PROTOCOL,
        "candidate_counts": candidate_counts,
        "original_counts": original_counts,
        "retained_counts": frozen._counts(kept),
        "raw_catalog_counts": dict(Counter(row["source"] for row in original_catalog)),
        "source_exclusion_reasons": dict(Counter(row["reason"] for row in source_exclusions)),
        "duplicate_audit": duplicate_audit,
        "evaluation_group_audit": eval_groups,
        "session_counts": dict(Counter(row["split"] for row in sessions)),
        "known_boundaries": [
            "All source-cleaned candidates are included; old cleaning exclusions remain excluded.",
            "Natural prevalence is retained; train exclusions can change train prevalence.",
            "SCCD bag gold is never copied to comments; unknown fallbacks are not fit eligible.",
            "ToxiCN expression=3 is unknown for attack/hate; HateBR supplies offensiveness gold.",
            "ScamSpot combines spam/scam and is not a pure fraud evaluation.",
            "Previously exposed held-out subsets are retained, not described as pristine tests.",
            "Dev/test dependencies are disclosed; bootstrap uses merged real sampling groups.",
        ],
    }
    folder.mkdir(parents=True, exist_ok=True)
    payloads = {
        "records.jsonl": kept,
        "sessions.jsonl": sessions,
        "candidate_records.jsonl": candidates,
        "original_catalog.jsonl": original_catalog,
        "source_exclusions.jsonl": source_exclusions,
        "exclusions.jsonl": exclusions,
        "duplicate_components.jsonl": components,
    }
    for name, rows in payloads.items():
        _write_jsonl(folder / name, rows)
    _write_json(folder / "audit.json", audit)
    if _configuration() != configuration:
        raise ValueError(
            "Frozen inputs or data code changed during preparation; manifest not finalized"
        )
    manifest = {
        "protocol": PROTOCOL,
        "configuration": configuration,
        "fingerprint": _hash(json.dumps(configuration, sort_keys=True, ensure_ascii=False)),
        "categories": list(CATEGORIES),
        "sources": list(SOURCES),
        "record_count": len(kept),
        "session_count": len(sessions),
        "counts": audit["retained_counts"],
        "original_counts": original_counts,
        "candidate_counts": candidate_counts,
        "session_counts": audit["session_counts"],
        "artifacts": {name: file_hash(folder / name) for name in (*payloads, "audit.json")},
    }
    _write_json(folder / "manifest.json", manifest)
    return manifest
