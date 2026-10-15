"""Audited five-source labels for the new, isolated semantic supervision protocol.

This module reads frozen cohorts without rewriting them.  A zero label with a
false observed mask is unknown, never an observed negative.  Deduplication is
exhaustive under the fixed normalized character 5-gram Jaccard rule; the prefix
index is a lossless candidate filter, not a probabilistic approximation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
SHORTTEXT = PROJECT / "benchmark/research_followup/20261004/shorttext_expansion"
SCCD = PROJECT / "benchmark/data/full677"
RAW_COMMENTS = PROJECT / "data/raw/comments.csv"
CATEGORIES = ("attack", "hate", "spam_scam")
SOURCES = ("sccd", "conda", "toxicn", "scamspot", "hatebr")
PROTOCOL = "cst-unified-semantic-data-20261005-v1"
NEAR_THRESHOLD = 0.85


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def normalized(text: str) -> str:
    """Retain content/URLs; NFKC, casefold and whitespace removal for matching only."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _grams(text: str) -> frozenset[str]:
    return frozenset(text[i : i + 5] for i in range(len(text) - 4))


def jaccard(left: str, right: str) -> float:
    left, right = normalized(left), normalized(right)
    if len(left) < 5 or len(right) < 5:
        return float(left == right)
    a, b = _grams(left), _grams(right)
    return len(a & b) / len(a | b)


def _binary(value: object) -> int:
    key = str(value).strip()
    if key not in {"0", "1"}:
        raise ValueError(f"Unexpected binary label: {value!r}")
    return int(key)


def _sccd_label(value: object) -> int:
    key = str(value).strip().casefold()
    if key in {"cb", "1"}:
        return 1
    if key in {"non-cb", "noncb", "0"}:
        return 0
    raise ValueError(f"Unexpected SCCD comment label: {value!r}")


def map_labels(source: str, original: dict) -> tuple[list[int], list[bool]]:
    """Explicit mapping. Unannotated risk heads are never inferred from other labels."""
    labels, mask = [0, 0, 0], [False, False, False]
    if source == "sccd":
        if original.get("fallback"):
            return labels, mask
        labels[0], mask[0] = _sccd_label(original["label"]), True
    elif source == "conda":
        value = original["intentClass"]
        if value not in {"E", "I", "A", "O"}:
            raise ValueError(f"Unexpected CONDA label: {value!r}")
        labels[0], mask[0] = int(value in {"E", "I"}), True
    elif source == "hatebr":
        labels[0], mask[0] = _binary(original["label_final"]), True
    elif source == "scamspot":
        labels[2], mask[2] = _binary(original["scam"]), True
    elif source == "toxicn":
        toxic = _binary(original["toxic"])
        toxic_type, expression = int(original["toxic_type"]), int(original["expression"])
        if toxic_type not in {0, 1, 2} or expression not in {0, 1, 2, 3}:
            raise ValueError("Unexpected ToxiCN fine-grained label")
        if (toxic == 0) != (toxic_type == 0):
            raise ValueError("Inconsistent ToxiCN toxic/toxic_type labels")
        if expression != 3:
            labels[:2], mask[:2] = [toxic, int(toxic_type == 2)], [True, True]
    else:
        raise ValueError(f"Unknown source: {source}")
    return labels, mask


def duplicate_components(records: list[dict]) -> tuple[dict[str, str], list[dict], dict]:
    """Find transitive exact/near components without O(n squared) pair enumeration."""
    started = time.perf_counter()
    exact: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        exact[normalized(record["text"])].append(record)
    texts = sorted(exact, key=_hash)
    grams = [_grams(text) if len(text) >= 5 else frozenset({text}) for text in texts]
    frequencies = Counter(gram for row in grams for gram in row)
    parent = list(range(len(texts)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    inverted: dict[str, list[int]] = defaultdict(list)
    checks, edges = 0, 0
    for i in sorted(range(len(texts)), key=lambda k: (len(grams[k]), _hash(texts[k]))):
        if len(texts[i]) < 5:
            continue
        tokens = grams[i]
        prefix_length = len(tokens) - (85 * len(tokens) + 99) // 100 + 1
        prefix = sorted(tokens, key=lambda token: (frequencies[token], token))[:prefix_length]
        candidates = {j for token in prefix for j in inverted[token]}
        for j in candidates:
            other = grams[j]
            if 100 * len(other) < 85 * len(tokens):
                continue
            checks += 1
            intersection = len(tokens & other)
            if 100 * intersection >= 85 * (len(tokens) + len(other) - intersection):
                parent[root(i)] = root(j)
                edges += 1
        for token in prefix:
            inverted[token].append(i)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for i, text in enumerate(texts):
        grouped[root(i)].extend(exact[text])
    mapping, components = {}, []
    for rows in grouped.values():
        hashes = sorted({_hash(normalized(row["text"])) for row in rows})
        identifier = _hash("|".join(hashes))
        for row in rows:
            mapping[row["record_id"]] = identifier
        components.append(
            {
                "duplicate_component": identifier,
                "record_ids": sorted(row["record_id"] for row in rows),
                "text_hashes": hashes,
                "sources": sorted({row["source"] for row in rows}),
                "splits": sorted({row["split"] for row in rows}),
                "session_ids": sorted({row["session_id"] for row in rows}),
            }
        )
    return (
        mapping,
        sorted(components, key=lambda row: row["duplicate_component"]),
        {
            "input_records": len(records),
            "unique_normalized_texts": len(texts),
            "components": len(components),
            "pair_checks": checks,
            "near_edges": edges,
            "seconds": time.perf_counter() - started,
            "rule": "NFKC/casefold/remove-whitespace; exact or character 5-gram Jaccard>=0.85",
            "candidate_filter": (
                "lossless frequency-ordered prefix intersection with set-size bound"
            ),
            "short_text_rule": "normalized texts shorter than 5 characters require exact equality",
        },
    )


def clean_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Keep frozen evaluation membership; remove train-side duplicates and mask conflicts."""
    mapping, components, duplicate_audit = duplicate_components(records)
    groups: dict[str, list[dict]] = defaultdict(list)
    for original in records:
        row = dict(original)
        row["labels"], row["observed_mask"] = list(row["labels"]), list(row["observed_mask"])
        row["duplicate_component"] = mapping[row["record_id"]]
        groups[row["duplicate_component"]].append(row)
    kept, exclusions, conflicts, dependencies = [], [], [], []
    for component in components:
        identifier = component["duplicate_component"]
        rows = groups[identifier]
        has_eval = any(row["split"] in {"dev", "test"} for row in rows)
        conflict_heads = [
            i
            for i in range(3)
            if len({row["labels"][i] for row in rows if row["observed_mask"][i]}) > 1
        ]
        if conflict_heads:
            conflicts.append(
                {
                    **component,
                    "categories": [CATEGORIES[i] for i in conflict_heads],
                    "observations": [
                        {
                            "record_id": row["record_id"],
                            "source": row["source"],
                            "split": row["split"],
                            "labels": row["labels"],
                            "observed_mask": row["observed_mask"],
                        }
                        for row in rows
                    ],
                    "action": "mask only conflicting observed train categories; preserve eval gold",
                }
            )
        if {"dev", "test"} <= set(component["splits"]):
            dependencies.append(component)
        for row in rows:
            if not normalized(row["text"]):
                exclusions.append({**row, "reason": "empty_normalized_text"})
                continue
            if row["split"] == "train" and has_eval and not row.get("is_fallback"):
                exclusions.append({**row, "reason": "train_component_overlaps_dev_or_test"})
                continue
            masked = [
                i for i in conflict_heads if row["split"] == "train" and row["observed_mask"][i]
            ]
            if masked:
                row["observed_mask"] = [
                    flag and i not in masked for i, flag in enumerate(row["observed_mask"])
                ]
                row["masked_conflict_categories"] = [CATEGORIES[i] for i in masked]
            kept.append(row)
    return (
        kept,
        exclusions,
        components,
        {
            **duplicate_audit,
            "excluded_records": len(exclusions),
            "exclusion_reasons": dict(Counter(row["reason"] for row in exclusions)),
            "conflicting_components": conflicts,
            "dev_test_dependencies": dependencies,
            "within_train_duplicates": "retained with component identity for OOF exclusion",
            "fallback_policy": (
                "unknown-labeled root context retained to preserve SCCD session membership"
            ),
        },
    )


def _csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _check_text(actual: str, expected: str, identifier: str) -> None:
    if actual.strip() != expected.strip():
        raise ValueError(f"Original full text does not match frozen record: {identifier}")


def _fallback(session: dict, reason: str) -> dict:
    message_id = session["session_id"] + ":root_fallback"
    return {
        "record_id": "sccd:" + message_id,
        "source": "sccd",
        "split": session["split"],
        "group_id": session["session_id"],
        "session_id": session["session_id"],
        "message_id": message_id,
        "order": 0,
        "parent_id": None,
        "text": session["post_text"],
        "labels": [0, 0, 0],
        "observed_mask": [False, False, False],
        "original_label": {"fallback": True},
        "source_ref": {"post_id": session["session_id"], "fallback_reason": reason},
        "duplicate_component": "",
        "is_fallback": True,
    }


def _load_sccd() -> tuple[list[dict], list[dict]]:
    sessions_raw, messages = (
        _read_jsonl(SCCD / "sessions.jsonl"),
        _read_jsonl(SCCD / "messages.jsonl"),
    )
    raw_by_id: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row_number, row in enumerate(_csv_rows(RAW_COMMENTS), 2):
        row["source_row"] = row_number
        raw_by_id[(row["post_id"].strip(), row["comment_id"].strip())].append(row)
    sessions = [
        {
            "session_id": row["post_id"],
            "split": row["split"],
            "label": int(row["label"]),
            "message_ids": list(row["message_ids"]),
            "post_text": row["post_raw_text"],
            "original_message_count": len(row["message_ids"]),
        }
        for row in sessions_raw
    ]
    by_session = {row["session_id"]: row for row in sessions}
    records = []
    for message in messages:
        sid, mid = message["post_id"], message["comment_id"]
        session = by_session[sid]
        if message["split"] != session["split"] or mid not in session["message_ids"]:
            raise ValueError("SCCD frozen message/session split or membership mismatch")
        originals = raw_by_id[(sid, mid)]
        if (
            not originals
            or len({(r["label"], r["comment_content"].strip()) for r in originals}) != 1
        ):
            raise ValueError(f"Ambiguous or missing SCCD raw-label join: {sid}/{mid}")
        original = originals[0]
        _check_text(message["raw_text"], original["comment_content"], mid)
        labels, mask = map_labels("sccd", original)
        if message.get("comment_label") is not None and message["comment_label"] != labels[0]:
            raise ValueError(f"SCCD frozen test label differs from raw gold: {mid}")
        records.append(
            {
                "record_id": f"sccd:{sid}:{mid}",
                "source": "sccd",
                "split": session["split"],
                "group_id": sid,
                "session_id": sid,
                "message_id": mid,
                "order": int(message["order"]),
                "parent_id": message.get("parent_id"),
                "text": message["raw_text"],
                "labels": labels,
                "observed_mask": mask,
                "original_label": {
                    key: original.get(key)
                    for key in ("label", "expression", "sarcasm", "target", "group category")
                },
                "source_ref": {
                    "path": str(RAW_COMMENTS),
                    "source_row": original["source_row"],
                    "post_id": sid,
                    "comment_id": mid,
                    "speaker_id": message.get("speaker_id"),
                    "comment_time": message.get("comment_time"),
                },
                "duplicate_component": "",
            }
        )
    for session in sessions:
        if not session["message_ids"]:
            record = _fallback(session, "original_empty_session")
            records.append(record)
            session["message_ids"] = [record["message_id"]]
            session["fallback_reason"] = "original_empty_session"
    return records, sessions


def _originals(source: str) -> dict[str, dict]:
    base = SHORTTEXT / "sources" / source
    result = {}
    if source == "toxicn":
        for partition in ("train", "test"):
            items = _read_json(base / f"ToxiCN_ex/ToxiCN/data/{partition}.json")
            for index, item in enumerate(items):
                result[f"toxicn:{partition}:{index}"] = item
    elif source == "conda":
        for partition in ("train", "valid"):
            for item in _csv_rows(base / f"data/CONDA_{partition}.csv"):
                identifier = f"conda:{partition}:{item['Id']}"
                if identifier in result:
                    raise ValueError(f"Duplicate original CONDA ID: {identifier}")
                result[identifier] = item
    elif source in {"scamspot", "hatebr"}:
        relative = "data/comments-rated.csv" if source == "scamspot" else "dataset/HateBR.csv"
        key = "number" if source == "scamspot" else "id"
        for ordinal, item in enumerate(_csv_rows(base / relative), 2):
            result[f"{source}:{ordinal}:{item[key]}"] = item
    return result


def _load_shorttext(source: str) -> list[dict]:
    base = SHORTTEXT / "prepared" / source
    sessions = {row["post_id"]: row for row in _read_jsonl(base / "sessions.jsonl")}
    provenance = {row["post_id"]: row for row in _read_jsonl(base / "provenance.jsonl")}
    originals = _originals(source)
    text_key = {
        "conda": "utterance",
        "toxicn": "content",
        "scamspot": "msg",
        "hatebr": "comentario",
    }[source]
    original_keys = {
        "conda": ("intentClass",),
        "toxicn": ("toxic", "toxic_type", "expression", "target"),
        "scamspot": ("scam",),
        "hatebr": ("label_final", "anotator1", "anotator2", "anotator3"),
    }[source]
    records = []
    for message in _read_jsonl(base / "messages.jsonl"):
        session = sessions[message["post_id"]]
        ref = provenance[message["post_id"]]
        if session["source_id"] != ref["source_id"] or session["split"] != ref["split"]:
            raise ValueError(f"Frozen provenance mismatch: {session['post_id']}")
        original = originals[ref["source_id"]]
        _check_text(message["raw_text"], original[text_key], ref["source_id"])
        labels, mask = map_labels(source, original)
        binary = (
            _binary(original["toxic"])
            if source == "toxicn"
            else labels[2 if source == "scamspot" else 0]
        )
        if session["label"] != binary or ref["label"] != binary:
            raise ValueError(f"Original label does not match frozen subset: {ref['source_id']}")
        records.append(
            {
                "record_id": f"{source}:{message['comment_id']}",
                "source": source,
                "split": session["split"],
                "group_id": str(ref.get("split_group") or session["post_id"]),
                "session_id": session["post_id"],
                "message_id": message["comment_id"],
                "order": int(message.get("order", 0)),
                "parent_id": None,
                "text": message["raw_text"],
                "labels": labels,
                "observed_mask": mask,
                "original_label": {key: original.get(key) for key in original_keys},
                "source_ref": dict(ref),
                "duplicate_component": "",
            }
        )
    if len(records) != len(sessions):
        raise ValueError(f"Expected one complete comment per frozen {source} sample")
    return records


def _input_fingerprints() -> dict[str, str]:
    """Verify old pinned hashes before every generation/resume, not just on first use."""
    result = {}

    def verify(path: Path, expected: str | None = None) -> None:
        actual = file_hash(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input fingerprint mismatch: {path}")
        result[str(path.resolve())] = actual

    manifest = _read_json(SCCD / "manifest.json")
    verify(SCCD / "manifest.json")
    verify(RAW_COMMENTS, manifest["source_hashes"]["comments.csv"])
    for name in ("sessions.jsonl", "messages.jsonl"):
        expected = manifest["files_sha256"].get(name)
        if expected is None:
            expected = manifest["files_sha256"].get(str((SCCD / name).resolve()))
        if expected is None:
            raise ValueError(f"SCCD manifest lacks frozen {name} fingerprint")
        verify(SCCD / name, expected)
    for source in SOURCES[1:]:
        base = SHORTTEXT / "prepared" / source
        cohort = _read_json(base / "manifest.json")
        verify(base / "manifest.json")
        for name in ("messages.jsonl", "sessions.jsonl", "provenance.jsonl"):
            path = base / name
            verify(path, cohort["artifacts"][str(path.resolve())])
        for original_path, expected in cohort["configuration"]["source_hashes"].items():
            verify(Path(original_path), expected)
    return dict(sorted(result.items()))


def _validate(records: list[dict], sessions: list[dict]) -> None:
    ids = [row["record_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Prepared record IDs are not unique")
    session_by_id = {row["session_id"]: row for row in sessions}
    if len(session_by_id) != len(sessions):
        raise ValueError("Prepared SCCD session IDs are not unique")
    seen_by_session: dict[str, set[str]] = defaultdict(set)
    supervised_train, evaluation = set(), set()
    for row in records:
        if row["source"] not in SOURCES or row["split"] not in {"train", "dev", "test"}:
            raise ValueError("Unexpected prepared source/split")
        if (
            len(row["labels"]) != 3
            or len(row["observed_mask"]) != 3
            or not row["duplicate_component"]
        ):
            raise ValueError("Invalid labels, mask, or duplicate component")
        if row["source"] == "sccd":
            if session_by_id[row["session_id"]]["split"] != row["split"]:
                raise ValueError("SCCD prepared session/message split mismatch")
            seen_by_session[row["session_id"]].add(row["message_id"])
        if row["split"] == "train" and any(row["observed_mask"]):
            supervised_train.add(row["duplicate_component"])
        if row["split"] in {"dev", "test"}:
            evaluation.add(row["duplicate_component"])
    if supervised_train & evaluation:
        raise ValueError("Supervised train text component overlaps validation/test")
    for session in sessions:
        if set(session["message_ids"]) != seen_by_session[session["session_id"]]:
            raise ValueError("Prepared session message membership incomplete")


def _counts(records: list[dict]) -> dict:
    counts = {}
    for source in SOURCES:
        counts[source] = {}
        for split in ("train", "dev", "test"):
            rows = [row for row in records if row["source"] == source and row["split"] == split]
            counts[source][split] = {
                "records": len(rows),
                "fallback_records": sum(bool(row.get("is_fallback")) for row in rows),
                "categories": {
                    name: {
                        "observed": sum(row["observed_mask"][i] for row in rows),
                        "positive": sum(
                            row["labels"][i] for row in rows if row["observed_mask"][i]
                        ),
                        "negative": sum(
                            1 - row["labels"][i] for row in rows if row["observed_mask"][i]
                        ),
                        "unknown": sum(not row["observed_mask"][i] for row in rows),
                    }
                    for i, name in enumerate(CATEGORIES)
                },
            }
    return counts


def prepare_dataset(output_dir: Path) -> dict:
    """Create prepared data once; existing data must match inputs, code and output hashes."""
    output_dir = Path(output_dir)
    prepared = output_dir / "prepared"
    inputs = _input_fingerprints()
    configuration = {
        "protocol": PROTOCOL,
        "categories": list(CATEGORIES),
        "sources": list(SOURCES),
        "near_threshold": NEAR_THRESHOLD,
        "data_code_sha256": file_hash(Path(__file__)),
        "input_hashes": inputs,
    }
    fingerprint = _hash(json.dumps(configuration, sort_keys=True, ensure_ascii=False))
    manifest_path = prepared / "manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest["fingerprint"] != fingerprint:
            raise ValueError("Prepared resume fingerprint mismatch; use a new output directory")
        for name, expected in manifest["artifacts"].items():
            if file_hash(prepared / name) != expected:
                raise ValueError(f"Prepared artifact fingerprint mismatch: {name}")
        _validate(_read_jsonl(prepared / "records.jsonl"), _read_jsonl(prepared / "sessions.jsonl"))
        return manifest
    if prepared.exists() and any(prepared.iterdir()):
        raise ValueError(
            "Incomplete prepared directory; preserve it and use a new output directory"
        )
    records, sessions = _load_sccd()
    for source in SOURCES[1:]:
        records.extend(_load_shorttext(source))
    original_counts = _counts(records)
    kept, exclusions, components, duplicate_audit = clean_records(records)
    session_records: dict[str, list[dict]] = defaultdict(list)
    for row in kept:
        if row["source"] == "sccd":
            session_records[row["session_id"]].append(row)
    additional_fallbacks = []
    for session in sessions:
        rows = sorted(session_records[session["session_id"]], key=lambda row: row["order"])
        if not rows:
            fallback = _fallback(session, "all_comments_excluded")
            additional_fallbacks.append(fallback)
            rows = [fallback]
            session["fallback_reason"] = "all_comments_excluded"
        session["message_ids"] = [row["message_id"] for row in rows]
        session["retained_message_count"] = sum(not row.get("is_fallback", False) for row in rows)
        session["excluded_message_count"] = (
            session["original_message_count"] - session["retained_message_count"]
        )
    if additional_fallbacks:
        # Include root fallback text in the same component universe for OOF, while
        # keeping it unknown rather than inheriting the conversation's gold label.
        kept.extend(additional_fallbacks)
        mapping, components, fallback_audit = duplicate_components(kept + exclusions)
        for row in kept + exclusions:
            row["duplicate_component"] = mapping[row["record_id"]]
        duplicate_audit["post_exclusion_fallback_component_audit"] = fallback_audit
    _validate(kept, sessions)
    kept.sort(
        key=lambda row: (
            SOURCES.index(row["source"]),
            row["split"],
            row["session_id"],
            row["order"],
        )
    )
    sessions.sort(key=lambda row: (row["split"], row["session_id"]))
    audit = {
        "protocol": PROTOCOL,
        "original_counts": original_counts,
        "retained_counts": _counts(kept),
        "duplicate_audit": duplicate_audit,
        "session_counts": dict(Counter(row["split"] for row in sessions)),
        "sccd_sessions_with_exclusions": sum(row["excluded_message_count"] > 0 for row in sessions),
        "sccd_fallback_sessions": [
            {
                "session_id": row["session_id"],
                "split": row["split"],
                "reason": row["fallback_reason"],
            }
            for row in sessions
            if "fallback_reason" in row
        ],
        "label_mapping": {
            "sccd": "raw comment CB/Non-CB only; conversation labels never copied to comments",
            "conda": "E/I attack=1, A/O attack=0; hate/spam unknown",
            "toxicn": "non-reporting toxic->attack; type2->hate; expression3 attack/hate unknown",
            "scamspot": "original spam OR scam binary -> spam_scam; attack/hate unknown",
            "hatebr": "original label_final offensiveness -> attack; hate/spam unknown",
        },
        "known_boundaries": [
            "New comment-level supervised protocol, not the frozen CST-r0 supervision.",
            "SCCD/CONDA/HateBR label definitions differ; per-source evaluation remains necessary.",
            "ToxiCN reporting is retained but not treated as a negative or an enacted attack.",
            "ScamSpot combines spam/scam; no pure fraud label is asserted.",
            "Fixed dev/test dependencies are retained and disclosed, not claimed independent.",
            "SCCD train sessions may lose repeated comments; root fallback labels remain unknown.",
        ],
    }
    prepared.mkdir(parents=True, exist_ok=True)
    _write_jsonl(prepared / "records.jsonl", kept)
    _write_jsonl(prepared / "sessions.jsonl", sessions)
    _write_jsonl(prepared / "exclusions.jsonl", exclusions)
    _write_jsonl(prepared / "duplicate_components.jsonl", components)
    _write_json(prepared / "audit.json", audit)
    # Recheck inputs before finalizing the manifest; concurrent changes invalidate this run.
    if _input_fingerprints() != inputs:
        raise ValueError("Frozen inputs changed during preparation; manifest not finalized")
    manifest = {
        "protocol": PROTOCOL,
        "fingerprint": fingerprint,
        "configuration": configuration,
        "categories": list(CATEGORIES),
        "record_count": len(kept),
        "session_count": len(sessions),
        "counts": audit["retained_counts"],
        "original_counts": original_counts,
        "session_counts": audit["session_counts"],
        "artifacts": {
            name: file_hash(prepared / name)
            for name in (
                "records.jsonl",
                "sessions.jsonl",
                "exclusions.jsonl",
                "duplicate_components.jsonl",
                "audit.json",
            )
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def read_records(prepared_dir: Path) -> list[dict]:
    return _read_jsonl(Path(prepared_dir) / "records.jsonl")


def read_sessions(prepared_dir: Path) -> list[dict]:
    return _read_jsonl(Path(prepared_dir) / "sessions.jsonl")
