"""Frozen, text-only short-message cohorts; no model fitting or paid calls.

Every source is acquired at a fixed revision, every acquired eligible text is
deduplicated, and every removal is auditable. Near matching uses an exact
character 5-gram Jaccard test after a lossless prefix candidate filter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

PROJECT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = PROJECT / "benchmark/research_followup/20261004/shorttext_expansion"
PROTOCOL = "cst-shorttext-20261004-v1"
DEFAULT_DATASETS = ("conda", "toxicn", "scamspot", "hatebr")
TARGETS = {name: {"train": 800, "dev": 200, "test": 200} for name in DEFAULT_DATASETS}
PINS = {
    "conda": ("usydnlp/CONDA", "3fb90675f881079c85c1de1f25df1bf8d521194f"),
    "toxicn": ("DUT-lujunyu/ToxiCN", "91d8f6a449967bd310364544d535d6310e45043b"),
    "scamspot": (
        "ScamSpot/scamspot_ig-comment-scraper",
        "84f5ef098052b06f41dd66bb6035429b32879da4",
    ),
    "hatebr": ("franciellevargas/HateBR", "a1301437a7b21620fbf0110a300c4c927a247daf"),
}
FILES = {
    "conda": ("README.md", "data/CONDA_train.csv", "data/CONDA_valid.csv", "data/CONDA_test.csv"),
    "toxicn": (
        "README.md",
        "ToxiCN_1.0.csv",
        "ToxiCN_ex/ToxiCN/data/train.json",
        "ToxiCN_ex/ToxiCN/data/test.json",
    ),
    "scamspot": ("README.md", "LICENSE", "data/comments-rated.csv"),
    "hatebr": ("README.md", "LICENSE", "dataset/HateBR.csv"),
}
SOURCE_SHA256 = {
    (
        "conda",
        "data/CONDA_train.csv",
    ): "fe7b18d19deae8c5a465163247dc929baea643f381e7f28596fec0b1559bddcd",
    (
        "conda",
        "data/CONDA_valid.csv",
    ): "4ed5b3eee87131014cdd373fe0303bc57c246cf10c5b864f08bd1ef8306d4949",
    (
        "conda",
        "data/CONDA_test.csv",
    ): "487eb6bc8336c5d5bea1ebad593f59b9f157209d987537572d99930ab01518af",
    (
        "scamspot",
        "data/comments-rated.csv",
    ): "097bdb0fce366bfcea86e504586dda379eb02f7367f2d02320aceb412421d901",
}
DATASET_NOTES = {
    "conda": {
        "language": "English; non-English conversations excluded by original authors",
        "risk_definition": "E explicit or I implicit toxicity versus A action or O other",
        "label_origin": "Original gamer-annotated utterance intent; not criminal-offense gold",
        "license": "No explicit dataset license found in the pinned author repository",
        "unit": "Released utterance; may merge consecutive same-player turns with [SEPA]",
        "split_policy": "Only public train+valid gold; split by match ID; official test unlabeled",
    },
    "toxicn": {
        "language": "Chinese",
        "risk_definition": "toxic=1 versus toxic=0; offense/hate, not a fraud label",
        "label_origin": "Original fine-grained human annotation on Zhihu/Tieba comments",
        "license": "CC BY-NC-ND 4.0; pinned README; no redistribution of modified source datasets",
        "unit": "Independent comment; text is content only",
        "split_policy": (
            "Official test stays test; original training pool split train/dev by component"
        ),
    },
    "scamspot": {
        "language": "English Instagram comments; not a language-gold annotation",
        "risk_definition": "scam=1 is the original spam OR scam label; not pure fraud",
        "label_origin": "Original manually rated comments; source binary label is retained",
        "license": (
            "Repository LICENSE is MIT (Software wording); "
            "no separate dataset-specific license found"
        ),
        "unit": "Independent Instagram comment; msg only",
        "split_policy": "Fixed component hash split; no fabricated conversation",
    },
    "hatebr": {
        "language": "Brazilian Portuguese",
        "risk_definition": "label_final=1 offensive versus 0 non-offensive; not gambling",
        "label_origin": "Three expert annotators; public version 2.0 binary offensiveness",
        "license": "CC BY-NC 4.0; pinned author LICENSE and README",
        "unit": "Independent Instagram comment; comentario only",
        "split_policy": (
            "Original Instagram post URL separated splits after deduplication; "
            "replacement selected before model scores exist"
        ),
    },
}


def stable(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_write(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def jsonl_read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalized(text: str) -> str:
    """Full content, NFKC/casefold/whitespace removal; no URL/mention masking."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def _grams(text: str) -> frozenset[str]:
    return (
        frozenset(text[i : i + 5] for i in range(len(text) - 4))
        if len(text) >= 5
        else frozenset({text})
    )


def jaccard(a: str, b: str) -> float:
    a, b = normalized(a), normalized(b)
    if len(a) < 5 or len(b) < 5:
        return float(a == b)
    x, y = _grams(a), _grams(b)
    return len(x & y) / len(x | y)


def _verify(path: Path, sha256: str | None = None, git_blob: str | None = None) -> None:
    if sha256 and digest(path) != sha256:
        raise ValueError(f"Source/artifact hash mismatch: {path}")
    if git_blob:
        hasher = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
        if hasher.hexdigest() != git_blob:
            raise ValueError(f"Frozen Git blob hash mismatch: {path}")


def fetch(url: str, path: Path, *, sha256: str | None = None, git_blob: str | None = None) -> dict:
    """Atomic source download; cached files must still pass the frozen hashes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix(path.suffix + ".part")
        for attempt in range(3):
            try:
                with (
                    urlopen(
                        Request(url, headers={"User-Agent": "CST-MIL-source-audit/1.0"}), timeout=60
                    ) as remote,
                    partial.open("wb") as handle,
                ):
                    while chunk := remote.read(1024 * 1024):
                        handle.write(chunk)
                _verify(partial, sha256, git_blob)
                partial.replace(path)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
    _verify(path, sha256, git_blob)
    return {
        "path": str(path.resolve()),
        "url": url,
        "sha256": digest(path),
        "git_blob_sha1": git_blob,
        "bytes": path.stat().st_size,
    }


def _label(value: object) -> int:
    if str(value).strip() not in {"0", "1"}:
        raise ValueError(f"Unexpected binary source label: {value!r}")
    return int(str(value).strip())


def parse_conda(handle: TextIO, partition: str) -> list[dict]:
    reader = csv.DictReader(handle)
    required = {"Id", "matchId", "conversationId", "utterance"}
    if not required <= set(reader.fieldnames or []):
        raise ValueError("CONDA required fields missing")
    if partition == "test":
        if "intentClass" in (reader.fieldnames or []):
            raise ValueError("Official CONDA test unexpectedly has gold; review frozen protocol")
        return []
    rows = []
    for ordinal, row in enumerate(reader, 2):
        if row.get("intentClass") not in {"E", "I", "A", "O"}:
            raise ValueError(f"Unexpected CONDA intent label at {ordinal}")
        rows.append(
            {
                "source_id": f"conda:{partition}:{row['Id']}",
                "source_partition": partition,
                "source_row": ordinal,
                "source_label": row["intentClass"],
                "label": int(row["intentClass"] in {"E", "I"}),
                "text": row["utterance"],
                "match_id": row["matchId"],
                "conversation_id": row["conversationId"],
            }
        )
    return rows


def parse_toxicn(payload: list[dict], partition: str) -> list[dict]:
    if not isinstance(payload, list):
        raise ValueError("ToxiCN JSON must be a list")
    rows = []
    for index, item in enumerate(payload):
        if not isinstance(item.get("content"), str):
            raise ValueError("ToxiCN content must be text")
        rows.append(
            {
                "source_id": f"toxicn:{partition}:{index}",
                "source_partition": partition,
                "source_split": partition,
                "source_row": index,
                "label": _label(item["toxic"]),
                "source_label": item["toxic"],
                "text": item["content"],
                "source_platform": item.get("platform"),
                "source_topic": item.get("topic"),
            }
        )
    return rows


def parse_scamspot(handle: TextIO) -> list[dict]:
    reader = csv.DictReader(handle)
    if not {"number", "scam", "msg"} <= set(reader.fieldnames or []):
        raise ValueError("ScamSpot required fields missing")
    return [
        {
            "source_id": f"scamspot:{ordinal}:{r['number']}",
            "source_partition": "all",
            "source_row": ordinal,
            "label": _label(r["scam"]),
            "source_label": r["scam"],
            "text": r["msg"],
        }
        for ordinal, r in enumerate(reader, 2)
    ]


def parse_hatebr(handle: TextIO) -> list[dict]:
    reader = csv.DictReader(handle)
    if not {"id", "comentario", "label_final"} <= set(reader.fieldnames or []):
        raise ValueError("HateBR required fields missing")
    return [
        {
            "source_id": f"hatebr:{ordinal}:{r['id']}",
            "source_partition": "all",
            "source_row": ordinal,
            "label": _label(r["label_final"]),
            "source_label": r["label_final"],
            "text": r["comentario"],
            "source_post": r.get("links_post") or None,
            "annotators": {k: r[k] for k in ("anotator1", "anotator2", "anotator3") if k in r},
        }
        for ordinal, r in enumerate(reader, 2)
    ]


def _length_summary(values: list[int]) -> dict:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "le30_fraction": None,
        }
    values = sorted(values)
    n = len(values)
    return {
        "n": n,
        "mean": sum(values) / n,
        "median": statistics.median(values),
        "p90": values[int((n - 1) * 0.9)],
        "p95": values[int((n - 1) * 0.95)],
        "max": values[-1],
        "le30_fraction": sum(v <= 30 for v in values) / n,
    }


def text_statistics(records: list[dict]) -> dict:
    return {
        "rows": len(records),
        "labels": dict(Counter(str(r.get("label")) for r in records)),
        "raw_chars": _length_summary([len(r["text"]) for r in records]),
        "nfkc_no_whitespace_chars": _length_summary(
            [len(re.sub(r"\s+", "", unicodedata.normalize("NFKC", r["text"]))) for r in records]
        ),
        "length_definition": (
            "Python Unicode codepoints; raw preserves whitespace; "
            "NFKC removes whitespace, preserves punctuation; no hard length exclusion"
        ),
    }


def deduplicate(
    records: list[dict], *, preserve_official_test: bool = False
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Exhaustive all-source near components; short strings require exact equality."""
    started = time.perf_counter()
    exact: dict[str, list[dict]] = defaultdict(list)
    audits: list[dict] = []
    for original in records:
        r = dict(original)
        if r["label"] not in {0, 1}:
            raise ValueError("Deduplication requires binary labels")
        text = normalized(r["text"])
        r["text_sha256"] = stable(text)
        if not text:
            audits.append(
                {k: v for k, v in r.items() if k != "text"} | {"reason": "empty_normalized_text"}
            )
        else:
            exact[text].append(r)
    texts = sorted(exact, key=stable)
    grams = [_grams(t) for t in texts]
    frequency = Counter(g for tokens in grams for g in tokens)
    parents = list(range(len(texts)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    inverted: dict[str, list[int]] = defaultdict(list)
    pairs = 0
    for i in sorted(range(len(texts)), key=lambda index: (len(grams[index]), stable(texts[index]))):
        if len(texts[i]) < 5:
            continue
        tokens = grams[i]
        prefix_len = len(tokens) - ((85 * len(tokens) + 99) // 100) + 1
        prefix = sorted(tokens, key=lambda g: (frequency[g], g))[:prefix_len]
        candidates = {j for token in prefix for j in inverted[token]}
        for j in candidates:
            other = grams[j]
            if 100 * len(other) < 85 * len(tokens):
                continue
            pairs += 1
            intersection = len(tokens & other)
            if 100 * intersection >= 85 * (len(tokens) + len(other) - intersection):
                parents[root(i)] = root(j)
        for token in prefix:
            inverted[token].append(i)
    components: dict[int, list[dict]] = defaultdict(list)
    for i, text in enumerate(texts):
        components[root(i)].extend(exact[text])
    kept, catalog = [], []
    conflict_count = official_count = 0
    for rows in components.values():
        rows.sort(key=lambda r: (stable(r["source_id"]), r["source_id"]))
        component_id = stable("|".join(sorted({r["text_sha256"] for r in rows})))
        conflict = len({r["label"] for r in rows}) > 1
        official_cross = preserve_official_test and {r.get("source_split") for r in rows} >= {
            "train",
            "test",
        }
        conflict_count += conflict
        official_count += official_cross
        catalog.append(
            {
                "component_id": component_id,
                "source_ids": [r["source_id"] for r in rows],
                "text_hashes": sorted({r["text_sha256"] for r in rows}),
                "labels": sorted({r["label"] for r in rows}),
                "conflicting_labels": conflict,
                "crosses_official_train_test": official_cross,
            }
        )
        if conflict or official_cross:
            reason = (
                "conflicting_component_labels"
                if conflict
                else "component_crosses_official_train_test"
            )
            rejected = rows
        else:
            representative = rows[0] | {"component_id": component_id}
            kept.append(representative)
            rejected = rows[1:]
            reason = "duplicate_component_nonrepresentative"
        audits.extend(
            {k: v for k, v in r.items() if k != "text"}
            | {"component_id": component_id, "reason": reason}
            for r in rejected
        )
    return (
        sorted(kept, key=lambda r: r["source_id"]),
        audits,
        sorted(catalog, key=lambda c: c["component_id"]),
        {
            "source_rows": len(records),
            "nonempty_exact_texts": len(texts),
            "components": len(catalog),
            "conflict_components": conflict_count,
            "official_cross_components": official_count,
            "retained_representatives": len(kept),
            "pair_checks": pairs,
            "seconds": time.perf_counter() - started,
            "scope": "All acquired eligible source texts, not only sampled candidates",
            "normalization": "NFKC + casefold + remove whitespace; URLs/mentions retained",
            "near_rule": (
                "Character 5-gram set Jaccard >= .85; exact-only for "
                "normalized strings shorter than 5"
            ),
            "component_policy": (
                "Transitive components; any conflicting binary labels quarantine whole "
                "component; stable SHA256(source_id) representative otherwise"
            ),
        },
    )


def assign_splits(records: list[dict], dataset: str, seed: int) -> list[dict]:
    result = []
    for original in records:
        r = dict(original)
        group = (
            r["match_id"]
            if dataset == "conda"
            else (r.get("source_post") or r["component_id"])
            if dataset == "hatebr"
            else r["component_id"]
        )
        bucket = int(stable(f"{seed}|{group}"), 16)
        if dataset == "toxicn":
            r["split"] = (
                "test" if r["source_split"] == "test" else ("dev" if bucket % 5 == 0 else "train")
            )
        else:
            r["split"] = "train" if bucket % 10 < 6 else ("dev" if bucket % 10 < 8 else "test")
        r["split_group"] = group
        result.append(r)
    return result


def select_balanced(
    records: list[dict], targets: dict[str, int], seed: int
) -> tuple[list[dict], dict, dict]:
    if any(n < 2 or n % 2 for n in targets.values()):
        raise ValueError("Each fixed target must be a positive even total")
    selected, counts, shortages = [], {}, {}
    for split, target in targets.items():
        groups = {
            label: sorted(
                [r for r in records if r["split"] == split and r["label"] == label],
                key=lambda r: (stable(f"{seed}|{r['source_id']}"), r["source_id"]),
            )
            for label in (0, 1)
        }
        size = min(target // 2, len(groups[0]), len(groups[1]))
        selected.extend(groups[0][:size] + groups[1][:size])
        counts[split] = {"0": size, "1": size}
        shortages[split] = 2 * size < target
    return selected, counts, shortages


def formal_eligible(counts: dict, minimum: int = 50) -> bool:
    return all(
        counts.get(split, {}).get(str(label), 0) >= minimum
        for split in ("dev", "test")
        for label in (0, 1)
    )


def canonical_unit(record: dict, dataset: str) -> tuple[dict, list[dict]]:
    post_id = dataset + ":" + stable(record["source_id"])[:24]
    session = {
        "post_id": post_id,
        "label": int(record["label"]),
        "split": record["split"],
        "post_raw_text": "",
        "source_id": record["source_id"],
        "dataset": dataset,
        "n_messages": 1,
        "comments": [],
    }
    messages = [
        {
            "post_id": post_id,
            "comment_id": post_id + ":message:0",
            "order": 0,
            "raw_text": record["text"],
            "model_text": record["text"],
            "parent_id": None,
            "speaker_id": None,
            "comment_time": None,
            "is_root": True,
            "comment_label": None,
        }
    ]
    return session, messages


def _input_fingerprint(records: list[dict]) -> str:
    return stable(
        json.dumps(
            sorted(
                (
                    r["source_id"],
                    stable(r["text"]),
                    r["label"],
                    r.get("source_split"),
                    r.get("match_id"),
                    r.get("source_post"),
                )
                for r in records
            ),
            ensure_ascii=False,
        )
    )


def prepare_records(
    records: list[dict],
    dataset: str,
    output: Path,
    *,
    seed: int = 42,
    targets: dict[str, int] | None = None,
    minimum_eval: int = 50,
    sources: list[dict] | None = None,
    source_statistics: dict | None = None,
) -> dict:
    """Pure-source preparation entry used by real acquisition and small QA fixtures."""
    output, sources = Path(output), sources or []
    targets = dict(targets or TARGETS[dataset])
    folder = output / "prepared" / dataset
    configuration = {
        "protocol": PROTOCOL,
        "dataset": dataset,
        "seed": seed,
        "targets": targets,
        "minimum_eval_per_class": minimum_eval,
        "data_code_sha256": digest(Path(__file__)),
        "input_fingerprint": _input_fingerprint(records),
        "source_hashes": {s["path"]: s["sha256"] for s in sources},
    }
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest, _, _ = load_dataset(dataset, output)
        if manifest["configuration"] != configuration:
            raise ValueError("Resume configuration changed; use a new output revision")
        return manifest
    for path, expected in configuration["source_hashes"].items():
        _verify(Path(path), expected)
    kept, exclusions, components, dedup_stats = deduplicate(
        records, preserve_official_test=dataset == "toxicn"
    )
    candidates = assign_splits(kept, dataset, seed)
    selected, counts, shortages = select_balanced(candidates, targets, seed)
    selected_ids = {r["source_id"] for r in selected}
    exclusions.extend(
        {k: v for k, v in r.items() if k != "text"} | {"reason": "not_selected_fixed_budget"}
        for r in candidates
        if r["source_id"] not in selected_ids
    )
    sessions, messages, provenance = [], [], []
    for r in selected:
        session, turns = canonical_unit(r, dataset)
        sessions.append(session)
        messages.extend(turns)
        provenance.append(
            {k: v for k, v in r.items() if k != "text"} | {"post_id": session["post_id"]}
        )
    # With a single component representative, there can be no component crossing.
    groups = defaultdict(set)
    for r in selected:
        groups[r["split_group"]].add(r["split"])
    if any(len(s) > 1 for s in groups.values()):
        raise ValueError("Group isolation invariant failed")
    catalog = [
        {k: v for k, v in r.items() if k != "text"}
        | {"raw_text_sha256": stable(r["text"]), "text_sha256": stable(normalized(r["text"]))}
        for r in records
    ]
    jsonl_write(folder / "sessions.jsonl", sessions)
    jsonl_write(folder / "messages.jsonl", messages)
    jsonl_write(folder / "provenance.jsonl", provenance)
    jsonl_write(folder / "source_catalog.jsonl", catalog)
    jsonl_write(folder / "components.jsonl", components)
    jsonl_write(folder / "exclusions.jsonl", exclusions)
    jsonl_write(
        folder / "candidates.jsonl",
        ({k: v for k, v in r.items() if k != "text"} for r in candidates),
    )
    for split in targets:
        jsonl_write(folder / split / "sessions.jsonl", (s for s in sessions if s["split"] == split))
        wanted = {s["post_id"] for s in sessions if s["split"] == split}
        jsonl_write(
            folder / split / "messages.jsonl", (m for m in messages if m["post_id"] in wanted)
        )
    save(folder / "source_index.json", sources)
    audit = {
        "raw": source_statistics or text_statistics(records),
        "dedup": dedup_stats,
        "eligible_candidates": text_statistics(candidates),
        "selected": text_statistics(selected),
        "exclusion_reasons": dict(Counter(r["reason"] for r in exclusions)),
        "available_by_split_label": {
            s: dict(Counter(str(r["label"]) for r in candidates if r["split"] == s))
            for s in targets
        },
        "notes": DATASET_NOTES.get(dataset, {}),
        "model_inputs": (
            "Only original Unicode message body; no metadata, message labels "
            "or invented reply edges"
        ),
        "message_evaluation": (
            "No Top-3 message accuracy; selected units are single messages "
            "and message gold is intentionally empty"
        ),
    }
    save(folder / "audit.json", audit)
    files = list(folder.glob("*.jsonl")) + [folder / "source_index.json", folder / "audit.json"]
    files += [folder / s / name for s in targets for name in ("sessions.jsonl", "messages.jsonl")]
    allowed = formal_eligible(counts, minimum_eval)
    manifest = {
        "dataset": dataset,
        "protocol_id": PROTOCOL,
        "configuration": configuration,
        "formal_comparison_allowed": allowed,
        "status": "ready"
        if formal_eligible(counts, minimum_eval)
        else "stopped_insufficient_eval_class_counts",
        "counts": counts,
        "target_counts": targets,
        "shortages": shortages,
        "n_sessions": len(sessions),
        "n_messages": len(messages),
        "splits": {s: [r["post_id"] for r in sessions if r["split"] == s] for s in targets},
        "cohort_fingerprint": stable(
            json.dumps(
                sorted((r["source_id"], r["text_sha256"], r["label"], r["split"]) for r in selected)
            )
        ),
        "repository": PINS.get(dataset, (None, None))[0],
        "revision": PINS.get(dataset, (None, None))[1],
        "source_hashes": configuration["source_hashes"],
        "artifacts": {str(path.resolve()): digest(path) for path in files},
    }
    save(manifest_path, manifest)
    return manifest


def load_dataset(
    dataset: str, output: Path = DEFAULT_OUTPUT
) -> tuple[dict, list[dict], list[dict]]:
    folder = Path(output) / "prepared" / dataset
    manifest = read(folder / "manifest.json")
    for path, expected in (manifest["artifacts"] | manifest["source_hashes"]).items():
        _verify(Path(path), expected)
    if manifest["configuration"]["data_code_sha256"] != digest(Path(__file__)):
        raise ValueError("Data code hash changed since preparation; use a new revision")
    return manifest, jsonl_read(folder / "sessions.jsonl"), jsonl_read(folder / "messages.jsonl")


def _acquire(dataset: str, output: Path) -> tuple[list[dict], dict[str, Path]]:
    repository, revision = PINS[dataset]
    folder = output / "sources" / dataset
    tree = fetch(
        f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1",
        folder / "git_tree.json",
    )
    payload = read(Path(tree["path"]))
    if payload.get("sha") != revision or payload.get("truncated"):
        raise ValueError("Source tree is not complete at the frozen commit")
    entries = {x["path"]: x for x in payload["tree"] if x["type"] == "blob"}
    sources, paths = [tree], {}
    for name in FILES[dataset]:
        if name not in entries:
            raise ValueError(f"Frozen source path missing: {dataset}/{name}")
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/{quote(name, safe='/')}"
        source = fetch(
            url,
            folder / name,
            sha256=SOURCE_SHA256.get((dataset, name)),
            git_blob=entries[name]["sha"],
        )
        sources.append(source)
        paths[name] = Path(source["path"])
    save(folder / "source_index.json", sources)
    sources.append(
        {
            "path": str((folder / "source_index.json").resolve()),
            "sha256": digest(folder / "source_index.json"),
            "url": None,
        }
    )
    return sources, paths


def _source_records(dataset: str, paths: dict[str, Path]) -> tuple[list[dict], dict]:
    records, stats = [], {}
    if dataset == "conda":
        for partition in ("train", "valid", "test"):
            path = paths[f"data/CONDA_{partition}.csv"]
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = parse_conda(handle, partition)
            if partition == "test":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    raw = list(csv.DictReader(handle))
                stats[partition] = text_statistics(
                    [{"text": r["utterance"], "label": None} for r in raw]
                ) | {"excluded_reason": "official_test_has_no_public_gold"}
            else:
                stats[partition] = text_statistics(rows)
            records.extend(rows)
    elif dataset == "toxicn":
        for partition in ("train", "test"):
            payload = json.loads(
                paths[f"ToxiCN_ex/ToxiCN/data/{partition}.json"].read_text(encoding="utf-8")
            )
            rows = parse_toxicn(payload, partition)
            stats[partition] = text_statistics(rows)
            records.extend(rows)
    else:
        path = paths["data/comments-rated.csv" if dataset == "scamspot" else "dataset/HateBR.csv"]
        with path.open(encoding="utf-8-sig", newline="") as handle:
            records = (parse_scamspot if dataset == "scamspot" else parse_hatebr)(handle)
        stats["all"] = text_statistics(records)
    stats["eligible_gold_pool"] = text_statistics(records)
    return records, stats


def _primary_attempt(url: str, folder: Path, name: str) -> dict:
    """Cache a real primary-source request, including blocked access evidence."""
    receipt_path, body_path = folder / f"{name}.json", folder / f"{name}.body"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt["url"] != url:
            raise ValueError("Primary evidence URL changed; use a new evidence revision")
        if receipt.get("body_sha256"):
            _verify(body_path, receipt["body_sha256"])
        return receipt
    folder.mkdir(parents=True, exist_ok=True)
    receipt = {"url": url, "method": "GET", "access_date": "2026-10-04"}
    try:
        with urlopen(Request(url, headers={"User-Agent": "CST-MIL-research/1.0"}), timeout=25) as r:
            body = r.read(4 * 1024 * 1024 + 1)
            receipt |= {
                "http_status": r.status,
                "final_url": r.geturl(),
                "content_type": r.headers.get("Content-Type"),
                "status": "retrieved",
                "truncated_at_4_mib": len(body) > 4 * 1024 * 1024,
            }
    except HTTPError as error:
        body = error.read(512 * 1024)
        receipt |= {
            "http_status": error.code,
            "final_url": error.geturl(),
            "status": "http_error",
            "error": str(error),
        }
    except OSError as error:
        body = b""
        receipt |= {"http_status": None, "status": "network_error", "error": str(error)}
    body_path.write_bytes(body)
    receipt |= {"body_path": str(body_path.resolve()), "body_sha256": digest(body_path)}
    save(receipt_path, receipt)
    return receipt


def _gambling_gate(output: Path) -> Path:
    """Record unverified prerequisites after actual primary-source attempts."""
    path = output / "prepared" / "gambling_gate.json"
    record_url = "https://zenodo.org/api/records/15424844"
    source = fetch(record_url, output / "sources" / "gambling_candidate" / "zenodo-record.json")
    metadata = read(Path(source["path"]))
    if metadata.get("id") != 15424844:
        raise ValueError("Gambling candidate record identity mismatch")
    folder = output / "sources" / "gambling_candidate"
    urls = (
        ("doi", "https://doi.org/10.1016/j.procs.2025.09.068"),
        ("publisher", "https://www.sciencedirect.com/science/article/pii/S1877050925027358"),
        (
            "author_publication",
            "https://alfhisa.github.io/publication/paper-4-online-gambling-comments-classification",
        ),
    )
    attempts = [_primary_attempt(url, folder, name) for name, url in urls]
    evidence = {
        "candidate_doi": "10.1016/j.procs.2025.09.068",
        "primary_access_attempts": attempts,
        "reviewed_primary_sources": [
            "Zenodo record 15424844 metadata and public file listing",
            "Author Muhammad Alfhi Saputra's publication page and Download Paper target",
            "Publisher-indexed abstract; direct full article access attempt",
        ],
        "verified": {
            "manual_subset": (
                "Author publication page and publisher abstract describe 10,101 comments; "
                "10 percent manually labeled Promotion/Non-Promotion, "
                "remaining comments used for pseudo-labeling"
            ),
            "author_paper_link": (
                "Author's Download Paper points to the same ScienceDirect article URL"
            ),
            "doi_response": (
                "DOI resolves to an Elsevier HTML redirect landing page, "
                "not the full paper; direct ScienceDirect returned HTTP 403"
            ),
            "zenodo_files": [f["key"] for f in metadata.get("files", [])],
        },
        "not_verified": [
            "Complete instructions and examples provided to original annotators",
            "Annotator count, independent double annotation and disagreement resolution",
            "Label review or adjudication procedure for the released manually labeled rows",
            "Full original paper and any supplementary annotation appendix",
        ],
        "interpretation": (
            "These items were not verified from the obtained primary materials; "
            "this is not evidence that the authors omitted the procedures. "
            "No gambling model was fitted and no model score influenced substitution."
        ),
    }
    evidence_path = folder / "evidence.json"
    if evidence_path.exists() and read(evidence_path) != evidence:
        raise ValueError("Primary evidence changed; use a new evidence revision")
    save(evidence_path, evidence)
    gate = {
        "candidate": "Dataset_Youtube_Online Gambling Promotions.xlsx",
        "zenodo_record": 15424844,
        "expected_workbook_sha256": (
            "2fdd04355e0972ebdc136069525070e552b28f7d45f776b5b360416506c57d71"
        ),
        "paper": "https://doi.org/10.1016/j.procs.2025.09.068",
        "record_source": source,
        "primary_evidence_path": str(evidence_path.resolve()),
        "primary_evidence_sha256": digest(evidence_path),
        "primary_access_attempts": attempts,
        "dataset_license_metadata": metadata.get("metadata", {}).get("license"),
        "complete_annotation_guideline_verified": False,
        "independent_label_review_verified": False,
        "status": "fallback_hatebr",
        "replacement": "hatebr",
        "reason": (
            "Actual publisher/DOI and author-page access attempts did not provide verified "
            "complete annotation instructions plus review/adjudication evidence; "
            "prerequisites were unmet before experiments, not asserted absent"
        ),
        "scope": (
            "HateBR is a Portuguese offense-domain substitute and does not "
            "establish gambling risk performance"
        ),
        "score_based_selection": False,
    }
    if path.exists() and read(path) != gate:
        raise ValueError("Gambling gate evidence changed; review before overwriting")
    save(path, gate)
    return path


def prepare_all(
    output: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    datasets: Iterable[str] | None = None,
    targets: dict[str, dict[str, int]] | None = None,
) -> Path:
    output = Path(output).resolve()
    if not output.is_relative_to(DEFAULT_OUTPUT.resolve()):
        raise ValueError(
            "Formal preparation output must remain under the authorized "
            "20261004 shorttext_expansion directory"
        )
    requested = tuple(str(name).lower() for name in (datasets or DEFAULT_DATASETS))
    if len(set(requested)) != len(requested) or any(
        name not in DEFAULT_DATASETS for name in requested
    ):
        raise ValueError("Unknown or repeated short-text dataset")
    chosen_targets = {name: dict((targets or {}).get(name, TARGETS[name])) for name in requested}
    if targets and any(name not in requested for name in targets):
        raise ValueError("Sample-size override for an unrequested dataset")
    for counts in chosen_targets.values():
        if set(counts) != {"train", "dev", "test"} or any(
            not isinstance(n, int) or n < 2 or n % 2 for n in counts.values()
        ):
            raise ValueError("Sample sizes must specify positive even train/dev/test totals")
    gate = _gambling_gate(output)
    entries = {}
    for dataset in requested:
        manifest_path = output / "prepared" / dataset / "manifest.json"
        if manifest_path.exists():
            manifest, _, _ = load_dataset(dataset, output)
            if (
                manifest["configuration"]["seed"] != seed
                or manifest["target_counts"] != chosen_targets[dataset]
            ):
                raise ValueError("Resume configuration changed; use a new output revision")
        else:
            sources, paths = _acquire(dataset, output)
            records, stats = _source_records(dataset, paths)
            manifest = prepare_records(
                records,
                dataset,
                output,
                seed=seed,
                sources=sources,
                targets=chosen_targets[dataset],
                source_statistics=stats,
            )
        entries[dataset] = {
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": digest(manifest_path),
            "status": manifest["status"],
            "formal_comparison_allowed": manifest["formal_comparison_allowed"],
            "counts": manifest["counts"],
        }
        print(
            json.dumps(
                {"dataset": dataset, "status": manifest["status"], "counts": manifest["counts"]}
            ),
            flush=True,
        )
    result = output / "prepared" / "manifest.json"
    save(
        result,
        {
            "protocol_id": PROTOCOL,
            "seed": seed,
            "datasets": entries,
            "gambling_gate": str(gate.resolve()),
            "gambling_gate_sha256": digest(gate),
            "data_code_sha256": digest(Path(__file__)),
        },
    )
    (output / "prepared" / "PREPARATION.md").write_text(
        "# Short-text preparation\n\n"
        "Frozen, independently prepared supplementary datasets; "
        "old SCCD and prior experiments are untouched. "
        "All acquired eligible texts enter exact and transitive "
        "character-5-gram Jaccard >=0.85 deduplication. "
        "Conflicting components are quarantined, a stable representative is selected otherwise. "
        "No hard 30-character cutoff is applied; raw and "
        "NFKC-without-whitespace length statistics are separate.\n\n"
        "CONDA uses public annotated messages only and match-separated splits; "
        "it does not use unannotated official test as negatives. "
        "ToxiCN preserves its official test, quarantining components spanning "
        "the original train/test boundary. HateBR uses real Instagram post URL groups. "
        "ScamSpot uses fixed component hashing. Each split is balanced, "
        "shrinks equally if necessary, "
        "and formal comparison stops if dev or test has fewer than 50 samples in either class.\n\n"
        "Each unit is one original Unicode message. Metadata, original source labels, "
        "user IDs and empty message gold do not enter features; parent_id is null. "
        "Do not report message Top-3 accuracy or MIL/reply-structure superiority. "
        "ScamSpot means spam OR scam. HateBR is an offense-domain fallback, "
        "not gambling evidence.\n\n"
        "See each dataset's source_index.json, audit.json, source_catalog.jsonl, "
        "components.jsonl, exclusions.jsonl "
        "and provenance.jsonl. load_dataset verifies artifact, source and data-code fingerprints. "
        "Source licences remain binding; originals are local research inputs, "
        "not packaged redistributable datasets.\n\n"
        "Command: `python -m cst_mil.research_followup.shorttext.data --seed 42`\n",
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--datasets", nargs="+", choices=DEFAULT_DATASETS, default=list(DEFAULT_DATASETS)
    )
    parser.add_argument(
        "--sample-sizes",
        type=json.loads,
        default=None,
        help="JSON mapping dataset names to train/dev/test total counts",
    )
    args = parser.parse_args()
    print(prepare_all(args.output, args.seed, args.datasets, args.sample_sizes))
