"""Fixed case selection and gold-label linkage, without changing old cohorts."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

from .common import (OLD_CASES, OUTPUT, data_profile, digest, freeze_protocol, identity,
                     load_data, read, save, write_jsonl)


def prepare(output: Path = OUTPUT) -> dict:
    protocol = freeze_protocol(output)
    records, sessions = load_data()
    groups = defaultdict(list)
    for row in records:
        if row["source"] == "sccd":
            groups[str(row["session_id"])].append(row)
    old = read(OLD_CASES)
    old_real = {str(c["session"].get("session_id", c["id"].removeprefix("sccd:")))
                for c in old["cases"] if c["kind"] == "real_dev"}
    if len(old_real) != 30:
        raise ValueError("Expected thirty previously reviewed real sessions")
    lengths = [len(groups[str(s["session_id"])]) for s in sessions if s["split"] == "train"]
    # Use the same previously fixed train-only development boundaries.
    bounds = old["length_boundaries_train_only"]
    buckets = defaultdict(list)
    for session in sessions:
        sid = str(session["session_id"])
        if session["split"] != "dev" or sid in old_real:
            continue
        band = int(np.searchsorted(bounds, len(groups[sid]), side="right"))
        buckets[int(session["label"]), band].append(session)
    chosen = []
    for label in (0, 1):
        for band in range(3):
            values = sorted(buckets[label, band], key=lambda s: identity({"seed": 42, "session_id": s["session_id"]}))
            if len(values) < 5:
                raise RuntimeError("Fixed new-review cell has fewer than five cases")
            chosen += values[:5]
    new_cases = []
    for session in chosen:
        sid = str(session["session_id"])
        messages = sorted(groups[sid], key=lambda r: (r["order"], r["message_id"]))
        new_cases.append({"id": f"sccd-new:{sid}", "kind": "new_dev_assisted_review", "session": session,
                          "previous_reviewed": False, "first_unseen_test": False,
                          "eval_groups": sorted({r["eval_group"] for r in messages}), "messages": messages})
    by_id = {(r["session_id"], r["message_id"]): r for r in records if r["source"] == "sccd"}
    regression = []
    linked, absent = 0, []
    for original in old["cases"]:
        case = dict(original)
        if case["kind"] == "real_dev":
            sid = str(case["session"].get("session_id", case["id"].removeprefix("sccd:")))
            current = []
            for message in case["messages"]:
                key = sid, str(message["comment_id"])
                if key not in by_id:
                    absent.append({"case_id": case["id"], "message_id": key[1], "reason": "not_in_frozen_full_cohort"})
                    continue
                current.append(by_id[key]); linked += 1
            case["messages"] = current
            case["eval_groups"] = sorted({r["eval_group"] for r in current})
        else:
            current = []
            sid = str(case["id"])
            for i, message in enumerate(case["messages"]):
                mid = str(message.get("comment_id", i))
                current.append({"record_id": f"{sid}:{mid}", "message_id": mid, "session_id": sid,
                    "source": "authored_challenge", "split": "development_challenge", "order": message.get("order", i),
                    "parent_id": message.get("parent_id"), "text": message["raw_text"],
                    "labels": [0, 0, 0], "observed_mask": [False, False, False],
                    "duplicate_component": f"authored:{sid}:{mid}", "eval_group": f"authored:{sid}"})
            case["messages"] = current
        regression.append(case)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "prepared"
    value = {"protocol_identity": protocol["identity"], "source_cases_sha256": digest(OLD_CASES),
             "old_cases": regression, "new_cases": new_cases, "length_boundaries_train_only": bounds,
             "current_train_length_tertiles_observation": np.quantile(lengths, (1/3, 2/3)).tolist(),
             "bucket_availability": {f"label{label}_band{band}": len(values) for (label,band), values in buckets.items()},
             "linked_old_gold_records": linked, "old_messages_absent_after_current_cleaning": absent,
             "annotation_policy": "Current observed masks linked by actual IDs; authored challenges never become risk gold; reviewer is model-assisted, not human"}
    if (target / "cases.json").exists() and read(target / "cases.json") != value:
        raise RuntimeError("Prepared review case selection changed on resume")
    save(target / "cases.json", value)
    save(target / "cohort_profile.json", data_profile(records, sessions))
    write_jsonl(target / "case_selection.jsonl", [{"case_id": c["id"], "kind": c["kind"],
        "session_id": c["session"].get("session_id", c["id"]), "n_messages": len(c["messages"]),
        "eval_groups": c.get("eval_groups", []), "human_gold": False} for c in regression + new_cases])
    return {"old_regression": len(regression), "new_assisted_cases": len(new_cases),
            "old_gold_records_linked": linked, "old_records_absent": len(absent)}
