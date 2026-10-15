"""Merge two isolated model-assisted judgments without creating a gold label."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import OUTPUT, identity, read, save, source_hashes, verify_hashes
from .review import REVIEW_DIMENSIONS, merge_assisted_reviews


def summarize_reviews(output: Path = OUTPUT) -> dict:
    folder = output / "topics"
    files = [folder / "reviews" / f"reviewer_{r}.json" for r in (1, 2)]
    values = [read(path) for path in files]
    mapping_path = folder / "packets/audit_only_identity_mapping.json"
    mapping = read(mapping_path)
    for reviewer, rows in enumerate(values, 1):
        index = read(folder / "packets" / f"reviewer_{reviewer}" / "index.json")
        expected = {
            (r["case_id"], alias)
            for r in index["cases"]
            for alias in mapping[f"reviewer_{reviewer}"].values()
        }
        actual = {(r["case_id"], r["anonymous_configuration"]) for r in rows}
        if len(rows) != 150 or actual != expected:
            raise ValueError(
                f"Reviewer {reviewer} has missing/duplicate/unexpected case configuration"
            )
        for row in rows:
            if not (row.get("evidence") or row.get("justification") or row.get("rationale")):
                raise ValueError("Assisted ratings require individual source-grounded explanations")
    result = merge_assisted_reviews(values[0], values[1], mapping)
    summaries = []
    for reviewer, rows in enumerate(result["review_records"], 1):
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["internal_variant"]].append(row)
        for variant, items in sorted(grouped.items()):
            for dimension in REVIEW_DIMENSIONS:
                observed = [r[dimension] for r in items if r.get(dimension) is not None]
                summaries.append(
                    {
                        "reviewer": reviewer,
                        "variant": variant,
                        "dimension": dimension,
                        "case_count": len(items),
                        "rated_count": len(observed),
                        "unrated_count": len(items) - len(observed),
                        "ordinal_mean_descriptive_only": float(np.mean(observed))
                        if observed
                        else None,
                        "ratings_0_1_2": [observed.count(i) for i in (0, 1, 2)],
                    }
                )
    result.update(
        descriptive_ordinal_summaries=summaries,
        disagreements_count=len(result["disagreements"]),
        review_context=(
            "two separate agent contexts; same AI family possible; "
            "not statistically independent human annotators"
        ),
        limitations=[
            "Anonymous configuration labels hide mapping, not visibly different outputs.",
            "All thirty cases are development cases and may share dependency clusters.",
            "Ordinal averages describe assisted judgments; "
            "they are not theme accuracy or human agreement.",
            "Disagreements and null ratings are retained; no majority label is invented.",
        ],
        input_hashes=source_hashes([*files, mapping_path, Path(__file__)]),
    )
    result["identity"] = identity(result)
    target = folder / "assisted_review_summary.json"
    if target.exists():
        old = read(target)
        if old != result:
            raise RuntimeError(
                "Assisted review summary identity changed; preserve previous material"
            )
        verify_hashes(old["input_hashes"])
        return old
    save(target, result)
    return result
