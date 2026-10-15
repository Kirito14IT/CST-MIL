"""Source-bound cost tables; inherited fitting and fresh raw inference stay separate."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import ABLATIONS, BASE, OUTPUT, SEEDS, digest, read, save, source_hashes, verify_hashes


def _csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_cost_tables(output: Path = OUTPUT) -> dict:
    root = output / "cost"
    summary = read(root / "summary.json")
    old_hashes = read(output / "preservation_before.json")
    inherited, new_fits, input_files = (
        [],
        [],
        [root / "summary.json", root / "dependency_inventory.json"],
    )
    for seed in SEEDS:
        path = BASE / "runs/cst_v1_1" / f"seed_{seed}/run_manifest.json"
        verify_hashes({str(path): old_hashes[str(path)]})
        source = read(path)
        inherited.append(
            {
                "seed": seed,
                "fitting_with_cached_vectors_seconds": source["training_seconds_cached"],
                "shared_TRAIN_DEV_encoding_seconds": source["shared_encoder_training_seconds"],
                "training_plus_encoding_seconds": source["training_seconds_cold"],
                "cold_definition": source["cold_training_definition"],
                "source": str(path),
                "source_sha256": digest(path),
                "measurement": (
                    "inherited accepted training of identical frozen full weights; "
                    "not this round's import time"
                ),
            }
        )
        input_files.append(path)
    for variant in ("context_a", "context_b", *ABLATIONS):
        for seed in SEEDS:
            folder = output / "runs" / variant / f"seed_{seed}"
            verify_hashes(read(folder / "complete.json")["artifacts"])
            run = read(folder / "manifest.json")
            new_fits.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "fit_seconds_cached": run["training_seconds_cached"],
                    "stage_wall_seconds_including_evidence": run["wall_stage_seconds"],
                    "peak_tree_RSS_bytes": run["resources"]["peak_tree_rss_bytes"],
                    "model_bytes": run["model_bytes"],
                    "encoder_bytes_separate": summary["encoder_asset_bytes"],
                    "threads": 4,
                    "GPU_required": False,
                    "scope": (
                        "resident shared fixed TRAIN/DEV vectors, sequential harness; "
                        "not cold encoder training"
                    ),
                }
            )
            input_files.append(folder / "manifest.json")
    repeats = summary["raw_repeats"]
    scalar = [
        {
            "metric": "complete_TEST_raw_inference_mean",
            "value": np.mean([r["total_seconds"] for r in repeats]),
            "unit": "seconds",
        },
        {
            "metric": "complete_TEST_raw_inference_sample_SD",
            "value": np.std([r["total_seconds"] for r in repeats], ddof=1),
            "unit": "seconds",
        },
        {
            "metric": "model_package_load",
            "value": summary["model_package_load_seconds"],
            "unit": "seconds",
        },
        {
            "metric": "encoder_first_load_and_TRAIN_warmup",
            "value": summary["encoder_initial_load_and_first_warmup_seconds"],
            "unit": "seconds",
        },
        {
            "metric": "raw_inference_peak_tree_RSS",
            "value": max(r["resources"]["peak_tree_rss_bytes"] for r in repeats),
            "unit": "bytes",
        },
        {
            "metric": "startup_peak_tree_RSS",
            "value": summary["startup_resources"]["peak_tree_rss_bytes"],
            "unit": "bytes",
        },
        {"metric": "trained_head_package", "value": summary["model_bytes"], "unit": "bytes"},
        {
            "metric": "required_shared_encoder_assets",
            "value": summary["encoder_asset_bytes"],
            "unit": "bytes",
        },
    ]
    _csv(root / "inherited_full_training.csv", inherited)
    _csv(root / "new_fit_costs.csv", new_fits)
    _csv(root / "raw_cost_overview.csv", scalar)
    result = {
        "status": "completed",
        "new_fit_rows": len(new_fits),
        "raw_input_records_each_pass": repeats[0]["record_count"],
        "resident_repeats": 3,
        "notes": [
            "Fresh raw timings include encoding and risk inference, not topic naming.",
            "RSS includes evaluation harness, full corpus and reference predictions; "
            "not solely service weights.",
            "Each SCCD unit is a full conversation; other sources use canonical batches up to64.",
            "Per-pass medians/P95 are latency quantiles, not confidence intervals.",
            "Original fitting times are inherited; importing existing weights is not training.",
            "Installed dependency logical bytes include development files; "
            "they are not a measured minimal runtime.",
            "No paid API was called; hardware, electricity and training remain local costs.",
        ],
        "input_hashes": source_hashes([*input_files, Path(__file__)]),
        "artifacts": source_hashes(
            root / name
            for name in (
                "inherited_full_training.csv",
                "new_fit_costs.csv",
                "raw_cost_overview.csv",
            )
        ),
    }
    save(root / "table_audit.json", result)
    return result
