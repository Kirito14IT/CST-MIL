"""python -m cst_mil.research_followup.full_joint_comparison --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import METHODS, SEEDS
from .common import DEFAULT_OUTPUT, campaign_lock, guard_output, preservation, save


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full joint five-source supervised comparison")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, choices=range(1, 5), default=4)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "status", "summarize", "plot", "verify-preservation"):
        p = sub.add_parser(name)
        if name == "summarize":
            p.add_argument("--bootstrap", type=int, default=2000)
    for name in ("train", "evaluate", "cost", "all"):
        p = sub.add_parser(name)
        p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
        p.add_argument(
            "--seeds",
            nargs="+",
            type=int,
            choices=SEEDS,
            default=[42] if name == "cost" else list(SEEDS),
        )
        p.add_argument("--resume", action="store_true")
    p = sub.add_parser("encode")
    p.add_argument("--split", choices=("train", "dev", "test"), required=True)
    for name in ("_train", "_evaluate", "_cost"):
        p = sub.add_parser(name)
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--seed", type=int, choices=SEEDS, required=True)
    p = sub.add_parser("_encode")
    p.add_argument("--split", choices=("train", "dev", "test"), required=True)
    args = parser.parse_args(argv)
    from cst_mil.bench.resources import configure_cpu

    cpu = configure_cpu(args.threads)
    output = guard_output(args.output)
    output.mkdir(parents=True, exist_ok=True)
    from .execution import (
        encode_split,
        evaluate_one,
        freeze_experiment,
        launch_stage,
        status,
        train_one,
    )

    if args.command == "status":
        result = status(output)
    elif args.command == "prepare":
        from .data import prepare

        preservation(output)
        manifest = prepare(output)
        save(output / "cpu_policy.json", cpu)
        result = {
            k: manifest[k]
            for k in ("record_count", "session_count", "candidate_counts", "session_counts")
        }
    elif args.command in {"encode", "_encode"}:
        if args.command == "encode":
            with campaign_lock(output):
                launch_stage(output, "encode", args.threads, split=args.split)
            result = {"split": args.split, "status": "complete"}
        else:
            result = encode_split(output, args.split, args.threads)
            result = {k: result[k] for k in ("split", "n_records", "cold_seconds")}
    elif args.command == "_train":
        result = train_one(output, args.method, args.seed, args.threads)
    elif args.command == "_evaluate":
        result = evaluate_one(output, args.method, args.seed, args.threads)
    elif args.command == "_cost":
        from .cost import measure_one

        result = measure_one(output, args.method, args.seed, args.threads)
    elif args.command == "summarize":
        from .analysis import summarize

        result = summarize(output, bootstrap=args.bootstrap)
    elif args.command == "plot":
        from .figures import plot

        result = plot(output)
    elif args.command == "verify-preservation":
        result = preservation(output, verify=True)
    else:
        with campaign_lock(output):
            if args.command == "cost" and args.seeds != [42]:
                raise ValueError("Cost protocol uses only seed42")
            state_name = {
                "train": "train_state.json",
                "all": "train_state.json",
                "evaluate": "test_state.json",
                "cost": "cost.json",
            }[args.command]
            existing = any(
                (output / "runs" / m / f"seed_{s}" / state_name).exists()
                for m in args.methods
                for s in args.seeds
            )
            if existing and not args.resume:
                raise ValueError(
                    "Existing stage evidence requires --resume and matching fingerprints"
                )
            if args.command == "all":
                from .data import prepare

                preservation(output)
                prepare(output)
                save(output / "cpu_policy.json", cpu)
            freeze_experiment(output, args.threads)
            semantic = any(m in {"cst_v1_1", "bertopic", "minilm_lr"} for m in args.methods)
            if args.command in {"train", "all"}:
                if semantic:
                    for split in ("train", "dev"):
                        launch_stage(output, "encode", args.threads, split=split)
                for seed in args.seeds:
                    for method in args.methods:
                        launch_stage(output, "train", args.threads, method, seed)
            if args.command in {"evaluate", "all"}:
                if semantic:
                    launch_stage(output, "encode", args.threads, split="test")
                for seed in args.seeds:
                    for method in args.methods:
                        launch_stage(output, "evaluate", args.threads, method, seed)
            if args.command in {"cost", "all"}:
                for method in args.methods:
                    launch_stage(output, "cost", args.threads, method, 42)
            if args.command == "all":
                from .analysis import summarize
                from .figures import plot

                summarize(output, bootstrap=2000)
                plot(output)
                preservation(output, verify=True)
        result = status(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
