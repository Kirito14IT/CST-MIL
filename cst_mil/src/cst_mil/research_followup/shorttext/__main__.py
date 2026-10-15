"""Explicit isolated entry point; no frozen campaign initialization."""

import argparse
import json
from pathlib import Path

from .core import METHODS, OUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["audit", "prepare", "run", "resume", "report"])
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--sample-sizes",
        type=json.loads,
        help="JSON mapping dataset to even train/dev/test totals; frozen defaults if omitted",
    )
    args = parser.parse_args()
    if args.action == "audit":
        from .core import preservation

        print(preservation())
    elif args.action == "prepare":
        from .data import prepare_all

        print(prepare_all(args.output, args.seed, args.datasets, args.sample_sizes))
    elif args.action in {"run", "resume"}:
        import subprocess
        import sys

        from .core import read
        from .execution import run_one

        datasets = args.datasets or list(read(args.output / "prepared/manifest.json")["datasets"])
        if len(datasets) == 1 and len(args.methods) == 1:
            run_one(datasets[0], args.methods[0], args.output, args.seed, args.threads)
        else:
            for dataset in datasets:
                for method in args.methods:
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            __package__,
                            "run",
                            "--datasets",
                            dataset,
                            "--methods",
                            method,
                            "--output",
                            str(args.output),
                            "--seed",
                            str(args.seed),
                            "--threads",
                            str(args.threads),
                        ],
                        check=True,
                    )
    else:
        from .analysis import build_report

        print(build_report(args.output))


if __name__ == "__main__":
    main()
