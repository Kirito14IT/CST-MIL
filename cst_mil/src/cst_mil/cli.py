"""Command-line entry point for the reproducible CST-MIL pilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cst-mil",
        description="CPU-only SCCD conversation risk pilot",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download-data", help="download pinned SCCD CSV files")
    download.add_argument("--data-dir", type=_path, default=Path("data/raw"))

    prepare = subparsers.add_parser("prepare", help="clean SCCD and create a session split")
    prepare.add_argument("--raw-dir", type=_path, default=Path("data/raw"))
    prepare.add_argument("--output", type=_path, default=Path("data/prepared"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--sessions", type=int, default=300)

    train = subparsers.add_parser("train", help="fit the one-run CST-MIL model")
    train.add_argument("--config", type=_path, default=Path("configs/pilot.yaml"))
    train.add_argument("--prepared", type=_path, default=Path("data/prepared"))
    train.add_argument("--run", type=_path, default=Path("artifacts/run_001"))

    evaluate = subparsers.add_parser("evaluate", help="evaluate a completed fitted model")
    evaluate.add_argument("--run", type=_path, default=Path("artifacts/run_001"))
    evaluate.add_argument("--prepared", type=_path, default=Path("data/prepared"))

    pilot = subparsers.add_parser("run-pilot", help="download, prepare, train, and evaluate once")
    pilot.add_argument("--config", type=_path, default=Path("configs/pilot.yaml"))
    pilot.add_argument("--data-dir", type=_path, default=Path("data/raw"))
    pilot.add_argument("--prepared", type=_path, default=Path("data/prepared"))
    pilot.add_argument("--run", type=_path, default=Path("artifacts/run_001"))

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from cst_mil.pipeline import (
        download_data_command,
        evaluate_command,
        prepare_command,
        run_pilot_command,
        train_command,
    )

    args = build_parser().parse_args(argv)
    if args.command == "download-data":
        download_data_command(args.data_dir)
    elif args.command == "prepare":
        prepare_command(args.raw_dir, args.output, args.seed, args.sessions)
    elif args.command == "train":
        train_command(args.config, args.prepared, args.run)
    elif args.command == "evaluate":
        evaluate_command(args.run, args.prepared)
    elif args.command == "run-pilot":
        run_pilot_command(args.config, args.data_dir, args.prepared, args.run)
    else:  # pragma: no cover - argparse enforces known commands
        raise RuntimeError(f"Unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
