"""Public single-method benchmark commands; imports follow CPU thread setup."""

from __future__ import annotations

import argparse
import json
import os


def main(argv: list[str] | None = None) -> int:
    from .resources import configure_cpu

    configure_cpu(4)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    from .common import read_json
    from .dataset import BENCHMARK, load_suite, prepare_suites
    from .runner import METHODS, ensure_protocol, run_method

    parser = argparse.ArgumentParser(
        prog="cst-bench", description="SCCD complete-session benchmark"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="prepare immutable pilot300, full677 and isolated smoke5")
    run = commands.add_parser("run", help="train once then resumably infer complete sessions")
    run.add_argument("--suite", choices=["pilot300", "full677", "smoke5"], required=True)
    run.add_argument("--method", choices=METHODS, required=True)
    run.add_argument("--revision", choices=["r0", "r1", "r2", "r3"])
    run.add_argument(
        "--limit", default="all", help="cumulative inference count (not training subset)"
    )
    run.add_argument("--resume", action="store_true")
    run.add_argument("--stage", choices=["all", "train"], default="all")
    status = commands.add_parser("status")
    status.add_argument("--suite", choices=["pilot300", "full677", "smoke5"], default="full677")
    smoke = commands.add_parser(
        "verify", help="five-session independent-process resume verification"
    )
    smoke.add_argument("--method", choices=METHODS, required=True)
    smoke.add_argument("--revision", choices=["r0", "r1", "r2", "r3"])
    commands.add_parser("pilot", help="run five fixed baselines and bounded dev-only CST revisions")
    compare = commands.add_parser(
        "compare", help="render comparison from completed held-out predictions"
    )
    compare.add_argument("--suite", choices=["pilot300", "full677"], default="pilot300")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        suites = prepare_suites()
        ensure_protocol()
        result = {
            k: {name: v[name] for name in ("n_sessions", "n_comments", "split_counts")}
            for k, v in suites.items()
        }
    elif args.command == "run":
        result = run_method(
            args.suite,
            args.method,
            revision=args.revision,
            limit=args.limit,
            resume=args.resume,
            stage=args.stage,
        )
    elif args.command == "status":
        manifest, _, _ = load_suite(args.suite)
        result = {"suite": args.suite, "total": manifest["n_sessions"], "methods": {}}
        for path in sorted((BENCHMARK / "runs" / args.suite).glob("*/status.json")):
            result["methods"][path.parent.name] = read_json(path)
    elif args.command == "verify":
        from .verification import verify_method

        result = verify_method(args.method, args.revision)
    elif args.command == "pilot":
        from .campaign import run_pilot_campaign

        result = run_pilot_campaign()
    elif args.command == "compare":
        from .reporting import create_comparison

        result = create_comparison(args.suite)
    else:
        parser.error("Unsupported command")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
