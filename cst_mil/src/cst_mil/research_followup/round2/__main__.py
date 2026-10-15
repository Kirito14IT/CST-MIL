"""Isolated second-round entry point; all expensive commands are sequential."""

import argparse
from pathlib import Path

from cst_mil.bench.resources import configure_cpu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "init",
            "quote",
            "gemini",
            "local-time",
            "heads",
            "stats",
            "report",
            "verify",
        ],
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--method", choices=["lda", "btm", "gsdmm", "nmf", "bertopic", "cst_mil_r0"]
    )
    parser.add_argument("--budget", type=float, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    cpu = configure_cpu(4)
    from cst_mil.bench.runner import run_lock

    from .core import OUT, check_preservation, initialize, save

    out = args.out or OUT
    initialize(out)
    save(out / "cpu_policy.json", cpu)
    with run_lock(out):
        if args.command in ["init", "verify"]:
            print(check_preservation(out), flush=True)
        elif args.command == "quote":
            from .gemini import prepare

            print(prepare(out)[1], flush=True)
        elif args.command == "gemini":
            from .gemini import run

            run(out, args.budget)
        elif args.command == "local-time":
            from .execution import measure_local

            if not args.method:
                parser.error("--method required; use one fresh process per method")
            measure_local(args.method, out)
        elif args.command == "heads":
            from .robustness import run

            run(out=out, seeds=args.seeds)
        elif args.command == "stats":
            from .robustness import summarize

            summarize(out=out, bootstrap=args.bootstrap)
        elif args.command == "report":
            from .report import build

            build(out)
    check_preservation(out)


if __name__ == "__main__":
    main()
