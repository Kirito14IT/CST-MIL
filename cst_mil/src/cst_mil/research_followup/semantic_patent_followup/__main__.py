"""Explicit independent stages; never modifies old campaigns or deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import OUTPUT, configure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "prepare",
            "bank",
            "candidates",
            "select",
            "ablations",
            "test",
            "topics",
            "analysis",
            "cost",
            "verify",
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    configure(args.threads)
    output = args.output.resolve()
    if args.stage == "prepare":
        from .data import prepare

        print(prepare(output))
    elif args.stage == "bank":
        from .representations import encode_bank

        print(encode_bank(args.split, output))
    elif args.stage in ("candidates", "select", "ablations", "test"):
        from . import execution

        getattr(execution, "evaluate_test" if args.stage == "test" else args.stage)(output)
    elif args.stage == "topics":
        from .topic_execution import run_topics

        print(run_topics(output))
    elif args.stage == "analysis":
        from .analysis import summarize

        print(summarize(output, count=args.bootstrap))
    elif args.stage == "cost":
        from .cost import measure

        print(measure(output))
    else:
        from .audit import verify

        print(verify(output))


if __name__ == "__main__":
    main()
