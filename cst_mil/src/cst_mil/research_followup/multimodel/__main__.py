from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze, finalize_review
from .core import MODELS, OUT, prepare, run_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fixed-cohort multi-model pilot; no full-suite API calls"
    )
    parser.add_argument("command", choices=("prepare", "run", "analyze", "review-finalize"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "prepare":
        selection, _, _ = prepare(args.out)
        result = dict(prepared=len(selection["sessions"]), models=MODELS, paid_calls=0)
    elif args.command == "run":
        if not args.model:
            parser.error("run requires explicit --model")
        r = run_model(args.model, args.out)
        result = {k: r[k] for k in ("model", "status", "completed", "attempts", "stop_reason")}
    elif args.command == "analyze":
        result = {
            m: dict(n_valid=r["n_valid"], ap=r["metrics"].get("ap"))
            for m, r in analyze(args.out).items()
        }
    else:
        result = finalize_review(args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
