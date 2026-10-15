from __future__ import annotations

import argparse
import json


def main():
    from cst_mil.bench.resources import configure_cpu
    policy = configure_cpu(4)
    from .io import OUT, save
    parser = argparse.ArgumentParser(description='Frozen full677 research follow-up')
    parser.add_argument('command', choices=['audit', 'ablate', 'profile', 'verify',
                                           'api-estimate', 'api-run'])
    parser.add_argument('--method')
    parser.add_argument('--budget', type=float)
    parser.add_argument('--input-price', type=float, help='USD per million input tokens')
    parser.add_argument('--output-price', type=float, help='USD per million output tokens')
    args = parser.parse_args()
    save(OUT / 'cpu_policy.json', policy)
    if args.command == 'audit':
        from .audit import run
        result = run()
    elif args.command == 'ablate':
        from .ablation import run, run_method
        result = run_method(args.method) if args.method else run()
    elif args.command == 'profile':
        from .audit import profile_existing
        result = profile_existing()
    elif args.command == 'verify':
        from .io import protect
        result = protect(verify=True)
    else:
        from .api import estimate, run
        result = estimate() if args.command == 'api-estimate' else run(
            budget=args.budget, input_price=args.input_price, output_price=args.output_price)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == '__main__':
    main()
