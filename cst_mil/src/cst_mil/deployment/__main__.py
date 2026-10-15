"""Install the public assets explicitly; starting the app never downloads them."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

from .assets import setup, verify_installation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    install = commands.add_parser("setup", help="Download/import and verify the frozen model assets")
    install.add_argument("--root", type=Path, help="Repository root (normally detected automatically)")
    install.add_argument("--model-archive", type=Path, help="Local verified release ZIP instead of downloading")
    install.add_argument("--encoder-source", type=Path, help="Local fixed-revision encoder snapshot")
    install.add_argument("--offline", action="store_true", help="Never use the network; require local/cached files")
    check = commands.add_parser("verify", help="Recheck the installed model, encoder, config and source fingerprints")
    check.add_argument("--root", type=Path)
    args = parser.parse_args()
    try:
        result = (setup(args.root, model_archive=args.model_archive,
                        encoder_source=args.encoder_source, offline=args.offline)
                  if args.command == "setup" else verify_installation(args.root))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, zipfile.BadZipFile) as error:
        print(f"Asset {args.command} failed: {error}", file=sys.stderr)
        print("No unverified installation is accepted. Rerun setup after correcting the cause.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
