"""`vector-runtime check`: prints one JSON report and exits 0 when it passed.

vector-runtime check   --weights DIR_OR_FILE [--max-bytes N] [--no-hash]
"""

from __future__ import annotations

import argparse
import json

from .check import DEFAULT_MAX_FILE_BYTES, check


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vector-runtime")
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("check", help="check model.safetensors against the architecture")
    c.add_argument("--weights", required=True)
    c.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_FILE_BYTES)
    c.add_argument("--no-hash", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            report = check(
                args.weights, max_file_bytes=args.max_bytes, compute_sha256=not args.no_hash
            )
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "errors": [str(exc)]}, indent=1))
        return 1
    print(json.dumps(report.to_dict(), indent=1))
    return 0 if report.ok else 1
