from __future__ import annotations

import argparse
import importlib
from typing import Sequence


COMMAND_MODULES = {
    "prepare-data": "prepare_data",
    "train": "train",
    "evaluate": "evaluate",
    "benchmark": "benchmark",
    "infer": "infer",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified entrypoint for the VOC YOLOv8-Light experiment project.")
    parser.add_argument("command", choices=COMMAND_MODULES.keys(), help="Command to run.")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the command module.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    module = importlib.import_module(COMMAND_MODULES[args.command])
    return int(module.main(args.args))


if __name__ == "__main__":
    raise SystemExit(main())
