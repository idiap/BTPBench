#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Jérémy Maceiras <jeremy.maceiras@idiap.ch>
# SPDX-License-Identifier: LicenseRef-primeaid-NC

import argparse
import json
import sys

from pathlib import Path

DEFAULT_COUNT = 60
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output" / "random-key-pool.json"


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for argparse."""
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("count must be an integer") from error
    if count < 1:
        raise argparse.ArgumentTypeError("count must be at least 1")
    return count


def build_key_pool(count: int) -> dict[str, object]:
    """Build a deterministic unselected-key pool payload."""
    if count < 1:
        raise ValueError("count must be at least 1")
    return {
        "metadata": {
            "control": "deterministic-unselected",
            "key_count": count,
        },
        "keys": {
            "random": {str(index): index for index in range(count)},
        },
    }


def write_key_pool(output: Path, count: int) -> Path:
    """Write a deterministic key pool and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_key_pool(count), indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic unselected control keys.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=_positive_int, default=DEFAULT_COUNT)
    return parser.parse_args()


def main() -> None:
    """Generate the requested key pool."""
    args = _parse_args()
    output = write_key_pool(args.output, args.count)
    sys.stdout.write(f"Wrote {output}\n")


if __name__ == "__main__":
    main()
