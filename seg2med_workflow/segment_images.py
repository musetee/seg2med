#!/usr/bin/env python3
"""Stage 2: create dataset segmentations with TotalSegmentator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seg2med_segmentation import run_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Segmentation YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and show work without segmenting.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_from_config(args.config, dry_run=args.dry_run)
    print(f"Planned: {len(result['planned'])}")
    for image, task, output in result["planned"][:10]:
        print(f"{task}: {image} -> {output}")
    if len(result["planned"]) > 10:
        print(f"... {len(result['planned']) - 10} more")
    if not args.dry_run:
        print(f"Created: {len(result['created'])}; skipped: {len(result['skipped'])}; failed: {len(result['failed'])}")
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
