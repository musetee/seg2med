#!/usr/bin/env python3
"""Stage 3: simulate CT/MR image priors from segmentation catalogs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Also support `python seg2med_workflow/create_image_priors.py` without relying
# on the working directory or an installed editable package.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seg2med_physiosynth import create_priors_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Physiology simulation YAML file.")
    parser.add_argument("--dry-run", action="store_true", help="Validate catalogs and report selected rows only.")
    parser.add_argument("--csv-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Override YAML csv_only: write only CSVs without image IO, or use --no-csv-only to generate priors.")
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None,
                        help="Override YAML overwrite: replace existing prior outputs, or use --no-overwrite to reuse them.")
    parser.add_argument("--absolute-paths", action=argparse.BooleanOptionalAction, default=None,
                        help="Override YAML absolute_paths: store absolute paths in CSVs, or use --no-absolute-paths for paths relative to data_root.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = create_priors_from_config(args.config, dry_run=args.dry_run, overwrite=args.overwrite,
                                       csv_only=args.csv_only, absolute_paths=args.absolute_paths)
    for name, output in outputs.items():
        rows = output["rows"] if isinstance(output, dict) else len(output)
        stats = output if isinstance(output, dict) else output.attrs
        counts = f"; generate: {stats['generated']}, reuse: {stats['reused']}" if "generated" in stats else ""
        if stats.get("csv_only", False):
            counts = "; CSV only (no image files written)"
        print(f"{name}: {rows} rows{counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
