#!/usr/bin/env python3
"""Stage 1: scan configured datasets and write a unified catalog."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Keep direct-file execution working as well as the preferred `python -m` form.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from seg2med_datasets.prepare import prepare_from_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Dataset preparation YAML file (required unless --print-list is used).")
    parser.add_argument("--print-list", action="store_true", help="Print available dataset kinds, required YAML fields, and modalities without scanning datasets.")
    parser.add_argument("--dry-run", action="store_true", help="Scan and summarize without writing tables.")
    parser.add_argument("--csv-only", action="store_true", help="Write only catalog/split CSV tables, skipping configured Parquet output. This script never generates prior images.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.print_list:
        from seg2med_datasets.datasets import describe_dataset_modalities, describe_dataset_requirements
        from seg2med_datasets.schema import MODALITY_NAMES

        print("Available dataset kinds (use as 'kind' in your YAML):")
        print("Required fields below are YAML keys for each dataset entry (+ means both; OR means either).")
        requirements = describe_dataset_requirements()
        for kind, description in describe_dataset_modalities().items():
            print(f"  {kind}")
            print(f"    required: {requirements[kind]}")
            print(f"    {description}")
        print("\nKnown modality IDs (for the 'modality' setting):")
        for value, name in MODALITY_NAMES.items():
            print(f"  {value}: {name}")
        return 0
    if not args.config:
        parser.error("--config is required unless --print-list is used")
    result = prepare_from_config(args.config, dry_run=args.dry_run, csv_only=args.csv_only)
    if args.dry_run:
        print(f"Datasets: {', '.join(result['datasets'])}")
        print(f"Images:   {result['images']}")
    else:
        summary = result["stats"]["by_modality"].reset_index()
        summary = summary.rename(columns={
            "modality": "ID", "modality_name": "Modality",
            "unique_cases": "Cases", "num_items": "Rows",
        })
        summary_text = (
            f"Catalog rows: {len(result['df'])}\n"
            f"Output:       {result['output_dir']}\n"
            "\nCases per modality:\n"
            + summary[["ID", "Modality", "Cases", "Rows"]].to_string(index=False)
            + ("\nCases = resolved unique (source, patient ID) pairs per modality; inventory includes excluded records.\n"
               if "identity_audit" in result else
               "\nCases = unique (dataset, patient ID) pairs per modality; rows include scans/slices and incomplete pairs.\n")
        )
        summary_path = Path(result["output_dir"]) / "dataset_summary.txt"
        if "identity_audit" in result:
            from seg2med_datasets.splitting import format_split_summary

            summary_text += "\n" + format_split_summary(result)
            summary_text += f"Patient mapping: {result['patient_mapping_path']}\n"
        summary_path.write_text(summary_text, encoding="utf-8")
        print(summary_text, end="")
        print(f"Summary saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
