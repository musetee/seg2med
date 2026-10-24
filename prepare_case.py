#!/usr/bin/env python3
"""Prepare one volume via CLI, or one/multiple cases from a YAML file."""

import argparse
import logging
from pathlib import Path

from seg2med_workflow.config import ConfigError, resolve_path
from seg2med_workflow.single_case import prepare_case_from_config


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Optional flat single-case YAML or YAML with a case list. CLI processing flags override YAML.")
    parser.add_argument("--dataset", help="Dataset name recorded in the segmentation CSV.")
    parser.add_argument("--id", help="Patient/case ID.")
    parser.add_argument("--img", help="Input .nii or .nii.gz volume.")
    parser.add_argument("--input-modality-name", "--modality-name", "--modality", dest="input_modality_name", help="Input modality name (e.g. mr_t1_gre) or integer ID.")
    parser.add_argument("--target-modality-name", help="Simulated prior modality; defaults to the input modality.")
    parser.add_argument("--segmentation-label-scheme", choices=["auto", "current", "legacy"],
                        help="Input mask label dictionary; auto reads embedded NIfTI metadata.")
    parser.add_argument("--prior-label-scheme", choices=["auto", "current", "legacy"],
                        help="Output MR/model convention. Use legacy for models trained with old MR labels.")
    parser.add_argument("--overwrite-seg", "--overwrite-segs", action=argparse.BooleanOptionalAction, default=None,
                        help="Regenerate masks; --no-overwrite-seg reuses existing masks.")
    parser.add_argument("--overwrite-prior", action=argparse.BooleanOptionalAction, default=None,
                        help="Regenerate priors; --no-overwrite-prior reuses existing priors.")
    parser.add_argument("--absolute-paths", action=argparse.BooleanOptionalAction, default=None,
                        help="CSV paths are absolute by default; relative paths use each manifest's directory as root.")
    parser.add_argument("--device", help="Segmentation device: cuda, cuda:N, gpu, gpu:N, cpu, or mps.")
    parser.add_argument("--seg-csv", help="Optional segmentation manifest path.")
    parser.add_argument("--prior-csv", help="Optional prior manifest path.")
    parser.add_argument("--dry-run", action="store_true", help="Show planned work without writing files or running models.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    overrides = {key: value for key, value in vars(args).items()
                 if key not in {"config", "dry_run"} and value is not None}
    # CLI path arguments use the shell's working directory, not the YAML directory.
    for key in ("img", "seg_csv", "prior_csv"):
        if key in overrides:
            overrides[key] = str(resolve_path(overrides[key], Path.cwd(), required=True))
    try:
        result = prepare_case_from_config(args.config, overrides=overrides, dry_run=args.dry_run)
    except (ConfigError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.dry_run:
        for case in result if isinstance(result, list) else [result]:
            print(f"Image: {case['img']}")
            print(f"Modalities: input={case['input_modality']}, prior={case['modality']}")
            print(f"Prior image: {case['prior']}")
            for task, path in case["segmentation_planned"]:
                print(f"{task}: {case['img']} -> {path}")
            print(f"Priors: {'generate' if case['prior_planned'] else 'reuse'}")
            print(f"Segmentation CSV: {case['seg_csv']}")
            print(f"Prior CSV: {case['prior_csv']}")
    if isinstance(result, list):
        print(f"{'Planned' if args.dry_run else 'Prepared'}: {len(result)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
