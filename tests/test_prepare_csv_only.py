"""CSV-only preparation should not write Parquet or image outputs."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml

from seg2med_workflow import prepare_datasets
from seg2med_datasets.prepare import prepare_from_config


class PrepareCsvOnlyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        images = self.root / "images"
        images.mkdir()
        for case in ("301", "302"):
            (images / f"ct-volume-{case}.nii").touch()
        self.config = self.root / "prepare.yaml"
        self.config.write_text(yaml.safe_dump({
            "data_root": str(self.root),
            "datasets": [{"kind": "anish", "root": "images"}],
            "output": {
                "directory": "tables", "parquet": "catalog.parquet",
                "check_paths": True,
                "split": {"enabled": True, "prefix": "cases"},
            },
        }), encoding="utf-8")

    def test_csv_only_writes_catalog_and_splits_without_parquet(self):
        with patch.object(pd.DataFrame, "to_parquet", side_effect=AssertionError("Parquet requested")):
            result = prepare_from_config(self.config, csv_only=True)
        self.assertEqual(len(result["df"]), 2)
        self.assertEqual({path.name for path in (self.root / "tables").iterdir()}, {
            "seg2med_catalog.csv", "cases_train.csv", "cases_validation.csv", "cases_test.csv",
            "cases_patient_splits.csv", "cases_identity_audit.csv", "cases_excluded.csv",
        })

    def test_default_still_honors_parquet_configuration(self):
        with patch.object(pd.DataFrame, "to_parquet") as write_parquet:
            prepare_from_config(self.config)
        self.assertEqual(write_parquet.call_count, 4)

    def test_cli_csv_only(self):
        with patch("sys.argv", ["seg2med_workflow.prepare_datasets", "--config", str(self.config), "--csv-only"]), \
                patch.object(pd.DataFrame, "to_parquet", side_effect=AssertionError("Parquet requested")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(prepare_datasets.main(), 0)
        self.assertTrue((self.root / "tables/seg2med_catalog.csv").is_file())
        self.assertTrue((self.root / "tables/dataset_summary.txt").is_file())

    def test_csv_only_dry_run_writes_nothing(self):
        result = prepare_from_config(self.config, csv_only=True, dry_run=True)
        self.assertEqual(result["images"], 2)
        self.assertFalse((self.root / "tables").exists())


if __name__ == "__main__":
    unittest.main()
