"""Preparation reports cases per modality without conflating IDs or slices."""

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from seg2med_workflow import prepare_datasets
from seg2med_datasets.catalog import build_and_save_catalog
from seg2med_datasets.datasets.common import catalog_config


class PrepareSummaryTests(unittest.TestCase):
    def test_counts_dataset_local_cases_separately_from_rows(self):
        configs = [
            catalog_config("ct_a", ["a.nii", "b.nii"], ["", ""], ["", ""], ["1", "1"], 0),
            catalog_config("ct_b", [""], ["mask.nii"], [""], ["1"], 0),
            catalog_config("mr", ["mr.nii"], [""], [""], ["1"], 1),
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = build_and_save_catalog(
                configs, output_dir=folder, out_csv="catalog.csv",
                out_parquet=None, check_paths=False, split_prefix=None,
            )
            stats = result["stats"]["by_modality"]
            self.assertEqual(stats.loc[(0, "ct"), "unique_cases"], 2)
            self.assertEqual(stats.loc[(0, "ct"), "num_items"], 3)
            self.assertEqual(stats.loc[(1, "mr_t1_gre"), "unique_cases"], 1)
            self.assertEqual(len(pd.read_csv(Path(folder) / "catalog.csv")), 4)

            result["output_dir"] = Path(folder)
            output = io.StringIO()
            with patch("sys.argv", ["seg2med_workflow.prepare_datasets", "--config", "unused.yaml"]), \
                    patch.object(prepare_datasets, "prepare_from_config", return_value=result), \
                    redirect_stdout(output):
                self.assertEqual(prepare_datasets.main(), 0)
            text = output.getvalue()
            self.assertIn("Cases per modality:", text)
            self.assertIn("Cases", text)
            self.assertIn("Rows", text)
            self.assertIn("mr_t1_gre", text)
            self.assertIn("unique (dataset, patient ID)", text)
            summary_path = Path(folder) / "dataset_summary.txt"
            saved_text = summary_path.read_text(encoding="utf-8")
            self.assertEqual(text, saved_text + f"Summary saved: {summary_path}\n")

            # Re-running refreshes the report rather than appending or keeping stale text.
            summary_path.write_text("outdated summary", encoding="utf-8")
            with patch("sys.argv", ["seg2med_workflow.prepare_datasets", "--config", "unused.yaml"]), \
                    patch.object(prepare_datasets, "prepare_from_config", return_value=result), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(prepare_datasets.main(), 0)
            self.assertEqual(summary_path.read_text(encoding="utf-8"), saved_text)


if __name__ == "__main__":
    unittest.main()
