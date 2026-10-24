import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from seg2med_physiosynth.priors import create_priors_from_config


class PriorOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        raw = self.root / "raw"
        raw.mkdir()
        for name, value in (("image", 50), ("organs", 1), ("tissues", 1)):
            nib.save(nib.Nifti1Image(np.full((5, 6, 2), value, dtype=np.int16), np.eye(4)), raw / f"{name}.nii.gz")
        cv2.imwrite(str(raw / "image.png"), np.full((5, 6), 100, dtype=np.uint8))
        cv2.imwrite(str(raw / "mask.png"), np.full((5, 6), 255, dtype=np.uint8))
        self.config = {
            "segmentation_label_scheme": "current",
            "data_root": str(self.root), "min_contour_area": 0,
            "combined_csv": "tables/all.csv", "splits": {},
        }
        for split in ("train", "test"):
            raster = split == "test"
            catalog = self.root / f"{split}.csv"
            pd.DataFrame([{
                "id": split, "modality": 7 if raster else 0, "aorta_diss": 0,
                "img": str(raw / ("image.png" if raster else "image.nii.gz")),
                "seg": str(raw / ("mask.png" if raster else "organs.nii.gz")),
                "seg_tissue": str(raw / ("mask.png" if raster else "tissues.nii.gz")),
            }]).to_csv(catalog, index=False)
            self.config["splits"][split] = {
                "catalog": str(catalog), "output_dir": f"priors/{split}",
                "output_csv": f"tables/{split}.csv",
            }
        self.yaml = self.root / "priors.yaml"
        self.write_config()

    def write_config(self):
        self.yaml.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def snapshot(self):
        return {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in (self.root / "priors").rglob("*") if file.is_file()}

    def test_existing_cases_skip_simulation_and_remain_in_combined_csv(self):
        create_priors_from_config(self.yaml)
        before = self.snapshot()
        with patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Volume was regenerated")), patch("seg2med_physiosynth.priors._prepare_raster", side_effect=AssertionError("Raster was regenerated")):
            result = create_priors_from_config(self.yaml)
        self.assertEqual(before, self.snapshot())
        for split in ("train", "test"):
            self.assertEqual(result[split].attrs["reused"], 1)
            self.assertEqual(result[split].attrs["generated"], 0)
            self.assertNotIn("split", result[split].columns)
        combined = pd.read_csv(self.root / "tables/all.csv")
        self.assertEqual(dict(zip(combined.id, combined.split)), {"train": "train", "test": "test"})
        self.assertEqual(len(result["combined"]), 2)
        for field in ("prior", "img", "prior_semantic"):
            self.assertTrue(all((self.root / value).is_file() for value in combined[field]))

    def test_partial_outputs_are_repaired_without_overwriting_existing_files(self):
        result = create_priors_from_config(self.yaml)
        for split in ("train", "test"):
            semantic = self.root / result[split].iloc[0].prior_semantic
            semantic.unlink()
        before = self.snapshot()
        result = create_priors_from_config(self.yaml)
        after = self.snapshot()
        for path, saved in before.items():
            self.assertEqual(saved, after[path])
        self.assertEqual(len(after), len(before) + 2)
        self.assertTrue(all(result[split].attrs["generated"] == 1 for split in ("train", "test")))

    def test_overwrite_yaml_and_python_overrides(self):
        create_priors_from_config(self.yaml)
        self.config["overwrite"] = True
        self.write_config()
        before = self.snapshot()
        create_priors_from_config(self.yaml, overwrite=False)
        self.assertEqual(before, self.snapshot())
        prior = self.root / "priors/train/train/image_prior_image.nii.gz"
        prior.write_bytes(b"old placeholder")
        create_priors_from_config(self.yaml)
        self.assertEqual(nib.load(prior).shape, (5, 6, 2))
        self.config["overwrite"] = False
        self.write_config()
        prior.write_bytes(b"old placeholder")
        create_priors_from_config(self.yaml, overwrite=True)
        self.assertEqual(nib.load(prior).shape, (5, 6, 2))

    def test_dry_run_does_not_write_outputs(self):
        result = create_priors_from_config(self.yaml, dry_run=True)
        self.assertEqual(result["combined"]["rows"], 2)
        self.assertEqual(result["train"]["generated"], 1)
        self.assertFalse((self.root / "priors").exists())
        self.assertFalse((self.root / "tables").exists())

    def test_cli_can_override_yaml(self):
        create_priors_from_config(self.yaml)
        self.config["overwrite"] = True
        self.write_config()
        before = self.snapshot()
        command = [sys.executable, "-B", "-m", "seg2med_workflow.create_image_priors", "--config", str(self.yaml), "--no-overwrite"]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("reuse: 1", completed.stdout)
        self.assertIn("combined: 2 rows", completed.stdout)
        self.assertEqual(before, self.snapshot())

    def test_combined_csv_cannot_replace_a_split_csv(self):
        self.config["combined_csv"] = "tables/train.csv"
        self.write_config()
        with self.assertRaisesRegex(ValueError, "combined_csv must differ"):
            create_priors_from_config(self.yaml)
        self.assertFalse((self.root / "priors").exists())

    def test_csv_only_exports_planned_paths_without_image_io(self):
        self.config["splits"]["train"]["write_excel"] = True
        self.write_config()
        # Mask columns need not be present when only constructing CSV paths.
        train_catalog = self.root / "train.csv"
        frame = pd.read_csv(train_catalog).drop(columns=["seg", "seg_tissue"])
        frame.to_csv(train_catalog, index=False)
        with patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Volume IO")), \
                patch("seg2med_physiosynth.priors._prepare_raster", side_effect=AssertionError("Raster IO")), \
                patch.object(pd.DataFrame, "to_excel", side_effect=AssertionError("Excel written")):
            result = create_priors_from_config(self.yaml, csv_only=True, overwrite=True)
        self.assertFalse((self.root / "priors").exists())
        self.assertEqual({path.name for path in (self.root / "tables").iterdir()}, {"train.csv", "test.csv", "all.csv"})
        combined = pd.read_csv(self.root / "tables/all.csv")
        self.assertEqual(set(combined.split), {"train", "test"})
        self.assertEqual(result["train"].attrs["generated"], 0)
        self.assertEqual(result["train"].iloc[0].prior, str(Path("priors/train/train/image_prior_image.nii.gz")))
        self.assertEqual(result["test"].iloc[0].prior, str(Path("priors/test/test/image_prior_image.png")))

    def test_csv_only_dry_run_writes_nothing(self):
        result = create_priors_from_config(self.yaml, csv_only=True, dry_run=True)
        self.assertEqual(result["combined"]["rows"], 2)
        self.assertEqual(result["train"]["generated"], 0)
        self.assertFalse((self.root / "tables").exists())
        self.assertFalse((self.root / "priors").exists())

    def test_cli_csv_only_preserves_existing_prior_files(self):
        create_priors_from_config(self.yaml)
        before = self.snapshot()
        command = [sys.executable, "-B", "-m", "seg2med_workflow.create_image_priors", "--config", str(self.yaml), "--csv-only", "--overwrite"]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CSV only (no image files written)", completed.stdout)
        self.assertEqual(before, self.snapshot())

    def test_yaml_csv_only_and_python_override(self):
        self.config["csv_only"] = True
        self.write_config()
        with patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Volume IO")), \
                patch("seg2med_physiosynth.priors._prepare_raster", side_effect=AssertionError("Raster IO")):
            result = create_priors_from_config(self.yaml)
        self.assertTrue(result["combined"].attrs["csv_only"])
        self.assertFalse((self.root / "priors").exists())
        result = create_priors_from_config(self.yaml, csv_only=False)
        self.assertFalse(result["combined"].attrs["csv_only"])
        self.assertTrue((self.root / "priors").exists())

    def test_cli_yaml_csv_only_and_negative_override(self):
        self.config["csv_only"] = True
        self.write_config()
        command = [sys.executable, "-B", "-m", "seg2med_workflow.create_image_priors", "--config", str(self.yaml)]
        completed = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("CSV only (no image files written)", completed.stdout)
        self.assertFalse((self.root / "priors").exists())
        completed = subprocess.run(command + ["--no-csv-only"], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("CSV only", completed.stdout)
        self.assertTrue((self.root / "priors").exists())

    def test_yaml_absolute_paths_for_csv_only_and_generated_outputs(self):
        self.config["absolute_paths"] = True
        self.write_config()
        for csv_only in (True, False):
            with self.subTest(csv_only=csv_only):
                result = create_priors_from_config(self.yaml, csv_only=csv_only)
                for name in ("train", "test", "combined"):
                    for field in ("img", "prior", "prior_semantic"):
                        self.assertTrue(all(Path(value).is_absolute() for value in result[name][field]))
                saved = pd.read_csv(self.root / "tables/all.csv")
                self.assertEqual(saved.prior.tolist(), result["combined"].prior.tolist())
        result = create_priors_from_config(self.yaml, csv_only=True, absolute_paths=False)
        self.assertFalse(Path(result["train"].iloc[0].prior).is_absolute())
        self.assertEqual(self.root / result["train"].iloc[0].prior,
                         self.root / "priors/train/train/image_prior_image.nii.gz")

    def test_cli_absolute_path_overrides(self):
        self.config["csv_only"] = True
        self.write_config()
        command = [sys.executable, "-B", "-m", "seg2med_workflow.create_image_priors", "--config", str(self.yaml)]
        completed = subprocess.run(command + ["--absolute-paths"], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertTrue(Path(pd.read_csv(self.root / "tables/train.csv").iloc[0].prior).is_absolute())
        self.config["absolute_paths"] = True
        self.write_config()
        completed = subprocess.run(command + ["--no-absolute-paths"], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertFalse(Path(pd.read_csv(self.root / "tables/train.csv").iloc[0].prior).is_absolute())


if __name__ == "__main__":
    unittest.main()
