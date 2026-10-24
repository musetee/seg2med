"""Relocated batch entry points work as modules and standalone files."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("prepare_datasets", "segment_images", "create_image_priors")


class WorkflowEntrypointTests(unittest.TestCase):
    def run_cli(self, args, *, cwd=ROOT):
        completed = subprocess.run([sys.executable, "-B", *args], cwd=cwd, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        return completed.stdout

    def test_module_help_and_root_cleanup(self):
        for name in SCRIPTS:
            with self.subTest(name=name):
                self.assertFalse((ROOT / f"{name}.py").exists())
                output = self.run_cli(["-m", f"seg2med_workflow.{name}", "--help"])
                self.assertIn("--config", output)

    def test_direct_file_help_from_another_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in SCRIPTS:
                with self.subTest(name=name):
                    output = self.run_cli([str(ROOT / "seg2med_workflow" / f"{name}.py"), "--help"], cwd=directory)
                    self.assertIn("--config", output)

    def test_module_dataset_listing(self):
        output = self.run_cli(["-m", "seg2med_workflow.prepare_datasets", "--print-list"])
        self.assertIn("anika", output)
        self.assertIn("ct_root + mr_root", output)
        self.assertIn("Known modality IDs", output)

    def test_all_stages_dry_run_preserves_yaml_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            raw = folder / "raw"
            raw.mkdir()
            (raw / "ct-volume-001.nii.gz").touch()
            with (folder / "catalog.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["id", "img", "seg", "seg_tissue", "modality"])
                writer.writeheader()
                writer.writerow({"id": "001", "img": "raw/ct-volume-001.nii.gz", "modality": 0,
                                 "seg": "raw/ct-volume-001_seg.nii.gz", "seg_tissue": "raw/ct-volume-001_seg_tissue.nii.gz"})
            configs = {
                "prepare_datasets": {"data_root": ".", "datasets": [{"kind": "anish", "root": "raw"}]},
                "segment_images": {"source": {"type": "catalog", "path": "catalog.csv"}},
                "create_image_priors": {"data_root": ".", "splits": {"train": {"catalog": "catalog.csv"}}},
            }
            for name, config in configs.items():
                config_path = folder / f"{name}.yaml"
                config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
                with self.subTest(name=name):
                    output = self.run_cli(["-m", f"seg2med_workflow.{name}", "--config", str(config_path), "--dry-run"])
                    self.assertTrue(output)
            self.assertFalse((folder / "generated").exists())
            self.assertFalse((folder / "train.csv").exists())
            self.assertEqual(len(list(raw.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
