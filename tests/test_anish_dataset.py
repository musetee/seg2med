import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from seg2med_datasets.catalog import build_dataframe, validate_dataframe
from seg2med_datasets.datasets import build_dataset_configs
from seg2med_datasets.prepare import prepare_from_config


class AnishDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.folder = self.root / "anish"
        self.folder.mkdir()

    def scan(self, **extra):
        configs = build_dataset_configs({"kind": "anish", "root": "anish", **extra}, self.root)
        frame = build_dataframe(configs)
        validate_dataframe(frame, check_paths=True)
        return frame

    def test_pairs_nii_and_nii_gz_and_preserves_missing_files(self):
        for name in (
            "ct-volume-301.nii", "ct-volume-301_seg.nii", "ct-volume-301_seg_tissue.nii.gz",
            "ct-volume-302.nii.gz", "ct-volume-302_seg.nii.gz",
            "ct-volume-303.nii", "ct-volume-304_seg.nii", "unrelated.nii",
            "ct-volume-301_prior_image.nii",
        ):
            (self.folder / name).touch()
        frame = self.scan(modalities=["ct"]).set_index("id")
        self.assertEqual(set(frame.index), {"301", "302", "303", "304"})
        self.assertTrue(frame.loc["301", "seg"].endswith("ct-volume-301_seg.nii"))
        self.assertTrue(frame.loc["301", "seg_tissue"].endswith("ct-volume-301_seg_tissue.nii.gz"))
        self.assertTrue(frame.loc["302", "seg"].endswith("ct-volume-302_seg.nii.gz"))
        self.assertEqual(frame.loc["302", "seg_tissue"], "")
        self.assertEqual(frame.loc["303", "seg"], "")
        self.assertEqual(frame.loc["304", "img"], "")
        self.assertTrue(frame.modality.eq(0).all())

    def test_csv_input_and_custom_columns_still_work(self):
        image = self.folder / "ct-volume-301.nii"
        image.touch()
        csv_path = self.root / "metadata.csv"
        pd.DataFrame([{"patient": "V301", "image": str(image), "mask": ""}]).to_csv(csv_path, index=False)
        configs = build_dataset_configs({
            "kind": "anish", "csv": "metadata.csv", "name": "old_anish",
            "columns": {"id": "patient", "image": "image", "segmentation": "mask"},
        }, self.root)
        frame = build_dataframe(configs)
        self.assertEqual(frame.iloc[0].id, "V301")
        self.assertEqual(frame.iloc[0].dataset, "old_anish")
        self.assertEqual(frame.iloc[0].img, str(image))
        self.assertEqual(frame.iloc[0].seg, "")

    def test_catalog_export_from_root(self):
        (self.folder / "ct-volume-301.nii").touch()
        (self.folder / "ct-volume-301_seg.nii").touch()
        config = self.root / "prepare.yaml"
        config.write_text(yaml.safe_dump({
            "data_root": str(self.root),
            "datasets": [{"kind": "anish", "name": "anish", "root": "anish", "modalities": ["ct"]}],
            "output": {"directory": "tables", "check_paths": True, "split": {"enabled": False}},
        }), encoding="utf-8")
        result = prepare_from_config(config)
        self.assertEqual(len(result["df"]), 1)
        frame = pd.read_csv(self.root / "tables/seg2med_catalog.csv", keep_default_na=False)
        self.assertEqual(frame.iloc[0].seg_tissue, "")
        self.assertEqual(frame.iloc[0].modality, 0)

    def test_duplicate_image_and_invalid_configuration_are_rejected(self):
        (self.folder / "ct-volume-301.nii").touch()
        (self.folder / "ct-volume-301.nii.gz").touch()
        with self.assertRaisesRegex(ValueError, "Multiple Anish img"):
            self.scan()
        with self.assertRaisesRegex(ValueError, r"modalities: \[ct\]"):
            self.scan(modalities=["mr_t1"])
        with self.assertRaisesRegex(ValueError, "either 'csv' or 'root'"):
            build_dataset_configs({"kind": "anish"}, self.root)


if __name__ == "__main__":
    unittest.main()
