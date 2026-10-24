"""Patient-level manifest isolation, stratification, and persistence."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
import yaml

from seg2med_datasets.catalog import build_and_save_catalog
from seg2med_datasets.datasets.common import catalog_config
from seg2med_datasets.prepare import prepare_from_config
from seg2med_datasets.schema import SCHEMA
from seg2med_datasets.splitting import modality_ids


class PatientSplittingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.configs = []

    def add(self, source, modality, ids, *, region="pelvis", kind="synthrad", missing=False):
        images, masks, tissues = [], [], []
        for index, patient in enumerate(ids):
            paths = [self.root / f"{source}_{modality}_{patient}_{index}{suffix}.nii"
                     for suffix in ("", "_seg", "_seg_tissue")]
            for path in paths:
                path.touch()
            images.append(str(paths[0]))
            masks.append("" if missing else str(paths[1]))
            tissues.append(str(paths[2]))
        config = catalog_config(f"{source}_{modality}", images, masks, tissues, ids, modality)
        config.update(split_source=source, split_kind=kind, split_region=region)
        self.configs.append(config)

    def export(self, **kwargs):
        return build_and_save_catalog(
            self.configs, output_dir=self.root / "tables", out_csv="inventory.csv",
            out_parquet=None, check_paths=False, split_prefix="cases", **kwargs,
        )

    def test_joint_stratification_of_paired_synthrad_and_multisequence_anika(self):
        ids = [f"1PA{index:03}" for index in range(40)] + [f"1PC{index:03}" for index in range(20)]
        self.add("synthrad", 0, ids)
        self.add("synthrad", 1, ids[:40])
        self.add("synthrad", 2, ids[40:])
        # Same patients, but scan IDs vary between CT, MR and repeated scans.
        self.add("m2olie", 0, [f"{index}_4" for index in range(100, 120)], kind="anika", region="abdomen")
        self.add("m2olie", 3, [f"{index}_6" for index in range(100, 120)] + ["100_8"], kind="anika", region="abdomen")
        self.add("m2olie", 4, [f"{index}_7" for index in range(100, 119)], kind="anika", region="abdomen")
        self.add("m2olie", 5, ["100_9"], kind="anika", region="abdomen")
        result = self.export()
        audit = result["identity_audit"]
        self.assertTrue(audit.groupby(["source", "patient_id"]).split.nunique().eq(1).all())
        self.assertEqual(len(result["patient_mapping"]), 80)
        self.assertEqual(len(result["excluded_df"]), 1)
        for name, target_t1, target_t2 in (("train", 28, 14), ("validation", 6, 3), ("test", 6, 3)):
            manifest = result["manifests"][name]
            self.assertEqual(int(manifest.modality.eq(1).sum()), target_t1)
            self.assertEqual(int(manifest.modality.eq(2).sum()), target_t2)
            self.assertEqual(list(manifest.columns), [*SCHEMA.ALL, "split"])
            self.assertFalse(manifest.modality.eq(5).any())
        # In-only patient remains eligible; no forced in/opp pairing.
        self.assertTrue(((audit.patient_id == "119") & (audit.modality == 3) & audit.exclusion_reason.eq("")).any())
        self.assertEqual(set(audit.loc[audit.source.eq("m2olie") & audit.patient_id.eq("100"), "id"]),
                         {"100_4", "100_6", "100_7", "100_8", "100_9"})

    def test_mapping_reused_reordered_extended_and_schema_preserved(self):
        self.add("source", 0, [str(index) for index in range(30)])
        first = self.export()
        first_map = first["patient_mapping"].set_index(["dataset", "patient_id"]).split.to_dict()
        # Reordering scanner rows must not affect assignments, even on a fresh map.
        for key in ("images", "segs", "seg_tissues", "pIDs"):
            self.configs[0][key].reverse()
        second = self.export()
        pd.testing.assert_frame_equal(first["patient_mapping"], second["patient_mapping"])
        fresh = self.export(patient_mapping="fresh.csv")
        pd.testing.assert_frame_equal(first["patient_mapping"], fresh["patient_mapping"])
        self.add("other", 0, ["0", "new"])
        extended = self.export()
        new_map = extended["patient_mapping"].set_index(["dataset", "patient_id"]).split.to_dict()
        self.assertTrue(all(new_map[key] == value for key, value in first_map.items()))
        self.assertEqual(len(new_map), 32)
        self.assertEqual(list(extended["df"].columns), list(SCHEMA.ALL))
        with self.assertRaisesRegex(ValueError, "different seed"):
            self.export(seed=1)

    def test_exclusions_and_unresolved_identities_retained_in_inventory(self):
        self.add("m2olie", 3, ["123_6", "unrecognized"], kind="anika")
        self.add("missing", 0, ["123"], missing=True)
        self.add("dixon", 6, ["123"])
        self.add("maskonly", 0, ["123"])
        self.configs[-1]["images"][0] = ""
        self.add("no_tissues", 0, ["123"])
        self.configs[-1]["seg_tissues"][0] = ""
        result = self.export()
        self.assertEqual(len(result["df"]), 6)
        reasons = ";".join(result["excluded_df"].exclusion_reason)
        self.assertIn("unresolved_patient_identity", reasons)
        self.assertIn("missing_seg", reasons)
        self.assertIn("missing_img", reasons)
        self.assertIn("missing_seg_tissue", reasons)
        self.assertIn("excluded_modality", reasons)
        self.assertEqual(sum(len(frame) for frame in result["manifests"].values()), 1)
        self.assertTrue((self.root / "tables/cases_identity_audit.csv").is_file())
        included = self.export(included_modalities="all")
        self.assertEqual(sum(len(frame) for frame in included["manifests"].values()), 2)

    def test_patient_regex_and_explicit_mapping(self):
        self.add("custom", 0, ["P1_scan1", "P1_scan2", "bad"])
        self.configs[0]["split_patient_id_regex"] = r"(?P<patient_id>P\d+)_scan\d+"
        result = self.export()
        self.assertEqual(len(result["patient_mapping"]), 1)
        self.assertEqual(len(result["excluded_df"]), 1)
        self.configs[0]["split_patient_id_map"] = {"P1_scan1": "P1", "P1_scan2": "P1", "bad": "P2"}
        result = self.export()
        self.assertEqual(len(result["patient_mapping"]), 2)
        self.assertTrue(result["excluded_df"].empty)

    def test_anatomy_strata_and_configurable_ratios(self):
        self.add("source", 0, [str(index) for index in range(40)], region="abdomen")
        self.add("source", 0, [str(index) for index in range(40, 80)], region="pelvis")
        result = self.export(train_ratio=0.5, validation_ratio=0.25, test_ratio=0.25)
        audit = result["identity_audit"]
        for region in ("abdomen", "pelvis"):
            self.assertEqual(audit.loc[audit.region.eq(region)].groupby("split").size().to_dict(),
                             {"train": 20, "validation": 10, "test": 10})
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            self.export(train_ratio=0.8)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            self.export(train_ratio=1.0, validation_ratio=-0.1, test_ratio=0.1)

    def test_prepare_attaches_shared_source_and_scan_id_policy(self):
        ct, mr = self.root / "CT", self.root / "MR"
        ct.mkdir()
        mr.mkdir()
        for folder, stem in ((ct, "ct-volume-123_i-Spiral - 4"),
                             (mr, "moved_mr-volume-123_t1_vibe_in_tra - 6")):
            for suffix in ("", "_seg", "_seg_tissue"):
                (folder / f"{stem}{suffix}.nii.gz").touch()
        config = self.root / "scan.yaml"
        config.write_text(yaml.safe_dump({
            "data_root": str(self.root),
            "datasets": [{"kind": "anika", "name": "m2olie", "ct_root": "CT", "mr_root": "MR", "region": "abdomen"}],
            "output": {"directory": "tables", "split": {"prefix": "cases"}},
        }), encoding="utf-8")
        result = prepare_from_config(config)
        self.assertEqual(len(result["patient_mapping"]), 1)
        self.assertEqual(result["patient_mapping"].iloc[0].patient_id, "123")
        self.assertEqual(set(result["identity_audit"].region), {"abdomen"})
        self.assertEqual(set(result["df"].id), {"123_4", "123_6"})

    def test_modality_names_and_invalid_selection(self):
        self.assertEqual(modality_ids(["ct", "mr_t1_gre", "mr_t2_space", 3, 4]), {0, 1, 2, 3, 4})
        with self.assertRaises(ValueError):
            modality_ids(["typo"])


if __name__ == "__main__":
    unittest.main()
