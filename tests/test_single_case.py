"""One-step CT/MR preparation without GPU inference or model downloads."""

import contextlib
import io
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

import prepare_case as cli
from seg2med_datasets.schema import MODALITY_NAMES, SCHEMA
from seg2med_segmentation.runner import segment_volume, run_from_config
from seg2med_workflow.single_case import PRIOR_COLUMNS, prepare_case, prepare_case_from_config


class SingleCaseTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.image = self.root / "mr.nii.gz"
        nib.save(nib.Nifti1Image(np.full((5, 6, 2), 200, dtype=np.int16), np.eye(4)), self.image)
        self.config = {"dataset": "tutorial", "id": "001", "img": str(self.image), "modality_name": "mr_t1_gre",
                       "segmentation_label_scheme": "current",
                       "prior": {"min_contour_area": 0}}

    def segment(self, image, path, **kwargs):
        reference = nib.load(str(image))
        nib.save(nib.Nifti1Image(np.ones(reference.shape, dtype=np.int16), reference.affine), path)

    def run_case(self, **settings):
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=self.segment) as backend:
            result = prepare_case({**self.config, **settings})
        return result, backend

    def snapshot(self):
        return {path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.root.glob("*.nii*")}

    def test_mr_pipeline_manifests_and_original_volume_preserved(self):
        original = self.image.read_bytes()
        result, backend = self.run_case()
        self.assertEqual([call.kwargs["task"] for call in backend.call_args_list], ["total_mr", "tissue_types_mr"])
        self.assertTrue(all(call.kwargs["device"] == "cuda" and call.kwargs["multi_label"] and not call.kwargs["fast"]
                            for call in backend.call_args_list))
        self.assertEqual(self.image.read_bytes(), original)
        for key in ("seg", "seg_tissue", "prior", "prior_semantic", "target", "seg_csv", "prior_csv"):
            self.assertTrue(result[key].is_file(), key)
            self.assertEqual(result[key].parent, self.root)
        seg = pd.read_csv(result["seg_csv"], dtype={"id": str})
        prior = pd.read_csv(result["prior_csv"], dtype={"id": str})
        self.assertEqual(list(seg.columns), list(SCHEMA.ALL))
        self.assertEqual(list(prior.columns), list(PRIOR_COLUMNS))
        self.assertEqual(seg.iloc[0].id, "001")
        self.assertEqual(seg.iloc[0].aorta_diss, 0)
        self.assertEqual(prior.iloc[0].aorta_diss, 0)
        self.assertEqual(seg.iloc[0].img, str(self.image))
        self.assertEqual(prior.iloc[0].img, str(result["target"]))
        self.assertNotEqual(result["target"], self.image)
        self.assertEqual(nib.load(result["prior"]).shape, (5, 6, 2))

    def test_ct_and_all_mr_modalities_route_to_correct_tasks(self):
        for modality, name in MODALITY_NAMES.items():
            with self.subTest(modality=name):
                result, backend = self.run_case(modality_name=name, overwrite_seg=True, overwrite_prior=True)
                expected = ["total", "tissue_types"] if modality == 0 else ["total_mr", "tissue_types_mr"]
                self.assertEqual([call.kwargs["task"] for call in backend.call_args_list], expected)
                self.assertEqual(result["modality"], modality)

    def test_complete_outputs_are_reused_without_loading_backends(self):
        self.run_case()
        before = self.snapshot()
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Backend called")), \
                patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Priors regenerated")):
            result = prepare_case(self.config)
        self.assertFalse(result["prior_planned"])
        self.assertEqual(result["segmentation_planned"], [])
        self.assertEqual(before, self.snapshot())

    def test_partial_outputs_are_completed_without_overwriting_existing_files(self):
        result, _ = self.run_case()
        result["seg_tissue"].unlink()
        result["prior_semantic"].unlink()
        before = self.snapshot()
        result, backend = self.run_case()
        self.assertEqual(backend.call_count, 1)
        self.assertEqual(backend.call_args.kwargs["task"], "tissue_types_mr")
        after = self.snapshot()
        for path, snapshot in before.items():
            self.assertEqual(snapshot, after[path])
        self.assertTrue(result["prior_semantic"].is_file())

    def test_overwrite_flags_are_independent(self):
        self.run_case()
        with patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Priors regenerated")):
            _, backend = self.run_case(overwrite_seg=True)
        self.assertEqual(backend.call_count, 2)
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Segmentation ran")):
            result = prepare_case({**self.config, "overwrite_prior": True})
        self.assertTrue(result["prior_planned"])

    def test_uncompressed_and_mixed_compression_masks_are_reused(self):
        image = self.root / "ct.nii"
        nib.save(nib.load(self.image), image)
        for suffix in ("_seg.nii", "_seg_tissue.nii.gz"):
            self.segment(image, self.root / f"ct{suffix}")
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Segmentation ran")):
            result = prepare_case({**self.config, "img": str(image), "modality_name": "ct"})
        self.assertEqual(result["seg"].suffix, ".nii")
        self.assertTrue(result["prior"].name.endswith("_prior_image.nii"))
        self.assertTrue(result["prior_semantic"].is_file())

    def test_dry_run_does_not_write_or_run_inference(self):
        before = {path for path in self.root.iterdir()}
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Backend called")):
            result = prepare_case(self.config, dry_run=True)
        self.assertEqual({path for path in self.root.iterdir()}, before)
        self.assertEqual(len(result["segmentation_planned"]), 2)
        self.assertTrue(result["prior_planned"])

    def test_relative_yaml_paths_cli_overrides_and_relative_csv_paths(self):
        config = self.root / "case.yaml"
        config.write_text(yaml.safe_dump({**self.config, "img": "mr.nii.gz", "overwrite_seg": True,
                                          "absolute_paths": False}), encoding="utf-8")
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=self.segment):
            result = prepare_case_from_config(config)
        row = pd.read_csv(result["prior_csv"]).iloc[0]
        self.assertEqual(row.prior, result["prior"].name)
        # CLI flags override YAML, including negative BooleanOptionalAction flags.
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Backend called")):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["--config", str(config), "--no-overwrite-seg", "--no-overwrite-prior"]), 0)

    def test_reusing_only_prior_files_uses_original_target_when_no_processed_target_exists(self):
        result, _ = self.run_case()
        result["target"].unlink()
        with patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Priors regenerated")):
            reused = prepare_case(self.config)
        self.assertEqual(reused["target"], self.image)

    def test_relative_custom_manifests_remain_resolvable(self):
        self.run_case()
        config = {**self.config, "absolute_paths": False,
                  "seg_csv": str(self.root / "tables/catalog.csv"),
                  "prior_csv": str(self.root / "other/priors.csv")}
        result = prepare_case(config)
        seg = pd.read_csv(result["seg_csv"]).iloc[0]
        prior = pd.read_csv(result["prior_csv"]).iloc[0]
        self.assertEqual((result["seg_csv"].parent / seg.img).resolve(), self.image)
        self.assertEqual((result["prior_csv"].parent / prior.prior).resolve(), result["prior"])

    def test_invalid_inputs_and_misaligned_masks_fail(self):
        for settings in ({"modality_name": "bad"}, {"modality_name": "99"}, {"id": ""},
                         {"seg_csv": str(self.image)}, {"segmentation": {"multi_label": False}}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                prepare_case({**self.config, **settings})
        result, _ = self.run_case()
        nib.save(nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4)), result["seg"])
        with self.assertRaisesRegex(ValueError, "image grid"):
            prepare_case(self.config)

    def test_backend_failure_does_not_write_a_success_manifest(self):
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError, "failed"):
                prepare_case(self.config)
        self.assertFalse(list(self.root.glob("*.csv")))

    def test_segmentation_cuda_alias_and_batch_reuse(self):
        api = types.ModuleType("totalsegmentator.python_api")
        api.totalsegmentator = Mock()
        with patch.dict(sys.modules, {"totalsegmentator.python_api": api}):
            for requested, expected in (("cuda", "gpu"), ("cuda:1", "gpu:1"), ("cpu", "cpu")):
                segment_volume(self.image, self.root / "out.nii.gz", task="total", device=requested)
                self.assertEqual(api.totalsegmentator.call_args.kwargs["device"], expected)
        result, _ = self.run_case()
        config = self.root / "batch.yaml"
        config.write_text(yaml.safe_dump({"source": {"type": "paths", "paths": [str(self.image)]},
                                          "tasks": ["total_mr", "tissue_types_mr"]}), encoding="utf-8")
        with patch("seg2med_segmentation.runner.segment_volume", side_effect=AssertionError("Backend called")):
            batch = run_from_config(config)
        self.assertEqual(len(batch["skipped"]), 2)

    def test_cli_only_input_dry_run(self):
        completed = subprocess.run([sys.executable, "-B", "prepare_case.py", "--dataset", "tutorial", "--id", "001",
                                    "--img", str(self.image), "--modality", "1", "--dry-run"],
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("total_mr", completed.stdout)
        self.assertFalse(list(self.root.glob("*.csv")))

    def multi_config(self, **shared):
        folder = self.root / "second"
        folder.mkdir(exist_ok=True)
        second = folder / "mr.nii.gz"
        nib.save(nib.load(self.image), second)
        path = self.root / "cases.yaml"
        value = {"dataset": "tutorial", "modality_name": "mr_t1_gre",
                 "segmentation_label_scheme": "current",
                 "prior": {"min_contour_area": 0}, "segmentation": {"device": "cuda:1", "fast": False},
                 "case": [{"id": "001", "img": "mr.nii.gz"},
                          {"id": "004", "img": "second/mr.nii.gz", "segmentation": {"fast": True}}], **shared}
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path, value

    def test_yaml_case_list_processes_cases_with_shared_and_nested_settings(self):
        config, _ = self.multi_config()
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=self.segment) as backend:
            results = prepare_case_from_config(config)
        self.assertEqual(len(results), 2)
        self.assertEqual(backend.call_count, 4)
        self.assertTrue(all(call.kwargs["device"] == "cuda:1" for call in backend.call_args_list))
        self.assertEqual([call.kwargs["fast"] for call in backend.call_args_list], [False, False, True, True])
        for result, expected in zip(results, ("001", "004")):
            self.assertEqual(pd.read_csv(result["prior_csv"], dtype={"id": str}).iloc[0].id, expected)
            self.assertTrue(result["prior"].is_file())
            self.assertEqual(result["seg_csv"].parent, result["img"].parent)

    def test_yaml_case_list_dry_run_and_cli_processing_override(self):
        config, _ = self.multi_config(overwrite_seg=True, overwrite_prior=True)
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Backend called")):
            plans = prepare_case_from_config(config, dry_run=True)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(cli.main(["--config", str(config), "--dry-run", "--no-overwrite-seg"]), 0)
        self.assertEqual(len(plans), 2)
        self.assertIn("Planned: 2 cases", output.getvalue())
        self.assertFalse(list(self.root.rglob("*.csv")))
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=self.segment):
            prepare_case_from_config(config)
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Backend called")), \
                patch("seg2med_physiosynth.priors._prepare_volume", side_effect=AssertionError("Priors regenerated")):
            results = prepare_case_from_config(config, overrides={"overwrite_seg": False, "overwrite_prior": False})
        self.assertEqual(len(results), 2)
        self.assertTrue(all(not result["prior_planned"] for result in results))

    def test_yaml_case_list_rejects_cli_metadata_and_duplicate_output_paths(self):
        config, value = self.multi_config()
        for overrides in ({"img": str(self.image)}, {"id": "new"}, {"seg_csv": str(self.root / "shared.csv")}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                prepare_case_from_config(config, overrides=overrides)
        value["case"][1]["img"] = "mr.nii.gz"
        config.write_text(yaml.safe_dump(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "distinct input"):
            prepare_case_from_config(config)
        self.assertFalse(list(self.root.rglob("*.csv")))

    def test_yaml_case_list_validates_every_case_before_processing(self):
        config, value = self.multi_config()
        value["case"][1]["img"] = "missing.nii.gz"
        config.write_text(yaml.safe_dump(value), encoding="utf-8")
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Backend called")):
            with self.assertRaises(FileNotFoundError):
                prepare_case_from_config(config)
        self.assertFalse(list(self.root.rglob("*.csv")))
        for invalid in ([], {}, ["bad"]):
            config.write_text(yaml.safe_dump({"case": invalid}), encoding="utf-8")
            with self.subTest(case=invalid), self.assertRaises(ValueError):
                prepare_case_from_config(config)

    def test_yaml_case_modality_overrides_shared_name(self):
        config, value = self.multi_config()
        value["case"][1]["modality"] = 0
        config.write_text(yaml.safe_dump(value), encoding="utf-8")
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=self.segment) as backend:
            results = prepare_case_from_config(config)
        self.assertEqual([result["modality"] for result in results], [1, 0])
        self.assertEqual([call.kwargs["task"] for call in backend.call_args_list],
                         ["total_mr", "tissue_types_mr", "total", "tissue_types"])

    def test_cross_modality_prior_names_and_no_false_ground_truth(self):
        same, _ = self.run_case()
        before = same["prior"].read_bytes()
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Segmentation ran")):
            cross = prepare_case({**self.config, "input_modality_name": "mr_t1_gre",
                                  "target_modality_name": "ct"})
        self.assertEqual(cross["prior"].name, "mr_prior_image_ct.nii.gz")
        self.assertEqual(cross["prior_semantic"].name, "mr_prior_semantic_ct.nii.gz")
        self.assertEqual(cross["prior_csv"].name, "mr_priors_ct.csv")
        row = pd.read_csv(cross["prior_csv"], keep_default_na=False).iloc[0]
        self.assertEqual(row.img, "")
        self.assertEqual(row.modality, 0)
        self.assertEqual(pd.read_csv(cross["seg_csv"]).iloc[0].modality, 1)
        self.assertEqual(same["prior"].read_bytes(), before)
        self.assertFalse(np.allclose(nib.load(cross["prior"]).get_fdata(), nib.load(same["prior"]).get_fdata()))

    def test_cross_modality_contour_and_processed_image_use_input_units(self):
        ct = np.full((5, 6, 2), -100, dtype=np.int16)
        ct[0, :, :] = -1000
        nib.save(nib.Nifti1Image(ct, np.eye(4)), self.image)
        result, backend = self.run_case(input_modality_name="ct", target_modality_name="mr_t1_gre")
        self.assertEqual(backend.call_args_list[0].kwargs["task"], "total")
        contour = nib.load(result["contour"]).get_fdata()
        self.assertGreater(contour.sum(), 0) # MR's threshold would remove this entire CT body.
        np.testing.assert_array_equal(nib.load(result["target"]).get_fdata()[0], -1000)

    def test_multiple_modalities_from_same_input_reuse_masks(self):
        path = self.root / "cross.yaml"
        path.write_text(yaml.safe_dump({"dataset": "test", "input_modality_name": "mr_t1_gre",
            "segmentation_label_scheme": "current",
            "prior": {"min_contour_area": 0}, "case": [
                {"id": "001", "img": "mr.nii.gz", "target_modality_name": "ct"},
                {"id": "001", "img": "mr.nii.gz", "target_modality_name": "mr_t2_space"}]}), encoding="utf-8")
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=self.segment) as backend:
            results = prepare_case_from_config(path)
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(len({result["prior"] for result in results}), 2)

    def test_null_target_preserves_legacy_output_names(self):
        result = prepare_case({**self.config, "input_modality_name": "mr_t1_gre",
                               "target_modality_name": None}, dry_run=True)
        self.assertEqual(result["prior"].name, "mr_prior_image.nii.gz")

    def test_explicit_same_target_adds_suffix_and_keeps_real_reference(self):
        original, _ = self.run_case()
        before = original["prior"].read_bytes()
        with patch("seg2med_workflow.single_case.segment_volume", side_effect=AssertionError("Segmentation ran")):
            result = prepare_case({**self.config, "target_modality_name": "mr_t1_gre"})
        self.assertEqual(result["prior"].name, "mr_prior_image_mr_t1_gre.nii.gz")
        self.assertEqual(result["prior_semantic"].name, "mr_prior_semantic_mr_t1_gre.nii.gz")
        self.assertEqual(result["prior_csv"].name, "mr_priors_mr_t1_gre.csv")
        self.assertEqual(pd.read_csv(result["prior_csv"]).iloc[0].img, str(result["target"]))
        self.assertEqual(original["prior"].read_bytes(), before)
        numeric = prepare_case({**self.config, "target_modality_name": 1}, dry_run=True)
        self.assertEqual(numeric["prior"], result["prior"])

    def test_null_and_explicit_same_modality_can_share_an_input(self):
        path = self.root / "naming.yaml"
        path.write_text(yaml.safe_dump({**self.config, "case": [
            {"target_modality_name": None}, {"target_modality_name": "mr_t1_gre"}]}), encoding="utf-8")
        plans = prepare_case_from_config(path, dry_run=True)
        self.assertEqual([plan["prior"].name for plan in plans],
                         ["mr_prior_image.nii.gz", "mr_prior_image_mr_t1_gre.nii.gz"])

    def test_script_and_app_cross_modality_remapping_agree(self):
        from seg2med_physiosynth.simulator import simulate_prior
        from seg2med_app.simulation.modality_simulator import simulate_modality

        # Deliberately different organ IDs in source and destination tables.
        ct = pd.DataFrame({"Order Number": [0, 1, 2], "Organ Name": ["body", "liver", "spleen"],
                           "HU Value": [0, 50, 40]})
        mr = pd.DataFrame({"Order Number": [0, 1, 2], "Organ Name": ["body", "spleen", "liver"],
                           "T1": [1000, 1200, 800], "T2": [80, 100, 60], "Rho": [1, 2, 3]})
        image = np.full((5, 6, 2), -100)
        organs = np.ones(image.shape, np.int16)
        tissues = np.zeros_like(organs)
        prior, semantic, contour = simulate_prior(image, organs, tissues, 1, input_modality=0,
            ct_parameters=ct, mr_parameters=mr, min_contour_area=0)
        expected = simulate_modality(contour, 1, organs, "ct", ct_parameters=ct, mr_parameters=mr)
        np.testing.assert_array_equal(prior, expected)
        np.testing.assert_array_equal(semantic, 3) # target liver ID 2 + body contour 1


if __name__ == "__main__":
    unittest.main()
