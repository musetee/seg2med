"""Manual YAML volumes use normal inference without inventing reference targets."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

from seg2med_train.Dataloader import PriorImageCsvLoader
from seg2med_train.utils.my_configs_yacs import init_cfg, config_path


class ManualPriorInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prior = self.root / "prior.nii.gz"
        nib.save(nib.Nifti1Image(np.full((8, 8, 2), 200, np.float32), np.eye(4)), self.prior)

    def config(self, **settings):
        values = {"model_name": "ddpm2d_seg2med", "mode": "test", "dataset": {
            "num_workers": 0, "resized_size": [8, 8, None],
            "case": [{"id": "001", "prior": "prior.nii.gz"}], "modality_name": "ct"}}
        if "case" in settings:
            values["dataset"].pop("case")
        for key, value in settings.items():
            if key == "dataset":
                values["dataset"].update(value)
            else:
                values[key] = value
        path = self.root / "inference.yaml"
        path.write_text(yaml.safe_dump(values), encoding="utf-8")
        return init_cfg(str(path))

    def loader(self, cfg):
        paths = config_path("manual", run_dir=str(self.root / "run"))
        Path(paths["saved_logs_folder"]).mkdir(parents=True, exist_ok=True)
        return PriorImageCsvLoader(cfg, paths, source_column="prior", normalize_source=True)

    def test_prior_only_and_multiple_cases_normalization(self):
        cfg = self.config(dataset={"case": [
            {"id": "001", "prior": "prior.nii.gz"},
            {"id": "002", "prior": "prior.nii.gz", "modality_name": "mr_t1_gre"}]})
        loader = self.loader(cfg)
        self.assertEqual(loader.test_patient_IDs, ["001", "002"])
        batches = list(loader.val_loader)
        self.assertFalse(bool(batches[0]["has_target"][0]))
        self.assertTrue(torch.all(batches[0]["source"] == 0.5))
        self.assertTrue(torch.all(batches[0]["target"] == 0))
        self.assertTrue(torch.all(batches[1]["source"] == 1))
        self.assertEqual(batches[0]["B_paths"], [""])

    def test_optional_reference_and_root_case_alias(self):
        cfg = self.config(case=[{
            "id": "001", "prior": "prior.nii.gz", "img": "prior.nii.gz", "modality": 0}])
        batch = next(iter(self.loader(cfg).val_loader))
        self.assertTrue(bool(batch["has_target"][0]))
        self.assertTrue(torch.all(batch["target"] == 0.5))

    def test_training_missing_modality_bad_file_and_duplicate_rejected(self):
        for settings in ({"mode": "train"}, {"dataset": {"modality_name": None}},
                         {"dataset": {"case": [{"id": "1", "prior": "missing.nii.gz"}]}},
                         {"dataset": {"case": [{"id": "1", "prior": "prior.nii.gz"}] * 2}}):
            with self.subTest(settings=settings), self.assertRaises((ValueError, FileNotFoundError)):
                self.loader(self.config(**settings))

    def test_reference_geometry_is_checked(self):
        other = self.root / "other.nii.gz"
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2)), np.eye(4)), other)
        cfg = self.config(dataset={"case": [{"id": "1", "prior": "prior.nii.gz", "img": "other.nii.gz"}]})
        with self.assertRaisesRegex(ValueError, "same grid"):
            self.loader(cfg)

    def test_empty_target_csv_is_supported_for_cross_modality_priors(self):
        import pandas as pd
        csv = self.root / "priors.csv"
        pd.DataFrame([{"id": "1", "prior": str(self.prior), "img": "", "modality": 0,
                       "aorta_diss": 0}]).to_csv(csv, index=False)
        cfg = self.config(dataset={"case": [], "test_csv": str(csv)})
        self.assertFalse(bool(next(iter(self.loader(cfg).val_loader))["has_target"][0]))

    def test_both_ddpm_runners_infer_without_saving_fake_targets(self):
        from unittest.mock import patch
        from seg2med_train.networks.ddpm.seg2med_ddpm import DiffusionModel, DiffusionModel_multimodal

        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, old_threads)
        for runner in (DiffusionModel, DiffusionModel_multimodal):
            cfg = self.config()
            cfg.model_name = "ddpm2d_seg2med_multimodal" if runner is DiffusionModel_multimodal else "ddpm2d_seg2med"
            cfg.ddpm.num_channels = (8, 8)
            cfg.ddpm.attention_levels = (False, False)
            cfg.ddpm.norm_num_groups = 8
            cfg.ddpm.num_head_channels = 8
            cfg.ddpm.num_res_units = 1
            cfg.ddpm.num_class_embeds = 2 if runner is DiffusionModel_multimodal else None
            cfg.ddpm.noise_type = "normal"
            cfg.ddpm.num_train_timesteps = 10
            cfg.ddpm.num_inference_steps = 2
            cfg.validation.save_dicom = True
            cfg.validation.save_png = True
            cfg.validation.save_npz = True
            loader = self.loader(cfg)
            paths = config_path("inference", run_dir=str(self.root / runner.__name__))
            owner = runner(cfg, paths, loader.train_loader, loader.val_loader, [], loader.test_patient_IDs)
            with patch("torch.cuda.is_available", return_value=False):
                owner.val()
            output = Path(paths["saved_img_folder"])
            self.assertEqual(len(list(output.rglob("*_output.tiff"))), 1)
            self.assertEqual(len(list(output.rglob("*_input.tiff"))), 1)
            self.assertFalse(list(output.rglob("*_target.*")))
            self.assertFalse(list(output.rglob("target")))
            self.assertEqual(len(list(output.rglob("*.dcm"))), 4)
            self.assertIsNone(owner.mse_loss)


if __name__ == "__main__":
    unittest.main()
