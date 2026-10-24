"""Train/validation CSV isolation and evaluation-only loader compatibility."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import nibabel as nib
import numpy as np
import pandas as pd

from seg2med_train.Dataloader import PriorImageCsvLoader
from seg2med_train.utils.my_configs_yacs import config_path, init_cfg


class TrainingLoaderTests(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = init_cfg()
        self.cfg.model_name = "ddpm2d_seg2med"
        self.cfg.dataset.data_dir = str(self.root)
        self.cfg.dataset.num_workers = 0
        self.cfg.dataset.resized_size = (8, 8, None)
        self.cfg.dataset.zoom = (1.0, 1.0, 1.0)
        for split, patient in (("train", "101"), ("val", "202")):
            source, target = self.root / f"{patient}_prior.nii.gz", self.root / f"{patient}.nii.gz"
            nib.save(nib.Nifti1Image(np.full((8, 8, 2), 100, dtype=np.float32), np.eye(4)), target)
            nib.save(nib.Nifti1Image(np.full((8, 8, 2), 0, dtype=np.float32), np.eye(4)), source)
            manifest = self.root / f"{split}.csv"
            pd.DataFrame([{"id": patient, "aorta_diss": 0, "prior": str(source), "img": str(target), "modality": 0}]).to_csv(manifest, index=False)
            setattr(self.cfg.dataset, f"{split}_csv", str(manifest))

    def loader(self):
        paths = config_path("loader", run_dir=str(self.root / "run"))
        Path(paths["saved_logs_folder"]).mkdir(parents=True, exist_ok=True)
        return PriorImageCsvLoader(self.cfg, paths, source_column="prior", normalize_source=True)

    def test_training_never_reads_test_and_validation_is_deterministic(self):
        self.cfg.dataset.test_csv = str(self.root / "test_DOES_NOT_EXIST.csv")
        loader = self.loader()
        self.assertEqual(loader.train_patient_IDs, ["101"])
        self.assertEqual(loader.val_patient_IDs, ["202"])
        self.assertEqual(loader.test_patient_IDs, [])
        first = next(iter(loader.val_loader))
        second = next(iter(loader.val_loader))
        self.assertTrue(torch.equal(first["source"], second["source"]))
        self.assertTrue(torch.equal(first["target"], second["target"]))
        self.assertEqual(first["sample_id"], second["sample_id"])

    def test_training_requires_independent_validation(self):
        self.cfg.dataset.val_csv = ""
        with self.assertRaisesRegex(ValueError, "requires dataset.val_csv"):
            self.loader()
        self.cfg.dataset.val_csv = self.cfg.dataset.train_csv
        with self.assertRaisesRegex(ValueError, "independent manifests"):
            self.loader()

    def test_patient_overlap_is_rejected_including_scan_specific_ids(self):
        for field, patient in (("train_csv", "101_4"), ("val_csv", "101_6")):
            path = Path(getattr(self.cfg.dataset, field))
            frame = pd.read_csv(path)
            frame["id"] = patient
            frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            self.loader()

    def test_test_mode_only_requires_test_csv(self):
        self.cfg.mode = "test"
        self.cfg.dataset.test_csv = self.cfg.dataset.val_csv
        self.cfg.dataset.train_csv = "missing_train.csv"
        self.cfg.dataset.val_csv = "missing_val.csv"
        loader = self.loader()
        self.assertEqual(loader.train_patient_IDs, [])
        self.assertEqual(loader.test_patient_IDs, ["202"])
        self.assertEqual(next(iter(loader.val_loader))["patient_ID"], ["202"])

    def test_entrypoint_smoke_with_independent_nifti_manifests(self):
        from seg2med_train.train import run
        for model_name in ("ddpm2d_seg2med", "ddpm2d_seg2med_multimodal"):
            with self.subTest(model_name=model_name):
                cfg = self.cfg.clone()
                cfg.model_name = model_name
                cfg.train.run_dir = str(self.root / model_name)
                cfg.train.num_epochs = 1
                cfg.train.save_ckpt_interval = 0
                cfg.ddpm.num_channels = (8, 8)
                cfg.ddpm.attention_levels = (False, False)
                cfg.ddpm.norm_num_groups = 8
                cfg.ddpm.num_head_channels = 8
                cfg.ddpm.num_res_units = 1
                cfg.ddpm.noise_type = "normal"
                cfg.ddpm.num_train_timesteps = 500
                cfg.ddpm.num_class_embeds = 2 if "multimodal" in model_name else None
                cfg.validation.preview.epoch_interval = 1
                cfg.validation.preview.num_inference_steps = 2
                cfg.validation.preview.num_samples = 1
                cfg.dataset.test_csv = "DOES_NOT_EXIST.csv"
                config = self.root / f"{model_name}.yaml"
                config.write_text(cfg.dump(), encoding="utf-8")
                with patch("torch.cuda.is_available", return_value=False):
                    run(["--config", str(config)])
                folder = Path(cfg.train.run_dir)
                self.assertTrue((folder / "saved_models/last.pt").is_file())
                self.assertTrue((folder / "saved_models/best.pt").is_file())
                self.assertTrue((folder / "saved_logs/validation_sampling.csv").is_file())
                self.assertTrue(list((folder / "saved_outputs/previews").rglob("*.png")))
                self.assertFalse((folder / "saved_tensorboard").exists())
                self.assertFalse((folder / "saved_inference").exists())


if __name__ == "__main__":
    unittest.main()
