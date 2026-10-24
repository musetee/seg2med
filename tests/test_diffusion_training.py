"""Small CPU integration tests for both production diffusion workflows."""

import csv
import copy
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from seg2med_train.networks.ddpm.seg2med_ddpm import DiffusionModel, DiffusionModel_multimodal
from seg2med_train.networks.ddpm.training import DiffusionTrainer, load_checkpoint, rng_state
from seg2med_train.utils.my_configs_yacs import config_path, init_cfg


class DiffusionTrainingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)
        cpu = patch("torch.cuda.is_available", return_value=False)
        cpu.start()
        self.addCleanup(cpu.stop)

    def config(self, multimodal=False):
        cfg = init_cfg()
        cfg.model_name = "ddpm2d_seg2med_multimodal" if multimodal else "ddpm2d_seg2med"
        cfg.dataset.batch_size = 2
        cfg.dataset.val_batch_size = 2
        cfg.dataset.num_workers = 0
        cfg.dataset.resized_size = (8, 8, None)
        cfg.ddpm.num_channels = (8, 8)
        cfg.ddpm.attention_levels = (False, False)
        cfg.ddpm.norm_num_groups = 8
        cfg.ddpm.num_head_channels = 8
        cfg.ddpm.num_res_units = 1
        cfg.ddpm.num_train_timesteps = 500
        cfg.ddpm.noise_type = "normal"
        cfg.ddpm.num_class_embeds = 3 if multimodal else None
        cfg.train.num_epochs = 2
        cfg.train.save_ckpt_interval = 0
        cfg.train.earlystopping_delta = 0.0
        cfg.validation.preview.num_samples = 4
        cfg.validation.preview.num_inference_steps = 2
        cfg.validation.preview.epoch_interval = 1
        cfg.validation.monitor = "val_modality_patient_denoising_mse" if multimodal else "val_patient_denoising_mse"
        return cfg

    def loader(self, reverse=False):
        rows = []
        generator = torch.Generator().manual_seed(91)
        for index, depth in ((0, 3), (1, 1), (2, 2)):
            rows.append({"source": torch.rand((1, 8, 8, depth), generator=generator),
                         "target": torch.rand((1, 8, 8, depth), generator=generator),
                         "patient_ID": f"P{index}", "patient_key": f"source:P{index}",
                         "sample_id": f"sample{index}", "modality": index, "Aorta_diss": 0})
        return DataLoader(list(reversed(rows)) if reverse else rows, batch_size=1, shuffle=False)

    def owner(self, cfg, folder):
        cls = DiffusionModel_multimodal if "multimodal" in cfg.model_name else DiffusionModel
        paths = config_path("smoke", run_dir=str(self.root / folder))
        return cls(cfg, paths, self.loader(), self.loader(), None, None)

    def test_both_workflows_validate_preview_checkpoint_and_resume(self):
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                cfg = self.config(multimodal)
                torch.manual_seed(12)
                model = self.owner(cfg, str(multimodal))
                model.train()
                logs = Path(model.log_path)
                saved = load_checkpoint(Path(model.saved_models_name) / "last.pt")
                self.assertEqual(saved["epoch"], 2)
                self.assertEqual(saved["global_step"], 6)
                self.assertTrue((Path(model.saved_models_name) / "best.pt").is_file())
                self.assertTrue((logs / "epoch_metrics.csv").is_file())
                for name in ("training_loss_curve.png", "validation_loss_curve.png", "loss_curves.png"):
                    self.assertTrue((logs / name).is_file(), name)
                self.assertEqual(saved["best_metric"], min(row[cfg.validation.monitor] for row in saved["history"]))
                with (logs / "validation_sampling.csv").open(newline="", encoding="utf-8") as stream:
                    records = list(csv.DictReader(stream))
                self.assertEqual(len(records), 6)
                before = (logs / "validation_sampling.csv").read_bytes()
                preview = Path(model.output_path) / "previews/epoch_0002"
                metadata = json.loads((preview / "preview_metadata.json").read_text())
                self.assertEqual(metadata["sampler"], "ddim")
                self.assertEqual(len(metadata["timesteps"]), 2)
                self.assertEqual(len(list(preview.glob("*.png"))), 4)
                initial_steps = model.scheduler.timesteps.clone()
                state = torch.get_rng_state().clone()
                metrics = model.trainer.validate()
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertTrue(model.model.training)
                cfg.dataset.val_batch_size = 1
                model.val_loader = self.loader(reverse=True)
                other = model.trainer.validate()
                for key in ("val_denoising_mse", "val_patient_denoising_mse", "val_modality_patient_denoising_mse"):
                    self.assertAlmostEqual(metrics[key], other[key], places=6)
                self.assertTrue(torch.equal(initial_steps, model.scheduler.timesteps))
                cfg.train.resume = True
                cfg.ckpt_path = str(Path(model.saved_models_name) / "last.pt")
                cfg.train.num_epochs = 3
                resumed = self.owner(cfg, str(multimodal))
                resumed.train()
                self.assertEqual((logs / "validation_sampling.csv").read_bytes(), before)
                last = load_checkpoint(Path(resumed.saved_models_name) / "last.pt")
                self.assertEqual(last["epoch"], 3)
                self.assertEqual(last["global_step"], 9)
                self.assertEqual(len(last["history"]), 3)
                self.assertIn("lr_scheduler", last)
                self.assertIn("rng_states", last)
                # Restored training RNG/optimizer must match uninterrupted training.
                continuous_cfg = self.config(multimodal)
                continuous_cfg.train.num_epochs = 3
                torch.manual_seed(12)
                continuous = self.owner(continuous_cfg, f"continuous_{multimodal}")
                continuous.train()
                expected = load_checkpoint(Path(continuous.saved_models_name) / "last.pt")
                for name in last["model"]:
                    self.assertTrue(torch.equal(last["model"][name], expected["model"][name]), name)

    def test_cosine_schedule_decays_to_min_lr_and_stays(self):
        cfg = self.config()
        cfg.train.num_epochs = 6
        cfg.train.scheduler.type = "cosine"
        cfg.train.scheduler.min_lr = 1e-5
        cfg.validation.preview.enabled = False
        owner = self.owner(cfg, "cosine")
        owner.init_diffusion()
        trainer = DiffusionTrainer(owner)
        with patch.object(trainer, "validate", return_value={cfg.validation.monitor: 1.0}), \
                patch("seg2med_train.plot_loss_curves.plot_loss_curves"):
            trainer.run()
        used = [row["lr_used"] for row in trainer.history]
        self.assertAlmostEqual(used[0], cfg.train.learning_rate)
        self.assertTrue(all(later < earlier for earlier, later in zip(used, used[1:])))
        self.assertAlmostEqual(trainer.history[-1]["lr"], 1e-5)
        trainer.lr_scheduler.step()
        self.assertAlmostEqual(owner.optimizer.param_groups[0]["lr"], 1e-5)

    def test_cosine_resume_with_changed_stopping_policy(self):
        cfg = self.config()
        cfg.train.scheduler.type = "cosine"
        cfg.validation.preview.enabled = False
        with patch("seg2med_train.plot_loss_curves.plot_loss_curves"):
            first = self.owner(cfg, "cosine_resume")
            first.train()
            cfg.train.resume = True
            cfg.ckpt_path = str(Path(first.saved_models_name) / "last.pt")
            cfg.train.num_epochs = 3
            cfg.train.earlystopping_patience = 5
            resumed = self.owner(cfg, "cosine_resume")
            resumed.train()
        last = load_checkpoint(Path(resumed.saved_models_name) / "last.pt")
        self.assertEqual(last["epoch"], 3)
        self.assertEqual(last["early_stopping"]["patience"], 5)
        metadata = json.loads((Path(resumed.log_path) / "run_metadata.json").read_text(encoding="utf-8"))
        self.assertRegex(metadata["code_version"]["commit"], r"^[0-9a-f]{40}$")

    def test_ema_volume_monitor_resume_and_inference_weights(self):
        def configure(cfg):
            cfg.train.ema.enabled = True
            cfg.train.ema.decay = 0.9
            cfg.validation.volume_eval.enabled = True
            cfg.validation.volume_eval.epoch_interval = 1
            cfg.validation.volume_eval.num_volumes = 2
            cfg.validation.volume_eval.num_inference_steps = 2
            cfg.validation.volume_eval.batch_size = 2
            cfg.validation.shared_volume_noise = True
            cfg.validation.monitor = "val_volume_mae"
            return cfg

        cfg = configure(self.config())
        torch.manual_seed(5)
        model = self.owner(cfg, "ema")
        model.train()
        logs, models = Path(model.log_path), Path(model.saved_models_name)
        saved = load_checkpoint(models / "last.pt")
        self.assertEqual(saved["ema"]["num_updates"], 6)
        self.assertTrue(any(not torch.equal(saved["ema_model"][name], saved["model"][name])
                            for name in saved["model"]))
        self.assertEqual(saved["best_metric"], min(row["val_volume_mae"] for row in saved["history"]))
        self.assertTrue((logs / "sampling_quality_curve.png").is_file())
        selection = json.loads((logs / "volume_eval_samples.json").read_text())
        self.assertEqual(len({row["patient_key"] for row in selection}), 2)
        report = json.loads((Path(model.output_path) / "volume_eval/epoch_0002/volume_metrics.json").read_text())
        self.assertEqual([row["slices"] for row in report["volumes"]],
                         [3 if row["volume_id"] == "sample0" else 1 if row["volume_id"] == "sample1" else 2
                          for row in selection])
        # Resume restores EMA state exactly like uninterrupted training.
        cfg.train.resume = True
        cfg.ckpt_path = str(models / "last.pt")
        cfg.train.num_epochs = 3
        self.owner(cfg, "ema").train()
        resumed = load_checkpoint(models / "last.pt")
        continuous_cfg = configure(self.config())
        continuous_cfg.train.num_epochs = 3
        torch.manual_seed(5)
        continuous = self.owner(continuous_cfg, "ema_continuous")
        continuous.train()
        expected = load_checkpoint(Path(continuous.saved_models_name) / "last.pt")
        for name in resumed["ema_model"]:
            self.assertTrue(torch.equal(resumed["ema_model"][name], expected["ema_model"][name]), name)
        # Inference uses EMA weights unless ddpm.use_ema is false.
        for use_ema, key in ((True, "ema_model"), (False, "model")):
            test_cfg = self.config()
            test_cfg.mode, test_cfg.ckpt_path, test_cfg.ddpm.use_ema = "test", str(models / "last.pt"), use_ema
            runner = self.owner(test_cfg, f"infer_{use_ema}")
            runner.init_diffusion()
            for name, value in runner.model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed[key][name]), name)

    def test_volume_metrics_measure_brightness_flicker(self):
        from seg2med_train.networks.ddpm.training import volume_metrics

        target = np.ones((3, 4, 4))
        output = target * np.array([1.0, 0.5, 1.0])[:, None, None]
        metrics = volume_metrics(output, target, 0.05, 0.0)
        self.assertAlmostEqual(metrics["mae"], 0.5 / 3)
        self.assertAlmostEqual(metrics["brightness_ratio"], 2.5 / 3)
        self.assertAlmostEqual(metrics["slice_jump"], 0.5)
        self.assertAlmostEqual(metrics["brightness_std"], np.std([1.0, 0.5, 1.0]))

    def test_shared_volume_noise(self):
        cfg = self.config()
        cfg.validation.shared_volume_noise = True
        owner = self.owner(cfg, "noise")
        owner.device = torch.device("cpu")
        targets = torch.zeros((3, 1, 8, 8))
        first = owner.initial_noise(targets)
        self.assertTrue(all(torch.equal(first[0], first[index]) for index in range(3)))
        self.assertTrue(torch.equal(owner.initial_noise(targets[:2])[0], first[0]))
        owner.volume_noise = None
        self.assertFalse(torch.equal(owner.initial_noise(targets)[0], first[0]))
        cfg.validation.shared_volume_noise = False
        independent = owner.initial_noise(targets)
        self.assertFalse(torch.equal(independent[0], independent[1]))

    def test_new_settings_default_for_older_runs(self):
        from seg2med_train.networks.ddpm.training import with_defaults

        defaults = {"seed": 42, "volume_eval": {"enabled": False, "num_volumes": 4}}
        self.assertEqual(with_defaults({"seed": 42}, defaults), defaults)
        self.assertNotEqual(with_defaults({"seed": 1}, defaults), defaults)

    def test_validation_patient_weighting_and_preview_rng_isolation(self):
        cfg = self.config(True)
        owner = self.owner(cfg, "metrics")
        owner.init_diffusion()
        trainer = DiffusionTrainer(owner)
        owner.model.train()
        before = rng_state()
        metrics = trainer.validate()
        trainer.preview(5)
        after = rng_state()
        self.assertEqual(before["python"], after["python"])
        self.assertTrue(np.array_equal(before["numpy"][1], after["numpy"][1]))
        self.assertTrue(torch.equal(before["torch"], after["torch"]))
        self.assertEqual(metrics["val_samples"], 6)
        self.assertEqual(metrics["val_patients"], 3)
        expected = np.mean([metrics[f"val_modality_{index}_patient_denoising_mse"] for index in range(3)])
        self.assertAlmostEqual(expected, metrics["val_modality_patient_denoising_mse"])

    def test_plateau_scheduler_and_early_stopping_use_validation_checks(self):
        cfg = self.config()
        cfg.train.num_epochs = 8
        cfg.train.earlystopping_patience = 2
        cfg.train.scheduler.patience = 0
        cfg.validation.preview.enabled = False
        owner = self.owner(cfg, "stop")
        owner.init_diffusion()
        trainer = DiffusionTrainer(owner)
        with patch.object(trainer, "validate", return_value={cfg.validation.monitor: 1.0}):
            trainer.run()
        last = load_checkpoint(Path(owner.saved_models_name) / "last.pt")
        best = load_checkpoint(Path(owner.saved_models_name) / "best.pt")
        self.assertEqual(last["epoch"], 3)
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(last["early_stopping"]["bad_checks"], 2)
        self.assertAlmostEqual(last["optimizer"]["param_groups"][0]["lr"], 0.00005)

    def test_resume_with_changed_stopping_policy(self):
        for change in ("delta", "patience"):
            with self.subTest(change=change), patch("seg2med_train.plot_loss_curves.plot_loss_curves"):
                cfg = self.config()
                cfg.train.num_epochs = 8
                cfg.train.earlystopping_delta = 0.1
                cfg.train.earlystopping_patience = 2
                cfg.validation.preview.enabled = False
                owner = self.owner(cfg, change)
                owner.init_diffusion()
                trainer = DiffusionTrainer(owner)
                # The old threshold ignores both small improvements and stops.
                with patch.object(trainer, "validate", side_effect=[
                        {cfg.validation.monitor: value} for value in (1.0, 0.99, 0.98)]):
                    trainer.run()
                path = Path(owner.saved_models_name) / "last.pt"
                saved = load_checkpoint(path)
                self.assertEqual(saved["epoch"], 3)
                self.assertEqual(saved["early_stopping"]["bad_checks"], 2)
                cfg.train.resume = True
                cfg.ckpt_path = str(path)
                cfg.train.num_epochs = 4
                # An unchanged policy must keep the stopped state.
                unchanged = self.owner(cfg, change)
                unchanged.init_diffusion()
                unchanged_trainer = DiffusionTrainer(unchanged)
                self.assertEqual(unchanged_trainer.bad_checks, 2)
                with patch.object(unchanged_trainer, "validate") as validate:
                    unchanged_trainer.run()
                    validate.assert_not_called()
                if change == "delta":
                    cfg.train.earlystopping_delta = 0.001
                else:
                    cfg.train.earlystopping_patience = 4
                resumed = self.owner(cfg, change)
                resumed.init_diffusion()
                with self.assertLogs("seg2med_train.networks.ddpm.training", level="INFO") as messages:
                    resumed_trainer = DiffusionTrainer(resumed)
                self.assertTrue(any("changed early stopping" in message for message in messages.output))
                self.assertEqual(resumed_trainer.bad_checks, 0)
                self.assertEqual(resumed_trainer.best, 0.98)
                self.assertEqual(resumed_trainer.history, saved["history"])
                self.assertEqual(resumed_trainer.lr_scheduler.state_dict(), saved["lr_scheduler"])
                self.assertEqual(resumed.optimizer.param_groups[0]["lr"],
                                 saved["optimizer"]["param_groups"][0]["lr"])
                self.assertEqual(resumed_trainer.global_step, saved["global_step"])
                for name, value in saved["model"].items():
                    self.assertTrue(torch.equal(resumed.model.state_dict()[name], value), name)
                with patch.object(resumed_trainer, "validate", return_value={cfg.validation.monitor: 0.97}):
                    resumed_trainer.run()
                result = load_checkpoint(path)
                self.assertEqual(result["epoch"], 4)
                self.assertEqual(result["history"][:3], saved["history"])
                self.assertEqual(result["early_stopping"]["min_delta"], cfg.train.earlystopping_delta)
                self.assertEqual(result["early_stopping"]["patience"], cfg.train.earlystopping_patience)
                self.assertEqual(result["early_stopping"]["bad_checks"], 0 if change == "delta" else 1)
                with (Path(resumed.log_path) / "epoch_metrics.csv").open(newline="") as stream:
                    self.assertEqual([int(row["epoch"]) for row in csv.DictReader(stream)], [1, 2, 3, 4])

    def test_resume_normalization_checks_only_active_modalities_and_logs_ranges(self):
        cfg = self.config()
        cfg.train.num_epochs = 1
        cfg.validation.preview.enabled = False

        def ct_owner():
            owner = self.owner(cfg, "normalization")
            owner.train_loader = DataLoader(owner.train_loader.dataset[:1], batch_size=1)
            owner.val_loader = DataLoader(owner.val_loader.dataset[:1], batch_size=1)
            owner.init_diffusion()
            return owner

        owner = ct_owner()
        with patch("seg2med_train.plot_loss_curves.plot_loss_curves"):
            trainer = DiffusionTrainer(owner)
            trainer.run()
        checkpoint = Path(owner.saved_models_name) / "last.pt"
        state = load_checkpoint(checkpoint)
        self.assertIn("preprocessing", state)
        metadata_path = Path(owner.log_path) / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        # Simulate a legacy CT run whose unused MR range was different.
        metadata["preprocessing"]["normalization"]["target"]["1"] = [0, 800]
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        cfg.train.resume = True
        cfg.ckpt_path = str(checkpoint)
        resumed = ct_owner()
        legacy = copy.deepcopy(state)
        legacy.pop("preprocessing")
        with patch("seg2med_train.networks.ddpm.training.load_checkpoint", return_value=legacy):
            with self.assertLogs("seg2med_train.networks.ddpm.training", level="INFO") as logs:
                restored = DiffusionTrainer(resumed)
        self.assertEqual(restored.active_modalities, {"0"})
        self.assertTrue(any("legacy checkpoint" in message for message in logs.output))
        self.assertTrue(any("target | modality=1 | unused" in message and "[0, 800]" in message
                            and "[0, 1200]" in message for message in logs.output))
        # A snapshot is checked independently of original run metadata.
        changed_checkpoint = copy.deepcopy(state)
        changed_checkpoint["preprocessing"]["normalization"]["source"]["0"] = [-200, 700]
        with patch("seg2med_train.networks.ddpm.training.load_checkpoint", return_value=changed_checkpoint):
            with self.assertLogs("seg2med_train.networks.ddpm.training", level="INFO") as logs:
                with self.assertRaisesRegex(ValueError, "active modality 0.*source, checkpoint"):
                    DiffusionTrainer(ct_owner())
        self.assertTrue(any("ACTIVE" in message and "[-200, 700]" in message for message in logs.output))
        # Validation-only modalities must also be protected.
        validation_mr = ct_owner()
        validation_mr.val_loader = self.loader()
        with self.assertRaisesRegex(ValueError, "active modality 1.*target"):
            DiffusionTrainer(validation_mr)
        # Changes to geometry remain forbidden, even when normalization is okay.
        metadata["preprocessing"]["dataset"]["rotate"] = True
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dataset/preprocessing settings must match"):
            DiffusionTrainer(ct_owner())

    def test_active_modalities_raw_metadata_and_unknown_fallback(self):
        from types import SimpleNamespace
        from monai.data import Dataset
        trainer = object.__new__(DiffusionTrainer)
        raw = [{"modality": 0}, {"modality": 1}]
        trainer.owner = SimpleNamespace(
            train_loader=SimpleNamespace(dataset=Dataset(raw)),
            val_loader=SimpleNamespace(dataset=torch.utils.data.Subset(raw, [1])))
        self.assertEqual(trainer._active_modalities(), {"0", "1"})
        trainer.owner.train_loader.dataset = object()
        self.assertIsNone(trainer._active_modalities())

    def test_logs_each_training_step_and_epoch_summary_for_both_models(self):
        for multimodal in (False, True):
            with self.subTest(multimodal=multimodal):
                cfg = self.config(multimodal)
                cfg.train.num_epochs = 1
                cfg.validation.preview.enabled = False
                owner = self.owner(cfg, f"logging_{multimodal}")
                with self.assertLogs("seg2med_train.networks.ddpm.training", level="INFO") as captured:
                    owner.train()
                steps = [record.getMessage() for record in captured.records if record.getMessage().startswith("Train |")]
                self.assertEqual(len(steps), 3)
                for index, message in enumerate(steps, start=1):
                    self.assertIn("epoch=1/1", message)
                    self.assertIn(f"step={index} | global_step={index}", message)
                    self.assertIn("loss=", message)
                    self.assertIn("denoising_mse=", message)
                    self.assertIn("lr=2.000e-04", message)
                self.assertTrue(any("Epoch summary |" in record.getMessage() for record in captured.records))


if __name__ == "__main__":
    unittest.main()
