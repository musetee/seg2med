"""Extension packages can register models, datasets, config defaults and manual-case path keys
without editing seg2med (CPU only, no PyTorch needed)."""

import sys
import tempfile
import types
import unittest
from pathlib import Path

from yacs.config import CfgNode as CN

from seg2med_train import dataloader_registry, model_selection
from seg2med_train.utils import my_configs_yacs


class ExtensionRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.models = dict(model_selection.MODEL_RUNNERS)
        self.datasets = dict(dataloader_registry.DATASET_REGISTRY)
        self.replaced = set(dataloader_registry._REPLACED)
        self.hooks = list(my_configs_yacs.CONFIG_DEFAULT_HOOKS)
        self.case_keys = list(my_configs_yacs.MANUAL_CASE_PATH_KEYS)
        module = types.ModuleType("ext_test_runner")
        module.Runner = type("Runner", (), {})
        sys.modules["ext_test_runner"] = module
        self.addCleanup(sys.modules.pop, "ext_test_runner", None)

    def tearDown(self):
        model_selection.MODEL_RUNNERS.clear()
        model_selection.MODEL_RUNNERS.update(self.models)
        dataloader_registry.DATASET_REGISTRY.clear()
        dataloader_registry.DATASET_REGISTRY.update(self.datasets)
        dataloader_registry._REPLACED.clear()
        dataloader_registry._REPLACED.update(self.replaced)
        my_configs_yacs.CONFIG_DEFAULT_HOOKS[:] = self.hooks
        my_configs_yacs.MANUAL_CASE_PATH_KEYS[:] = self.case_keys

    def test_registered_model_is_listed_and_resolved(self):
        model_selection.register_model("ext_model", "ext_test_runner", "Runner")
        self.assertIn("ext_model", model_selection.available_model_names())
        self.assertIs(model_selection.get_model_runner("ext_model"), sys.modules["ext_test_runner"].Runner)

    def test_model_registration_is_idempotent_but_rejects_conflicts(self):
        model_selection.register_model("ext_model", "ext_test_runner", "Runner")
        model_selection.register_model("ext_model", "ext_test_runner", "Runner")
        with self.assertRaises(ValueError):
            model_selection.register_model("ext_model", "ext_test_runner", "Other")
        with self.assertRaises(ValueError):
            model_selection.register_model("ddpm2d_seg2med", "ext_test_runner", "Runner")

    def test_invalid_model_registrations_are_rejected_without_side_effects(self):
        before = model_selection.available_model_names()
        for name, module, cls in ((None, "ext_test_runner", "Runner"), ("", "ext_test_runner", "Runner"),
                                  (" ext", "ext_test_runner", "Runner"), (3, "ext_test_runner", "Runner"),
                                  ("ext_model", "", "Runner"), ("ext_model", "ext test", "Runner"),
                                  ("ext_model", "ext_test_runner", None), ("ext_model", "ext_test_runner", "Run ner")):
            with self.subTest(name=name, module=module, cls=cls), self.assertRaises((TypeError, ValueError)):
                model_selection.register_model(name, module, cls)
        self.assertEqual(model_selection.available_model_names(), before)

    def test_invalid_dataset_names_are_rejected(self):
        before = dataloader_registry.available_dataset_names()
        for name in (None, "", " ext", 3):
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                dataloader_registry.register_dataset(name)
        self.assertEqual(dataloader_registry.available_dataset_names(), before)

    def test_conflicting_dataset_registration_is_rejected_unless_replace(self):
        @dataloader_registry.register_dataset("ext_dataset")
        def first(opt, paths):
            return "first"

        def second(opt, paths):
            return "second"

        with self.assertRaises(ValueError):
            dataloader_registry.register_dataset("ext_dataset")(second)
        self.assertIs(dataloader_registry.get_dataset_loader("ext_dataset"), first)
        with self.assertRaises(ValueError):
            dataloader_registry.register_dataset("image_prior_csv")(second)
        dataloader_registry.register_dataset("ext_dataset", replace=True)(second)
        self.assertIs(dataloader_registry.get_dataset_loader("ext_dataset"), second)

    def test_explicit_replacement_of_a_core_loader_survives_the_core_import(self):
        def mine(opt, paths):
            return "mine"

        dataloader_registry.register_dataset("image_prior_csv", replace=True)(mine)
        core = types.FunctionType(mine.__code__, mine.__globals__, "load_image_prior_csv")
        core.__module__, core.__qualname__ = "seg2med_train.Dataloader", "load_image_prior_csv"
        dataloader_registry.register_dataset("image_prior_csv")(core)  # what importing Dataloader.py does
        self.assertIs(dataloader_registry.get_dataset_loader("image_prior_csv"), mine)
        other = types.FunctionType(mine.__code__, mine.__globals__, "other")
        other.__module__, other.__qualname__ = "someone_else", "other"
        with self.assertRaises(ValueError):
            dataloader_registry.register_dataset("image_prior_csv")(other)

    def test_dataset_reregistration_of_the_same_function_is_idempotent(self):
        def load(opt, paths):
            return None

        dataloader_registry.register_dataset("ext_dataset")(load)
        dataloader_registry.register_dataset("ext_dataset")(load)
        # a module reload creates a new function object with the same module and qualified name
        reloaded = types.FunctionType(load.__code__, load.__globals__, load.__name__)
        reloaded.__qualname__, reloaded.__module__ = load.__qualname__, load.__module__
        dataloader_registry.register_dataset("ext_dataset")(reloaded)
        self.assertIs(dataloader_registry.get_dataset_loader("ext_dataset"), reloaded)

    def test_externally_registered_dataset_is_listed(self):
        @dataloader_registry.register_dataset("ext_dataset")
        def load(opt, paths):
            return None

        self.assertIn("ext_dataset", dataloader_registry.available_dataset_names())
        self.assertIs(dataloader_registry.get_dataset_loader("ext_dataset"), load)
        self.assertIn("image_prior_csv", dataloader_registry.available_dataset_names())

    def test_config_defaults_hook_adds_mergeable_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "run.yaml")
            path.write_text("ext_section:\n  scale: 5\n", encoding="utf-8")
            with self.assertRaises(KeyError):
                my_configs_yacs.init_cfg(str(path))

            @my_configs_yacs.register_config_defaults
            def add(cfg):
                cfg.ext_section = CN()
                cfg.ext_section.scale = 1

            my_configs_yacs.register_config_defaults(add)
            self.assertEqual(my_configs_yacs.CONFIG_DEFAULT_HOOKS.count(add), 1)
            self.assertEqual(my_configs_yacs.init_cfg().ext_section.scale, 1)
            self.assertEqual(my_configs_yacs.init_cfg(str(path)).ext_section.scale, 5)

    def test_registered_manual_case_key_is_resolved_relative_to_yaml(self):
        my_configs_yacs.register_manual_case_path_key("ext_mask")
        my_configs_yacs.register_manual_case_path_key("ext_mask")
        self.assertEqual(my_configs_yacs.MANUAL_CASE_PATH_KEYS.count("ext_mask"), 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "infer.yaml")
            path.write_text("case:\n  - id: a\n    prior: p.nii.gz\n    ext_mask: m.nii.gz\n", encoding="utf-8")
            case = my_configs_yacs.init_cfg(str(path)).dataset.case[0]
        self.assertEqual(Path(case["ext_mask"]), Path(tmp).resolve() / "m.nii.gz")
        self.assertEqual(Path(case["prior"]), Path(tmp).resolve() / "p.nii.gz")


if __name__ == "__main__":
    unittest.main()
