"""Old/new per-file dictionaries and model convention selection are independent."""

from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd

from seg2med_physiosynth.label_schemes import (
    DEFAULT_MR_LEGACY_PARAMETERS, embedded_labels, source_anatomy, mr_output_parameters,
)
from seg2med_physiosynth.simulator import DEFAULT_CT_PARAMETERS, DEFAULT_MR_PARAMETERS
from seg2med_workflow.single_case import prepare_case


def with_labels(array, table):
    """Build a realistic full dictionary, independent of labels present in this crop."""
    image = nib.Nifti1Image(np.asarray(array, dtype=np.int16), np.eye(4))
    root = ET.Element("CaretExtension")
    labels = ET.SubElement(root, "LabelTable")
    for row in pd.read_csv(table).to_dict("records"):
        key = int(row["Order Number"])
        if 0 < key < 200:
            ET.SubElement(labels, "Label", Key=str(key)).text = row["Organ Name"]
    image.header.extensions.append(nib.nifti1.Nifti1Extension(0, ET.tostring(root)))
    return image


class PhysioSynthLabelSchemeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_legacy_table_matches_actual_56_label_dictionary(self):
        table = pd.read_csv(DEFAULT_MR_LEGACY_PARAMETERS)
        labels = dict(zip(table["Order Number"], table["Organ Name"]))
        self.assertEqual(set(labels), {0, *range(1, 57), 201, 202, 203})
        self.assertEqual({key: labels[key] for key in [32, 33, 39, 41, 47, 56]}, {
            32: "fibula", 33: "tibia", 39: "gluteus_maximus_right",
            41: "gluteus_medius_right", 47: "iliopsoas_right", 56: "brain"})
        self.assertNotIn("kidney_cyst_right", labels.values())
        current = pd.read_csv(DEFAULT_MR_PARAMETERS).set_index("Organ Name")
        legacy = table.set_index("Organ Name")
        common = sorted(set(current.index) & set(legacy.index))
        np.testing.assert_array_equal(current.loc[common, ["T1", "T2", "Rho"]],
                                      legacy.loc[common, ["T1", "T2", "Rho"]])

    def test_auto_detects_both_dictionaries_on_a_crop_containing_only_label_1(self):
        for path, expected in ((DEFAULT_MR_PARAMETERS, "current"), (DEFAULT_MR_LEGACY_PARAMETERS, "legacy")):
            image = with_labels(np.ones((2, 2, 2)), path)
            source, scheme = source_anatomy(image, "mr")
            self.assertEqual(scheme, expected)
            self.assertEqual(source.loc[source["Order Number"] == 39, "Organ Name"].iloc[0],
                             "hip_right" if expected == "current" else "gluteus_maximus_right")

    def test_missing_metadata_requires_explicit_input_scheme(self):
        image = nib.Nifti1Image(np.full((2, 2, 2), 49, dtype=np.int16), np.eye(4))
        with self.assertRaisesRegex(ValueError, "no embedded label dictionary"):
            source_anatomy(image, "mr")
        self.assertEqual(source_anatomy(image, "mr", "legacy")[1], "legacy")
        self.assertEqual(source_anatomy(image, "mr", "current")[1], "current")

    def test_explicit_scheme_cannot_override_contradictory_metadata(self):
        image = with_labels(np.ones((2, 2, 2)), DEFAULT_MR_LEGACY_PARAMETERS)
        with self.assertRaisesRegex(ValueError, "contradicts"):
            source_anatomy(image, "mr", "current")

    def test_ambiguous_metadata_requires_explicit_mr_output_choice(self):
        image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4))
        image.header.extensions.append(nib.nifti1.Nifti1Extension(0,
            b'<CaretExtension><LabelTable><Label Key="1">spleen</Label></LabelTable></CaretExtension>'))
        _, scheme = source_anatomy(image, "mr")
        self.assertEqual(scheme, "custom")
        with self.assertRaisesRegex(ValueError, "custom/ambiguous"):
            mr_output_parameters("auto", scheme, "mr")
        self.assertEqual(mr_output_parameters("legacy", scheme, "mr")[0], DEFAULT_MR_LEGACY_PARAMETERS)

    def test_model_scheme_independent_of_input_scheme(self):
        self.assertEqual(mr_output_parameters("auto", "legacy", "mr")[1], "legacy")
        self.assertEqual(mr_output_parameters("auto", "current", "ct")[1], "current")
        self.assertEqual(mr_output_parameters("legacy", "current", "ct")[1], "legacy")
        self.assertEqual(mr_output_parameters("current", "legacy", "mr")[1], "current")

    def case(self, name, kind, table, ids):
        folder = self.root / name
        folder.mkdir()
        image = folder / f"{kind}.nii.gz"
        shape = (4, 3, 2)
        nib.save(nib.Nifti1Image(np.full(shape, 200, dtype=np.int16), np.eye(4)), image)
        values = np.tile(np.array(ids)[None, :, None], (4, 1, 2))
        nib.save(with_labels(values, table), folder / f"{kind}_seg.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.int16), np.eye(4)), folder / f"{kind}_seg_tissue.nii.gz")
        return {"id": name, "dataset": "test", "img": str(image),
                "input_modality_name": "ct" if kind == "ct" else "mr_t2_space",
                "prior": {"min_contour_area": 0}}

    def test_old_and_new_inputs_produce_identical_ct_semantics_and_signals(self):
        results = []
        for name, table, ids in (("old", DEFAULT_MR_LEGACY_PARAMETERS, [39, 41, 47]),
                                 ("new", DEFAULT_MR_PARAMETERS, [41, 43, 49])):
            config = self.case(name, "mr", table, ids)
            result = prepare_case({**config, "target_modality_name": "ct"})
            semantic = nib.load(result["prior_semantic"]).get_fdata()
            np.testing.assert_array_equal(semantic[0, :, 0], [82, 84, 90])
            results.append(nib.load(result["prior"]).get_fdata())
        np.testing.assert_array_equal(*results)

    def test_native_legacy_default_and_explicit_current_output(self):
        config = self.case("old", "mr", DEFAULT_MR_LEGACY_PARAMETERS, [39, 41, 47])
        old = prepare_case(config)
        np.testing.assert_array_equal(nib.load(old["prior_semantic"]).get_fdata()[0, :, 0], [40, 42, 48])
        new = prepare_case({**config, "prior_label_scheme": "current", "overwrite_prior": True})
        np.testing.assert_array_equal(nib.load(new["prior_semantic"]).get_fdata()[0, :, 0], [42, 44, 50])

    def test_ct_and_current_mr_generate_legacy_model_semantics(self):
        for name, kind, table, ids in (("ct", "ct", DEFAULT_CT_PARAMETERS, [81, 83, 89]),
                                       ("mr", "mr", DEFAULT_MR_PARAMETERS, [41, 43, 49])):
            config = self.case(name, kind, table, ids)
            result = prepare_case({**config, "target_modality_name": "mr_t2_space", "prior_label_scheme": "legacy"})
            np.testing.assert_array_equal(nib.load(result["prior_semantic"]).get_fdata()[0, :, 0], [40, 42, 48])

    def test_app_upload_uses_embedded_legacy_names_and_source_ids(self):
        from seg2med_app.anatomy import load_upload, cleanup_workspace, build_scene
        from seg2med_app.gradio_app import simulate, simulate_upload_volume
        config = self.case("app", "mr", DEFAULT_MR_LEGACY_PARAMETERS, [39, 41, 47])
        image = Path(config["img"])
        state = load_upload(image.with_name("mr_seg.nii.gz"), image.with_name("mr_seg_tissue.nii.gz"),
                            image, kind="mr", min_area=0)
        self.addCleanup(cleanup_workspace, state)
        self.assertEqual(state.label_scheme, "legacy")
        self.assertEqual(state.sources["gluteus_maximus_right"]["label"], 39)
        payload = build_scene(state, 0, [])
        simulate(state, payload, 2)
        self.assertTrue(np.isfinite(state.prior).all())
        simulate_upload_volume(state, payload, 2, prior_label_scheme="legacy")
        volume = nib.load(next(path for path in state.volume_exports if "prior_image" in path)).get_fdata()
        np.testing.assert_allclose(volume[:, :, 0], state.prior)

    def test_catalog_mixed_input_conventions_share_legacy_output(self):
        import yaml
        from seg2med_physiosynth.priors import create_priors_from_config

        rows = []
        for name, table, ids in (("old", DEFAULT_MR_LEGACY_PARAMETERS, [39, 41, 47]),
                                 ("new", DEFAULT_MR_PARAMETERS, [41, 43, 49])):
            case = self.case(name, "mr", table, ids)
            image = Path(case["img"])
            rows.append({"id": name, "img": str(image), "seg": str(image.with_name("mr_seg.nii.gz")),
                         "seg_tissue": str(image.with_name("mr_seg_tissue.nii.gz")), "modality": 2})
        catalog = self.root / "catalog.csv"
        pd.DataFrame(rows).to_csv(catalog, index=False)
        config = self.root / "priors.yaml"
        config.write_text(yaml.safe_dump({"data_root": str(self.root), "min_contour_area": 0,
            "prior_label_scheme": "legacy", "segmentation_label_scheme": "auto",
            "splits": {"test": {"catalog": str(catalog), "output_dir": "out", "output_csv": "out.csv"}}}), encoding="utf-8")
        result = create_priors_from_config(config)["test"]
        for relative in result.prior_semantic:
            np.testing.assert_array_equal(nib.load(self.root / relative).get_fdata()[0, :, 0], [40, 42, 48])

    def test_custom_legacy_parameters_do_not_change_source_label_meaning(self):
        from seg2med_physiosynth import formulas

        config = self.case("custom", "mr", DEFAULT_MR_PARAMETERS, [41, 43, 49])
        table = pd.read_csv(DEFAULT_MR_LEGACY_PARAMETERS)
        table.loc[table["Organ Name"] == "gluteus_maximus_right", "Rho"] = 123
        path = self.root / "custom_legacy.csv"
        table.to_csv(path, index=False)
        result = prepare_case({**config, "prior_label_scheme": "legacy", "parameters": {"mr_legacy": str(path)}})
        np.testing.assert_array_equal(nib.load(result["prior_semantic"]).get_fdata()[0, :, 0], [40, 42, 48])
        self.assertAlmostEqual(nib.load(result["prior"]).get_fdata()[0, 0, 0],
                               formulas.t2_space(1232.9, 37.2, 123), places=5)

    def test_unsafe_or_conflicting_embedded_dictionary_is_rejected(self):
        for content in (b'<!DOCTYPE x><CaretExtension><LabelTable/></CaretExtension>',
                        b'<CaretExtension><LabelTable><Label Key="1">liver</Label><Label Key="1">spleen</Label></LabelTable></CaretExtension>'):
            image = nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.int16), np.eye(4))
            image.header.extensions.append(nib.nifti1.Nifti1Extension(0, content))
            with self.assertRaises(ValueError):
                embedded_labels(image)


if __name__ == "__main__":
    unittest.main()
