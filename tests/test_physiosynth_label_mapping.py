"""Independent TotalSegmentator label-contract and conversion regressions."""

import unittest

import numpy as np
import pandas as pd

from seg2med_physiosynth.anatomy import convert_segmentation_mask
from seg2med_physiosynth.simulator import (
    DEFAULT_CT_PARAMETERS, DEFAULT_MR_PARAMETERS, simulate_prior,
)

# Official total_mr class map, verified against TotalSegmentator 2.13.0.
# Keep independent of params_mr.csv so an internally consistent wrong table fails.
MR_NAMES = """spleen kidney_right kidney_left gallbladder liver stomach pancreas
adrenal_gland_right adrenal_gland_left lung_left lung_right esophagus small_bowel
duodenum colon urinary_bladder prostate sacrum vertebrae intervertebral_discs
spinal_cord heart aorta inferior_vena_cava portal_vein_and_splenic_vein
iliac_artery_left iliac_artery_right iliac_vena_left iliac_vena_right
humerus_left humerus_right scapula_left scapula_right clavicula_left clavicula_right
femur_left femur_right hip_left hip_right gluteus_maximus_left gluteus_maximus_right
gluteus_medius_left gluteus_medius_right gluteus_minimus_left gluteus_minimus_right
autochthon_left autochthon_right iliopsoas_left iliopsoas_right brain""".split()
EXPECTED_MR = dict(enumerate(MR_NAMES, 1))


class PhysioSynthLabelMappingTests(unittest.TestCase):
    def test_default_mr_table_matches_independent_official_label_contract(self):
        table = pd.read_csv(DEFAULT_MR_PARAMETERS)
        organs = table[(table["Order Number"] > 0) & (table["Order Number"] < 200)]
        self.assertEqual(dict(zip(organs["Order Number"], organs["Organ Name"])), EXPECTED_MR)
        self.assertEqual(set(table["Order Number"]), {0, *range(1, 51), 201, 202, 203})
        self.assertFalse(table["Order Number"].duplicated().any())

    def test_installed_totalsegmentator_matches_parameter_tables_when_available(self):
        try:
            from totalsegmentator.map_to_binary import class_map
        except ImportError:
            self.skipTest("Optional TotalSegmentator backend is not installed.")
        for task, path in (("total_mr", DEFAULT_MR_PARAMETERS), ("total", DEFAULT_CT_PARAMETERS)):
            with self.subTest(task=task):
                frame = pd.read_csv(path)
                organs = frame[(frame["Order Number"] > 0) & (frame["Order Number"] < 200)]
                self.assertEqual(dict(zip(organs["Order Number"], organs["Organ Name"])), class_map[task])

    def test_reported_mr_to_ct_mismatches_are_corrected(self):
        labels = np.array([49, 37, 41, 40], dtype=np.int16)
        before = labels.copy()
        converted = convert_segmentation_mask(labels, DEFAULT_MR_PARAMETERS, DEFAULT_CT_PARAMETERS)
        np.testing.assert_array_equal(converted + 1, [90, 77, 82, 81])
        np.testing.assert_array_equal(labels, before)
        reverse = convert_segmentation_mask(converted, DEFAULT_CT_PARAMETERS, DEFAULT_MR_PARAMETERS)
        np.testing.assert_array_equal(reverse, labels)

    def test_scripted_prior_uses_correct_semantic_ids_and_hu(self):
        labels = np.tile([49, 37, 41, 40], (4, 1)).astype(np.int16)[..., None]
        prior, semantic, contour = simulate_prior(
            np.full(labels.shape, 200), labels, np.zeros_like(labels), 0, input_modality=1,
            ct_parameters=DEFAULT_CT_PARAMETERS, mr_parameters=DEFAULT_MR_PARAMETERS, min_contour_area=0)
        np.testing.assert_array_equal(contour, 1)
        np.testing.assert_array_equal(semantic[0, :, 0], [90, 77, 82, 81])
        np.testing.assert_array_equal(prior[0, :, 0], [60, 700, 60, 60])

    def test_fuzzy_and_unrelated_manual_matches_are_not_used(self):
        source = pd.DataFrame({"Order Number": [1, 2, 3],
                               "Organ Name": ["gluteus_maximus_righ", "intervertebral_discs", "quadriceps_femoris_right"]})
        target = pd.DataFrame({"Order Number": [81, 79],
                               "Organ Name": ["gluteus_maximus_right", "spinal_cord"]})
        with self.assertLogs("seg2med_physiosynth.anatomy", level="WARNING") as logs:
            converted = convert_segmentation_mask(np.array([1, 2, 3]), source, target)
        np.testing.assert_array_equal(converted, 0)
        self.assertEqual(len(logs.output), 3)

    def test_explicit_ct_to_mr_aggregations(self):
        ct = pd.read_csv(DEFAULT_CT_PARAMETERS)
        lookup = dict(zip(ct["Organ Name"], ct["Order Number"]))
        names = ["lung_upper_lobe_left", "lung_lower_lobe_left", "lung_middle_lobe_right",
                 "vertebrae_C1", "vertebrae_T12", "vertebrae_L5", "vertebrae_S1"]
        labels = np.array([lookup[name] for name in names])
        np.testing.assert_array_equal(convert_segmentation_mask(labels, DEFAULT_CT_PARAMETERS, DEFAULT_MR_PARAMETERS),
                                      [10, 10, 11, 19, 19, 19, 19])

    def test_coarse_mr_regions_are_not_arbitrarily_split_into_ct_subregions(self):
        with self.assertLogs("seg2med_physiosynth.anatomy", level="WARNING"):
            converted = convert_segmentation_mask(np.array([10, 11, 19, 20]), DEFAULT_MR_PARAMETERS, DEFAULT_CT_PARAMETERS)
        np.testing.assert_array_equal(converted, 0)

    def test_ambiguous_parameter_tables_are_rejected(self):
        source = pd.DataFrame({"Order Number": [1], "Organ Name": ["liver"]})
        for target in (pd.DataFrame({"Order Number": [5, 5], "Organ Name": ["liver", "spleen"]}),
                       pd.DataFrame({"Order Number": [5, 6], "Organ Name": ["liver", "liver"]})):
            with self.subTest(target=target), self.assertRaises(ValueError):
                convert_segmentation_mask(np.ones(1), source, target)


if __name__ == "__main__":
    unittest.main()
