"""Medical intensity restoration must not change normalized model arrays."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from seg2med_train.export_intensity import MedicalIntensityExport
from seg2med_train.slice_export import InferenceSliceWriter


class DdpmDicomIntensityTests(unittest.TestCase):
    def test_current_ct_range_and_no_mutation_or_extra_clipping(self):
        convert = MedicalIntensityExport()
        normalized = np.array([0, 0.5, 1, 1.1], dtype=np.float32)
        original = normalized.copy()
        for label in ("input", "target", "output"):
            restored, units = convert(normalized, label, 0)
            np.testing.assert_allclose(restored, [-300, 200, 700, 800], atol=0.001)
            self.assertEqual(units, "HU")
        np.testing.assert_array_equal(normalized, original)

    def test_checkpoint_output_range_and_current_target_source_ranges(self):
        convert = MedicalIntensityExport(preprocessing={
            "normalization": {"target": {"1": [10, 2010]}}})
        values = np.array([0, 0.5, 1])
        np.testing.assert_allclose(convert(values, "output", 1)[0], [10, 1010, 2010])
        np.testing.assert_allclose(convert(values, "target", 1)[0], [0, 600, 1200])
        np.testing.assert_allclose(convert(values, "input", 1)[0], [0, 4.5, 9])
        self.assertEqual(convert(values, "output", 1)[1], "US")

    def test_legacy_checkpoint_warns_and_falls_back(self):
        with self.assertLogs("seg2med_train.export_intensity", level="WARNING"):
            convert = MedicalIntensityExport(preprocessing={})
        self.assertEqual(convert(np.array([0.5]), "output", 0)[0][0], 200)

    def test_semantic_input_keeps_labels(self):
        convert = MedicalIntensityExport(normalize_source=False)
        labels = np.array([0, 1, 50])
        values, units = convert(labels, "input", 0)
        np.testing.assert_array_equal(values, labels)
        self.assertEqual(units, "US")

    def test_invalid_or_missing_checkpoint_range_is_rejected(self):
        for ranges in ({"0": [1, 1]}, {"1": [0, 10]}):
            convert = MedicalIntensityExport(preprocessing={"normalization": {"target": ranges}})
            with self.assertRaises(ValueError):
                convert(np.zeros(1), "output", 0)

    def test_streaming_dicom_round_trip_and_npz_stays_normalized(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            writer = InferenceSliceWriter(
                patient_id="P1", modality=0, epoch=1, step=0,
                img_folder=root / "img", save_dicom=True, save_npz=True,
                dicom_intensity=MedicalIntensityExport())
            normalized = np.array([[[[0, 0.5, 1], [0.25, 0.75, 0.1]]]], dtype=np.float32)
            for _ in range(2):
                writer.save_batch(normalized, normalized, normalized)
            for label in ("input", "target", "output"):
                files = sorted((root / "dicom" / label).rglob("*.dcm"))
                self.assertEqual(len(files), 2)
                for path in files:
                    ds = pydicom.dcmread(path)
                    self.assertEqual(ds.RescaleType, "HU")
                    np.testing.assert_allclose(
                        apply_modality_lut(ds.pixel_array, ds),
                        (normalized[0, 0] * 1000 - 300).T, atol=0.02)
                npz = next((root / "img" / label).glob("*.npz"))
                with np.load(npz) as saved:
                    np.testing.assert_array_equal(saved[label], normalized[0, 0])

    def test_volume_dicom_restored_while_tiff_stays_normalized(self):
        import torch
        import tifffile
        from seg2med_train.networks.results_eval import output_sample_results

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tensor = torch.tensor([0.0, 0.5, 1.0, 0.25]).reshape(1, 1, 2, 2, 1)
            output_sample_results(tensor, tensor, ["P1"], tensor, 0, 1, 0,
                                  root / "img", save_dicom=True,
                                  dicom_intensity=MedicalIntensityExport())
            expected = (tensor[0, 0, :, :, 0].numpy() * 1000 - 300).T
            for label in ("input", "output", "target"):
                ds = pydicom.dcmread(next((root / "dicom" / label).rglob("*.dcm")))
                np.testing.assert_allclose(apply_modality_lut(ds.pixel_array, ds), expected, atol=0.02)
                tiff = next((root / "img").glob(f"*_{label}.tiff"))
                np.testing.assert_array_equal(tifffile.imread(tiff), tensor[0, 0].numpy().transpose(2, 1, 0))

    def test_both_runners_keep_checkpoint_ranges_and_share_export_path(self):
        import torch
        from types import SimpleNamespace
        from unittest.mock import patch
        from seg2med_train.networks.ddpm.seg2med_ddpm import DiffusionModel, DiffusionModel_multimodal

        preprocessing = {"normalization": {"target": {"0": [-1000, 2000]}}}
        self.assertIs(DiffusionModel.val, DiffusionModel_multimodal.val)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            path.touch()
            for runner in (DiffusionModel, DiffusionModel_multimodal):
                owner = runner.__new__(runner)
                owner.config = SimpleNamespace(ckpt_path=str(path), mode="test",
                                               train=SimpleNamespace(resume=False))
                owner.device = "cpu"
                owner.model = torch.nn.Linear(1, 1)
                checkpoint = {"model": owner.model.state_dict(), "preprocessing": preprocessing}
                with patch("seg2med_train.networks.ddpm.training.load_checkpoint", return_value=checkpoint):
                    owner.init_model_ckpt()
                self.assertEqual(owner.export_preprocessing, preprocessing)
                convert = MedicalIntensityExport(preprocessing=owner.export_preprocessing)
                np.testing.assert_allclose(convert(np.array([0, 1]), "output", 0)[0], [-1000, 2000])


if __name__ == "__main__":
    unittest.main()
