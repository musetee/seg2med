"""DICOM exports live beside img, not inside it."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
import pydicom

from seg2med_train.slice_export import InferenceSliceWriter
from seg2med_train.networks.results_eval import output_sample_results


class DicomOutputPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.epoch = self.root / "saved_outputs/epoch_100"
        self.img = self.epoch / "img"

    def test_streamed_batches_use_sibling_directory_and_keep_slice_numbering(self):
        writer = InferenceSliceWriter(patient_id="P1", modality=0, epoch=100, step=0,
                                      img_folder=self.img, save_dicom=True)
        for depth in (2, 1):
            array = np.zeros((depth, 1, 2, 3), dtype=np.float32)
            writer.save_batch(array, array, array)
        for label in ("input", "output", "target"):
            files = sorted((self.epoch / "dicom" / label).rglob("*.dcm"))
            self.assertEqual(len(files), 3)
            series = [pydicom.dcmread(path) for path in files]
            self.assertEqual([int(item.InstanceNumber) for item in series], [1, 2, 3])
            self.assertEqual(len({item.SeriesInstanceUID for item in series}), 1)
        self.assertFalse((self.img / "dicom").exists())

    def test_extra_label_series_keeps_raw_mask_values(self):
        # Intensity export rescales model-space images; label maps must stay 0/1.
        scale = lambda array, label, modality: (np.asarray(array) * 1000.0, "US")
        writer = InferenceSliceWriter(patient_id="P1", modality=1, epoch=100, step=0, img_folder=self.img,
                                      save_dicom=True, dicom_intensity=scale, extra_labels=("lesion_mask",))
        image = np.full((2, 1, 2, 3), 0.5, dtype=np.float32)
        mask = np.zeros((2, 1, 2, 3), dtype=np.float32)
        mask[0, 0, 1, 2] = 1
        writer.save_batch(image, image, image, extras={"lesion_mask": mask})
        series = [pydicom.dcmread(path) for path in sorted((self.epoch / "dicom" / "lesion_mask").rglob("*.dcm"))]
        decoded = np.stack([item.pixel_array * float(item.RescaleSlope) + float(item.RescaleIntercept)
                            for item in series])
        np.testing.assert_array_equal(decoded, mask[:, 0].transpose(0, 2, 1))
        self.assertEqual({int(item.SeriesNumber) for item in series}, {4})
        output = pydicom.dcmread(next((self.epoch / "dicom" / "output").rglob("*.dcm")))
        self.assertEqual(series[0].StudyInstanceUID, output.StudyInstanceUID)
        self.assertNotEqual(series[0].SeriesInstanceUID, output.SeriesInstanceUID)
        # TLA+ mutation counterexample (formal/a006_slice_writer): a batch without
        # the extra series must be rejected before any series is written.
        with self.assertRaisesRegex(ValueError, "lesion_mask"):
            writer.save_batch(image, image, image)
        for label in ("input", "output", "target", "lesion_mask"):
            self.assertEqual(len(list((self.epoch / "dicom" / label).rglob("*.dcm"))), 2, label)
        writer.save_batch(image, image, image, extras={"lesion_mask": mask})
        numbers = sorted(int(pydicom.dcmread(path).InstanceNumber)
                         for path in (self.epoch / "dicom" / "lesion_mask").rglob("*.dcm"))
        self.assertEqual(numbers, [1, 2, 3, 4])

    def export_volume(self, **options):
        tensor = torch.zeros((1, 1, 2, 3, 4))
        with contextlib.redirect_stdout(io.StringIO()):
            output_sample_results(tensor, tensor, ["P1"], tensor, 0, 100, 0,
                                  self.img, **options)

    def test_volume_export_uses_sibling_directory_and_keeps_tiffs_in_img(self):
        self.export_volume(save_dicom=True)
        self.assertEqual(len(list(self.img.glob("*.tiff"))), 3)
        for label in ("input", "output", "target"):
            self.assertEqual(len(list((self.epoch / "dicom" / label).rglob("*.dcm"))), 4)
        self.assertFalse((self.img / "dicom").exists())

    def test_explicit_dicom_directory_override_is_preserved(self):
        custom = self.root / "custom_dicom"
        self.export_volume(save_dicom=True, dicom_folder=custom)
        self.assertEqual(len(list(custom.rglob("*.dcm"))), 12)
        self.assertFalse((self.epoch / "dicom").exists())

    def test_disabled_dicom_does_not_create_a_directory(self):
        self.export_volume()
        self.assertFalse((self.epoch / "dicom").exists())


if __name__ == "__main__":
    unittest.main()
