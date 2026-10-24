"""Loss plotting works with sparse validation and previously saved CSVs."""

import csv
import tempfile
import unittest
from pathlib import Path

from seg2med_train.plot_loss_curves import _prepare_history, _series, main, plot_loss_curves


class LossCurveTests(unittest.TestCase):
    def test_csv_cli_and_sparse_validation(self):
        rows = [
            {"epoch": 1, "train_loss": 0.5, "monitor": "val_patient_denoising_mse"},
            {"epoch": 2, "train_loss": 0.3, "monitor": "val_patient_denoising_mse",
             "val_patient_denoising_mse": 0.4},
            {"epoch": 3, "train_loss": 0.2, "monitor": "val_patient_denoising_mse"},
        ]
        prepared, _, _, monitor = _prepare_history(rows)
        self.assertEqual(monitor, "val_patient_denoising_mse")
        self.assertEqual(_series(prepared, monitor), [(2, 0.4)])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "epoch_metrics.csv"
            with source.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[1]))
                writer.writeheader()
                writer.writerows(rows)
            original = source.read_bytes()
            self.assertEqual(main(["--csv", str(source), "--dpi", "40"]), 0)
            for name in ("training_loss_curve.png", "validation_loss_curve.png", "loss_curves.png"):
                self.assertEqual((directory / name).read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(source.read_bytes(), original)

    def test_training_only_and_explicit_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_loss_curves([{"epoch": 1, "train_loss": 0.5}], Path(temporary) / "plots", dpi=40)
            self.assertTrue(all(path.is_file() for path in paths.values()))

    def test_validation_metric_selection_and_nonfinite_values(self):
        rows = [{"epoch": "4", "train_loss": "0.3", "val_denoising_mse": "nan",
                 "val_patient_denoising_mse": "0.2", "preview_generated_image_mse": "999"}]
        prepared, columns, _, selected = _prepare_history(rows)
        self.assertEqual(selected, "val_patient_denoising_mse")
        self.assertNotIn("preview_generated_image_mse", columns)
        self.assertEqual(_series(prepared, "val_denoising_mse"), [])
        self.assertEqual(_prepare_history(rows, "val_denoising_mse")[3], "val_denoising_mse")
        with self.assertRaises(ValueError):
            _prepare_history(rows, "missing_metric")

    def test_invalid_histories(self):
        for rows in ([], [{"epoch": 1}], [{"epoch": "bad", "train_loss": 1}],
                     [{"epoch": 1, "train_loss": 1}] * 2):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                _prepare_history(rows)


if __name__ == "__main__":
    unittest.main()
