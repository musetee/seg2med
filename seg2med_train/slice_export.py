"""Incremental slice exports while a patient's volume is being inferred."""

from pathlib import Path

import numpy as np

from .dicom_export import save_dicom_series


class InferenceSliceWriter:
    def __init__(self, *, patient_id, modality, epoch, step, img_folder,
                 npy_folder=None, save_png=False, save_npz=False, save_dicom=False,
                 dicom_intensity=None, save_target=True, extra_labels=()):
        self.patient_id = patient_id
        self.modality = modality
        self.dicom_intensity = dicom_intensity
        self.save_target = save_target
        # Additional model inputs (e.g. lesion_mask) exported as their own series.
        # They are label maps: written with raw values, never intensity-restored.
        self.extra_labels = tuple(extra_labels)
        if {"input", "output", "target"} & set(self.extra_labels):
            raise ValueError("extra_labels must not reuse input/output/target.")
        self.prefix = f"epoch_{epoch}_{patient_id}_modality_{modality}_step_{step}"
        self.img_folder = Path(img_folder)
        self.dicom_folder = self.img_folder.parent / "dicom"
        self.archive_folder = Path(npy_folder) if npy_folder is not None else self.img_folder
        self.save_png, self.save_npz, self.save_dicom = save_png, save_npz, save_dicom
        self.next_slice = 1
        if save_png:
            import imageio

            self.imageio = imageio
        if save_dicom:
            try:
                from pydicom.uid import generate_uid
            except ImportError as error:
                raise ImportError("DICOM export requires pydicom. Install it with 'pip install pydicom'.") from error
            self.study_uid = generate_uid()
            self.series_uids = {label: generate_uid() for label in ("input", "output", "target", *self.extra_labels)}
        self.series_numbers = {"input": 1, "output": 2, "target": 3,
                               **{label: 4 + index for index, label in enumerate(self.extra_labels)}}

    def save_batch(self, inputs, targets, outputs, extras=None):
        """Write completed (slices, channels, X, Y) batches before inference resumes."""
        if not (self.save_png or self.save_npz or self.save_dicom):
            return
        extras = extras or {}
        if set(extras) != set(self.extra_labels):
            raise ValueError(f"Expected extra slices for {list(self.extra_labels)}, got {sorted(extras)}.")
        arrays = {}
        for label, tensor in (("input", inputs), ("target", targets), ("output", outputs), *extras.items()):
            if label == "target" and not self.save_target:
                continue
            array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            if array.ndim != 4 or array.shape[1] != 1:
                raise ValueError(f"Expected single-channel slice batch for {label}, got {array.shape}.")
            arrays[label] = array[:, 0]
        if len({array.shape for array in arrays.values()}) != 1:
            raise ValueError("Input, target, output and extra slice batches must have matching shapes.")

        for label, array in arrays.items():
            for index, plane in enumerate(array):
                prefix = f"{self.prefix}_slice_{self.next_slice + index:05d}_{label}"
                if self.save_png:
                    folder = self.img_folder / label
                    folder.mkdir(parents=True, exist_ok=True)
                    preview = (plane * 255).clip(0, 255).astype(np.uint8).T
                    self.imageio.imwrite(str(folder / f"{prefix}.png"), preview)
                if self.save_npz:
                    folder = self.archive_folder / label
                    folder.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(folder / f"{prefix}.npz", **{label: plane})
            if self.save_dicom:
                dicom_array, units = (self.dicom_intensity(array, label, self.modality)
                                      if self.dicom_intensity is not None and label not in self.extra_labels
                                      else (array, "US"))
                save_dicom_series(
                    dicom_array.transpose(0, 2, 1), self.dicom_folder / label / self.prefix,
                    patient_id=self.patient_id, label=label, study_uid=self.study_uid,
                    series_number=self.series_numbers[label],
                    series_uid=self.series_uids[label], instance_start=self.next_slice,
                    rescale_type=units,
                )
        self.next_slice += arrays["output"].shape[0]
