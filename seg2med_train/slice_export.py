"""Incremental slice exports while a patient's volume is being inferred."""

from pathlib import Path

import numpy as np

from .dicom_export import save_dicom_series


class InferenceSliceWriter:
    def __init__(self, *, patient_id, modality, epoch, step, img_folder,
                 npy_folder=None, save_png=False, save_npz=False, save_dicom=False):
        self.patient_id = patient_id
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
            self.series_uids = {label: generate_uid() for label in ("input", "output", "target")}

    def save_batch(self, inputs, targets, outputs):
        """Write completed (slices, channels, X, Y) batches before inference resumes."""
        if not (self.save_png or self.save_npz or self.save_dicom):
            return
        arrays = {}
        for label, tensor in (("input", inputs), ("target", targets), ("output", outputs)):
            array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
            if array.ndim != 4 or array.shape[1] != 1:
                raise ValueError(f"Expected single-channel slice batch for {label}, got {array.shape}.")
            arrays[label] = array[:, 0]
        if len({array.shape for array in arrays.values()}) != 1:
            raise ValueError("Input, target, and output slice batches must have matching shapes.")

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
                save_dicom_series(
                    array.transpose(0, 2, 1), self.dicom_folder / label / self.prefix,
                    patient_id=self.patient_id, label=label, study_uid=self.study_uid,
                    series_number={"input": 1, "output": 2, "target": 3}[label],
                    series_uid=self.series_uids[label], instance_start=self.next_slice,
                )
        self.next_slice += arrays["output"].shape[0]
