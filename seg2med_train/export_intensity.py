"""Invert training normalization on export copies, never on model tensors."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


class MedicalIntensityExport:
    def __init__(self, *, preprocessing=None, normalize_source=True, normalizer=None):
        from .Dataloader import NormalizePriorTargetd

        normalizer = normalizer or NormalizePriorTargetd("source", "target")
        self.ranges = {"input": normalizer.source_ranges,
                       "target": normalizer.target_ranges,
                       "output": normalizer.target_ranges}
        self.normalize_source = normalize_source
        if preprocessing is not None:
            saved = preprocessing.get("normalization", {}).get("target")
            if saved:
                self.ranges["output"] = saved
                logger.info("DICOM output intensities use checkpoint target normalization ranges: %s", saved)
            else:
                logger.warning("Checkpoint has no target normalization ranges; DICOM output uses current "
                               "Dataloader.py ranges. Verify these match the checkpoint's training ranges.")

    def __call__(self, array, label, modality):
        """Return a new intensity array and DICOM units; do not add clipping.

        Inversion cannot recover intensities clipped during preprocessing.
        MR intensities are arbitrary units, not quantitative tissue parameters.
        """
        if label == "input" and not self.normalize_source:
            return np.asarray(array).copy(), "US"
        ranges = self.ranges[label]
        key = str(int(modality))
        bounds = ranges.get(key, ranges.get(int(modality)))
        if bounds is None:
            raise ValueError(f"No {label} export normalization range for modality {modality}.")
        lower, upper = map(float, bounds)
        if not np.isfinite([lower, upper]).all() or upper <= lower:
            raise ValueError(f"Invalid {label} export normalization range: {bounds}.")
        restored = np.asarray(array, dtype=np.float64) * (upper - lower) + lower
        return restored, "HU" if int(modality) == 0 else "US"
