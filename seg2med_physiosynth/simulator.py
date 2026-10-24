"""Pure array operations for constructing physics-informed image priors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TYPE_CHECKING

import numpy as np

from . import formulas

if TYPE_CHECKING:
    import pandas as pd

PHYSIOSYNTH_ROOT = Path(__file__).resolve().parent
DEFAULT_CT_PARAMETERS = PHYSIOSYNTH_ROOT / "params_ct.csv"
DEFAULT_MR_PARAMETERS = PHYSIOSYNTH_ROOT / "params_mr.csv"

SignalFormula = Callable[[float, float, float], float]

MODALITY_NAMES = {
    0: "ct",
    1: "mr_t1_gre",
    2: "mr_t2_space",
    3: "mr_vibe_in",
    4: "mr_vibe_opp",
    5: "mr_vibe_dixon_tra_in",
    6: "mr_vibe_dixon_tra_opp",
    7: "mri_2d_grayscale",
}


@dataclass(frozen=True)
class ModalitySpec:
    threshold: float
    background: float
    formula: SignalFormula | None
    parameter_type: str


MODALITY_SPECS = {
    0: ModalitySpec(-500, -1000, None, "ct"),
    1: ModalitySpec(100, 0, formulas.gre_t1, "mr"),
    2: ModalitySpec(10, 0, formulas.t2_space, "mr"),
    3: ModalitySpec(30, 0, formulas.vibe, "mr"),
    4: ModalitySpec(30, 0, formulas.vibe_opposed, "mr"),
    5: ModalitySpec(30, 0, formulas.vibe_dixon_in, "mr"),
    6: ModalitySpec(30, 0, formulas.vibe_dixon_opposed, "mr"),
    7: ModalitySpec(1, 0, formulas.gre_t1, "mr"),
}


def create_body_contour(image: np.ndarray, threshold: float, min_area: float = 10000) -> np.ndarray:
    """Create a slice-wise filled body mask from an intensity image."""
    import cv2

    image = np.asarray(image)
    if image.ndim not in {2, 3}:
        raise ValueError(f"Body contour input must be 2D or 3D, got shape {image.shape}")

    def one_slice(array: np.ndarray) -> np.ndarray:
        binary = np.ascontiguousarray((array > threshold).astype(np.uint8))
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = np.zeros_like(binary)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(result, [contour], -1, 1, thickness=cv2.FILLED)
        return result

    if image.ndim == 2:
        return one_slice(image)
    result = np.zeros_like(image, dtype=np.uint8)
    for index in range(image.shape[-1]):
        result[..., index] = one_slice(image[..., index])
    return result


def merge_segmentations(organs: np.ndarray, tissues: np.ndarray, tissue_offset: int = 200) -> np.ndarray:
    if organs.shape != tissues.shape:
        raise ValueError(f"Organ and tissue segmentations have different shapes: {organs.shape} != {tissues.shape}")
    organs = np.asarray(organs).astype(np.int32, copy=False)
    tissue_labels = np.asarray(tissues).astype(np.int32, copy=True)
    tissue_labels[tissue_labels > 0] += tissue_offset
    merged = tissue_labels
    merged[organs > 0] = organs[organs > 0]
    return merged


def _labels_with_contour(segmentation: np.ndarray, contour: np.ndarray) -> np.ndarray:
    if segmentation.shape != contour.shape:
        raise ValueError(f"Segmentation and contour have different shapes: {segmentation.shape} != {contour.shape}")
    # TotalSegmentator labels start at one, while parameter tables start at
    # zero. Adding the contour maps body to 1 and organ N to N+1.
    return segmentation.astype(np.int32, copy=False) + contour.astype(np.int32, copy=False)


def assign_ct(labels: np.ndarray, parameter_csv: str | Path | pd.DataFrame) -> np.ndarray:
    import pandas as pd

    frame = parameter_csv if isinstance(parameter_csv, pd.DataFrame) else pd.read_csv(parameter_csv)
    required = {"Order Number", "HU Value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"CT parameter table must contain {sorted(required)}")
    result = np.full(labels.shape, -1000, dtype=np.float32)
    starts_at_zero = int(frame["Order Number"].min()) == 0
    for order, value in zip(frame["Order Number"], frame["HU Value"]):
        label = int(order) + (1 if starts_at_zero else 0)
        result[labels == label] = float(value)
    return result


def assign_mr(labels: np.ndarray, parameter_csv: str | Path | pd.DataFrame, formula: SignalFormula) -> np.ndarray:
    import pandas as pd

    frame = parameter_csv if isinstance(parameter_csv, pd.DataFrame) else pd.read_csv(parameter_csv)
    required = {"Order Number", "T1", "T2", "Rho"}
    if not required.issubset(frame.columns):
        raise ValueError(f"MR parameter table must contain {sorted(required)}")
    result = np.zeros(labels.shape, dtype=np.float32)
    starts_at_zero = int(frame["Order Number"].min()) == 0
    for order, t1, t2, rho in zip(frame["Order Number"], frame["T1"], frame["T2"], frame["Rho"]):
        label = int(order) + (1 if starts_at_zero else 0)
        result[labels == label] = formula(float(t1), float(t2), float(rho))
    return result


def simulate_from_segmentation(
    segmentation: np.ndarray,
    contour: np.ndarray,
    modality: int,
    *,
    ct_parameters: str | Path | pd.DataFrame = DEFAULT_CT_PARAMETERS,
    mr_parameters: str | Path | pd.DataFrame = DEFAULT_MR_PARAMETERS,
    signal_formula: SignalFormula | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate from merged organ/tissue labels and an existing body contour.

    Labels use the original TotalSegmentator IDs (tissues 201/202/203), before
    adding the binary contour. Returns float32 prior and integer semantic labels.
    Tables may be CSV paths or DataFrames, e.g. from an interactive upload.
    """
    try:
        spec = MODALITY_SPECS[int(modality)]
    except KeyError as exc:
        raise ValueError(f"Unsupported modality id: {modality}") from exc
    if not np.isin(contour, (0, 1)).all():
        raise ValueError("Body contour must be binary (0 or 1).")
    semantic = _labels_with_contour(np.asarray(segmentation), np.asarray(contour))
    if spec.parameter_type == "ct":
        if signal_formula is not None:
            raise ValueError("A custom MR signal formula cannot be used for CT.")
        prior = assign_ct(semantic, ct_parameters)
    else:
        prior = assign_mr(semantic, mr_parameters, signal_formula or spec.formula)
    return prior, semantic


def simulate_prior(
    image: np.ndarray,
    organ_segmentation: np.ndarray,
    tissue_segmentation: np.ndarray,
    modality: int,
    *,
    ct_parameters: str | Path,
    mr_parameters: str | Path,
    body_threshold: float | None = None,
    min_contour_area: float = 10000,
    input_modality: int | None = None,
    source_anatomy=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(prior, semantic_labels, body_contour)`` for one volume.

    ``modality`` selects the simulated signal. ``input_modality`` optionally
    selects source-image contour thresholds and cross-modality anatomy mapping.
    Omitting it preserves the original same-modality behavior.
    """
    try:
        spec = MODALITY_SPECS[int(modality)]
    except KeyError as exc:
        raise ValueError(f"Unsupported modality id: {modality}") from exc
    if image.shape != organ_segmentation.shape or image.shape != tissue_segmentation.shape:
        raise ValueError(
            "Image, organ segmentation, and tissue segmentation must have the same shape: "
            f"{image.shape}, {organ_segmentation.shape}, {tissue_segmentation.shape}"
        )
    input_spec = MODALITY_SPECS[int(input_modality)] if input_modality is not None else spec
    threshold = input_spec.threshold if body_threshold is None else float(body_threshold)
    contour = create_body_contour(image, threshold, min_area=min_contour_area)
    merged = merge_segmentations(organ_segmentation, tissue_segmentation)
    if source_anatomy is not None or input_spec.parameter_type != spec.parameter_type:
        from .anatomy import convert_segmentation_mask
        source_table = (source_anatomy if source_anatomy is not None else
                        ct_parameters if input_spec.parameter_type == "ct" else mr_parameters)
        target_table = ct_parameters if spec.parameter_type == "ct" else mr_parameters
        merged = convert_segmentation_mask(merged, source_table, target_table)
    prior, semantic = simulate_from_segmentation(
        merged, contour, modality, ct_parameters=ct_parameters, mr_parameters=mr_parameters)
    return prior, semantic, contour
