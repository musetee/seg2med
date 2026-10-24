"""Shared anatomical-label mapping for CT/MR prior conversion."""

import logging
import re

import numpy as np

MANUAL_ANATOMY_MAPPING = {
    # Explicit many-to-one aggregations supported by total -> total_mr.
    "lung_upper_lobe_left": "lung_left",
    "lung_lower_lobe_left": "lung_left",
    "lung_upper_lobe_right": "lung_right",
    "lung_middle_lobe_right": "lung_right",
    "lung_lower_lobe_right": "lung_right",
}


def convert_segmentation_mask(source_mask, source_csv, target_csv, body_contour_value=0, *, logger=None):
    """Map unshifted labels by exact tissue name or explicit aggregation.

    Unmapped labels become unlabeled body, with a warning. The input is not
    mutated. Parameter tables may be CSV paths or pandas DataFrames. Broad MR
    structures cannot be split into CT substructures from a label alone; never
    guess their destination through fuzzy matching.
    """
    import pandas as pd

    logger = logger or logging.getLogger(__name__)
    source = source_csv if isinstance(source_csv, pd.DataFrame) else pd.read_csv(source_csv)
    target = target_csv if isinstance(target_csv, pd.DataFrame) else pd.read_csv(target_csv)
    for frame in (source, target):
        if not {"Order Number", "Organ Name"}.issubset(frame.columns):
            raise ValueError("Anatomy tables must contain 'Order Number' and 'Organ Name'.")
        if frame["Order Number"].duplicated().any():
            raise ValueError("Anatomy tables must have unique 'Order Number' values.")
        if frame["Organ Name"].isna().any() or frame["Organ Name"].astype(str).str.strip().eq("").any():
            raise ValueError("Anatomy tables must have nonempty organ names.")
    if target["Organ Name"].duplicated().any():
        raise ValueError("Target anatomy names must be unique; duplicate names make conversion ambiguous.")
    source_names = dict(zip(source["Order Number"], source["Organ Name"]))
    target_ids = {}
    for order, name in zip(target["Order Number"], target["Organ Name"]):
        target_ids.setdefault(name, int(order))
    result = np.zeros(np.asarray(source_mask).shape, dtype=np.int32)
    for label in np.unique(source_mask):
        if label == 0:
            continue
        name = source_names.get(label)
        match = name if name in target_ids else MANUAL_ANATOMY_MAPPING.get(name)
        if match is None and isinstance(name, str) and re.fullmatch(r"vertebrae_[CTLS]\d+", name):
            match = "vertebrae"
        if match not in target_ids:
            match = None
        if match is None:
            logger.warning("No target anatomy match for label %s (%s); treating it as unlabeled body.", label, name)
        result[source_mask == label] = target_ids.get(match, body_contour_value)
    return result
