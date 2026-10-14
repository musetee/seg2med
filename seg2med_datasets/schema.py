from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
class ColumnSchema:
    """Column names shared by catalog preparation and training."""

    DATASET: str = "dataset"
    ID: str = "id"
    IMG: str = "img"
    SEG: str = "seg"
    SEG_TISSUE: str = "seg_tissue"
    MODALITY_ID: str = "modality"
    MODALITY_NAME: str = "modality_name"
    AORTA_DISS: str = "aorta_diss"
    PRIOR: str = "prior"
    PRIOR_SEMANTIC: str = "prior_semantic"

    CFG_NAME: str = "name"
    CFG_IMAGES: str = "images"
    CFG_SEGS: str = "segs"
    CFG_SEG_TISSUES: str = "seg_tissues"
    CFG_PIDS: str = "pIDs"
    CFG_MODALITY: str = "modality"

    @property
    def ALL(self) -> Tuple[str, ...]:
        return (
            self.DATASET,
            self.ID,
            self.SEG,
            self.SEG_TISSUE,
            self.IMG,
            self.MODALITY_ID,
            self.MODALITY_NAME,
            self.AORTA_DISS,
        )


SCHEMA = ColumnSchema()


