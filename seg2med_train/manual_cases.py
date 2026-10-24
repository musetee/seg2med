"""Translate inference-only YAML cases into the existing manifest schema."""

from pathlib import Path

from seg2med_datasets.schema import MODALITY_NAMES


def case_rows(config, source_column):
    cases = config.dataset.case
    if config.mode != "test" or config.model_name not in {"ddpm2d_seg2med", "ddpm2d_seg2med_multimodal"}:
        raise ValueError("Manual dataset.case inputs require mode: test and a seg2med DDPM2D model.")
    if not isinstance(cases, list) or not cases:
        raise ValueError("dataset.case must be a non-empty list.")
    names = {name: key for key, name in MODALITY_NAMES.items()}
    rows, identities = [], set()
    for index, entry in enumerate(cases, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"case {index} must be a mapping.")
        patient = str(entry.get("id", "")).strip()
        source = entry.get(source_column)
        if not patient or not source:
            raise ValueError(f"case {index} requires id and {source_column}.")
        modality = entry.get("modality_name", entry.get("modality", config.dataset.modality_name))
        try:
            token = str(modality).strip().lower()
            modality = names[token] if token in names else int(token)
        except (ValueError, TypeError) as error:
            raise ValueError(f"case {index}: specify modality_name per case or under dataset (e.g. ct, mr_t1_gre).") from error
        if modality not in MODALITY_NAMES:
            raise ValueError(f"case {index}: unsupported modality {modality}.")
        identity = (patient, modality)
        if identity in identities:
            raise ValueError(f"Duplicate manual case id/modality: {identity}.")
        identities.add(identity)
        target = entry.get("img") or ""
        for value in (source, target):
            if value and not Path(value).is_file():
                raise FileNotFoundError(f"Manual case file does not exist: {value}")
        # Check alignment before applying shared spatial transforms.
        import nibabel as nib
        import numpy as np
        prior = nib.load(str(source))
        if len(prior.shape) != 3:
            raise ValueError(f"Manual priors must be 3D NIfTI volumes: {source}")
        if target:
            image = nib.load(str(target))
            if image.shape != prior.shape or not np.allclose(image.affine, prior.affine, atol=1e-3, rtol=0):
                raise ValueError(f"Manual case {patient}: image and prior must have the same grid.")
        rows.append({"id": patient, source_column: str(source), "img": str(target),
                     "modality": modality, "aorta_diss": entry.get("aorta_diss", 0)})
    return rows
