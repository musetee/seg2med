"""One input volume -> masks, physics priors, and compatible CSV manifests."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

from seg2med_datasets.schema import MODALITY_NAMES, SCHEMA
from seg2med_segmentation.runner import segment_volume
from .config import ConfigError, as_bool, load_yaml, resolve_path

logger = logging.getLogger(__name__)
DEFAULTS_PATH = Path(__file__).with_name("case_defaults.yaml")
PRIOR_COLUMNS = ("id", "aorta_diss", "prior", "img", "modality", "prior_semantic")


def _modality(value: Any) -> int:
    names = {name: key for key, name in MODALITY_NAMES.items()}
    token = str(value).strip().lower()
    try:
        result = names[token] if token in names else int(token)
    except ValueError as error:
        raise ConfigError(f"Unknown modality {value!r}. Choose from {', '.join(names)}.") from error
    if result not in MODALITY_NAMES:
        raise ConfigError(f"Unknown modality ID {result}.")
    return result


def _companion(image: Path, suffix: str, *, segmentation=False) -> Path:
    stem = image.name[:-7] if image.name.lower().endswith(".nii.gz") else image.stem
    extension = ".nii.gz" if image.name.lower().endswith(".nii.gz") else ".nii"
    candidate = image.with_name(f"{stem}{suffix}{extension}")
    alternate = image.with_name(f"{stem}{suffix}{'.nii' if extension == '.nii.gz' else '.nii.gz'}")
    if candidate.is_file():
        return candidate
    if alternate.is_file():
        return alternate
    return image.with_name(f"{stem}{suffix}.nii.gz") if segmentation else candidate


def _validate_geometry(image: Path, organs: Path, tissues: Path) -> None:
    import nibabel as nib
    import numpy as np

    reference = nib.load(str(image))
    if len(reference.shape) != 3:
        raise ConfigError(f"Single-case preparation requires a 3D NIfTI volume, got {reference.shape}.")
    for path in (organs, tissues):
        mask = nib.load(str(path))
        if mask.shape != reference.shape or not np.allclose(mask.affine, reference.affine, atol=1e-3, rtol=0):
            raise ConfigError(f"Mask does not match the image grid: {path}")


def _write_manifest(path: Path, row: dict, columns) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerow(row)


def prepare_case(config: dict[str, Any], *, base_dir: str | Path = ".", dry_run: bool = False) -> dict[str, Any]:
    """Prepare one case in place, never overwriting the original image.

    Relative input/override paths are relative to base_dir. Default manifests
    and generated companions live beside the image. Reuse needs no optional
    segmentation backend; dry runs do not import it or write any files.
    """
    defaults, _ = load_yaml(DEFAULTS_PATH)
    settings = {**defaults, **config}
    from seg2med_physiosynth.label_schemes import validate_scheme
    validate_scheme(settings["segmentation_label_scheme"])
    validate_scheme(settings["prior_label_scheme"])
    for section in ("segmentation", "prior"):
        supplied = config.get(section, {})
        if not isinstance(supplied, dict):
            raise ConfigError(f"'{section}' must be a mapping.")
        settings[section] = {**defaults[section], **supplied}
    for key in ("dataset", "id", "img"):
        if settings.get(key) is None or not str(settings[key]).strip():
            raise ConfigError(f"Single-case preparation requires '{key}'.")
    modality_value = settings.get("input_modality_name", settings.get("modality_name", settings.get("modality name", settings.get("modality"))))
    modality = _modality(modality_value)
    target_value = settings.get("target_modality_name")
    target_modality = modality if target_value is None else _modality(target_value)
    cross_modality = target_modality != modality
    suffix = f"_{MODALITY_NAMES[target_modality]}" if target_value is not None else ""
    image = resolve_path(settings["img"], base_dir, required=True)
    assert image is not None
    if not image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image}")
    if not image.name.lower().endswith((".nii", ".nii.gz")):
        raise ConfigError("Single-case preparation currently accepts .nii or .nii.gz medical volumes.")
    segmentation = settings["segmentation"]
    if not as_bool(segmentation["multi_label"]):
        raise ConfigError("Single-case preparation requires multi_label: true for its label-map manifests.")
    tasks = segmentation["ct_tasks" if modality == 0 else "mr_tasks"]
    if not isinstance(tasks, list) or len(tasks) != 2 or not all(isinstance(task, str) for task in tasks):
        raise ConfigError("Segmentation defaults must specify the organ and tissue tasks, in that order.")
    overwrite_seg = as_bool(config.get("overwrite_seg", config.get("overwrite_segs", settings["overwrite_seg"])))
    overwrite_prior = as_bool(settings["overwrite_prior"])
    organs = _companion(image, "_seg", segmentation=True)
    tissues = _companion(image, "_seg_tissue", segmentation=True)
    prior = _companion(image, "_prior_image" + suffix)
    semantic = _companion(image, "_prior_semantic" + suffix)
    target = _companion(image, "_target")
    stem = image.name[:-7] if image.name.lower().endswith(".nii.gz") else image.stem
    contour = image.with_name(f"{stem}_contour{'.nii.gz' if image.name.lower().endswith('.nii.gz') else '.nii'}")
    seg_csv = resolve_path(settings.get("seg_csv", image.with_name(f"{stem}_seg.csv")), base_dir, required=True)
    prior_csv = resolve_path(settings.get("prior_csv", image.with_name(f"{stem}_priors{suffix}.csv")), base_dir, required=True)
    companions = (organs, tissues, prior, semantic, target, contour)
    resolved = [path.resolve() for path in companions]
    if image in resolved or len(set(resolved)) != len(resolved):
        raise ConfigError("Generated companions must be distinct files and must not link to the input image.")
    protected = {image, *resolved}
    if seg_csv == prior_csv or seg_csv in protected or prior_csv in protected:
        raise ConfigError("Manifest paths must differ from each other and from input/generated image files.")
    if seg_csv.suffix.lower() != ".csv" or prior_csv.suffix.lower() != ".csv":
        raise ConfigError("Segmentation and prior manifest filenames must end in .csv.")
    work = [(task, path) for task, path in zip(tasks, (organs, tissues)) if overwrite_seg or not path.is_file()]
    generate_prior = overwrite_prior or not (prior.is_file() and semantic.is_file())
    result = {"img": image, "seg": organs, "seg_tissue": tissues, "prior": prior,
              "prior_semantic": semantic, "target": target if generate_prior or target.is_file() else image,
              "contour": contour,
              "seg_csv": seg_csv, "prior_csv": prior_csv, "segmentation_planned": work,
              "prior_planned": generate_prior, "modality": target_modality, "input_modality": modality,
              "created_masks": [], "dry_run": dry_run}
    logger.info("Case %s | input=%s | prior=%s | image=%s", settings["id"],
                MODALITY_NAMES[modality], MODALITY_NAMES[target_modality], image)
    logger.info("Segmentation: %d task(s) to run; priors: %s", len(work), "generate" if generate_prior else "reuse")
    if dry_run:
        return result
    if not generate_prior:
        logger.warning("Reusing existing priors unchanged; label-scheme and parameter changes require overwrite_prior: true.")

    for task, path in work:
        logger.info("Segmenting %s -> %s", task, path)
        segment_volume(image, path, task=task, device=str(settings.get("device", segmentation["device"])),
                       multi_label=True, fast=as_bool(segmentation["fast"]))
        if not path.is_file():
            raise RuntimeError(f"Segmentation task {task!r} did not produce its output: {path}")
        result["created_masks"].append(path)
    _validate_geometry(image, organs, tissues)
    absolute = as_bool(settings["absolute_paths"])
    def csv_path(path, manifest):
        if absolute:
            return str(path)
        import os
        return os.path.relpath(path, start=manifest.parent)

    seg_row = {"dataset": str(settings["dataset"]), "id": str(settings["id"]),
               "seg": csv_path(organs, seg_csv), "seg_tissue": csv_path(tissues, seg_csv), "img": csv_path(image, seg_csv),
               "modality": modality, "modality_name": MODALITY_NAMES[modality], "aorta_diss": 0}
    _write_manifest(seg_csv, seg_row, SCHEMA.ALL)
    if work and not overwrite_prior and (prior.is_file() or semantic.is_file()):
        logger.warning("Masks changed but existing prior files are being retained; use overwrite_prior to refresh them.")
    if generate_prior:
        from seg2med_physiosynth.priors import DEFAULT_CT_PARAMETERS, DEFAULT_MR_PARAMETERS, _prepare_volume

        parameters = settings.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ConfigError("'parameters' must be a mapping.")
        ct = resolve_path(parameters.get("ct", DEFAULT_CT_PARAMETERS), base_dir, required=True)
        mr = resolve_path(parameters.get("mr", DEFAULT_MR_PARAMETERS), base_dir, required=True)
        threshold = settings["prior"]["body_threshold"]
        logger.info("Generating image and semantic priors (original image is preserved)")
        _prepare_volume(image, organs, tissues, image.parent, target_modality,
                        ct_parameters=ct, mr_parameters=mr,
                        min_contour_area=float(settings["prior"]["min_contour_area"]),
                        threshold=None if threshold is None else float(threshold), overwrite=overwrite_prior,
                        output_paths=(target, prior, semantic), input_modality=modality,
                        segmentation_label_scheme=settings["segmentation_label_scheme"],
                        prior_label_scheme=settings["prior_label_scheme"],
                        mr_legacy_parameters=resolve_path(parameters.get("mr_legacy"), base_dir))
        if not all(path.is_file() for path in (target, prior, semantic)):
            raise RuntimeError("Prior generation did not produce all required files.")
    prior_row = {"id": str(settings["id"]), "aorta_diss": 0, "prior": csv_path(prior, prior_csv),
                 # A source CT/MR is not ground truth for a different target modality.
                 "img": "" if cross_modality else csv_path(result["target"], prior_csv),
                 "modality": target_modality, "prior_semantic": csv_path(semantic, prior_csv)}
    _write_manifest(prior_csv, prior_row, PRIOR_COLUMNS)
    logger.info("Segmentation manifest: %s", seg_csv)
    logger.info("Prior manifest: %s", prior_csv)
    return result


def prepare_case_from_config(config_path: str | Path | None = None, *, overrides: dict | None = None,
                             dry_run: bool = False) -> dict[str, Any] | list[dict[str, Any]]:
    """Accept a flat single-case config or a YAML-only ``case: [...]`` list.

    List entries inherit shared settings; nested option mappings merge by key.
    CLI metadata overrides remain single-case only, while processing flags can
    override shared/per-case values for the entire list.
    """
    config, base = load_yaml(config_path) if config_path else ({}, Path.cwd())
    overrides = overrides or {}
    if "case" not in config:
        settings = {**config, **overrides}
        if "overwrite_segs" in overrides and "overwrite_seg" not in overrides:
            settings["overwrite_seg"] = overrides["overwrite_segs"]
        return prepare_case(settings, base_dir=base, dry_run=dry_run)
    entries = config["case"]
    if not isinstance(entries, list) or not entries:
        raise ConfigError("'case' must be a non-empty list of case mappings.")
    single_fields = {"img", "id", "dataset", "modality", "modality_name", "modality name", "input_modality_name", "target_modality_name"}
    if single_fields & overrides.keys():
        raise ConfigError("CLI image/case metadata arguments are single-input only; edit the YAML case entries instead.")
    shared = {key: value for key, value in config.items() if key != "case"}
    merged = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ConfigError(f"case entry {index} must be a mapping.")
        settings = {**shared, **entry, **overrides}
        # Treat the accepted modality spellings as one setting when applying
        # per-case overrides (e.g. a numeric modality overrides a shared name).
        for layer in (shared, entry, overrides):
            if "overwrite_seg" in layer or "overwrite_segs" in layer:
                settings["overwrite_seg"] = layer.get("overwrite_seg", layer.get("overwrite_segs"))
            for key in ("input_modality_name", "modality_name", "modality name", "modality"):
                if key in layer:
                    settings["input_modality_name"] = layer[key]
                    break
        for section in ("segmentation", "prior", "parameters"):
            values = {}
            for layer in (shared, entry, overrides):
                options = layer.get(section, {})
                if not isinstance(options, dict):
                    raise ConfigError(f"case entry {index}: '{section}' must be a mapping.")
                values.update(options)
            if values:
                settings[section] = values
        merged.append(settings)
    # Validate the whole list and all output locations before writing the first
    # case. A shared explicit CSV path must not silently overwrite earlier cases.
    plans = [prepare_case(settings, base_dir=base, dry_run=True) for settings in merged]
    inputs = {plan["img"].resolve() for plan in plans}
    if len({(plan["img"].resolve(), plan["prior"].resolve()) for plan in plans}) != len(plans):
        raise ConfigError("Each YAML case must reference a distinct input image / target naming combination.")
    source_modalities = {}
    destinations = {}
    for plan in plans:
        image = plan["img"].resolve()
        if image in source_modalities and source_modalities[image] != plan["input_modality"]:
            raise ConfigError(f"Conflicting input modalities for {image}.")
        source_modalities[image] = plan["input_modality"]
        for key in ("seg", "seg_tissue", "prior", "prior_semantic", "target", "contour", "seg_csv", "prior_csv"):
            path = plan[key].resolve()
            # Reuse of priors without a processed target legitimately reads the
            # original image; it does not write to that target path.
            if key == "target" and path == plan["img"].resolve():
                continue
            shared = key in {"seg", "seg_tissue", "target", "contour", "seg_csv"} and destinations.get(path) == (image, key)
            if path in inputs or (path in destinations and not shared):
                raise ConfigError(f"Cases have overlapping input/output paths: {path}. Use distinct per-case output paths.")
            destinations[path] = (image, key)
    if dry_run:
        return plans
    results = []
    for index, settings in enumerate(merged, start=1):
        logger.info("Preparing case %d/%d", index, len(merged))
        results.append(prepare_case(settings, base_dir=base))
    return results
