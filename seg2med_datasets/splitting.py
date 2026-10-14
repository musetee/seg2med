"""Patient-grouped, approximately stratified manifest construction.

Only sidecars carry canonical identities; model-facing catalog columns and scan
IDs remain unchanged. No imaging libraries or volume contents are needed here.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from seg2med_workflow.config import ConfigError
from .schema import MODALITY_NAMES

SPLITS = ("train", "validation", "test")
DEFAULT_MODALITIES = (0, 1, 2, 3, 4)


def modality_ids(value: Any) -> set[int]:
    if value == "all":
        return set(MODALITY_NAMES)
    if not isinstance(value, (list, tuple)):
        raise ConfigError("output.split.modalities must be a list of IDs/names or 'all'.")
    names = {name: key for key, name in MODALITY_NAMES.items()}
    names.update(mr_t1=1, mr_t2=2)
    try:
        selected = {names[item] if item in names else int(item) for item in value}
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Unknown split modality in {value!r}") from exc
    if not selected.issubset(MODALITY_NAMES):
        raise ConfigError(f"Unknown split modality IDs: {sorted(selected - set(MODALITY_NAMES))}")
    return selected


def inventory_identities(frame: pd.DataFrame, configs: list[dict[str, Any]]) -> pd.DataFrame:
    """Resolve adapter scan IDs to source-local patient IDs, conservatively."""
    policies = {config["name"]: config for config in configs}
    records = []
    row_policies = frame.attrs.get("split_policies")
    for position, row in enumerate(frame.to_dict("records")):
        policy = row_policies[position] if row_policies is not None else policies[row["dataset"]]
        source = str(policy.get("split_source", row["dataset"]))
        raw_id = str(row["id"]).strip()
        patient = raw_id
        issue = ""
        mapping = policy.get("split_patient_id_map", {})
        pattern = policy.get("split_patient_id_regex")
        if mapping:
            value = mapping.get(raw_id)
            patient = str(value).strip() if value is not None else ""
            if not patient:
                issue = "patient ID missing from explicit mapping"
        elif pattern:
            match = re.fullmatch(pattern, raw_id)
            if match:
                patient = match.groupdict().get("patient_id") or (match.group(1) if match.groups() else match.group(0))
            else:
                issue = "patient ID did not match split_patient_id_regex"
        elif policy.get("split_kind") == "anika":
            # Adapter IDs are <patient-number>_<scan-number>, not patients.
            match = re.fullmatch(r"(?P<patient>\d+)_[^_]+", raw_id)
            if match:
                patient = match["patient"]
            else:
                issue = "unrecognized Anika patient_scan ID"
        if not patient:
            issue = issue or "empty patient identity"
        region = str(policy.get("split_region", "")).strip().lower()
        if not region:
            paths = "/".join(str(row[key]).replace("\\", "/").lower() for key in ("img", "seg", "seg_tissue"))
            regions = {value for value in ("abdomen", "pelvis", "brain") if value in paths.split("/")}
            region = next(iter(regions)) if len(regions) == 1 else "unknown"
        records.append({"dataset": source, "raw_id": raw_id, "patient_id": patient,
                        "region": region, "identity_issue": issue})
    return pd.DataFrame(records, index=frame.index)


def _features(group: pd.DataFrame) -> set[str]:
    source = group.iloc[0]["dataset"]
    labels = {f"source:{source}"}
    for region in group.region.unique():
        labels.add(f"region:{source}:{region}")
    # Joint patient-level labels propagate MR stratification to paired CT.
    for region, modality in group[["region", "modality"]].drop_duplicates().itertuples(index=False, name=None):
        labels.add(f"sequence:{source}:{region}:{modality}")
    return labels


def assign_patients(inventory: pd.DataFrame, *, ratios: tuple[float, float, float],
                    seed: int, mapping_path: Path) -> pd.DataFrame:
    """Greedy multilabel stratification with fixed, reusable patient assignments."""
    if any(not math.isfinite(value) or value < 0 for value in ratios) or not math.isclose(sum(ratios), 1.0):
        raise ConfigError("Train/validation/test ratios must be nonnegative and sum to 1.")
    resolved = inventory.loc[inventory.identity_issue.eq("")]
    groups = {key: _features(group) for key, group in resolved.groupby(["dataset", "patient_id"], sort=True)}
    counts = Counter(label for labels in groups.values() for label in labels)
    assigned: dict[tuple[str, str], str] = {}
    saved = None
    if mapping_path.is_file():
        saved = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
        required = {"dataset", "patient_id", "split", "seed", "train_ratio", "validation_ratio", "test_ratio"}
        if not required.issubset(saved.columns):
            raise ConfigError(f"Invalid patient mapping: {mapping_path}")
        if saved.duplicated(["dataset", "patient_id"]).any() or not saved.split.isin(SPLITS).all():
            raise ConfigError(f"Duplicate patients or invalid splits in {mapping_path}")
        settings = {"seed": seed, **dict(zip(("train_ratio", "validation_ratio", "test_ratio"), ratios))}
        for column, expected in settings.items():
            if not saved[column].map(lambda value: math.isclose(float(value), expected)).all():
                raise ConfigError(f"Saved mapping uses different {column}; use a new split prefix or mapping filename.")
        assigned = {(row.dataset, row.patient_id): row.split for row in saved.itertuples()}
    label_counts = {name: Counter() for name in SPLITS}
    totals = Counter()
    for key, labels in groups.items():
        if key in assigned:
            name = assigned[key]
            label_counts[name].update(labels)
            totals[name] += 1
    rng = random.Random(seed)
    pending = [key for key in sorted(groups) if key not in assigned]
    rng.shuffle(pending)
    remaining = Counter(label for key in pending for label in groups[key])
    while pending:
        # Iterative multilabel stratification: finish the rarest remaining
        # stratum first. Global deficits must not swamp a scarce MR sequence.
        rare = min((label for label, count in remaining.items() if count),
                   key=lambda label: (remaining[label], label))
        key = next(key for key in pending if rare in groups[key])
        labels = groups[key]
        def score(index: int) -> tuple[float, float, float]:
            name, ratio = SPLITS[index], ratios[index]
            return (
                counts[rare] * ratio - label_counts[name][rare],
                sum((counts[label] * ratio - label_counts[name][label]) / counts[label]
                    for label in sorted(labels)),
                len(groups) * ratio - totals[name],
            )
        options = [index for index, ratio in enumerate(ratios) if ratio > 0]
        rng.shuffle(options)
        name = SPLITS[max(options, key=score)]
        assigned[key] = name
        label_counts[name].update(labels)
        totals[name] += 1
        remaining.subtract(labels)
        pending.remove(key)
    rows = [{"dataset": key[0], "patient_id": key[1], "split": name, "seed": seed,
             "train_ratio": ratios[0], "validation_ratio": ratios[1], "test_ratio": ratios[2]}
            for key, name in sorted(assigned.items())]
    return pd.DataFrame(rows, columns=["dataset", "patient_id", "split", "seed", "train_ratio", "validation_ratio", "test_ratio"])


def build_manifests(frame: pd.DataFrame, configs: list[dict[str, Any]], *,
                    ratios: tuple[float, float, float], seed: int, mapping_path: Path,
                    included_modalities: Any = DEFAULT_MODALITIES) -> dict[str, Any]:
    identities = inventory_identities(frame, configs)
    inventory = identities.assign(modality=frame.modality)
    mapping = assign_patients(inventory, ratios=ratios, seed=seed, mapping_path=mapping_path)
    lookup = {(row.dataset, row.patient_id): row.split for row in mapping.itertuples()}
    assignments = [lookup.get((row.dataset, row.patient_id), "") if not row.identity_issue else ""
                   for row in identities.itertuples()]
    allowed = modality_ids(included_modalities)
    reasons = []
    for index, row in frame.iterrows():
        issues = []
        if identities.loc[index, "identity_issue"]:
            issues.append("unresolved_patient_identity")
        if int(row.modality) not in allowed:
            issues.append("excluded_modality")
        for field in ("img", "seg", "seg_tissue"):
            value = str(row[field]).strip()
            if not value or not Path(value).is_file():
                issues.append(f"missing_{field}")
        reasons.append(";".join(issues))
    audit = frame.copy()
    audit["source"] = identities.dataset
    audit["patient_id"] = identities.patient_id
    audit["region"] = identities.region
    audit["identity_issue"] = identities.identity_issue
    audit["split"] = assignments
    audit["exclusion_reason"] = reasons
    resolved = audit.loc[audit.identity_issue.eq("")]
    if resolved.groupby(["source", "patient_id"]).split.nunique().gt(1).any():
        raise AssertionError("Patient overlap in inventory assignments")
    manifests = {}
    patient_sets = []
    for name in SPLITS:
        mask = audit.split.eq(name) & audit.exclusion_reason.eq("")
        manifests[name] = frame.loc[mask].assign(split=name).reset_index(drop=True)
        patient_sets.append(set(audit.loc[mask, ["source", "patient_id"]].itertuples(index=False, name=None)))
    for index, patients in enumerate(patient_sets):
        for other in patient_sets[index + 1:]:
            if patients & other:
                raise AssertionError("Patient overlap between manifests")
    return {"manifests": manifests, "patient_mapping": mapping, "identity_audit": audit,
            "excluded_df": audit.loc[audit.exclusion_reason.ne("")].reset_index(drop=True)}


def format_split_summary(result: dict[str, Any]) -> str:
    audit = result["identity_audit"]
    lines = ["Patient-level split summary (zero patient overlap verified):"]
    mapping = result["patient_mapping"]
    if not mapping.empty:
        settings = mapping.iloc[0]
        lines.append(f"Target patient ratios: train={settings.train_ratio:g}, validation={settings.validation_ratio:g}, "
                     f"test={settings.test_ratio:g}; seed={settings.seed}")
    resolved = audit.loc[audit.identity_issue.eq("")]
    lines.append(f"Inventory: {len(resolved[['source', 'patient_id']].drop_duplicates())} resolved patients; {len(audit)} rows")
    for name in SPLITS:
        rows = audit.loc[audit.split.eq(name) & audit.exclusion_reason.eq("")]
        patients = rows[["source", "patient_id"]].drop_duplicates().shape[0]
        scans = rows[["dataset", "img"]].drop_duplicates().shape[0]
        lines.append(f"\n{name}: {patients} patients; {scans} scans/image paths; {len(rows)} rows")
        if not rows.empty:
            counts = rows.groupby(["source", "region", "modality_name"]).agg(
                patients=("patient_id", "nunique"), rows=("id", "size"))
            lines.append(counts.to_string())
    excluded = audit.loc[audit.exclusion_reason.ne("")]
    unresolved = audit.loc[audit.identity_issue.ne("")]
    lines += [f"\nExcluded records: {len(excluded)} (retained in full inventory)",
              f"Unresolved patient identity records: {len(unresolved)}",
              f"Unknown anatomy records: {int(audit.region.eq('unknown').sum())}"]
    if not excluded.empty:
        lines.append(excluded.groupby(["source", "modality_name", "exclusion_reason"]).size().to_string())
    return "\n".join(lines) + "\n"
