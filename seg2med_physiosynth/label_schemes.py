"""Read per-file anatomy dictionaries instead of guessing from label maxima."""

from pathlib import Path
import xml.etree.ElementTree as ET

from .simulator import DEFAULT_CT_PARAMETERS, DEFAULT_MR_PARAMETERS

DEFAULT_MR_LEGACY_PARAMETERS = Path(__file__).with_name("params_mr_legacy.csv")
SCHEMES = {"auto", "current", "legacy"}


def validate_scheme(value):
    if value not in SCHEMES:
        raise ValueError(f"Unknown label scheme {value!r}; choose auto, current, or legacy.")
    return value


def embedded_labels(image):
    """Extract raw ID -> organ name from a TotalSegmentator Caret extension."""
    labels = {}
    for extension in image.header.extensions:
        try:
            content = extension.get_content()
        except NotImplementedError:
            continue
        if isinstance(content, str):
            content = content.encode("utf-8")
        if not isinstance(content, bytes) or b"<LabelTable" not in content:
            continue
        if len(content) > 2_000_000 or b"<!DOCTYPE" in content.upper() or b"<!ENTITY" in content.upper():
            raise ValueError("Unsafe or oversized segmentation label dictionary.")
        try:
            root = ET.fromstring(content.rstrip(b"\x00"))
            for node in root.findall(".//LabelTable/Label"):
                key, name = int(node.attrib["Key"]), (node.text or "").strip()
                if key <= 0 or key >= 200:
                    continue
                if not name or (key in labels and labels[key] != name):
                    raise ValueError("Empty or conflicting segmentation label names.")
                labels[key] = name
        except (ET.ParseError, KeyError) as error:
            raise ValueError("Invalid embedded segmentation label dictionary.") from error
    return labels


def _names(table):
    return {int(row["Order Number"]): str(row["Organ Name"])
            for row in table.to_dict("records") if 0 < int(row["Order Number"]) < 200}


def source_anatomy(image, kind, scheme="auto"):
    """Return source-name table and detected MR scheme (or current CT).

    Explicit schemes provide a fallback for metadata-free files, not permission
    to override a contradictory embedded dictionary.
    """
    import pandas as pd

    validate_scheme(scheme)
    if kind not in {"ct", "mr"}:
        raise ValueError("Segmentation kind must be ct or mr.")
    paths = ({"current": DEFAULT_CT_PARAMETERS} if kind == "ct" else
             {"current": DEFAULT_MR_PARAMETERS, "legacy": DEFAULT_MR_LEGACY_PARAMETERS})
    tables = {name: pd.read_csv(path) for name, path in paths.items()}
    labels = embedded_labels(image)
    if not labels:
        if scheme == "auto":
            raise ValueError("Segmentation has no embedded label dictionary. Set segmentation_label_scheme "
                             "explicitly to current or legacy; label numbers alone are ambiguous.")
        if kind == "ct" and scheme == "legacy":
            raise ValueError("No metadata-free legacy CT preset is available; supply a mask with a label dictionary.")
        return tables[scheme], scheme
    dictionaries = {name: _names(table) for name, table in tables.items()}
    matches = [name for name, dictionary in dictionaries.items()
               if all(dictionary.get(key) == value for key, value in labels.items())]
    if scheme != "auto" and scheme not in matches:
        raise ValueError(f"segmentation_label_scheme={scheme} contradicts the embedded {kind} label dictionary.")
    detected = scheme if scheme != "auto" else (matches[0] if len(matches) == 1 else "custom")
    # Tissue masks are a separate, stable 1/2/3 convention merged as 201/202/203.
    tissue_table = tables["current"]
    rows = [{"Order Number": key, "Organ Name": name} for key, name in labels.items()]
    rows += tissue_table.loc[tissue_table["Order Number"].isin([201, 202, 203]),
                             ["Order Number", "Organ Name"]].to_dict("records")
    return pd.DataFrame(rows), detected


def mr_output_parameters(scheme, source_scheme, input_kind, *, current=DEFAULT_MR_PARAMETERS,
                         legacy=DEFAULT_MR_LEGACY_PARAMETERS):
    """Auto preserves a recognized MR input scheme; CT -> MR defaults current."""
    validate_scheme(scheme)
    if scheme == "auto":
        scheme = source_scheme if input_kind == "mr" else "current"
        if scheme not in {"current", "legacy"}:
            raise ValueError("MR input dictionary is custom/ambiguous; set prior_label_scheme to current or legacy.")
    return (legacy if scheme == "legacy" else current), scheme
