"""Export tensor images as derived DICOM secondary-capture series."""

from datetime import datetime
from pathlib import Path

import numpy as np


def save_dicom_series(array, directory, *, patient_id, label, study_uid, series_number,
                      series_uid=None, instance_start=1, rescale_type="US"):
    """Write TIFF-oriented (Y, X) or (Z, Y, X) data as ordered slices.

    Floating values are quantized to unsigned 16-bit pixels with shared rescale
    metadata across the supplied batch. Streaming callers may reuse series_uid
    and advance instance_start. No physical geometry or source-scan metadata is
    invented. Callers may supply normalized tensors or restored intensities;
    rescale_type describes the decoded units (e.g. HU for restored CT values).
    """
    try:
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
        from pydicom.valuerep import format_number_as_ds
    except ImportError as error:
        raise ImportError("DICOM export requires pydicom. Install it with 'pip install pydicom'.") from error

    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3 or not values.size:
        raise ValueError("DICOM export expects a non-empty 2D image or 3D stack.")
    if not np.isfinite(values).all():
        raise ValueError("DICOM export cannot encode NaN or infinite pixel values.")
    if max(values.shape[1:]) > 65535:
        raise ValueError("DICOM rows and columns must be at most 65535.")

    minimum, maximum = float(values.min()), float(values.max())
    value_range = maximum - minimum
    if not np.isfinite(value_range):
        raise ValueError("DICOM pixel range is too large to rescale.")
    # Integers in a 16-bit range are encoded exactly, including negative values.
    integral = np.equal(values, np.rint(values)).all()
    slope = 1.0 if value_range == 0 or (integral and value_range <= 65535) else value_range / 65535
    # DICOM Decimal String values are limited to 16 characters.
    slope_text, intercept_text = format_number_as_ds(slope), format_number_as_ds(minimum)
    slope, intercept = float(slope_text), float(intercept_text)
    if slope <= 0:
        raise ValueError("DICOM rescale slope must be positive.")
    pixels = np.rint((values - intercept) / slope).clip(0, 65535).astype("<u2")

    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid if series_uid is not None else generate_uid()
    timestamp = datetime.now()
    for index, plane in enumerate(pixels):
        instance_number = instance_start + index
        filename = destination / f"slice_{instance_number:05d}.dcm"
        sop_uid = generate_uid()
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = sop_uid
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(filename), {}, file_meta=file_meta, preamble=b"\0" * 128)
        dataset.is_little_endian = True
        dataset.is_implicit_VR = False
        dataset.SOPClassUID = SecondaryCaptureImageStorage
        dataset.SOPInstanceUID = sop_uid
        dataset.StudyInstanceUID = study_uid
        dataset.SeriesInstanceUID = series_uid
        dataset.SpecificCharacterSet = "ISO_IR 192"
        dataset.PatientID = str(patient_id)
        dataset.PatientName = ""
        dataset.PatientBirthDate = ""
        dataset.PatientSex = ""
        dataset.StudyDate = timestamp.strftime("%Y%m%d")
        dataset.StudyTime = timestamp.strftime("%H%M%S")
        dataset.ContentDate = dataset.StudyDate
        dataset.ContentTime = dataset.StudyTime
        dataset.StudyID = ""
        dataset.AccessionNumber = ""
        dataset.ReferringPhysicianName = ""
        dataset.Modality = "OT"
        dataset.Manufacturer = "seg2med"
        dataset.ConversionType = "WSD"
        dataset.ImageType = ["DERIVED", "SECONDARY"]
        dataset.SeriesDescription = f"seg2med {label} tensor export"
        dataset.DerivationDescription = (
            "Derived tensor values; 16-bit rescaled pixels. Original physical geometry is unavailable."
        )
        dataset.SeriesNumber = int(series_number)
        dataset.InstanceNumber = instance_number
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = "MONOCHROME2"
        dataset.Rows, dataset.Columns = plane.shape
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 0
        dataset.RescaleSlope = slope_text
        dataset.RescaleIntercept = intercept_text
        dataset.RescaleType = rescale_type
        dataset.WindowCenter = format_number_as_ds(minimum + value_range / 2)
        dataset.WindowWidth = format_number_as_ds(max(value_range, 1.0))
        dataset.PixelData = plane.tobytes()
        dataset.save_as(str(filename), write_like_original=False)
