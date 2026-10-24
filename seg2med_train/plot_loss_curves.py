"""Plot DDPM training/validation curves from epoch_metrics.csv or metric history.

Example:
    python -m seg2med_train.plot_loss_curves --csv logs/run/saved_logs/epoch_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


MONITOR_METRICS = ("val_patient_denoising_mse", "val_modality_patient_denoising_mse", "val_denoising_mse")
# Metrics from actually generated images (DDIM previews / full validation volumes).
SAMPLING_METRICS = ("val_volume_mae", "val_volume_brightness_std", "val_volume_slice_jump",
                    "preview_generated_image_mse")


def _number(value, column):
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value in {column}: {value!r}") from error
    return result if math.isfinite(result) else None


def _series(rows, column):
    points = [(row["epoch"], _number(row.get(column), column)) for row in rows]
    return [(epoch, value) for epoch, value in points if value is not None]


def _prepare_history(history, validation_metric=None):
    rows = [dict(row) for row in history]
    if not rows:
        raise ValueError("Metric history is empty.")
    columns = set().union(*(row.keys() for row in rows))
    if not {"epoch", "train_loss"}.issubset(columns):
        raise ValueError("Metrics must contain 'epoch' and 'train_loss' columns.")
    for row in rows:
        epoch = _number(row.get("epoch"), "epoch")
        if epoch is None or epoch < 0 or not epoch.is_integer():
            raise ValueError(f"Expected a nonnegative integer epoch, got {row.get('epoch')!r}.")
        row["epoch"] = int(epoch)
    rows.sort(key=lambda row: row["epoch"])
    if len({row["epoch"] for row in rows}) != len(rows):
        raise ValueError("Metrics contain duplicate epochs; resolve them before plotting.")
    if not _series(rows, "train_loss"):
        raise ValueError("No finite training loss values were found.")
    validation_columns = sorted(column for column in columns
                                if column.startswith("val_") and column.endswith("denoising_mse"))
    sampling_columns = [column for column in SAMPLING_METRICS if column in columns and _series(rows, column)]
    if validation_metric:
        if validation_metric not in validation_columns + sampling_columns:
            raise ValueError(f"Validation metric {validation_metric!r} is unavailable. "
                             f"Choose from: {validation_columns + sampling_columns}")
        selected = validation_metric
    else:
        monitors = {str(row["monitor"]).strip() for row in rows if row.get("monitor")}
        if len(monitors) > 1:
            raise ValueError("Monitoring metric changed within the CSV; specify --validation-metric explicitly.")
        selected = next(iter(monitors), None)
        if (selected and selected not in validation_columns and selected not in MONITOR_METRICS
                and selected not in SAMPLING_METRICS):
            raise ValueError(f"Unrecognized validation monitoring metric: {selected}")
        if selected is None:
            selected = next((column for column in MONITOR_METRICS if _series(rows, column)), None)
    return rows, validation_columns, sampling_columns, selected


def _line(axis, rows, column, *, label=None, **options):
    points = _series(rows, column)
    if points:
        epochs, losses = zip(*points)
        axis.plot(epochs, losses, label=label or column, marker=".", markersize=4, **options)
    return bool(points)


def _style(axis, title, ylabel):
    axis.set_title(title)
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    from matplotlib.ticker import MaxNLocator
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    if axis.get_legend_handles_labels()[0]:
        axis.legend(fontsize=8)


def _training_panel(axis, rows):
    _line(axis, rows, "train_loss", label="Training loss (total objective)", linewidth=1.8)
    mse = _series(rows, "train_denoising_mse")
    # With plain MSE training these are identical; avoid drawing two overlapping
    # lines. Auxiliary training losses remain distinguishable when enabled.
    if mse and mse != _series(rows, "train_loss"):
        _line(axis, rows, "train_denoising_mse", label="Training denoising MSE", linestyle="--")
    _style(axis, "Training loss", "Loss")


def _validation_panel(axis, rows, columns, selected):
    labels = {"val_denoising_mse": "Validation MSE (slice average)",
              "val_patient_denoising_mse": "Validation MSE (patient average)",
              "val_modality_patient_denoising_mse": "Validation MSE (equal-modality patient average)"}
    has_values = False
    for column in columns:
        is_monitor = column == selected
        label = labels.get(column, column)
        has_values |= _line(axis, rows, column, label=label + (" [monitor]" if is_monitor else ""),
                            linewidth=2.2 if is_monitor else 1.0,
                            linestyle="-" if is_monitor else "--", alpha=1.0 if is_monitor else 0.65)
    if not has_values:
        axis.text(0.5, 0.5, "No validation results yet", ha="center", va="center", transform=axis.transAxes)
    _style(axis, "Validation denoising loss", "Denoising MSE")


def _sampling_panel(axis, rows, columns, selected):
    labels = {"val_volume_mae": "Volume MAE (foreground)",
              "val_volume_brightness_std": "Slice brightness ratio std",
              "val_volume_slice_jump": "Mean slice-to-slice brightness jump",
              "preview_generated_image_mse": "Preview generated-image MSE"}
    for column in columns:
        is_monitor = column == selected
        _line(axis, rows, column, label=labels[column] + (" [monitor]" if is_monitor else ""),
              linewidth=2.2 if is_monitor else 1.0, linestyle="-" if is_monitor else "--")
    _style(axis, "Generated-image quality (lower is better)", "Metric")


def plot_loss_curves(history, output_dir, *, validation_metric=None, dpi=150):
    """Save the same three PNGs for live training and offline CSV plotting.

    Missing/nonfinite loss values are omitted, never replaced with zero. Values
    are plotted without smoothing. Preview generated-image MSE is not included
    in denoising loss plots. This function needs no model, CUDA, or display server.
    """
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    rows, columns, sampling_columns, selected = _prepare_history(history, validation_metric)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {"training": output / "training_loss_curve.png",
             "validation": output / "validation_loss_curve.png",
             "combined": output / "loss_curves.png"}
    if sampling_columns:
        paths["sampling"] = output / "sampling_quality_curve.png"
    for kind, path in paths.items():
        panels = 3 if sampling_columns else 2
        figure = Figure(figsize=(7.5 * panels, 5) if kind == "combined" else (9, 5), constrained_layout=True)
        FigureCanvasAgg(figure)
        try:
            if kind == "combined":
                axes = figure.subplots(1, panels)
                _training_panel(axes[0], rows)
                _validation_panel(axes[1], rows, columns, selected)
                if sampling_columns:
                    _sampling_panel(axes[2], rows, sampling_columns, selected)
            elif kind == "training":
                _training_panel(figure.subplots(), rows)
            elif kind == "sampling":
                _sampling_panel(figure.subplots(), rows, sampling_columns, selected)
            else:
                _validation_panel(figure.subplots(), rows, columns, selected)
            figure.savefig(path, dpi=dpi)
        finally:
            figure.clear()
    return paths


def plot_csv(csv_path, output_dir=None, *, validation_metric=None, dpi=150):
    csv_path = Path(csv_path)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        history = list(csv.DictReader(stream))
    return plot_loss_curves(history, output_dir if output_dir is not None else csv_path.parent,
                            validation_metric=validation_metric, dpi=dpi)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Existing epoch_metrics.csv file.")
    parser.add_argument("--output-dir", type=Path, help="Plot directory; defaults to the CSV's directory.")
    parser.add_argument("--validation-metric", help="Highlight this validation metric instead of the CSV's monitor.")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution (default: 150 DPI).")
    args = parser.parse_args(argv)
    try:
        paths = plot_csv(args.csv, args.output_dir, validation_metric=args.validation_metric, dpi=args.dpi)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    for path in paths.values():
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
