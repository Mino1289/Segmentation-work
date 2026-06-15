from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_DATASETS = {
    "DINOv2": Path(__file__).with_name("metrics-dinov2.csv"),
    "DINOv3": Path(__file__).with_name("metrics-dinov3.csv"),
    "UPerNet": Path(__file__).with_name("metrics-upernet.csv"),
}


def load_metrics(csv_path: Path) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            row: dict[str, float | int | None] = {"epoch": int(raw_row["epoch"]) }
            for key, value in raw_row.items():
                if key == "epoch":
                    continue
                row[key] = float(value) if value not in (None, "") else None
            rows.append(row)
    return rows


def collect_metric_names(series: dict[str, list[dict[str, float | int | None]]]) -> list[str]:
    metric_names: set[str] = set()
    for rows in series.values():
        for row in rows:
            metric_names.update(
                key
                for key in row.keys()
                if key != "epoch" and row.get(key) is not None
            )

    preferred_order = ["train_loss", "train_acc", "val_acc", "val_mIoU", "val_biou"]
    ordered = [name for name in preferred_order if name in metric_names]
    ordered.extend(sorted(metric_names - set(ordered)))
    return ordered


def pretty_label(metric_name: str) -> str:
    labels = {
        "train_loss": "Train loss",
        "train_acc": "Train accuracy",
        "val_acc": "Validation accuracy",
        "val_mIoU": "Validation mIoU",
        "val_biou": "Validation boundary IoU",
    }
    return labels.get(metric_name, metric_name)


def plot_metrics(series: dict[str, list[dict[str, float | int | None]]], output: Path | None) -> None:
    metric_names = collect_metric_names(series)
    if not metric_names:
        raise ValueError("No metric values found in the provided CSV files.")

    n_metrics = len(metric_names)
    n_cols = 2 if n_metrics > 1 else 1
    n_rows = math.ceil(n_metrics / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.5 * n_cols, 4.2 * n_rows),
        sharex=True,
    )
    axes_list = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, metric_name in zip(axes_list, metric_names, strict=False):
        for model_name, rows in series.items():
            epochs = [row["epoch"] for row in rows if row.get(metric_name) is not None]
            values = [row[metric_name] for row in rows if row.get(metric_name) is not None]
            if epochs and values:
                axis.plot(epochs, values, marker="o", linewidth=2, markersize=4, label=model_name)

        axis.set_title(pretty_label(metric_name))
        axis.set_xlabel("Epoch")
        axis.set_ylabel(metric_name)
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)

    for axis in axes_list[len(metric_names):]:
        axis.axis("off")

    fig.suptitle("Training metrics comparison", fontsize=16, y=1.02)
    fig.tight_layout()

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output}")

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from the three training CSV files.")
    parser.add_argument(
        "--dinov2",
        type=Path,
        default=DEFAULT_DATASETS["DINOv2"],
        help="Path to metrics-dinov2.csv",
    )
    parser.add_argument(
        "--dinov3",
        type=Path,
        default=DEFAULT_DATASETS["DINOv3"],
        help="Path to metrics-dinov3.csv",
    )
    parser.add_argument(
        "--upernet",
        type=Path,
        default=DEFAULT_DATASETS["UPerNet"],
        help="Path to metrics-upernet.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("metrics_comparison.png"),
        help="Where to save the figure (PNG).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Display the figure without saving it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    series = {
        "DINOv2": load_metrics(args.dinov2),
        "DINOv3": load_metrics(args.dinov3),
        "UPerNet": load_metrics(args.upernet),
    }

    output = None if args.no_save else args.output
    plot_metrics(series, output)


if __name__ == "__main__":
    main()