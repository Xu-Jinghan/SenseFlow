import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "restoration_case1native_noise1e-6_recovery_overview_20260415"

DEFAULT_SOURCES = {
    ("TIA", 0): REPO_ROOT / "outputs" / "restoration_case1native_noise1e-6_tia_fps_sweep_20260415" / "restoration_accuracy_summary.csv",
    ("Integration", 0): REPO_ROOT / "outputs" / "restoration_case1native_noise1e-6_integration_fps_sweep_20260415" / "restoration_accuracy_summary.csv",
    ("TIA", 5): REPO_ROOT / "outputs" / "restoration_case1native_noise1e-6_tia_spatialR5_fps_sweep_20260415" / "restoration_accuracy_summary.csv",
    ("Integration", 5): REPO_ROOT / "outputs" / "restoration_case1native_noise1e-6_integration_spatialR5_fps_sweep_20260415" / "restoration_accuracy_summary.csv",
}

READOUT_ORDER = ["TIA", "Integration"]
SPATIAL_ORDER = [0, 5]
FPS_ORDER = [20, 100, 200]

TITLE_SIZE = 28
SUBTITLE_SIZE = 17
LABEL_SIZE = 19
TICK_SIZE = 16
ANNOTATION_SIZE = 14
CAPTION_SIZE = 14
SPINE_WIDTH = 2.8
GRID_WIDTH = 1.3

BASELINE_COLOR = "#E9AFA3"
RESTORED_COLOR = "#87CBB9"
IDEAL_COLOR = "#7E8A97"
CONNECTOR_COLOR = "#586A75"
DELTA_COLOR = "#2F4858"
PANEL_FACE = "#FBFCFD"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine restoration results across readout and spatial-variation conditions "
            "and render a dumbbell-style recovery overview figure."
        )
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_rows(csv_path, readout, spatial_variation_pct):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for raw_row in csv.DictReader(handle):
            if spatial_variation_pct == 0:
                baseline_nonideal = float(raw_row["baseline_nonideal_acc"])
                restored_acc = float(raw_row["final_restored_acc"])
                restored_gain = float(raw_row["restored_minus_nonideal"])
                gap_recovery_ratio = float(raw_row["gap_recovery_ratio"])
                delta_vs_no_spatial = 0.0
            else:
                baseline_nonideal = float(raw_row["baseline_nonideal_acc_spatialR5"])
                restored_acc = float(raw_row["final_restored_acc_spatialR5"])
                restored_gain = float(raw_row["restored_minus_nonideal_spatialR5"])
                gap_recovery_ratio = float(raw_row["gap_recovery_ratio_spatialR5"])
                delta_vs_no_spatial = float(raw_row["delta_vs_no_spatial"])

            rows.append(
                {
                    "readout": readout,
                    "spatial_variation_pct": spatial_variation_pct,
                    "fps": int(float(raw_row["fps"])),
                    "raw_acc": float(raw_row["raw_acc"]),
                    "ideal_acc": float(raw_row["ideal_acc"]),
                    "baseline_nonideal_acc": baseline_nonideal,
                    "restored_acc": restored_acc,
                    "restored_gain": restored_gain,
                    "gap_recovery_ratio": gap_recovery_ratio,
                    "delta_vs_no_spatial": delta_vs_no_spatial,
                    "summary_json": raw_row["summary_json"],
                }
            )
    return rows


def build_rows():
    combined = []
    for key, csv_path in DEFAULT_SOURCES.items():
        readout, spatial_variation_pct = key
        combined.extend(load_rows(csv_path, readout, spatial_variation_pct))
    combined.sort(key=lambda row: (READOUT_ORDER.index(row["readout"]), row["spatial_variation_pct"], row["fps"]))
    return combined


def save_combined_csv(rows, output_path):
    fieldnames = [
        "readout",
        "spatial_variation_pct",
        "fps",
        "raw_acc",
        "ideal_acc",
        "baseline_nonideal_acc",
        "restored_acc",
        "restored_gain",
        "gap_recovery_ratio",
        "delta_vs_no_spatial",
        "summary_json",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def style_axis(ax):
    ax.set_facecolor(PANEL_FACE)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#27333A")
    ax.tick_params(axis="both", width=SPINE_WIDTH, length=6, labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="x", color="#C8D2D8", linewidth=GRID_WIDTH, alpha=0.65)
    ax.set_axisbelow(True)


def render_overview(rows, png_path, pdf_path):
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 10.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.86, hspace=0.28, wspace=0.16)

    for row_idx, readout in enumerate(READOUT_ORDER):
        for col_idx, spatial_variation_pct in enumerate(SPATIAL_ORDER):
            ax = axes[row_idx, col_idx]
            panel_rows = [
                row for row in rows
                if row["readout"] == readout and row["spatial_variation_pct"] == spatial_variation_pct
            ]
            panel_rows.sort(key=lambda row: FPS_ORDER.index(row["fps"]))
            y_positions = np.arange(len(panel_rows), dtype=np.float64)

            for y_pos, row in zip(y_positions, panel_rows):
                baseline = row["baseline_nonideal_acc"]
                restored = row["restored_acc"]
                ideal = row["ideal_acc"]
                gain = row["restored_gain"]

                ax.plot(
                    [baseline, restored],
                    [y_pos, y_pos],
                    color=CONNECTOR_COLOR,
                    linewidth=4.6,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.scatter(
                    [baseline],
                    [y_pos],
                    s=175,
                    facecolor="white",
                    edgecolor=BASELINE_COLOR,
                    linewidth=3.4,
                    zorder=4,
                )
                ax.scatter(
                    [restored],
                    [y_pos],
                    s=205,
                    facecolor=RESTORED_COLOR,
                    edgecolor="#284B63",
                    linewidth=2.4,
                    zorder=5,
                )
                ax.scatter(
                    [ideal],
                    [y_pos],
                    s=130,
                    marker="D",
                    facecolor=IDEAL_COLOR,
                    edgecolor="white",
                    linewidth=1.4,
                    zorder=6,
                )
                ax.text(
                    restored + 1.2,
                    y_pos,
                    f"+{gain:.1f}",
                    fontsize=ANNOTATION_SIZE,
                    fontweight="bold",
                    color=DELTA_COLOR,
                    va="center",
                    ha="left",
                )

            ax.set_title(
                f"{readout} | Spatial variation {spatial_variation_pct}%",
                fontsize=LABEL_SIZE,
                fontweight="bold",
            )
            ax.set_yticks(y_positions)
            ax.set_yticklabels([f"FPS {row['fps']}" for row in panel_rows], fontsize=TICK_SIZE, fontweight="bold")
            style_axis(ax)
            ax.invert_yaxis()

    for ax in axes[1, :]:
        ax.set_xlabel("Top-1 accuracy (%)", fontsize=LABEL_SIZE, fontweight="bold")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sensor setting", fontsize=LABEL_SIZE, fontweight="bold")

    all_values = [row["baseline_nonideal_acc"] for row in rows] + [row["restored_acc"] for row in rows] + [row["ideal_acc"] for row in rows]
    x_min = max(0.0, min(all_values) - 8.0)
    x_max = min(100.0, max(all_values) + 8.0)
    for ax in axes.ravel():
        ax.set_xlim(x_min, x_max)

    fig.suptitle(
        "Restoration Network Recovery Overview",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        y=0.96,
    )
    fig.text(
        0.5,
        0.91,
        "Open circle = degraded nonideal baseline | Filled circle = restored output | Diamond = ideal reference",
        ha="center",
        va="center",
        fontsize=SUBTITLE_SIZE,
        fontweight="bold",
        color="#42505A",
    )
    fig.text(
        0.5,
        0.05,
        "Line length and +value indicate how much classification accuracy is recovered by the restoration frontend.",
        ha="center",
        va="center",
        fontsize=CAPTION_SIZE,
        fontweight="bold",
        color="#42505A",
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    rows = build_rows()

    combined_csv = output_dir / "restoration_recovery_combined_summary.csv"
    png_path = output_dir / "restoration_recovery_overview.png"
    pdf_path = output_dir / "restoration_recovery_overview.pdf"

    save_combined_csv(rows, combined_csv)
    render_overview(rows, png_path, pdf_path)

    print(f"Combined summary CSV: {combined_csv}")
    print(f"Overview PNG: {png_path}")
    print(f"Overview PDF: {pdf_path}")


if __name__ == "__main__":
    main()
