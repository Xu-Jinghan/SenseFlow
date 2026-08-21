import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]
READOUTS = [
    ("tia", "TIA", "#F6BD60"),
    ("integration", "Integration", "#84A59D"),
]
NOISE_STYLES = [
    ("no_noise", "No Noise", "", 0.34, 0.92),
    ("x100", "Noise x100", "//", 0.24, 0.85),
    ("x1000", "Noise x1000", "xx", 0.14, 0.78),
]

TITLE_SIZE = 32
LABEL_SIZE = 28
TICK_SIZE = 22
VALUE_SIZE = 18
SPINE_WIDTH = 2.4
BAR_EDGE_WIDTH = 1.4
Y_MAX = 112.0
VALUE_OFFSET = 1.1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a paper-style 2D overlay bar chart for case1 native baseline. "
            "TIA and Integration use fixed colors, while noise levels are distinguished by hatches."
        )
    )
    parser.add_argument(
        "--no-noise-csv",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_native_vs_x2_video_sequence"
            / "native_fps200_supplement"
            / "native_fps20_200_accuracy_records.csv"
        ),
    )
    parser.add_argument(
        "--x100-csv",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_native_noise_scale_sweep"
            / "noise_x100"
            / "case1_native_noise_x100_accuracy_records.csv"
        ),
    )
    parser.add_argument(
        "--x1000-csv",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_native_noise_scale_sweep"
            / "noise_x1000"
            / "case1_native_noise_x1000_accuracy_records.csv"
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_native_noise_scale_sweep"
            / "case1_native_noise_2d_overlay_tia_integration_paper_style.png"
        ),
    )
    return parser.parse_args()


def load_rows(csv_path, no_noise=False):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if no_noise and row.get("param_group") != "native":
                continue
            rows.append(row)
    return rows


def build_lookup(rows):
    return {
        (row["readout_label"], float(row["video_fps"])): float(row["accuracy_nonideal"])
        for row in rows
        if row["readout_label"] in {"tia", "integration"}
    }


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#222222")
    ax.tick_params(axis="both", width=SPINE_WIDTH, length=6, labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.16, linewidth=0.9)


def main():
    args = parse_args()

    lookups = {
        "no_noise": build_lookup(load_rows(args.no_noise_csv, no_noise=True)),
        "x100": build_lookup(load_rows(args.x100_csv)),
        "x1000": build_lookup(load_rows(args.x1000_csv)),
    }

    fig, ax = plt.subplots(1, 1, figsize=(15.8, 7.4))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.86)

    group_centers = np.arange(len(FPS_VALUES), dtype=np.float64) * 1.45
    readout_offsets = {
        "tia": -0.22,
        "integration": 0.22,
    }

    for readout_key, _readout_label, color in READOUTS:
        x_center = group_centers + readout_offsets[readout_key]
        for noise_key, _noise_label, hatch, width, alpha in NOISE_STYLES:
            values = [lookups[noise_key][(readout_key, fps)] for fps in FPS_VALUES]
            bars = ax.bar(
                x_center,
                values,
                width=width,
                color=color,
                edgecolor="#222222",
                linewidth=BAR_EDGE_WIDTH,
                hatch=hatch,
                alpha=alpha,
                zorder=3,
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() * 0.5,
                    value + VALUE_OFFSET,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=VALUE_SIZE,
                    fontweight="bold",
                    color="#111111",
                    zorder=5,
                )

    ax.set_title(
        "Case1 Native Baseline | TIA and Integration\nColor = Readout, Hatch = Noise Level",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"FPS {int(fps)}" for fps in FPS_VALUES], fontsize=TICK_SIZE, fontweight="bold")
    ax.set_xlabel("Video FPS", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("CIFAR-10 Accuracy", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylim(0.0, Y_MAX)
    style_axis(ax)

    readout_handles = [Patch(facecolor=color, edgecolor="#222222", label=label) for _, label, color in READOUTS]
    noise_handles = [
        Patch(facecolor="#FFFFFF", edgecolor="#222222", hatch=hatch, label=label)
        for _, label, hatch, _, _ in NOISE_STYLES
    ]

    readout_legend = ax.legend(
        handles=readout_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.98),
        frameon=False,
        title="Readout",
        title_fontsize=16,
        fontsize=15,
    )
    ax.add_artist(readout_legend)
    noise_legend = ax.legend(
        handles=noise_handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 0.77),
        frameon=False,
        title="Noise Level",
        title_fontsize=16,
        fontsize=15,
    )
    noise_legend._legend_box.align = "left"

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)
    print(f"2D overlay chart: {output_path}")


if __name__ == "__main__":
    main()
