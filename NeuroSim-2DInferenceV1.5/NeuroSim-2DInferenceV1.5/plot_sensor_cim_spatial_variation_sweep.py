import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]

DEFAULT_SUMMARY_CSV = (
    REPO_ROOT
    / "outputs"
    / "case1_native_sensor_cim_sweep_100imgs_20260415"
    / "summary.csv"
)

CASE_ORDER = [
    (0, 0, "Sensor ideal + CIM ideal", "S0+C0", "#B8E0D2"),
    (0, 1, "Sensor ideal + CIM nonideal", "S0+C1", "#F9DCC4"),
    (1, 0, "Sensor nonideal + CIM ideal", "S1+C0", "#CDE7F0"),
    (1, 1, "Sensor nonideal + CIM nonideal", "S1+C1", "#E8D7F1"),
]

TITLE_SIZE = 26
LABEL_SIZE = 22
TICK_SIZE = 19
VALUE_SIZE = 18
CASE_LABEL_SIZE = 14
SPINE_WIDTH = 2.6
BAR_EDGE_WIDTH = 1.6
VALUE_OFFSET = 1.0
Y_TOP_MIN = 110.0
Y_HEADROOM = 12.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a grouped bar chart for the 100-image sensor/CIM nonideality sweep "
            "across spatial-variation settings."
        )
    )
    parser.add_argument(
        "--summary-csv",
        default=str(DEFAULT_SUMMARY_CSV),
        help="CSV generated from the sensor/CIM sweep.",
    )
    parser.add_argument(
        "--output-path",
        default="auto",
        help="Output PNG path. Defaults to the summary CSV directory.",
    )
    parser.add_argument(
        "--pdf-path",
        default="auto",
        help="Optional PDF output path. Defaults to the PNG path with a .pdf suffix.",
    )
    return parser.parse_args()


def resolve_output_paths(args, summary_csv):
    if args.output_path not in {"", "auto"}:
        png_path = Path(args.output_path).expanduser()
    else:
        png_path = summary_csv.with_name("sensor_cim_spatial_variation_grouped_bar.png")

    if args.pdf_path not in {"", "auto"}:
        pdf_path = Path(args.pdf_path).expanduser()
    else:
        pdf_path = png_path.with_suffix(".pdf")

    return png_path, pdf_path


def load_rows(csv_path):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "sensor_nonideal": int(row["sensor_nonideal"]),
                    "cim_nonideal": int(row["cim_nonideal"]),
                    "spatial_variation_r_pct": float(row["spatial_variation_r_pct"]),
                    "top1_accuracy": float(row["top1_accuracy"]),
                }
            )
    if not rows:
        raise ValueError(f"No rows found in summary CSV: {csv_path}")
    return rows


def build_lookup(rows):
    lookup = {}
    for row in rows:
        key = (
            row["spatial_variation_r_pct"],
            row["sensor_nonideal"],
            row["cim_nonideal"],
        )
        lookup[key] = row["top1_accuracy"]
    return lookup


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#222222")
    ax.tick_params(axis="both", width=SPINE_WIDTH, length=7, labelsize=TICK_SIZE)
    ax.grid(axis="y", alpha=0.18, linewidth=1.1, color="#8FA3B0")
    ax.set_axisbelow(True)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def render_chart(rows, png_path, pdf_path):
    lookup = build_lookup(rows)
    variation_values = sorted({row["spatial_variation_r_pct"] for row in rows})
    if not variation_values:
        raise ValueError("No spatial-variation values were found.")
    max_accuracy = max(row["top1_accuracy"] for row in rows)

    group_centers = np.arange(len(variation_values), dtype=np.float64) * 1.5
    width = 0.20
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    fig, ax = plt.subplots(1, 1, figsize=(13.0, 7.0))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.26, top=0.88)

    for idx, (sensor_nonideal, cim_nonideal, _label, short_label, color) in enumerate(CASE_ORDER):
        values = [
            lookup[(variation, sensor_nonideal, cim_nonideal)]
            for variation in variation_values
        ]
        bars = ax.bar(
            group_centers + offsets[idx],
            values,
            width=width,
            color=color,
            edgecolor="#222222",
            linewidth=BAR_EDGE_WIDTH,
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                value + VALUE_OFFSET,
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=VALUE_SIZE,
                fontweight="bold",
                rotation=90,
                color="#111111",
                zorder=5,
            )
            ax.text(
                bar.get_x() + bar.get_width() * 0.5,
                -0.12,
                short_label,
                ha="center",
                va="top",
                fontsize=CASE_LABEL_SIZE,
                fontweight="bold",
                color="#3E4C59",
                transform=ax.get_xaxis_transform(),
            )

    ax.set_title(
        "Sensor/CIM Sweep on 100 CIFAR-10 Samples",
        fontsize=TITLE_SIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Sensor spatial variation of responsivity R", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [f"{variation:g}%" for variation in variation_values],
        fontsize=TICK_SIZE,
        fontweight="bold",
    )
    ax.set_ylim(0.0, max(Y_TOP_MIN, max_accuracy + Y_HEADROOM))
    style_axis(ax)
    fig.text(
        0.5,
        0.06,
        "Within each group: S0+C0, S0+C1, S1+C0, S1+C1",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#46525C",
    )

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=320, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    summary_csv = Path(args.summary_csv).expanduser()
    rows = load_rows(summary_csv)
    png_path, pdf_path = resolve_output_paths(args, summary_csv)
    render_chart(rows, png_path, pdf_path)
    print(f"Grouped bar chart PNG: {png_path}")
    print(f"Grouped bar chart PDF: {pdf_path}")


if __name__ == "__main__":
    main()
