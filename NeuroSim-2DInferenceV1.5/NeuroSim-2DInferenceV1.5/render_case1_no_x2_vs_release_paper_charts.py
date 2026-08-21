import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]
READOUTS = [
    ("tia", "TIA", "#F6BD60"),
    ("integration", "Integration", "#84A59D"),
    ("adc4", "ADC 4-bit", "#A8DADC"),
    ("adc8", "ADC 8-bit", "#CDB4DB"),
]

TITLE_SIZE = 36
LABEL_SIZE = 32
TICK_SIZE = 26
VALUE_SIZE = 24
SPINE_WIDTH = 2.8
BAR_EDGE_WIDTH = 1.6
Y_MAX = 112.0
VALUE_OFFSET = 1.4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render paper-style case1 no-x2 vs x2-release bar charts and a combined side-by-side figure."
    )
    parser.add_argument(
        "--no-x2-csv",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_no_x2_lowprange"
            / "case1_x2_ion50_accuracy_records.csv"
        ),
    )
    parser.add_argument(
        "--x2-release-csv",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_x2_release_lowprange"
            / "case1_x2_ion50_accuracy_records.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "artifacts" / "case1_paper_charts"),
    )
    return parser.parse_args()


def load_records(csv_path):
    with Path(csv_path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def to_lookup(records):
    return {(float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"]) for r in records}


def style_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#222222")
    ax.tick_params(axis="both", width=SPINE_WIDTH, length=7, labelsize=TICK_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="y", alpha=0.18, linewidth=0.9)


def render_single_chart(lookup, title, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(12.6, 7.2), constrained_layout=True)
    x = np.arange(len(FPS_VALUES), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for idx, (key, label, color) in enumerate(READOUTS):
        values = [lookup[(fps, key)] for fps in FPS_VALUES]
        bars = ax.bar(
            x + offsets[idx],
            values,
            width=width,
            color=color,
            edgecolor="#222222",
            linewidth=BAR_EDGE_WIDTH,
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
            )

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"FPS {int(fps)}" for fps in FPS_VALUES], fontsize=TICK_SIZE, fontweight="bold")
    ax.set_xlabel("Video FPS", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylabel("CIFAR-10 Accuracy", fontsize=LABEL_SIZE, fontweight="bold")
    ax.set_ylim(0.0, Y_MAX)
    style_axis(ax)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_combined_chart(no_x2_lookup, x2_lookup, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(22.5, 7.2), sharey=True, constrained_layout=True)
    x = np.arange(len(FPS_VALUES), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for ax, title, lookup in [
        (axes[0], "W/O On-state-drift", no_x2_lookup),
        (axes[1], "W/ On-state-drift", x2_lookup),
    ]:
        for idx, (key, label, color) in enumerate(READOUTS):
            values = [lookup[(fps, key)] for fps in FPS_VALUES]
            bars = ax.bar(
                x + offsets[idx],
                values,
                width=width,
                color=color,
                edgecolor="#222222",
                linewidth=BAR_EDGE_WIDTH,
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
                )
        ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"FPS {int(fps)}" for fps in FPS_VALUES], fontsize=TICK_SIZE, fontweight="bold")
        ax.set_xlabel("Video FPS", fontsize=LABEL_SIZE, fontweight="bold")
        ax.set_ylim(0.0, Y_MAX)
        style_axis(ax)

    axes[0].set_ylabel("CIFAR-10 Accuracy", fontsize=LABEL_SIZE, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    no_x2_lookup = to_lookup(load_records(args.no_x2_csv))
    x2_lookup = to_lookup(load_records(args.x2_release_csv))

    no_x2_single = output_dir / "case1_no_x2_paper.png"
    x2_single = output_dir / "case1_x2_release_paper.png"
    combined = output_dir / "case1_no_x2_vs_release_paper.png"

    render_single_chart(no_x2_lookup, "W/O On-state-drift", no_x2_single)
    render_single_chart(x2_lookup, "W/ On-state-drift", x2_single)
    render_combined_chart(no_x2_lookup, x2_lookup, combined)

    print(f"No x2 chart: {no_x2_single}")
    print(f"x2 release chart: {x2_single}")
    print(f"Combined chart: {combined}")


if __name__ == "__main__":
    main()
