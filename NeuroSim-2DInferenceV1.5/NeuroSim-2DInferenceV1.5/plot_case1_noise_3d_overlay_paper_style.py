import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D, proj3d  # noqa: F401


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]
NOISE_CONDITIONS = [
    ("x1000", "Noise x1000"),
    ("x100", "Noise x100"),
    ("no_noise", "No Noise"),
]
COMBINED_NOISE_CONDITIONS = [
    ("x1000", "1e-6 A/Hz^0.5"),
    ("x100", "1e-7 A/Hz^0.5"),
    ("x10", "1e-9 A/Hz^0.5"),
]
READOUT_STYLES = {
    "tia": {
        "title": "Case1 Native Baseline TIA Accuracy",
        "zlabel": "TIA Accuracy (%)",
        "color": "#F8D89B",
        "output_name": "case1_native_noise_3d_overlay_tia_paper_style.png",
    },
    "integration": {
        "title": "Case1 Native Baseline Integration Accuracy",
        "zlabel": "Integration Accuracy (%)",
        "color": "#B8D6CC",
        "output_name": "case1_native_noise_3d_overlay_integration_paper_style.png",
    },
}
ALPHAS = {
    "x1000": 0.42,
    "x100": 0.66,
    "no_noise": 0.92,
}
COMBINED_ALPHA = 1.0
COMBINED_OUTPUT_NAME = "case1_native_noise_3d_overlay_tia_integration_paper_style.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render paper-style 3D overlay charts for case1 native baseline, using one hue per readout "
            "and transparency to indicate the noise condition."
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
        "--x10-csv",
        default=str(
            Path(__file__).resolve().parent
            / "artifacts"
            / "case1_native_noise_scale_sweep"
            / "noise_x10"
            / "case1_native_noise_x10_accuracy_records.csv"
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
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "artifacts" / "case1_native_noise_scale_sweep"),
    )
    return parser.parse_args()


def load_rows(csv_path, readout_label, no_noise=False):
    rows = []
    with Path(csv_path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("readout_label") != readout_label:
                continue
            if no_noise and row.get("param_group") != "native":
                continue
            rows.append(row)
    return rows


def to_lookup(rows):
    return {float(row["video_fps"]): float(row["accuracy_nonideal"]) for row in rows}


def build_noise_legend_handles(base_color):
    return [
        Patch(facecolor=base_color, edgecolor="#222222", alpha=ALPHAS[key], label=label)
        for key, label in NOISE_CONDITIONS
    ]


def draw_readout_overlay(ax, style, lookups, title_suffix=None):
    group_spacing = 1.75
    dx = 0.55
    dy = 0.72
    x_positions = np.arange(len(FPS_VALUES), dtype=np.float64) * group_spacing
    y_positions = {
        "x1000": 0.00,
        "x100": 1.05,
        "x10": 2.10,
        "no_noise": 2.10,
    }

    for cond_key, _cond_label in NOISE_CONDITIONS:
        xs = x_positions
        ys = np.full_like(xs, y_positions[cond_key])
        dz = np.array([lookups[cond_key][fps] for fps in FPS_VALUES], dtype=np.float64)
        ax.bar3d(
            xs,
            ys,
            np.zeros_like(xs),
            dx,
            dy,
            dz,
            color=style["color"],
            alpha=ALPHAS[cond_key],
            edgecolor="#222222",
            linewidth=0.95,
            shade=True,
            zsort="average",
        )
        for x, y, z in zip(xs, ys, dz):
            ax.text(
                x + dx * 0.5,
                y + dy * 0.5,
                z + 1.2,
                f"{z:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#111111",
            )

    title = style["title"]
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    ax.set_title(
        title,
        pad=26,
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xticks(x_positions + dx * 0.5)
    ax.set_xticklabels([f"FPS {int(fps)}" for fps in FPS_VALUES], fontsize=11, fontweight="bold")
    ax.set_xlabel("Video FPS", labelpad=16, fontsize=12, fontweight="bold")

    ytick_centers = [y_positions[key] + dy * 0.5 for key, _ in NOISE_CONDITIONS]
    ax.set_yticks(ytick_centers)
    ax.set_yticklabels([label for _key, label in NOISE_CONDITIONS], fontsize=11, fontweight="bold")
    ax.set_ylabel("Noise Condition", labelpad=18, fontsize=12, fontweight="bold")

    ax.set_zlabel(style["zlabel"], labelpad=12, fontsize=12, fontweight="bold")
    ax.set_zlim(0.0, 100.0)
    ax.set_zticks(np.arange(0, 101, 10))
    ax.tick_params(axis="z", labelsize=10, width=1.2)
    ax.view_init(elev=26, azim=-56)
    ax.set_box_aspect((4.4, 2.4, 2.9))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_alpha(0.08)
        axis.pane.set_edgecolor("#999999")


def overlay_projected_text_labels(ax, labels):
    ax.figure.canvas.draw()
    for x, y, z, text in labels:
        x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
        ax.annotate(
            text,
            xy=(x2, y2),
            xycoords="data",
            xytext=(0, 0),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
            color="#111111",
            annotation_clip=False,
            zorder=1000,
        )


def render_readout_overlay(readout_label, style, lookups, output_path):
    fig = plt.figure(figsize=(12.8, 9.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.90)
    draw_readout_overlay(ax, style, lookups, title_suffix="Front to back: Noise x1000, Noise x100, No Noise")

    legend = ax.legend(
        handles=build_noise_legend_handles(style["color"]),
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        title="Noise Level",
    )
    legend._legend_box.align = "left"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def render_combined_overlay(output_path, all_lookups):
    fig = plt.figure(figsize=(16.8, 10.2))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.03, right=0.98, bottom=0.08, top=0.90)

    group_spacing = 1.65
    readout_spacing = 0.78
    dx = 0.42
    dy = 0.68
    y_positions = {
        "x1000": 0.00,
        "x100": 1.05,
        "x10": 2.10,
    }
    x_group_starts = np.arange(len(FPS_VALUES), dtype=np.float64) * (group_spacing + readout_spacing)
    x_positions_by_readout = {
        "tia": x_group_starts,
        "integration": x_group_starts + readout_spacing,
    }
    projected_labels = []

    for readout_label in ["tia", "integration"]:
        style = READOUT_STYLES[readout_label]
        for cond_key, _cond_label in COMBINED_NOISE_CONDITIONS:
            xs = x_positions_by_readout[readout_label]
            ys = np.full_like(xs, y_positions[cond_key])
            dz = np.array([all_lookups[readout_label][cond_key][fps] for fps in FPS_VALUES], dtype=np.float64)
            ax.bar3d(
                xs,
                ys,
                np.zeros_like(xs),
                dx,
                dy,
                dz,
                color=style["color"],
                alpha=COMBINED_ALPHA,
                edgecolor="#222222",
                linewidth=0.9,
                shade=True,
                zsort="average",
            )
            for x, y, z in zip(xs, ys, dz):
                projected_labels.append((x + dx * 0.5, y + dy * 0.5, min(z + 1.0, 103.0), f"{z:.1f}"))

    ax.set_title(
        "Case1 Native Baseline Accuracy",
        pad=26,
        fontsize=24,
        fontweight="bold",
    )

    xticks = []
    xticklabels = []
    for fps, tia_x, integration_x in zip(FPS_VALUES, x_positions_by_readout["tia"], x_positions_by_readout["integration"]):
        xticks.append((tia_x + integration_x) * 0.5 + dx * 0.5)
        xticklabels.append(f"FPS {int(fps)}")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=18, fontweight="bold")
    ax.set_xlabel("Video FPS", labelpad=24, fontsize=20, fontweight="bold")

    ytick_centers = [y_positions[key] + dy * 0.5 for key, _ in COMBINED_NOISE_CONDITIONS]
    ax.set_yticks(ytick_centers)
    ax.set_yticklabels([label for _key, label in COMBINED_NOISE_CONDITIONS], fontsize=16, fontweight="bold")
    ax.set_zlabel("Accuracy (%)", labelpad=18, fontsize=20, fontweight="bold")
    ax.set_zlim(0.0, 104.0)
    ax.set_zticks(np.arange(0, 101, 10))
    ax.tick_params(axis="x", pad=4)
    ax.tick_params(axis="y", pad=22)
    ax.tick_params(axis="z", labelsize=16, width=1.4)
    ax.view_init(elev=24, azim=-62)
    ax.set_box_aspect((6.4, 2.7, 2.9))

    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_alpha(0.08)
        axis.pane.set_edgecolor("#999999")

    overlay_projected_text_labels(ax, projected_labels)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=320, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    all_lookups = {}

    for readout_label, style in READOUT_STYLES.items():
        lookups = {
            "no_noise": to_lookup(load_rows(args.no_noise_csv, readout_label, no_noise=True)),
            "x10": to_lookup(load_rows(args.x10_csv, readout_label)),
            "x100": to_lookup(load_rows(args.x100_csv, readout_label)),
            "x1000": to_lookup(load_rows(args.x1000_csv, readout_label)),
        }
        all_lookups[readout_label] = lookups
        output_path = output_dir / style["output_name"]
        render_readout_overlay(readout_label, style, lookups, output_path)
        print(f"{readout_label} chart: {output_path}")

    combined_output = output_dir / COMBINED_OUTPUT_NAME
    render_combined_overlay(combined_output, all_lookups)
    print(f"combined chart: {combined_output}")


if __name__ == "__main__":
    main()
