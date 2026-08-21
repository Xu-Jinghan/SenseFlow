import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from plot_cifar10_fixed_eta_r_tr_noise_paper import (
    build_matrix,
    filter_rows,
    load_rows,
    set_paper_style,
)
from plot_dual_scan_overview_with_paper_structures import load_paper_points


TEXT_SCALE = 2.0

FPS_STYLES = {
    20.0: {"color": "#1F77B4", "linestyle": "solid", "label": "FPS=20"},
    100.0: {"color": "#D62728", "linestyle": "solid", "label": "FPS=100"},
}

STRUCTURE_STYLES = {
    "Photoconductor": {"color": "#2CA02C", "marker": "o", "size": 220},
    "Phototransistor": {"color": "#1F77B4", "marker": "^", "size": 220},
    "Photodiode": {"color": "#D62728", "marker": "s", "size": 220},
}

DEFAULT_LINEWIDTH = 1.9 * 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot only the 75% contour for the Spatial=3% slices and overlay selected "
            "paper-device structures. Each subplot combines FPS=20 and FPS=100."
        )
    )
    parser.add_argument("--aggregate-a", required=True)
    parser.add_argument("--eta-a", type=float, required=True)
    parser.add_argument("--noise-a", type=float, required=True)
    parser.add_argument("--readout-a", required=True)
    parser.add_argument("--structures-a", nargs="+", default=["Photoconductor", "Phototransistor"])
    parser.add_argument("--aggregate-b", required=True)
    parser.add_argument("--eta-b", type=float, required=True)
    parser.add_argument("--noise-b", type=float, required=True)
    parser.add_argument("--readout-b", required=True)
    parser.add_argument("--structures-b", nargs="+", default=["Photodiode"])
    parser.add_argument("--paper-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps-values", nargs="+", type=float, default=[20.0, 100.0])
    parser.add_argument("--spatial-value", type=float, default=3.0)
    parser.add_argument("--threshold", type=float, default=75.0)
    parser.add_argument("--linewidth", type=float, default=DEFAULT_LINEWIDTH)
    return parser.parse_args()


def dataset_title(eta, noise, readout, spatial_value):
    return rf"noise={noise:.0e} | Spatial={spatial_value:.0f}%"


def apply_axis_style(ax, x_limits, y_limits, show_xlabel, show_ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(which="major", color="#D6D6D6", linewidth=0.8, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(2.6)
    ax.tick_params(length=8.5, width=2.4, labelsize=10 * TEXT_SCALE)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    if show_xlabel:
        ax.set_xlabel("Responsivity R", fontsize=12 * TEXT_SCALE, fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("response time (s)", fontsize=12 * TEXT_SCALE, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(10 * TEXT_SCALE)
        tick_label.set_fontweight("bold")


def draw_threshold_contour(ax, rows, fps, spatial_value, threshold, r_values, tr_values, linewidth):
    matrix = build_matrix(rows, fps, spatial=spatial_value, r_values=r_values, tr_values=tr_values)
    style = FPS_STYLES.get(float(fps), {"color": "#202020", "linestyle": "solid"})
    contour = ax.contour(
        *np.meshgrid(r_values, tr_values),
        matrix,
        levels=[threshold],
        colors=[style["color"]],
        linewidths=linewidth,
        linestyles=[style["linestyle"]],
        zorder=3,
    )
    return contour


def plot_structure_points(ax, points, structures):
    for structure in structures:
        subset = [point for point in points if point["structure"] == structure]
        if not subset:
            continue
        style = STRUCTURE_STYLES[structure]
        ax.scatter(
            [point["R_fast"] for point in subset],
            [point["tau_avg_fast"] for point in subset],
            marker=style["marker"],
            s=style["size"],
            facecolors=style["color"],
            edgecolors="#202020",
            linewidths=0.7,
            alpha=0.96,
            zorder=4,
        )


def plot_overlay(rows_a, rows_b, points_a, points_b, fps_values, spatial_value, threshold, linewidth, output_stem, meta):
    all_rows = rows_a + rows_b
    r_values = sorted({row["R_single"] for row in all_rows})
    tr_values = sorted({row["trise_tfall_equal_s"] for row in all_rows})
    x_limits = (min(r_values), max(r_values))
    y_limits = (min(tr_values), max(tr_values))

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(18.0, 10.2),
        sharex=True,
        sharey=True,
    )

    row_configs = [
        {
            "ax": axes[0],
            "rows": rows_a,
            "points": points_a,
            "structures": meta["structures_a"],
            "title": dataset_title(meta["eta_a"], meta["noise_a"], meta["readout_a"], spatial_value),
        },
        {
            "ax": axes[1],
            "rows": rows_b,
            "points": points_b,
            "structures": meta["structures_b"],
            "title": dataset_title(meta["eta_b"], meta["noise_b"], meta["readout_b"], spatial_value),
        },
    ]

    for col_idx, config in enumerate(row_configs):
        ax = config["ax"]
        for fps in fps_values:
            draw_threshold_contour(
                ax=ax,
                rows=config["rows"],
                fps=fps,
                spatial_value=spatial_value,
                threshold=threshold,
                r_values=r_values,
                tr_values=tr_values,
                linewidth=linewidth,
            )
        plot_structure_points(ax, config["points"], config["structures"])
        apply_axis_style(
            ax=ax,
            x_limits=x_limits,
            y_limits=y_limits,
            show_xlabel=True,
            show_ylabel=(col_idx == 0),
        )
        ax.set_title(config["title"], fontsize=12 * TEXT_SCALE, fontweight="bold", pad=14)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=FPS_STYLES[20.0]["color"],
            linewidth=linewidth,
            linestyle=FPS_STYLES[20.0]["linestyle"],
            label=FPS_STYLES[20.0]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=FPS_STYLES[100.0]["color"],
            linewidth=linewidth,
            linestyle=FPS_STYLES[100.0]["linestyle"],
            label=FPS_STYLES[100.0]["label"],
        ),
        Line2D(
            [0],
            [0],
            marker=STRUCTURE_STYLES["Photoconductor"]["marker"],
            linestyle="none",
            markersize=18,
            markerfacecolor=STRUCTURE_STYLES["Photoconductor"]["color"],
            markeredgecolor="#202020",
            label="Photoconductor",
        ),
        Line2D(
            [0],
            [0],
            marker=STRUCTURE_STYLES["Phototransistor"]["marker"],
            linestyle="none",
            markersize=18,
            markerfacecolor=STRUCTURE_STYLES["Phototransistor"]["color"],
            markeredgecolor="#202020",
            label="Phototransistor",
        ),
        Line2D(
            [0],
            [0],
            marker=STRUCTURE_STYLES["Photodiode"]["marker"],
            linestyle="none",
            markersize=18,
            markerfacecolor=STRUCTURE_STYLES["Photodiode"]["color"],
            markeredgecolor="#202020",
            label="Photodiode",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10 * TEXT_SCALE,
    )
    fig.suptitle(
        "75% Contour Overlay at Spatial=3%",
        fontsize=16 * TEXT_SCALE,
        fontweight="bold",
        y=0.99,
    )
    fig.subplots_adjust(top=0.82, wspace=0.08)

    fig.savefig(str(output_stem) + ".png", dpi=320, bbox_inches="tight")
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    set_paper_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_a = filter_rows(
        load_rows(args.aggregate_a),
        eta=args.eta_a,
        noise=args.noise_a,
        spatial_values=[args.spatial_value],
        fps_values=args.fps_values,
    )
    rows_b = filter_rows(
        load_rows(args.aggregate_b),
        eta=args.eta_b,
        noise=args.noise_b,
        spatial_values=[args.spatial_value],
        fps_values=args.fps_values,
    )

    points_a = load_paper_points(args.paper_csv, args.structures_a)
    points_b = load_paper_points(args.paper_csv, args.structures_b)

    metadata = {
        "aggregate_a": args.aggregate_a,
        "eta_a": args.eta_a,
        "noise_a": args.noise_a,
        "readout_a": args.readout_a,
        "structures_a": args.structures_a,
        "aggregate_b": args.aggregate_b,
        "eta_b": args.eta_b,
        "noise_b": args.noise_b,
        "readout_b": args.readout_b,
        "structures_b": args.structures_b,
        "paper_csv": args.paper_csv,
        "fps_values_hz": [float(value) for value in args.fps_values],
        "spatial_value_pct": float(args.spatial_value),
        "threshold": float(args.threshold),
        "threshold_linewidth": float(args.linewidth),
        "linewidth_factor_from_original": float(args.linewidth / 1.9),
        "artifacts": [
            "dual_spatial3_threshold75_paper_overlay.png",
            "dual_spatial3_threshold75_paper_overlay.pdf",
        ],
    }

    plot_overlay(
        rows_a=rows_a,
        rows_b=rows_b,
        points_a=points_a,
        points_b=points_b,
        fps_values=[float(value) for value in args.fps_values],
        spatial_value=float(args.spatial_value),
        threshold=float(args.threshold),
        linewidth=float(args.linewidth),
        output_stem=output_dir / "dual_spatial3_threshold75_paper_overlay",
        meta=metadata,
    )

    (output_dir / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
