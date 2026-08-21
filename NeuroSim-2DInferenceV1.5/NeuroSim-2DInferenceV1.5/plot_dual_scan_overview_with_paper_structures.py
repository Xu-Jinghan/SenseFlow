import argparse
import csv
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


STRUCTURE_COLORS = {
    "Photoconductor": "#D55E00",
    "Phototransistor": "#0072B2",
    "Photodiode": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create an 8-panel overview figure from two fixed-condition scans and overlay "
            "selected paper-device structures."
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
    parser.add_argument("--spatial-values", nargs="+", type=float, default=[1.0, 3.0])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[75.0, 85.0])
    return parser.parse_args()


def parse_float(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_paper_points(path, structures):
    requested = {str(structure).strip() for structure in structures}
    points = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entry_id = str(row.get("entry_id", "")).strip()
            if entry_id.startswith("default_") or entry_id.startswith("paper_stub_"):
                continue
            structure = str(row.get("structure", "")).strip()
            if structure not in requested:
                continue
            r_fast = parse_float(row.get("R_fast", ""))
            tau_rise = parse_float(row.get("tau_rise_fast", ""))
            tau_fall = parse_float(row.get("tau_fall_fast", ""))
            if r_fast is None or tau_rise is None or tau_fall is None:
                continue
            tau_avg = 0.5 * (tau_rise + tau_fall)
            if r_fast <= 0.0 or tau_avg <= 0.0:
                continue
            points.append(
                {
                    "entry_id": entry_id,
                    "specific_material": str(row.get("specific_material", "")).strip(),
                    "structure": structure,
                    "R_fast": r_fast,
                    "tau_avg_fast": tau_avg,
                }
            )
    return points


def apply_axis_style(ax, x_limits, y_limits, show_xlabel, show_ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(which="major", color="#D6D6D6", linewidth=0.75, alpha=0.7)
    ax.tick_params(length=4.0, width=1.1, labelsize=8.5)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    if show_xlabel:
        ax.set_xlabel("Responsivity R", fontsize=10)
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"$(t_r+t_f)/2$ (s)", fontsize=10)
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def add_panel_badge(ax, text):
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BFBFBF", "alpha": 0.94},
    )


def dataset_label(eta, noise, readout):
    return rf"$\eta={eta:.1f}$ | noise={noise:.0e} | {readout}"


def plot_overview(
    rows_a,
    rows_b,
    points_a,
    points_b,
    fps_values,
    spatial_values,
    thresholds,
    label_a,
    label_b,
    output_stem,
):
    panel_order = [(fps, spatial) for fps in fps_values for spatial in spatial_values]
    all_rows = rows_a + rows_b
    r_values = sorted({row["R_single"] for row in all_rows})
    tr_values = sorted({row["trise_tfall_equal_s"] for row in all_rows})
    x_mesh, y_mesh = np.meshgrid(r_values, tr_values)
    x_limits = (min(r_values), max(r_values))
    y_limits = (min(tr_values), max(tr_values))
    vmin = min(row["accuracy_nonideal_cifar10"] for row in all_rows)
    vmax = max(row["accuracy_nonideal_cifar10"] for row in all_rows)
    levels = np.linspace(vmin, vmax, 14)
    threshold_colors = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14.8, 7.6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    contourf = None
    row_configs = [
        {"rows": rows_a, "points": points_a, "label": label_a},
        {"rows": rows_b, "points": points_b, "label": label_b},
    ]

    for row_idx, config in enumerate(row_configs):
        for col_idx, (fps, spatial) in enumerate(panel_order):
            ax = axes[row_idx, col_idx]
            matrix = build_matrix(config["rows"], fps, spatial, r_values, tr_values)
            contourf = ax.contourf(
                x_mesh,
                y_mesh,
                matrix,
                levels=levels,
                cmap="viridis",
                extend="both",
            )
            contour = ax.contour(
                x_mesh,
                y_mesh,
                matrix,
                levels=levels[::2],
                colors="#202020",
                linewidths=0.72,
                alpha=0.60,
            )
            ax.clabel(contour, inline=True, fontsize=6.8, fmt="%.0f")
            for idx, threshold in enumerate(thresholds):
                boundary = ax.contour(
                    x_mesh,
                    y_mesh,
                    matrix,
                    levels=[threshold],
                    colors=[threshold_colors[idx % len(threshold_colors)]],
                    linewidths=1.9,
                )
                ax.clabel(boundary, inline=True, fontsize=6.9, fmt={threshold: f"{threshold:.0f}%"})

            if np.isfinite(matrix).any():
                best_idx = np.nanargmax(matrix)
                best_tr_idx, best_r_idx = np.unravel_index(best_idx, matrix.shape)
                ax.scatter(
                    [r_values[best_r_idx]],
                    [tr_values[best_tr_idx]],
                    marker="*",
                    s=180,
                    facecolor="white",
                    edgecolor="#111111",
                    linewidth=0.9,
                    zorder=4,
                )
                add_panel_badge(ax, f"best = {np.nanmax(matrix):.1f}%")

            for structure in sorted({point["structure"] for point in config["points"]}):
                subset = [point for point in config["points"] if point["structure"] == structure]
                ax.scatter(
                    [point["R_fast"] for point in subset],
                    [point["tau_avg_fast"] for point in subset],
                    marker="o",
                    s=44,
                    facecolors=STRUCTURE_COLORS.get(structure, "#6F6F6F"),
                    edgecolors="white",
                    linewidths=0.8,
                    alpha=0.95,
                    zorder=5,
                )

            apply_axis_style(
                ax,
                x_limits=x_limits,
                y_limits=y_limits,
                show_xlabel=(row_idx == 1),
                show_ylabel=(col_idx == 0),
            )
            ax.set_title(f"FPS={int(fps)} | Spatial={int(spatial)}%", fontsize=11, pad=7, fontweight="bold")

    fig.text(0.012, 0.74, label_a, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(0.012, 0.26, label_b, rotation=90, ha="center", va="center", fontsize=12, fontweight="bold")

    legend_handles = [
        Line2D([0], [0], color=threshold_colors[idx % len(threshold_colors)], linewidth=1.9, label=f"{threshold:.0f}% contour")
        for idx, threshold in enumerate(thresholds)
    ]
    structure_union = sorted({point["structure"] for point in (points_a + points_b)})
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7.2,
                markerfacecolor=STRUCTURE_COLORS.get(structure, "#6F6F6F"),
                markeredgecolor="white",
                label=structure,
            )
            for structure in structure_union
        ]
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="none",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor="#111111",
            label="Best scan point",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=min(6, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        title="Thresholds and paper structures",
        fontsize=9.0,
        title_fontsize=9.4,
    )
    fig.suptitle(
        "Overview Contours with Paper-Device Overlay",
        fontsize=15,
        fontweight="bold",
        y=1.05,
    )
    cbar = fig.colorbar(contourf, ax=axes.ravel().tolist(), fraction=0.02, pad=0.012)
    cbar.set_label("Accuracy (%)")
    cbar.ax.tick_params(labelsize=8.8, width=1.0, length=3.4)

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
        spatial_values=args.spatial_values,
        fps_values=args.fps_values,
    )
    rows_b = filter_rows(
        load_rows(args.aggregate_b),
        eta=args.eta_b,
        noise=args.noise_b,
        spatial_values=args.spatial_values,
        fps_values=args.fps_values,
    )

    points_a = load_paper_points(args.paper_csv, args.structures_a)
    points_b = load_paper_points(args.paper_csv, args.structures_b)

    plot_overview(
        rows_a=rows_a,
        rows_b=rows_b,
        points_a=points_a,
        points_b=points_b,
        fps_values=[float(value) for value in args.fps_values],
        spatial_values=[float(value) for value in args.spatial_values],
        thresholds=args.thresholds,
        label_a=dataset_label(args.eta_a, args.noise_a, args.readout_a),
        label_b=dataset_label(args.eta_b, args.noise_b, args.readout_b),
        output_stem=output_dir / "overview_dual_scan_paper_overlay",
    )

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
        "fps_values_hz": [float(value) for value in args.fps_values],
        "spatial_values_pct": [float(value) for value in args.spatial_values],
        "thresholds": args.thresholds,
        "artifacts": [
            "overview_dual_scan_paper_overlay.png",
            "overview_dual_scan_paper_overlay.pdf",
        ],
    }
    (output_dir / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
