import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from plot_cifar10_fixed_eta_r_tr_noise_paper import (
    add_panel_badge,
    apply_axis_style,
    build_matrix,
    filter_rows,
    load_rows,
    save_png_pdf,
    set_paper_style,
    sorted_unique,
)

MATERIAL_COLORS = {
    "Perovskite": "#D55E00",
    "2D semiconductor": "#0072B2",
    "Oxide semiconductor": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay photoconductor-paper scatter points on top of the paper-style "
            "CIFAR-10 contour grid at a fixed eta/noise slice."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--paper-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--readout-label", default="integration")
    parser.add_argument("--structure", default="Photoconductor")
    parser.add_argument("--spatial-values", nargs="+", type=float, default=None)
    parser.add_argument("--fps-values", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[75.0, 85.0])
    parser.add_argument("--axis-mode", default="sweep", choices=["sweep", "paper_union"])
    parser.add_argument("--annotate-paper-points", type=int, default=0)
    parser.add_argument("--color-by", default="none", choices=["none", "material"])
    return parser.parse_args()


def parse_float(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_material_label(value):
    text = str(value).strip()
    if not text:
        return "Unknown"
    lowered = text.lower()
    if lowered == "perovskite":
        return "Perovskite"
    if lowered == "2d semiconductor":
        return "2D semiconductor"
    if lowered == "oxide semiconductor":
        return "Oxide semiconductor"
    return text


def structure_slug(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()).strip("_")


def structure_display_label(value):
    text = str(value).strip()
    return text or "Device"


def load_paper_points(path, structure):
    points = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            entry_id = str(row.get("entry_id", "")).strip()
            if entry_id.startswith("default_") or entry_id.startswith("paper_stub_"):
                continue
            if str(row.get("structure", "")).strip() != structure:
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
                    "material_raw": str(row.get("material", "")).strip(),
                    "material_subcategory": str(row.get("material_subcategory", "")).strip(),
                    "material_label": normalize_material_label(row.get("material", "")),
                    "specific_material": str(row.get("specific_material", "")).strip(),
                    "structure": structure,
                    "R_fast": r_fast,
                    "tau_rise_fast": tau_rise,
                    "tau_fall_fast": tau_fall,
                    "tau_avg_fast": tau_avg,
                }
            )
    if not points:
        raise ValueError(f"No valid paper points found for structure={structure}.")
    return points


def assign_paper_indices(points):
    indexed = []
    for idx, point in enumerate(points, start=1):
        indexed.append({**point, "paper_index": idx})
    return indexed


def annotate_points_range(points, r_values, tr_values):
    r_min, r_max = min(r_values), max(r_values)
    tr_min, tr_max = min(tr_values), max(tr_values)
    annotated = []
    for point in points:
        in_sweep = (
            r_min <= point["R_fast"] <= r_max
            and tr_min <= point["tau_avg_fast"] <= tr_max
        )
        annotated.append({**point, "in_sweep_range": in_sweep})
    return annotated


def compute_axis_limits(r_values, tr_values, paper_points, axis_mode):
    if axis_mode == "sweep":
        return (min(r_values), max(r_values)), (min(tr_values), max(tr_values))

    r_candidates = list(r_values) + [point["R_fast"] for point in paper_points]
    tr_candidates = list(tr_values) + [point["tau_avg_fast"] for point in paper_points]
    r_min = 10 ** (np.floor(np.log10(min(r_candidates))) - 0.05)
    r_max = 10 ** (np.ceil(np.log10(max(r_candidates))) + 0.05)
    tr_min = 10 ** (np.floor(np.log10(min(tr_candidates))) - 0.05)
    tr_max = 10 ** (np.ceil(np.log10(max(tr_candidates))) + 0.05)
    return (r_min, r_max), (tr_min, tr_max)


def apply_overlay_axis_style(
    ax,
    r_values,
    tr_values,
    x_limits,
    y_limits,
    show_xlabel,
    show_ylabel,
    axis_mode,
):
    if axis_mode == "sweep":
        apply_axis_style(ax, r_values, tr_values, show_xlabel, show_ylabel)
        return

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(which="major", color="#D6D6D6", linewidth=0.75, alpha=0.7)
    ax.tick_params(length=4.0, width=1.15)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    if show_xlabel:
        ax.set_xlabel("Responsivity R")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"$(t_r+t_f)/2$ (s)")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)


def export_overlay_points(points, output_csv):
    fieldnames = [
        "paper_index",
        "entry_id",
        "material_raw",
        "material_subcategory",
        "material_label",
        "specific_material",
        "structure",
        "R_fast",
        "tau_rise_fast",
        "tau_fall_fast",
        "tau_avg_fast",
        "in_sweep_range",
    ]
    with Path(output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(points)


def material_categories(points):
    categories = sorted({point["material_label"] for point in points})
    return categories


def color_for_point(point, color_by):
    if color_by == "material":
        return MATERIAL_COLORS.get(point["material_label"], "#6F6F6F")
    return "#FFF4B8"


def plot_overlay_contour_grid(
    rows,
    paper_points,
    fps_values,
    spatial_values,
    r_values,
    tr_values,
    eta,
    noise,
    readout_label,
    thresholds,
    axis_mode,
    annotate_paper_points,
    color_by,
    structure,
    output_stem,
):
    nrows = len(spatial_values)
    ncols = len(fps_values)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.15 * ncols + 0.8, 2.9 * nrows + 1.15),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    x_mesh, y_mesh = np.meshgrid(r_values, tr_values)
    vmin = min(row["accuracy_nonideal_cifar10"] for row in rows)
    vmax = max(row["accuracy_nonideal_cifar10"] for row in rows)
    levels = np.linspace(vmin, vmax, 14)
    contourf = None
    threshold_colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    x_limits, y_limits = compute_axis_limits(r_values, tr_values, paper_points, axis_mode)
    in_sweep_points = [point for point in paper_points if point["in_sweep_range"]]
    label_offsets = [(6, 5), (7, -6), (-7, 6), (-8, -5), (9, 1), (-10, 1)]
    structure_label = structure_display_label(structure)

    for row_idx, spatial in enumerate(spatial_values):
        for col_idx, fps in enumerate(fps_values):
            ax = axes[row_idx, col_idx]
            matrix = build_matrix(rows, fps, spatial, r_values, tr_values)
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
                linewidths=0.75,
                alpha=0.55,
            )
            ax.clabel(contour, inline=True, fontsize=7.2, fmt="%.0f")
            for idx, threshold in enumerate(thresholds):
                boundary = ax.contour(
                    x_mesh,
                    y_mesh,
                    matrix,
                    levels=[threshold],
                    colors=[threshold_colors[idx % len(threshold_colors)]],
                    linewidths=2.1,
                )
                ax.clabel(boundary, inline=True, fontsize=7.8, fmt={threshold: f"{threshold:.0f}%"})
            if np.isfinite(matrix).any():
                best_idx = np.nanargmax(matrix)
                best_tr_idx, best_r_idx = np.unravel_index(best_idx, matrix.shape)
                ax.scatter(
                    [r_values[best_r_idx]],
                    [tr_values[best_tr_idx]],
                    marker="*",
                    s=230,
                    facecolor="white",
                    edgecolor="#111111",
                    linewidth=0.9,
                    zorder=4,
                )
                add_panel_badge(ax, f"best = {np.nanmax(matrix):.1f}%")
            if axis_mode == "paper_union":
                ax.add_patch(
                    Rectangle(
                        (min(r_values), min(tr_values)),
                        max(r_values) - min(r_values),
                        max(tr_values) - min(tr_values),
                        fill=False,
                        linestyle=(0, (4, 2)),
                        linewidth=1.2,
                        edgecolor="#5A5A5A",
                        alpha=0.9,
                        zorder=4,
                    )
                )
            if paper_points:
                if color_by == "material":
                    for category in material_categories(paper_points):
                        category_points = [point for point in paper_points if point["material_label"] == category]
                        color = MATERIAL_COLORS.get(category, "#6F6F6F")
                        ax.scatter(
                            [point["R_fast"] for point in category_points],
                            [point["tau_avg_fast"] for point in category_points],
                            marker="o",
                            s=66,
                            facecolors=color,
                            edgecolors="white",
                            linewidths=0.95,
                            alpha=0.96,
                            zorder=5,
                        )
                else:
                    ax.scatter(
                        [point["R_fast"] for point in paper_points],
                        [point["tau_avg_fast"] for point in paper_points],
                        marker="o",
                        s=64,
                        facecolors="#FFF4B8",
                        edgecolors="#A50F15",
                        linewidths=1.05,
                        alpha=0.95,
                        zorder=5,
                    )
                if annotate_paper_points:
                    for point in paper_points:
                        dx, dy = label_offsets[(point["paper_index"] - 1) % len(label_offsets)]
                        edge_color = MATERIAL_COLORS.get(point["material_label"], "#D9A441") if color_by == "material" else "#D9A441"
                        ax.annotate(
                            str(point["paper_index"]),
                            xy=(point["R_fast"], point["tau_avg_fast"]),
                            xytext=(dx, dy),
                            textcoords="offset points",
                            ha="center",
                            va="center",
                            fontsize=7.2,
                            color="#7F0000",
                            bbox={
                                "boxstyle": "round,pad=0.15",
                                "facecolor": "white",
                                "edgecolor": edge_color,
                                "alpha": 0.88,
                            },
                            zorder=6,
                        )
            apply_overlay_axis_style(
                ax,
                r_values,
                tr_values,
                x_limits,
                y_limits,
                show_xlabel=(row_idx == nrows - 1),
                show_ylabel=(col_idx == 0),
                axis_mode=axis_mode,
            )
            if row_idx == 0:
                ax.set_title(f"FPS = {int(fps)} Hz", pad=8)

    for row_idx, spatial in enumerate(spatial_values):
        fig.text(
            0.01,
            1.0 - ((row_idx + 0.5) / nrows),
            f"Spatial {spatial:.0f}%",
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    legend_handles = [
        Line2D([0], [0], color=threshold_colors[idx % len(threshold_colors)], linewidth=2.1, label=f"{threshold:.0f}%")
        for idx, threshold in enumerate(thresholds)
    ]
    if color_by == "material":
        legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                    linestyle="none",
                    markersize=7.2,
                    markerfacecolor=MATERIAL_COLORS.get(category, "#6F6F6F"),
                    markeredgecolor="white",
                    label=category,
                )
                for category in material_categories(paper_points)
            ]
        )
    else:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7.5,
                markerfacecolor="#FFF4B8",
                markeredgecolor="#A50F15",
                label=f"Paper {structure_label} points",
            )
        )
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#5A5A5A",
                linewidth=1.2,
                linestyle=(0, (4, 2)),
                label="Computed sweep window",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="none",
                markersize=10,
                markerfacecolor="white",
                markeredgecolor="#111111",
                label="Best sweep point",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5 if color_by == "material" else 4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        title="Thresholds and overlays",
        fontsize=9.0,
        title_fontsize=9.3,
    )
    total_points = len(paper_points)
    in_sweep_count = len(in_sweep_points)
    fig.suptitle(
        f"CIFAR-10 Contours + {structure_label} Paper Overlay | {readout_label} readout | "
        + rf"$\eta={eta:.1f}$ | noise={noise:.0e} A/$\sqrt{{Hz}}$",
        fontsize=14,
        fontweight="bold",
        y=1.09,
    )
    fig.text(
        0.5,
        1.01,
        f"Paper points use R_fast vs (tau_rise_fast + tau_fall_fast)/2. "
        + (
            f"Colored by material. {in_sweep_count}/{total_points} lie inside the original computed sweep window."
            if color_by == "material"
            else f"{in_sweep_count}/{total_points} lie inside the original computed sweep window."
        ),
        ha="center",
        va="bottom",
        fontsize=9.4,
        color="#4A4A4A",
    )
    cbar = fig.colorbar(contourf, ax=axes.ravel().tolist(), fraction=0.024, pad=0.015)
    cbar.set_label("Accuracy (%)")
    cbar.ax.tick_params(labelsize=9.5, width=1.0, length=3.6)
    save_png_pdf(fig, output_stem)


def main():
    args = parse_args()
    set_paper_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.aggregate_csv)
    filtered = filter_rows(
        rows,
        eta=args.eta,
        noise=args.noise,
        spatial_values=args.spatial_values,
        fps_values=args.fps_values,
    )
    fps_values = sorted_unique(filtered, "video_fps")
    spatial_values = sorted_unique(filtered, "spatial_variation_r_pct")
    r_values = sorted_unique(filtered, "R_single")
    tr_values = sorted_unique(filtered, "trise_tfall_equal_s")

    paper_points = load_paper_points(args.paper_csv, args.structure)
    paper_points = assign_paper_indices(paper_points)
    paper_points = annotate_points_range(paper_points, r_values, tr_values)
    slug = structure_slug(args.structure)
    suffix = "_fullrange" if args.axis_mode == "paper_union" else ""
    if args.color_by == "material":
        suffix = f"{suffix}_by_material"
    export_overlay_points(paper_points, output_dir / f"{slug}_paper_points_overlay{suffix}.csv")

    plot_overlay_contour_grid(
        rows=filtered,
        paper_points=paper_points,
        fps_values=fps_values,
        spatial_values=spatial_values,
        r_values=r_values,
        tr_values=tr_values,
        eta=args.eta,
        noise=args.noise,
        readout_label=args.readout_label,
        thresholds=args.thresholds,
        axis_mode=args.axis_mode,
        annotate_paper_points=bool(args.annotate_paper_points),
        color_by=args.color_by,
        structure=args.structure,
        output_stem=output_dir / f"c10_contour_grid_paper_with_{slug}_overlay{suffix}",
    )

    material_counts = {}
    for category in material_categories(paper_points):
        material_counts[category] = sum(1 for point in paper_points if point["material_label"] == category)

    metadata = {
        "aggregate_csv": args.aggregate_csv,
        "paper_csv": args.paper_csv,
        "fixed_eta": args.eta,
        "fixed_noise": args.noise,
        "readout_label": args.readout_label,
        "paper_structure": args.structure,
        "paper_point_definition": {
            "x": "R_fast",
            "y": "(tau_rise_fast + tau_fall_fast) / 2",
        },
        "axis_mode": args.axis_mode,
        "color_by": args.color_by,
        "paper_points_total": len(paper_points),
        "paper_points_in_sweep_range": sum(1 for point in paper_points if point["in_sweep_range"]),
        "material_counts": material_counts,
        "artifacts": [
            f"c10_contour_grid_paper_with_{slug}_overlay{suffix}.png",
            f"c10_contour_grid_paper_with_{slug}_overlay{suffix}.pdf",
            f"{slug}_paper_points_overlay{suffix}.csv",
        ],
    }
    readme_name = f"{slug}_overlay{suffix}_README.json"
    (output_dir / readme_name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
