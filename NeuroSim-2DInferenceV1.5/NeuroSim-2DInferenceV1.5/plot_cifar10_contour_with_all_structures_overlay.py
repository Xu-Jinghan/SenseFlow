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
    build_matrix,
    filter_rows,
    load_rows,
    save_png_pdf,
    set_paper_style,
    sorted_unique,
)


STRUCTURE_COLORS = {
    "Photoconductor": "#D55E00",
    "Phototransistor": "#0072B2",
    "Photodiode": "#009E73",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Overlay multiple paper-device structures on the paper-style CIFAR-10 contour grid "
            "at a fixed eta/noise slice."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--paper-csv", required=True)
    parser.add_argument("--supplement-csv", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--readout-label", default="integration")
    parser.add_argument(
        "--structures",
        nargs="+",
        default=["Photoconductor", "Phototransistor", "Photodiode"],
    )
    parser.add_argument("--spatial-values", nargs="+", type=float, default=None)
    parser.add_argument("--fps-values", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[75.0, 85.0])
    parser.add_argument("--axis-mode", default="paper_union", choices=["sweep", "paper_union"])
    parser.add_argument("--annotate-paper-points", type=int, default=0)
    return parser.parse_args()


def parse_float(value):
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def structure_slug(values):
    return "_".join(
        "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value).strip()).strip("_")
        for value in values
    )


def load_structure_points(path, structures):
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
                    "material_raw": str(row.get("material", "")).strip(),
                    "material_subcategory": str(row.get("material_subcategory", "")).strip(),
                    "specific_material": str(row.get("specific_material", "")).strip(),
                    "structure": structure,
                    "R_fast": r_fast,
                    "tau_rise_fast": tau_rise,
                    "tau_fall_fast": tau_fall,
                    "tau_avg_fast": tau_avg,
                }
            )
    if not points:
        raise ValueError("No valid paper points found for the requested structures.")
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


def apply_overlay_axis_style(ax, x_limits, y_limits, show_xlabel, show_ylabel):
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


def load_supplement_rows(path):
    if path is None:
        return []
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["video_fps"] = float(row["video_fps"])
            row["spatial_variation_r_pct"] = float(row["spatial_variation_r_pct"])
            row["R_single"] = float(row["R_single"])
            row["eta_single"] = float(row["eta_single"])
            row["trise_tfall_equal_s"] = float(row["trise_tfall_equal_s"])
            row["noise_1f_density_1hz_a_root_hz"] = float(row["noise_1f_density_1hz_a_root_hz"])
            row["accuracy_nonideal_cifar10"] = float(row["accuracy_nonideal_cifar10"])
            rows.append(row)
    return rows


def plot_overlay_contour_grid(
    rows,
    paper_points,
    supplement_rows,
    fps_values,
    spatial_values,
    r_values,
    tr_values,
    eta,
    noise,
    readout_label,
    thresholds,
    annotate_paper_points,
    output_stem,
):
    nrows = len(spatial_values)
    ncols = len(fps_values)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.15 * ncols + 0.8, 2.9 * nrows + 1.2),
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
    x_limits, y_limits = compute_axis_limits(r_values, tr_values, paper_points, "paper_union")
    in_sweep_points = [point for point in paper_points if point["in_sweep_range"]]
    label_offsets = [(6, 5), (7, -6), (-7, 6), (-8, -5), (9, 1), (-10, 1)]
    supplement_label_offsets = [(5, 5), (6, -6), (-6, 6), (-7, -5), (8, 0), (-8, 0)]

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

            for structure in sorted({point["structure"] for point in paper_points}):
                structure_points = [point for point in paper_points if point["structure"] == structure]
                ax.scatter(
                    [point["R_fast"] for point in structure_points],
                    [point["tau_avg_fast"] for point in structure_points],
                    marker="o",
                    s=68,
                    facecolors=STRUCTURE_COLORS.get(structure, "#6F6F6F"),
                    edgecolors="white",
                    linewidths=0.95,
                    alpha=0.96,
                    zorder=5,
                )

            if supplement_rows:
                subset = [
                    row
                    for row in supplement_rows
                    if row["video_fps"] == fps
                    and row["spatial_variation_r_pct"] == spatial
                ]
                if subset:
                    ax.scatter(
                        [row["R_single"] for row in subset],
                        [row["trise_tfall_equal_s"] for row in subset],
                        marker="s",
                        s=58,
                        c=[row["accuracy_nonideal_cifar10"] for row in subset],
                        cmap="viridis",
                        vmin=vmin,
                        vmax=vmax,
                        edgecolors="#111111",
                        linewidths=0.75,
                        alpha=0.98,
                        zorder=5.4,
                    )
                    for idx, row in enumerate(subset):
                        dx, dy = supplement_label_offsets[idx % len(supplement_label_offsets)]
                        ax.annotate(
                            f"{row['accuracy_nonideal_cifar10']:.0f}",
                            xy=(row["R_single"], row["trise_tfall_equal_s"]),
                            xytext=(dx, dy),
                            textcoords="offset points",
                            ha="center",
                            va="center",
                            fontsize=6.0,
                            color="#1A1A1A",
                            bbox={
                                "boxstyle": "round,pad=0.12",
                                "facecolor": "white",
                                "edgecolor": "#555555",
                                "alpha": 0.82,
                            },
                            zorder=5.8,
                        )

            if annotate_paper_points:
                for point in paper_points:
                    dx, dy = label_offsets[(point["paper_index"] - 1) % len(label_offsets)]
                    ax.annotate(
                        str(point["paper_index"]),
                        xy=(point["R_fast"], point["tau_avg_fast"]),
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        fontsize=7.0,
                        color="#5A1E00",
                        bbox={
                            "boxstyle": "round,pad=0.15",
                            "facecolor": "white",
                            "edgecolor": STRUCTURE_COLORS.get(point["structure"], "#AAAAAA"),
                            "alpha": 0.88,
                        },
                        zorder=6,
                    )

            apply_overlay_axis_style(
                ax,
                x_limits,
                y_limits,
                show_xlabel=(row_idx == nrows - 1),
                show_ylabel=(col_idx == 0),
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
    structure_counts = {
        structure: sum(1 for point in paper_points if point["structure"] == structure)
        for structure in sorted({point["structure"] for point in paper_points})
    }
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markersize=7.5,
                markerfacecolor=STRUCTURE_COLORS.get(structure, "#6F6F6F"),
                markeredgecolor="white",
                label=f"{structure} ({structure_counts[structure]})",
            )
            for structure in sorted(structure_counts)
        ]
    )
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="s",
                linestyle="none",
                markersize=7.2,
                markerfacecolor="#8E8E8E",
                markeredgecolor="#111111",
                label="Sparse supplement scan",
            ),
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
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        title="Thresholds and structure overlays",
        fontsize=9.0,
        title_fontsize=9.3,
    )
    fig.suptitle(
        f"CIFAR-10 Contours + Multi-Structure Paper Overlay | {readout_label} readout | "
        + rf"$\eta={eta:.1f}$ | noise={noise:.0e} A/$\sqrt{{Hz}}$",
        fontsize=14,
        fontweight="bold",
        y=1.09,
    )
    fig.text(
        0.5,
        1.01,
        f"Paper points use R_fast vs (tau_rise_fast + tau_fall_fast)/2. "
        f"{len(in_sweep_points)}/{len(paper_points)} lie inside the original computed sweep window."
        + (
            f" Added {len(supplement_rows)} sparse log-scan points with accuracy labels."
            if supplement_rows
            else ""
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
    supplement_rows = [
        row
        for row in load_supplement_rows(args.supplement_csv)
        if abs(row["eta_single"] - args.eta) <= 1e-12
        and abs(row["noise_1f_density_1hz_a_root_hz"] - args.noise) <= 1e-20
        and (
            args.fps_values is None
            or row["video_fps"] in {float(value) for value in args.fps_values}
        )
        and (
            args.spatial_values is None
            or row["spatial_variation_r_pct"] in {float(value) for value in args.spatial_values}
        )
    ]
    fps_values = sorted_unique(filtered, "video_fps")
    spatial_values = sorted_unique(filtered, "spatial_variation_r_pct")
    r_values = sorted_unique(filtered, "R_single")
    tr_values = sorted_unique(filtered, "trise_tfall_equal_s")

    paper_points = load_structure_points(args.paper_csv, args.structures)
    paper_points = assign_paper_indices(paper_points)
    paper_points = annotate_points_range(paper_points, r_values, tr_values)

    slug = structure_slug(args.structures)
    export_overlay_points(
        paper_points,
        output_dir / f"{slug}_paper_points_overlay_fullrange_by_structure.csv",
    )

    plot_overlay_contour_grid(
        rows=filtered,
        paper_points=paper_points,
        supplement_rows=supplement_rows,
        fps_values=fps_values,
        spatial_values=spatial_values,
        r_values=r_values,
        tr_values=tr_values,
        eta=args.eta,
        noise=args.noise,
        readout_label=args.readout_label,
        thresholds=args.thresholds,
        annotate_paper_points=bool(args.annotate_paper_points),
        output_stem=output_dir / (
            "c10_contour_grid_paper_with_all_structures_overlay_fullrange_with_sparse_supplement"
            if supplement_rows
            else "c10_contour_grid_paper_with_all_structures_overlay_fullrange"
        ),
    )

    structure_counts = {
        structure: sum(1 for point in paper_points if point["structure"] == structure)
        for structure in sorted({point["structure"] for point in paper_points})
    }
    metadata = {
        "aggregate_csv": args.aggregate_csv,
        "paper_csv": args.paper_csv,
        "supplement_csv": args.supplement_csv,
        "fixed_eta": args.eta,
        "fixed_noise": args.noise,
        "readout_label": args.readout_label,
        "structures": args.structures,
        "paper_point_definition": {
            "x": "R_fast",
            "y": "(tau_rise_fast + tau_fall_fast) / 2",
        },
        "color_by": "structure",
        "paper_points_total": len(paper_points),
        "paper_points_in_sweep_range": sum(1 for point in paper_points if point["in_sweep_range"]),
        "supplement_point_count": len(supplement_rows),
        "structure_counts": structure_counts,
        "artifacts": [
            (
                "c10_contour_grid_paper_with_all_structures_overlay_fullrange_with_sparse_supplement.png"
                if supplement_rows
                else "c10_contour_grid_paper_with_all_structures_overlay_fullrange.png"
            ),
            (
                "c10_contour_grid_paper_with_all_structures_overlay_fullrange_with_sparse_supplement.pdf"
                if supplement_rows
                else "c10_contour_grid_paper_with_all_structures_overlay_fullrange.pdf"
            ),
            f"{slug}_paper_points_overlay_fullrange_by_structure.csv",
        ],
    }
    readme_name = (
        "all_structures_overlay_fullrange_with_sparse_supplement_README.json"
        if supplement_rows
        else "all_structures_overlay_fullrange_README.json"
    )
    (output_dir / readme_name).write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
