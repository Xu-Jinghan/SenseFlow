import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator

from plot_cifar10_fixed_eta_r_tr_noise_paper import (
    build_matrix,
    filter_rows,
    load_rows,
    set_paper_style,
)


DEFAULT_MINOR_CONTOURS = [70.0, 80.0, 90.0, 95.0]
HIGHLIGHT_CONTOUR_75 = 75.0
HIGHLIGHT_CONTOUR_85 = 85.0
HIGHLIGHT_CONTOUR_LINEWIDTH = 7.2
TEXT_SCALE = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render a subset of the dual-scan overview panels without paper-device overlays. "
            "This is useful for isolating the left/right half of the overview figure."
        )
    )
    parser.add_argument("--aggregate-a", required=True)
    parser.add_argument("--eta-a", type=float, required=True)
    parser.add_argument("--noise-a", type=float, required=True)
    parser.add_argument("--readout-a", required=True)
    parser.add_argument("--aggregate-b", required=True)
    parser.add_argument("--eta-b", type=float, required=True)
    parser.add_argument("--noise-b", type=float, required=True)
    parser.add_argument("--readout-b", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps-values", nargs="+", type=float, default=[20.0])
    parser.add_argument("--spatial-values", nargs="+", type=float, default=[1.0, 3.0])
    parser.add_argument("--minor-contours", nargs="+", type=float, default=DEFAULT_MINOR_CONTOURS)
    parser.add_argument("--title", default="Overview Contours Without Paper Overlay")
    parser.add_argument("--vmin", type=float, default=50.0)
    parser.add_argument("--vmax", type=float, default=100.0)
    parser.add_argument("--label-contours", action="store_true")
    return parser.parse_args()


def dataset_label(eta, noise, readout):
    return rf"$\eta={eta:.1f}$ | noise={noise:.0e} | {readout}"


def make_soft_coolwarm():
    base = plt.get_cmap("coolwarm")

    def soften(color, white_mix):
        rgb = np.array(color[:3], dtype=float)
        softened = rgb * (1.0 - white_mix) + white_mix
        return tuple(softened.tolist()) + (1.0,)

    return LinearSegmentedColormap.from_list(
        "soft_coolwarm",
        [
            soften(base(0.12), 0.52),
            soften(base(0.50), 0.82),
            soften(base(0.88), 0.50),
        ],
    )


def apply_axis_style(ax, x_limits, y_limits, show_xlabel, show_ylabel):
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.grid(which="major", color="#D6D6D6", linewidth=0.75, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
    ax.tick_params(length=8.0, width=2.2, labelsize=8.5 * TEXT_SCALE)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    if show_xlabel:
        ax.set_xlabel("Responsivity R", fontsize=10 * TEXT_SCALE, fontweight="bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel("response time (s)", fontsize=10 * TEXT_SCALE, fontweight="bold")
    else:
        ax.set_ylabel("")
        ax.tick_params(labelleft=False)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontsize(8.5 * TEXT_SCALE)
        tick_label.set_fontweight("bold")


def add_panel_badge(ax, text):
    ax.text(
        0.03,
        0.96,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6 * TEXT_SCALE,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#BFBFBF", "alpha": 0.94},
    )


def draw_contour(ax, x_mesh, y_mesh, matrix, level, color, linewidth, linestyle):
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return None
    min_value = float(np.min(finite))
    max_value = float(np.max(finite))
    if not (min_value <= level <= max_value):
        return None

    contour = ax.contour(
        x_mesh,
        y_mesh,
        matrix,
        levels=[level],
        colors=[color],
        linewidths=linewidth,
        linestyles=[linestyle],
        zorder=4,
    )
    return contour


def select_label_anchor(contour, label_index, label_count):
    segments = [segment for segment in contour.allsegs[0] if len(segment) >= 2]
    if not segments:
        return None

    def segment_length(segment):
        diffs = np.diff(segment, axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

    segment = max(segments, key=segment_length)
    points = segment[1:-1] if len(segment) > 3 else segment
    if len(points) == 0:
        points = segment

    log_x = np.log10(points[:, 0])
    x_min = float(np.min(log_x))
    x_max = float(np.max(log_x))
    fractions = np.linspace(0.42, 0.82, max(1, label_count))
    fraction = float(fractions[min(label_index, len(fractions) - 1)])
    target_log_x = x_min + fraction * (x_max - x_min)
    point_index = int(np.argmin(np.abs(log_x - target_log_x)))
    return tuple(points[point_index])


def add_contour_label(ax, contour, level, color, label_index, label_count):
    anchor = select_label_anchor(contour, label_index, label_count)
    if anchor is None:
        return None

    fontweight = "semibold" if level in {HIGHLIGHT_CONTOUR_75, HIGHLIGHT_CONTOUR_85} else "normal"
    text = ax.text(
        anchor[0],
        anchor[1],
        f"{level:.0f}%",
        color=color,
        fontsize=6.9 * TEXT_SCALE,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=6,
    )
    text.set_path_effects([path_effects.withStroke(linewidth=2.2, foreground="white", alpha=0.92)])
    return text


def plot_subset(
    rows_a,
    rows_b,
    fps_values,
    spatial_values,
    minor_contours,
    label_a,
    label_b,
    title,
    vmin,
    vmax,
    label_contours,
    output_stem,
):
    panel_order = [(fps, spatial) for fps in fps_values for spatial in spatial_values]
    all_rows = rows_a + rows_b
    r_values = sorted({row["R_single"] for row in all_rows})
    tr_values = sorted({row["trise_tfall_equal_s"] for row in all_rows})
    x_mesh, y_mesh = np.meshgrid(r_values, tr_values)
    x_limits = (min(r_values), max(r_values))
    y_limits = (min(tr_values), max(tr_values))
    fill_levels = np.linspace(vmin, vmax, 21)
    cmap = make_soft_coolwarm()
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        2,
        len(panel_order),
        figsize=(4.7 * len(panel_order) + 2.0, 11.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(2, len(panel_order))

    contourf = None
    row_configs = [
        {"rows": rows_a, "label": label_a},
        {"rows": rows_b, "label": label_b},
    ]
    all_contour_levels = [*minor_contours, HIGHLIGHT_CONTOUR_75, HIGHLIGHT_CONTOUR_85]
    sorted_label_levels = sorted(all_contour_levels)
    label_index_lookup = {level: idx for idx, level in enumerate(sorted_label_levels)}

    for row_idx, config in enumerate(row_configs):
        for col_idx, (fps, spatial) in enumerate(panel_order):
            ax = axes[row_idx, col_idx]
            matrix = build_matrix(config["rows"], fps, spatial, r_values, tr_values)
            contourf = ax.contourf(
                x_mesh,
                y_mesh,
                matrix,
                levels=fill_levels,
                cmap=cmap,
                norm=norm,
                extend="both",
            )

            for level in minor_contours:
                contour = draw_contour(
                    ax=ax,
                    x_mesh=x_mesh,
                    y_mesh=y_mesh,
                    matrix=matrix,
                    level=level,
                    color="#C6C6C6",
                    linewidth=0.9,
                    linestyle="solid",
                )
                if label_contours and contour is not None:
                    add_contour_label(
                        ax=ax,
                        contour=contour,
                        level=level,
                        color="#9A9A9A",
                        label_index=label_index_lookup[level],
                        label_count=len(sorted_label_levels),
                    )

            contour_75 = draw_contour(
                ax=ax,
                x_mesh=x_mesh,
                y_mesh=y_mesh,
                matrix=matrix,
                level=HIGHLIGHT_CONTOUR_75,
                color="#D84B3C",
                linewidth=HIGHLIGHT_CONTOUR_LINEWIDTH,
                linestyle="dashed",
            )
            if label_contours and contour_75 is not None:
                add_contour_label(
                    ax=ax,
                    contour=contour_75,
                    level=HIGHLIGHT_CONTOUR_75,
                    color="#C73D32",
                    label_index=label_index_lookup[HIGHLIGHT_CONTOUR_75],
                    label_count=len(sorted_label_levels),
                )

            contour_85 = draw_contour(
                ax=ax,
                x_mesh=x_mesh,
                y_mesh=y_mesh,
                matrix=matrix,
                level=HIGHLIGHT_CONTOUR_85,
                color="#D84B3C",
                linewidth=HIGHLIGHT_CONTOUR_LINEWIDTH,
                linestyle="solid",
            )
            if label_contours and contour_85 is not None:
                add_contour_label(
                    ax=ax,
                    contour=contour_85,
                    level=HIGHLIGHT_CONTOUR_85,
                    color="#C73D32",
                    label_index=label_index_lookup[HIGHLIGHT_CONTOUR_85],
                    label_count=len(sorted_label_levels),
                )

            if np.isfinite(matrix).any():
                add_panel_badge(ax, f"best = {np.nanmax(matrix):.1f}%")

            apply_axis_style(
                ax,
                x_limits=x_limits,
                y_limits=y_limits,
                show_xlabel=(row_idx == 1),
                show_ylabel=(col_idx == 0),
            )
            ax.set_title(
                f"FPS={int(fps)} | Spatial={int(spatial)}%",
                fontsize=11 * TEXT_SCALE,
                pad=10,
                fontweight="bold",
            )

    legend_handles = [
        Line2D([0], [0], color="#D84B3C", linewidth=HIGHLIGHT_CONTOUR_LINEWIDTH, linestyle="dashed", label="75% contour"),
        Line2D([0], [0], color="#D84B3C", linewidth=HIGHLIGHT_CONTOUR_LINEWIDTH, linestyle="solid", label="85% contour"),
        Line2D([0], [0], color="#C6C6C6", linewidth=0.9, linestyle="solid", label="70/80/90/95% contours"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.055),
        fontsize=8.6 * TEXT_SCALE,
    )
    fig.suptitle(
        title,
        fontsize=15 * TEXT_SCALE,
        fontweight="bold",
        y=1.14,
    )

    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        fraction=0.024,
        pad=0.015,
        extend="both",
    )
    cbar.set_label("Accuracy (%)", fontsize=10 * TEXT_SCALE, fontweight="bold")
    cbar.set_ticks([50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    cbar.ax.tick_params(labelsize=8.8 * TEXT_SCALE, width=2.2, length=6.4)
    cbar.outline.set_linewidth(2.4)
    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

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

    plot_subset(
        rows_a=rows_a,
        rows_b=rows_b,
        fps_values=[float(value) for value in args.fps_values],
        spatial_values=[float(value) for value in args.spatial_values],
        minor_contours=[float(value) for value in args.minor_contours],
        label_a=dataset_label(args.eta_a, args.noise_a, args.readout_a),
        label_b=dataset_label(args.eta_b, args.noise_b, args.readout_b),
        title=args.title,
        vmin=args.vmin,
        vmax=args.vmax,
        label_contours=args.label_contours,
        output_stem=output_dir / "overview_subset_no_paper",
    )

    metadata = {
        "aggregate_a": args.aggregate_a,
        "eta_a": args.eta_a,
        "noise_a": args.noise_a,
        "readout_a": args.readout_a,
        "aggregate_b": args.aggregate_b,
        "eta_b": args.eta_b,
        "noise_b": args.noise_b,
        "readout_b": args.readout_b,
        "fps_values_hz": [float(value) for value in args.fps_values],
        "spatial_values_pct": [float(value) for value in args.spatial_values],
        "minor_contours": [float(value) for value in args.minor_contours],
        "highlight_contours": {
            "75": {"color": "red", "linestyle": "dashed", "linewidth": HIGHLIGHT_CONTOUR_LINEWIDTH},
            "85": {"color": "red", "linestyle": "solid", "linewidth": HIGHLIGHT_CONTOUR_LINEWIDTH},
        },
        "label_contours": args.label_contours,
        "inline_contour_labels": False,
        "best_scan_point_marker": False,
        "colormap": {
            "base": "coolwarm",
            "style": "softened pastel blue-white-red",
            "range": [args.vmin, args.vmax],
        },
        "artifacts": [
            "overview_subset_no_paper.png",
            "overview_subset_no_paper.pdf",
        ],
    }
    (output_dir / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
