import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Render paper-style CIFAR-10 parameter-space figures on the R vs tr=tf plane "
            "for a fixed eta/noise slice."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--readout-label", default="integration")
    parser.add_argument("--spatial-values", nargs="+", type=float, default=None)
    parser.add_argument("--fps-values", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[75.0, 85.0])
    return parser.parse_args()


def set_paper_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.3,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.major.width": 1.2,
            "ytick.major.width": 1.2,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
        }
    )


def load_rows(path):
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


def filter_rows(rows, eta, noise, spatial_values=None, fps_values=None):
    filtered = [
        row
        for row in rows
        if abs(row["eta_single"] - eta) <= 1e-12
        and abs(row["noise_1f_density_1hz_a_root_hz"] - noise) <= 1e-20
    ]
    if spatial_values is not None:
        spatial_set = {float(value) for value in spatial_values}
        filtered = [row for row in filtered if row["spatial_variation_r_pct"] in spatial_set]
    if fps_values is not None:
        fps_set = {float(value) for value in fps_values}
        filtered = [row for row in filtered if row["video_fps"] in fps_set]
    if not filtered:
        raise ValueError("No rows matched the requested eta/noise slice.")
    return filtered


def sorted_unique(rows, field):
    return sorted({row[field] for row in rows})


def build_matrix(rows, fps, spatial, r_values, tr_values):
    matrix = np.full((len(tr_values), len(r_values)), np.nan, dtype=float)
    lookup = {
        (row["video_fps"], row["spatial_variation_r_pct"], row["R_single"], row["trise_tfall_equal_s"]): row[
            "accuracy_nonideal_cifar10"
        ]
        for row in rows
    }
    for tr_idx, tr_value in enumerate(tr_values):
        for r_idx, r_value in enumerate(r_values):
            value = lookup.get((fps, spatial, r_value, tr_value))
            if value is not None:
                matrix[tr_idx, r_idx] = value
    return matrix


def r_tick_labels(values):
    labels = []
    for value in values:
        if value < 0.01:
            labels.append("5e-3")
        elif value < 0.1:
            labels.append(f"{value:.2f}".rstrip("0").rstrip("."))
        else:
            labels.append(f"{value:.1f}".rstrip("0").rstrip("."))
    return labels


def tr_tick_labels(values):
    return [f"{value:.0e}" for value in values]


def apply_axis_style(ax, r_values, tr_values, show_xlabel, show_ylabel):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(min(r_values), max(r_values))
    ax.set_ylim(min(tr_values), max(tr_values))
    ax.set_xticks(r_values)
    ax.set_xticklabels(r_tick_labels(r_values))
    ax.set_yticks(tr_values)
    ax.set_yticklabels(tr_tick_labels(tr_values))
    ax.grid(which="major", color="#D6D6D6", linewidth=0.75, alpha=0.7)
    ax.tick_params(length=4.0, width=1.15)
    if show_xlabel:
        ax.set_xlabel("Responsivity R")
    else:
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    if show_ylabel:
        ax.set_ylabel(r"$t_r=t_f$ (s)")
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
        fontsize=9.5,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BFBFBF", "alpha": 0.95},
    )


def save_png_pdf(fig, output_stem):
    fig.savefig(str(output_stem) + ".png", dpi=320, bbox_inches="tight")
    fig.savefig(str(output_stem) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_distribution_grid(rows, fps_values, spatial_values, r_values, tr_values, eta, noise, readout_label, output_stem):
    nrows = len(spatial_values)
    ncols = len(fps_values)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.15 * ncols + 0.8, 2.9 * nrows + 0.75),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    vmin = min(row["accuracy_nonideal_cifar10"] for row in rows)
    vmax = max(row["accuracy_nonideal_cifar10"] for row in rows)
    scatter = None

    for row_idx, spatial in enumerate(spatial_values):
        for col_idx, fps in enumerate(fps_values):
            ax = axes[row_idx, col_idx]
            matrix = build_matrix(rows, fps, spatial, r_values, tr_values)
            xx, yy = np.meshgrid(r_values, tr_values)
            ax.plot(xx, yy, color="#E7E7E7", linewidth=0.8, zorder=1)
            ax.plot(xx.T, yy.T, color="#E7E7E7", linewidth=0.8, zorder=1)
            scatter = ax.scatter(
                xx.flatten(),
                yy.flatten(),
                c=matrix.flatten(),
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                s=150,
                edgecolors="#1A1A1A",
                linewidths=0.6,
                zorder=3,
            )
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
            apply_axis_style(
                ax,
                r_values,
                tr_values,
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

    fig.suptitle(
        f"CIFAR-10 Parameter-Space Distribution | {readout_label} readout | "
        + rf"$\eta={eta:.1f}$ | noise={noise:.0e} A/$\sqrt{{Hz}}$",
        fontsize=14,
        fontweight="bold",
    )
    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.024, pad=0.015)
    cbar.set_label("Accuracy (%)")
    cbar.ax.tick_params(labelsize=9.5, width=1.0, length=3.6)
    save_png_pdf(fig, output_stem)


def plot_contour_grid(rows, fps_values, spatial_values, r_values, tr_values, eta, noise, readout_label, thresholds, output_stem):
    nrows = len(spatial_values)
    ncols = len(fps_values)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3.15 * ncols + 0.8, 2.9 * nrows + 0.9),
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
            apply_axis_style(
                ax,
                r_values,
                tr_values,
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
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=max(1, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        title="Threshold contours",
        fontsize=9.2,
        title_fontsize=9.4,
    )
    fig.suptitle(
        f"CIFAR-10 Contours | {readout_label} readout | "
        + rf"$\eta={eta:.1f}$ | noise={noise:.0e} A/$\sqrt{{Hz}}$",
        fontsize=14,
        fontweight="bold",
        y=1.08,
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

    with (output_dir / "filtered_slice.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered[0].keys()))
        writer.writeheader()
        writer.writerows(filtered)

    plot_distribution_grid(
        rows=filtered,
        fps_values=fps_values,
        spatial_values=spatial_values,
        r_values=r_values,
        tr_values=tr_values,
        eta=args.eta,
        noise=args.noise,
        readout_label=args.readout_label,
        output_stem=output_dir / "c10_distribution_grid_paper",
    )
    plot_contour_grid(
        rows=filtered,
        fps_values=fps_values,
        spatial_values=spatial_values,
        r_values=r_values,
        tr_values=tr_values,
        eta=args.eta,
        noise=args.noise,
        readout_label=args.readout_label,
        thresholds=args.thresholds,
        output_stem=output_dir / "c10_contour_grid_paper",
    )

    metadata = {
        "aggregate_csv": args.aggregate_csv,
        "fixed_eta": args.eta,
        "fixed_noise": args.noise,
        "readout_label": args.readout_label,
        "spatial_values_pct": spatial_values,
        "fps_values_hz": fps_values,
        "R_values": r_values,
        "trise_tfall_equal_values_s": tr_values,
        "thresholds": args.thresholds,
        "artifacts": [
            "filtered_slice.csv",
            "c10_distribution_grid_paper.png",
            "c10_distribution_grid_paper.pdf",
            "c10_contour_grid_paper.png",
            "c10_contour_grid_paper.pdf",
        ],
    }
    (output_dir / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
