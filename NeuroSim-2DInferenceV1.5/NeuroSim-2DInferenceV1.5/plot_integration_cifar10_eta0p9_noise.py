import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot CIFAR-10 accuracy over the R vs trise=tfall plane under different FPS and spatial variation "
            "for the integration-readout fixed-eta scan."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta", type=float, default=0.9)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[75.0, 85.0])
    return parser.parse_args()


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


def build_matrix(rows, fps, spatial, r_values, tr_values):
    matrix = np.full((len(tr_values), len(r_values)), np.nan, dtype=float)
    for tr_idx, tr_value in enumerate(tr_values):
        for r_idx, r_value in enumerate(r_values):
            matched = [
                row for row in rows
                if row["video_fps"] == fps
                and row["spatial_variation_r_pct"] == spatial
                and row["R_single"] == r_value
                and row["trise_tfall_equal_s"] == tr_value
            ]
            if matched:
                matrix[tr_idx, r_idx] = matched[0]["accuracy_nonideal_cifar10"]
    return matrix


def style_2d_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(2.2)
    ax.tick_params(axis="both", which="major", width=2.0, length=6.5, labelsize=12)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4.0)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(15)
    ax.title.set_weight("bold")


def style_3d_axis(ax):
    ax.tick_params(axis="both", which="major", width=1.8, length=6.0, labelsize=11)
    ax.tick_params(axis="z", which="major", width=1.8, length=6.0, labelsize=11)
    ax.xaxis.label.set_size(13)
    ax.yaxis.label.set_size(13)
    ax.zaxis.label.set_size(13)
    ax.title.set_size(14)
    ax.title.set_weight("bold")
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis.line.set_linewidth(2.2)
        except Exception:
            pass


def _color_for_value(value, vmax):
    return "white" if value < vmax * 0.6 else "black"


def plot_distribution(rows, fps, spatial_values, r_values, tr_values, eta, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.2), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    vmin = min(row["accuracy_nonideal_cifar10"] for row in rows)
    vmax = max(row["accuracy_nonideal_cifar10"] for row in rows)
    scatter = None
    for ax, spatial in zip(axes, spatial_values):
        subset = [row for row in rows if row["video_fps"] == fps and row["spatial_variation_r_pct"] == spatial]
        x_vals = [np.log10(row["R_single"]) for row in subset]
        y_vals = [np.log10(row["trise_tfall_equal_s"]) for row in subset]
        c_vals = [row["accuracy_nonideal_cifar10"] for row in subset]
        scatter = ax.scatter(
            x_vals,
            y_vals,
            c=c_vals,
            cmap="viridis",
            s=140,
            edgecolors="black",
            linewidths=0.9,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"Spatial={spatial:.0f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.3g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        style_2d_axis(ax)
    fig.suptitle(f"CIFAR-10 Parameter Distribution | integration | FPS={int(fps)} | eta={eta:.1f}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(scatter, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("CIFAR-10 Accuracy (%)")
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_surface(rows, fps, spatial_values, r_values, tr_values, eta, output_path):
    fig = plt.figure(figsize=(5.0 * len(spatial_values), 5.8))
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    surface = None
    for idx, spatial in enumerate(spatial_values, start=1):
        ax = fig.add_subplot(1, len(spatial_values), idx, projection="3d")
        matrix = build_matrix(rows, fps, spatial, r_values, tr_values)
        surface = ax.plot_surface(
            x_mesh,
            y_mesh,
            np.ma.masked_invalid(matrix),
            cmap="viridis",
            edgecolor="k",
            linewidth=0.85,
            antialiased=True,
            alpha=0.95,
        )
        ax.set_title(f"Spatial={spatial:.0f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_zlabel("CIFAR-10 Accuracy (%)")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.3g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        ax.view_init(elev=27, azim=-58)
        style_3d_axis(ax)
    fig.suptitle(f"CIFAR-10 3D Surface | integration | FPS={int(fps)} | eta={eta:.1f}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(surface, ax=fig.axes, shrink=0.78, pad=0.03)
    cbar.set_label("CIFAR-10 Accuracy (%)")
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_contour(rows, fps, spatial_values, r_values, tr_values, eta, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.2), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    contourf = None
    for ax, spatial in zip(axes, spatial_values):
        matrix = build_matrix(rows, fps, spatial, r_values, tr_values)
        contourf = ax.contourf(x_mesh, y_mesh, matrix, levels=12, cmap="viridis")
        contour = ax.contour(x_mesh, y_mesh, matrix, levels=10, colors="k", linewidths=1.2, alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.0f")
        ax.set_title(f"Spatial={spatial:.0f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.3g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        style_2d_axis(ax)
    fig.suptitle(f"CIFAR-10 Contours | integration | FPS={int(fps)} | eta={eta:.1f}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(contourf, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("CIFAR-10 Accuracy (%)")
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(rows, fps, spatial_values, r_values, tr_values, eta, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.2), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    image = None
    for ax, spatial in zip(axes, spatial_values):
        matrix = build_matrix(rows, fps, spatial, r_values, tr_values)
        image = ax.imshow(matrix, cmap="viridis", origin="lower", aspect="auto")
        vmax = np.nanmax(matrix)
        ax.set_title(f"Spatial={spatial:.0f}%")
        ax.set_xlabel("R")
        ax.set_ylabel("tr=tf (s)")
        ax.set_xticks(np.arange(len(r_values)))
        ax.set_xticklabels([f"{value:.3g}" for value in r_values])
        ax.set_yticks(np.arange(len(tr_values)))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        for tr_idx, tr_value in enumerate(tr_values):
            for r_idx, r_value in enumerate(r_values):
                value = matrix[tr_idx, r_idx]
                if np.isnan(value):
                    continue
                ax.text(
                    r_idx,
                    tr_idx,
                    f"{value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=_color_for_value(value, vmax),
                )
        style_2d_axis(ax)
    fig.suptitle(f"CIFAR-10 Heatmap | integration | FPS={int(fps)} | eta={eta:.1f}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("CIFAR-10 Accuracy (%)")
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_thresholds(rows, fps, spatial_values, r_values, tr_values, eta, thresholds, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.3), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    contourf = None
    colors = ["red", "orange", "cyan", "magenta"]
    for ax, spatial in zip(axes, spatial_values):
        matrix = build_matrix(rows, fps, spatial, r_values, tr_values)
        contourf = ax.contourf(x_mesh, y_mesh, matrix, levels=12, cmap="viridis")
        contour = ax.contour(x_mesh, y_mesh, matrix, levels=10, colors="k", linewidths=1.2, alpha=0.35)
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.0f")
        for idx, threshold in enumerate(thresholds):
            boundary = ax.contour(
                x_mesh,
                y_mesh,
                matrix,
                levels=[threshold],
                colors=[colors[idx % len(colors)]],
                linewidths=3.2,
            )
            ax.clabel(boundary, inline=True, fontsize=10, fmt={threshold: f"Acc={threshold:.0f}"})
        ax.set_title(f"Spatial={spatial:.0f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.3g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        style_2d_axis(ax)
    fig.suptitle(f"CIFAR-10 Contours with Thresholds | integration | FPS={int(fps)} | eta={eta:.1f}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(contourf, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("CIFAR-10 Accuracy (%)")
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.aggregate_csv)
    filtered = [
        row for row in rows
        if abs(row["noise_1f_density_1hz_a_root_hz"] - args.noise) <= 1e-20
        and abs(row["eta_single"] - args.eta) <= 1e-12
    ]
    if not filtered:
        raise ValueError("No rows matched the requested noise slice.")

    fps_values = sorted({row["video_fps"] for row in filtered})
    spatial_values = sorted({row["spatial_variation_r_pct"] for row in filtered})
    r_values = sorted({row["R_single"] for row in filtered})
    tr_values = sorted({row["trise_tfall_equal_s"] for row in filtered})

    with (output_dir / "filtered_slice.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered[0].keys()))
        writer.writeheader()
        writer.writerows(filtered)

    for fps in fps_values:
        plot_distribution(filtered, fps, spatial_values, r_values, tr_values, args.eta, output_dir / f"c10_distribution_fps_{int(fps):03d}.png")
        plot_surface(filtered, fps, spatial_values, r_values, tr_values, args.eta, output_dir / f"c10_surface_fps_{int(fps):03d}.png")
        plot_contour(filtered, fps, spatial_values, r_values, tr_values, args.eta, output_dir / f"c10_contour_fps_{int(fps):03d}.png")
        plot_heatmap(filtered, fps, spatial_values, r_values, tr_values, args.eta, output_dir / f"c10_heatmap_fps_{int(fps):03d}.png")
        plot_thresholds(filtered, fps, spatial_values, r_values, tr_values, args.eta, args.thresholds, output_dir / f"c10_contour_thresholds_fps_{int(fps):03d}.png")

    readme = {
        "aggregate_csv": args.aggregate_csv,
        "fixed_noise": args.noise,
        "fixed_eta": args.eta,
        "readout": "integration",
        "spatial_values_pct": spatial_values,
        "R_values": r_values,
        "trise_tfall_equal_values_s": tr_values,
        "thresholds": args.thresholds,
    }
    (output_dir / "README.json").write_text(json.dumps(readme, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
