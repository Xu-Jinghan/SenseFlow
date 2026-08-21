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
            "Filter the dual-dataset trise/noise sweep at fixed eta and noise, then plot "
            "accuracy over the R vs trise=tfall plane under different FPS and spatial variation."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--noise", type=float, default=1e-8)
    parser.add_argument("--metric", default="joint_min", choices=["joint_min", "cifar100", "cifar10"])
    parser.add_argument("--spatial-values", nargs="+", type=float, default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[55.0, 65.0])
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
            row["accuracy_nonideal_cifar100"] = float(row["accuracy_nonideal_cifar100"])
            row["accuracy_nonideal_cifar10"] = float(row["accuracy_nonideal_cifar10"])
            row["joint_min_accuracy"] = min(row["accuracy_nonideal_cifar100"], row["accuracy_nonideal_cifar10"])
            rows.append(row)
    return rows


def select_metric(row, metric):
    if metric == "cifar100":
        return row["accuracy_nonideal_cifar100"]
    if metric == "cifar10":
        return row["accuracy_nonideal_cifar10"]
    return row["joint_min_accuracy"]


def build_matrix(rows, fps, spatial, r_values, tr_values, metric):
    matrix = np.full((len(tr_values), len(r_values)), np.nan, dtype=float)
    c100 = np.full_like(matrix, np.nan)
    c10 = np.full_like(matrix, np.nan)
    for tr_idx, tr_value in enumerate(tr_values):
        for r_idx, r_value in enumerate(r_values):
            matched = [
                row for row in rows
                if row["video_fps"] == fps
                and row["spatial_variation_r_pct"] == spatial
                and row["R_single"] == r_value
                and row["trise_tfall_equal_s"] == tr_value
            ]
            if not matched:
                continue
            row = matched[0]
            matrix[tr_idx, r_idx] = select_metric(row, metric)
            c100[tr_idx, r_idx] = row["accuracy_nonideal_cifar100"]
            c10[tr_idx, r_idx] = row["accuracy_nonideal_cifar10"]
    return matrix, c100, c10


def metric_label(metric):
    if metric == "cifar100":
        return "CIFAR-100 Accuracy (%)"
    if metric == "cifar10":
        return "CIFAR-10 Accuracy (%)"
    return "Joint Min Accuracy (%)"


def metric_short(metric):
    if metric == "cifar100":
        return "c100"
    if metric == "cifar10":
        return "c10"
    return "joint"


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


def plot_threshold_overlays(rows, fps, spatial_values, r_values, tr_values, metric, thresholds, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.3), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    contourf = None
    threshold_colors = ["red", "orange", "cyan", "magenta"]

    for ax, spatial in zip(axes, spatial_values):
        matrix, _, _ = build_matrix(rows, fps, spatial, r_values, tr_values, metric)
        contourf = ax.contourf(
            x_mesh,
            y_mesh,
            matrix,
            levels=12,
            cmap="viridis",
        )
        contour = ax.contour(
            x_mesh,
            y_mesh,
            matrix,
            levels=10,
            colors="k",
            linewidths=1.2,
            alpha=0.35,
        )
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.0f")

        for threshold_idx, threshold in enumerate(thresholds):
            boundary = ax.contour(
                x_mesh,
                y_mesh,
                matrix,
                levels=[threshold],
                colors=[threshold_colors[threshold_idx % len(threshold_colors)]],
                linewidths=3.2,
            )
            ax.clabel(
                boundary,
                inline=True,
                fontsize=10,
                fmt={threshold: f"Acc={threshold:.0f}"},
            )

        ax.set_title(f"Spatial={spatial:.2f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.2g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        style_2d_axis(ax)

    fig.suptitle(
        f"R vs tr=tf Contours with Thresholds | FPS={int(fps)} | eta=0.8 | noise=1e-08 | {metric_label(metric)}",
        fontsize=18,
        fontweight="bold",
    )
    cbar = fig.colorbar(contourf, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label(metric_label(metric))
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    cbar.ax.yaxis.label.set_size(13)
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_surfaces(rows, fps, spatial_values, r_values, tr_values, metric, output_path):
    fig = plt.figure(figsize=(5.0 * len(spatial_values), 5.8))
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    surface = None
    for idx, spatial in enumerate(spatial_values, start=1):
        ax = fig.add_subplot(1, len(spatial_values), idx, projection="3d")
        matrix, _, _ = build_matrix(rows, fps, spatial, r_values, tr_values, metric)
        masked = np.ma.masked_invalid(matrix)
        surface = ax.plot_surface(
            x_mesh,
            y_mesh,
            masked,
            cmap="viridis",
            edgecolor="k",
            linewidth=0.85,
            antialiased=True,
            alpha=0.95,
        )
        ax.set_title(f"Spatial={spatial:.2f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_zlabel(metric_label(metric))
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.2g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        ax.view_init(elev=27, azim=-58)
        style_3d_axis(ax)
    fig.suptitle(f"R vs tr=tf 3D Surface | FPS={int(fps)} | eta=0.8 | noise=1e-08 | {metric_label(metric)}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(surface, ax=fig.axes, shrink=0.78, pad=0.03)
    cbar.set_label(metric_label(metric))
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    cbar.ax.yaxis.label.set_size(13)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_contours(rows, fps, spatial_values, r_values, tr_values, metric, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.2), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    contourf = None
    for ax, spatial in zip(axes, spatial_values):
        matrix, _, _ = build_matrix(rows, fps, spatial, r_values, tr_values, metric)
        contourf = ax.contourf(
            x_mesh,
            y_mesh,
            matrix,
            levels=12,
            cmap="viridis",
        )
        contour = ax.contour(
            x_mesh,
            y_mesh,
            matrix,
            levels=10,
            colors="k",
            linewidths=1.2,
            alpha=0.6,
        )
        ax.clabel(contour, inline=True, fontsize=9, fmt="%.0f")
        ax.set_title(f"Spatial={spatial:.2f}%")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("log10(tr=tf)")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.2g}" for value in r_values])
        ax.set_yticks(np.log10(tr_values))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        style_2d_axis(ax)
    fig.suptitle(f"R vs tr=tf Contours | FPS={int(fps)} | eta=0.8 | noise=1e-08 | {metric_label(metric)}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(contourf, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label(metric_label(metric))
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    cbar.ax.yaxis.label.set_size(13)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(rows, fps, spatial_values, r_values, tr_values, metric, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.2), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    image = None
    for ax, spatial in zip(axes, spatial_values):
        matrix, c100, c10 = build_matrix(rows, fps, spatial, r_values, tr_values, metric)
        image = ax.imshow(matrix, cmap="viridis", origin="lower", aspect="auto")
        ax.set_title(f"Spatial={spatial:.2f}%")
        ax.set_xlabel("R")
        ax.set_ylabel("tr=tf (s)")
        ax.set_xticks(np.arange(len(r_values)))
        ax.set_xticklabels([f"{value:.2g}" for value in r_values], rotation=0)
        ax.set_yticks(np.arange(len(tr_values)))
        ax.set_yticklabels([f"{value:.0e}" for value in tr_values])
        vmax = np.nanmax(matrix)
        for tr_idx, tr_value in enumerate(tr_values):
            for r_idx, r_value in enumerate(r_values):
                value = matrix[tr_idx, r_idx]
                if np.isnan(value):
                    continue
                ax.text(
                    r_idx,
                    tr_idx,
                    f"{value:.0f}\n{c100[tr_idx, r_idx]:.0f}/{c10[tr_idx, r_idx]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=_color_for_value(value, vmax),
                )
        style_2d_axis(ax)
    fig.suptitle(f"R vs tr=tf Heatmap | FPS={int(fps)} | eta=0.8 | noise=1e-08 | {metric_label(metric)}", fontsize=18, fontweight="bold")
    cbar = fig.colorbar(image, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label(metric_label(metric))
    cbar.ax.tick_params(labelsize=12, width=1.8, length=6.0)
    cbar.ax.yaxis.label.set_size(13)
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)


def export_filtered_table(rows, output_path):
    fieldnames = [
        "scenario_name",
        "video_fps",
        "spatial_variation_r_pct",
        "R_single",
        "eta_single",
        "trise_tfall_equal_s",
        "noise_1f_density_1hz_a_root_hz",
        "accuracy_nonideal_cifar100",
        "accuracy_nonideal_cifar10",
        "joint_min_accuracy",
    ]
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.aggregate_csv)
    filtered = [
        row for row in rows
        if abs(row["eta_single"] - args.eta) <= 1e-12
        and abs(row["noise_1f_density_1hz_a_root_hz"] - args.noise) <= 1e-20
    ]
    if not filtered:
        raise ValueError("No rows matched the requested eta/noise slice.")

    fps_values = sorted({row["video_fps"] for row in filtered})
    spatial_values = sorted({row["spatial_variation_r_pct"] for row in filtered})
    if args.spatial_values:
        requested = {float(value) for value in args.spatial_values}
        spatial_values = [value for value in spatial_values if value in requested]
        filtered = [row for row in filtered if row["spatial_variation_r_pct"] in requested]
    if not spatial_values:
        raise ValueError("No spatial-variation values remain after filtering.")
    r_values = sorted({row["R_single"] for row in filtered})
    tr_values = sorted({row["trise_tfall_equal_s"] for row in filtered})

    export_filtered_table(filtered, output_dir / "filtered_eta_noise_slice.csv")
    (output_dir / "filtered_eta_noise_slice.json").write_text(json.dumps(filtered, indent=2), encoding="utf-8")

    for fps in fps_values:
        plot_surfaces(
            filtered,
            fps=fps,
            spatial_values=spatial_values,
            r_values=r_values,
            tr_values=tr_values,
            metric=args.metric,
            output_path=output_dir / f"{metric_short(args.metric)}_surface_fps_{int(fps):03d}.png",
        )
        plot_contours(
            filtered,
            fps=fps,
            spatial_values=spatial_values,
            r_values=r_values,
            tr_values=tr_values,
            metric=args.metric,
            output_path=output_dir / f"{metric_short(args.metric)}_contour_fps_{int(fps):03d}.png",
        )
        plot_heatmaps(
            filtered,
            fps=fps,
            spatial_values=spatial_values,
            r_values=r_values,
            tr_values=tr_values,
            metric=args.metric,
            output_path=output_dir / f"{metric_short(args.metric)}_heatmap_fps_{int(fps):03d}.png",
        )
        plot_threshold_overlays(
            filtered,
            fps=fps,
            spatial_values=spatial_values,
            r_values=r_values,
            tr_values=tr_values,
            metric=args.metric,
            thresholds=args.thresholds,
            output_path=output_dir / f"{metric_short(args.metric)}_contour_thresholds_fps_{int(fps):03d}.png",
        )

    readme = {
        "aggregate_csv": args.aggregate_csv,
        "fixed_eta": args.eta,
        "fixed_noise": args.noise,
        "metric": args.metric,
        "thresholds": args.thresholds,
        "fps_values": fps_values,
        "spatial_values_pct": spatial_values,
        "R_values": r_values,
        "trise_tfall_equal_values_s": tr_values,
        "note": (
            "Each figure fixes one FPS and sweeps spatial variation across subplots. "
            "Axes are R and trise=tfall. Heatmap labels are 'selected_metric / cifar100 / cifar10'. "
            "Threshold-overlay contour figures are generated automatically."
        ),
    }
    (output_dir / "README.json").write_text(json.dumps(readme, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
