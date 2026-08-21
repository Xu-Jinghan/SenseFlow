import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot contour maps on the R vs trise=tfall plane at fixed eta and noise, "
            "with user-specified accuracy threshold boundaries overlaid."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--eta", type=float, default=0.8)
    parser.add_argument("--noise", type=float, default=1e-8)
    parser.add_argument("--metric", default="joint_min", choices=["joint_min", "cifar100", "cifar10"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[55.0, 65.0])
    parser.add_argument("--spatial-values", nargs="+", type=float, default=None)
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
            matrix[tr_idx, r_idx] = select_metric(matched[0], metric)
    return matrix


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


def plot_contours(rows, fps, spatial_values, r_values, tr_values, metric, thresholds, output_path):
    fig, axes = plt.subplots(1, len(spatial_values), figsize=(4.8 * len(spatial_values), 5.3), constrained_layout=True)
    if len(spatial_values) == 1:
        axes = [axes]
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), np.log10(tr_values))
    contourf = None

    threshold_colors = ["red", "orange", "cyan", "magenta"]
    for ax, spatial in zip(axes, spatial_values):
        matrix = build_matrix(rows, fps, spatial, r_values, tr_values, metric)
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

    for fps in fps_values:
        plot_contours(
            filtered,
            fps=fps,
            spatial_values=spatial_values,
            r_values=r_values,
            tr_values=tr_values,
            metric=args.metric,
            thresholds=args.thresholds,
            output_path=output_dir / f"{metric_short(args.metric)}_contour_thresholds_fps_{int(fps):03d}.png",
        )

    metadata = {
        "aggregate_csv": args.aggregate_csv,
        "fixed_eta": args.eta,
        "fixed_noise": args.noise,
        "metric": args.metric,
        "thresholds": args.thresholds,
        "fps_values": fps_values,
        "spatial_values_pct": spatial_values,
    }
    (output_dir / "threshold_overlay_readme.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
