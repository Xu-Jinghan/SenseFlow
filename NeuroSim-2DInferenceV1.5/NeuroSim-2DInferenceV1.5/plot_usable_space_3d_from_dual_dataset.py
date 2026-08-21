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
            "Plot 3D usable-space boundary and joint accuracy surfaces from the dual-dataset "
            "trise=tfall and noise sweep."
        )
    )
    parser.add_argument("--aggregate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cifar100-threshold", type=float, default=55.0)
    parser.add_argument("--cifar10-threshold", type=float, default=75.0)
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


def build_grid_summary(rows, cifar100_threshold, cifar10_threshold):
    fps_values = sorted({row["video_fps"] for row in rows})
    noise_values = sorted({row["noise_1f_density_1hz_a_root_hz"] for row in rows})
    r_values = sorted({row["R_single"] for row in rows})
    eta_values = sorted({row["eta_single"] for row in rows})

    summary_rows = []
    for fps in fps_values:
        for noise in noise_values:
            for r_value in r_values:
                for eta_value in eta_values:
                    subset = [
                        row for row in rows
                        if row["video_fps"] == fps
                        and row["noise_1f_density_1hz_a_root_hz"] == noise
                        and row["R_single"] == r_value
                        and row["eta_single"] == eta_value
                    ]
                    if not subset:
                        continue

                    best_joint = max(subset, key=lambda row: row["joint_min_accuracy"])
                    usable_subset = [
                        row for row in subset
                        if row["accuracy_nonideal_cifar100"] > cifar100_threshold
                        and row["accuracy_nonideal_cifar10"] > cifar10_threshold
                    ]
                    if usable_subset:
                        boundary_row = max(usable_subset, key=lambda row: row["spatial_variation_r_pct"])
                        boundary_spatial = boundary_row["spatial_variation_r_pct"]
                        boundary_trise = boundary_row["trise_tfall_equal_s"]
                    else:
                        boundary_spatial = np.nan
                        boundary_trise = np.nan

                    summary_rows.append(
                        {
                            "video_fps": fps,
                            "noise_1f_density_1hz_a_root_hz": noise,
                            "R_single": r_value,
                            "eta_single": eta_value,
                            "best_joint_accuracy": best_joint["joint_min_accuracy"],
                            "best_joint_trise_tfall_equal_s": best_joint["trise_tfall_equal_s"],
                            "boundary_spatial_variation_r_pct": boundary_spatial,
                            "boundary_trise_tfall_equal_s": boundary_trise,
                        }
                    )
    return summary_rows, fps_values, noise_values, r_values, eta_values


def _matrix_from_summary(summary_rows, fps, noise, r_values, eta_values, value_key):
    matrix = np.full((len(eta_values), len(r_values)), np.nan, dtype=float)
    trise_matrix = np.full((len(eta_values), len(r_values)), np.nan, dtype=float)
    for eta_idx, eta_value in enumerate(eta_values):
        for r_idx, r_value in enumerate(r_values):
            matched = [
                row for row in summary_rows
                if row["video_fps"] == fps
                and row["noise_1f_density_1hz_a_root_hz"] == noise
                and row["R_single"] == r_value
                and row["eta_single"] == eta_value
            ]
            if not matched:
                continue
            row = matched[0]
            matrix[eta_idx, r_idx] = row[value_key]
            trise_matrix[eta_idx, r_idx] = row["best_joint_trise_tfall_equal_s"]
    return matrix, trise_matrix


def _format_noise_label(noise_value):
    return f"{noise_value:.0e}"


def _format_trise_annotation(value):
    if np.isnan(value):
        return "-"
    return f"{value:.0e}"


def plot_surface_figure(summary_rows, fps_values, noise_value, r_values, eta_values, value_key, z_label, title_prefix, output_path):
    fig = plt.figure(figsize=(20, 4.8))
    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), eta_values)

    for subplot_idx, fps in enumerate(fps_values, start=1):
        ax = fig.add_subplot(1, len(fps_values), subplot_idx, projection="3d")
        value_matrix, trise_matrix = _matrix_from_summary(summary_rows, fps, noise_value, r_values, eta_values, value_key)
        masked = np.ma.masked_invalid(value_matrix)
        surf = ax.plot_surface(
            x_mesh,
            y_mesh,
            masked,
            cmap="viridis",
            edgecolor="k",
            linewidth=0.5,
            antialiased=True,
            alpha=0.95,
        )
        ax.set_title(f"FPS={int(fps)}")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("eta")
        ax.set_zlabel(z_label)
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.2g}" for value in r_values], rotation=0)
        ax.set_yticks(eta_values)
        ax.view_init(elev=27, azim=-58)

        for eta_idx, eta_value in enumerate(eta_values):
            for r_idx, r_value in enumerate(r_values):
                z_val = value_matrix[eta_idx, r_idx]
                if np.isnan(z_val):
                    continue
                ax.text(
                    np.log10(r_value),
                    eta_value,
                    z_val,
                    _format_trise_annotation(trise_matrix[eta_idx, r_idx]),
                    fontsize=7,
                    ha="center",
                    va="bottom",
                )

    fig.suptitle(f"{title_prefix} | noise={_format_noise_label(noise_value)}", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(surf, ax=fig.axes, shrink=0.72, pad=0.03)
    cbar.set_label(z_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_summary_csv(summary_rows, path):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.aggregate_csv)
    summary_rows, fps_values, noise_values, r_values, eta_values = build_grid_summary(
        rows,
        cifar100_threshold=args.cifar100_threshold,
        cifar10_threshold=args.cifar10_threshold,
    )

    export_summary_csv(summary_rows, output_dir / "usable_boundary_summary.csv")
    (output_dir / "usable_boundary_summary.json").write_text(
        json.dumps(summary_rows, indent=2),
        encoding="utf-8",
    )

    for noise_value in noise_values:
        plot_surface_figure(
            summary_rows,
            fps_values=fps_values,
            noise_value=noise_value,
            r_values=r_values,
            eta_values=eta_values,
            value_key="boundary_spatial_variation_r_pct",
            z_label="Max Usable Spatial Variation (%)",
            title_prefix="Usable-Space Boundary",
            output_path=output_dir / f"usable_boundary_noise_{_format_noise_label(noise_value)}.png",
        )
        plot_surface_figure(
            summary_rows,
            fps_values=fps_values,
            noise_value=noise_value,
            r_values=r_values,
            eta_values=eta_values,
            value_key="best_joint_accuracy",
            z_label="Best Joint Accuracy (%)",
            title_prefix="Accuracy Inside Parameter Space",
            output_path=output_dir / f"joint_accuracy_noise_{_format_noise_label(noise_value)}.png",
        )

    readme = {
        "aggregate_csv": args.aggregate_csv,
        "cifar100_threshold": args.cifar100_threshold,
        "cifar10_threshold": args.cifar10_threshold,
        "generated_files": sorted(path.name for path in output_dir.iterdir()),
        "note": (
            "For each FPS and noise panel, the boundary surface uses axes log10(R), eta, and the maximum spatial variation "
            "that still satisfies CIFAR100 and CIFAR10 thresholds. Text labels on the surface are the best trise=tfall "
            "value selected at that grid cell."
        ),
    }
    (output_dir / "README.json").write_text(json.dumps(readme, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
