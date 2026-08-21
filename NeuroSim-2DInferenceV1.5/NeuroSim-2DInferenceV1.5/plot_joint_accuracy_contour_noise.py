import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a 2D contour map of joint accuracy on the eta-R plane for a fixed noise level, "
            "with the Accuracy=55 boundary highlighted."
        )
    )
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--accuracy-threshold", type=float, default=55.0)
    parser.add_argument("--output-path", required=True)
    return parser.parse_args()


def load_rows(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row["video_fps"] = float(row["video_fps"])
            row["noise_1f_density_1hz_a_root_hz"] = float(row["noise_1f_density_1hz_a_root_hz"])
            row["R_single"] = float(row["R_single"])
            row["eta_single"] = float(row["eta_single"])
            row["best_joint_accuracy"] = float(row["best_joint_accuracy"])
            row["best_joint_trise_tfall_equal_s"] = float(row["best_joint_trise_tfall_equal_s"])
            rows.append(row)
    return rows


def build_matrix(rows, fps, noise, r_values, eta_values, value_key):
    matrix = np.full((len(eta_values), len(r_values)), np.nan, dtype=float)
    trise_matrix = np.full((len(eta_values), len(r_values)), np.nan, dtype=float)
    for eta_idx, eta_value in enumerate(eta_values):
        for r_idx, r_value in enumerate(r_values):
            matched = [
                row for row in rows
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


def main():
    args = parse_args()
    rows = load_rows(args.summary_csv)
    fps_values = sorted({row["video_fps"] for row in rows})
    r_values = sorted({row["R_single"] for row in rows})
    eta_values = sorted({row["eta_single"] for row in rows})

    x_mesh, y_mesh = np.meshgrid(np.log10(r_values), eta_values)
    fig, axes = plt.subplots(1, len(fps_values), figsize=(20, 4.8), constrained_layout=True)
    if len(fps_values) == 1:
        axes = [axes]

    contourf_handle = None
    for ax, fps in zip(axes, fps_values):
        accuracy_matrix, trise_matrix = build_matrix(
            rows,
            fps=fps,
            noise=args.noise,
            r_values=r_values,
            eta_values=eta_values,
            value_key="best_joint_accuracy",
        )

        contourf_handle = ax.contourf(
            x_mesh,
            y_mesh,
            accuracy_matrix,
            levels=np.linspace(np.nanmin(accuracy_matrix), np.nanmax(accuracy_matrix), 12),
            cmap="viridis",
        )
        boundary = ax.contour(
            x_mesh,
            y_mesh,
            accuracy_matrix,
            levels=[args.accuracy_threshold],
            colors="red",
            linewidths=2.4,
        )
        ax.clabel(boundary, inline=True, fmt={args.accuracy_threshold: f"Acc={args.accuracy_threshold:.0f}"}, fontsize=9)
        ax.set_title(f"FPS={int(fps)}")
        ax.set_xlabel("log10(R)")
        ax.set_ylabel("eta")
        ax.set_xticks(np.log10(r_values))
        ax.set_xticklabels([f"{value:.2g}" for value in r_values], rotation=0)
        ax.set_yticks(eta_values)

        for eta_idx, eta_value in enumerate(eta_values):
            for r_idx, r_value in enumerate(r_values):
                accuracy = accuracy_matrix[eta_idx, r_idx]
                if np.isnan(accuracy):
                    continue
                ax.text(
                    np.log10(r_value),
                    eta_value,
                    f"{accuracy:.0f}\n{trise_matrix[eta_idx, r_idx]:.0e}",
                    fontsize=7,
                    ha="center",
                    va="center",
                    color="white" if accuracy < np.nanmax(accuracy_matrix) * 0.65 else "black",
                )

    fig.suptitle(f"Joint Accuracy Contour at noise={args.noise:.0e} (text: acc / best tr=tf)", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(contourf_handle, ax=axes, shrink=0.88, pad=0.03)
    cbar.set_label("Best Joint Accuracy (%)")
    fig.savefig(args.output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
