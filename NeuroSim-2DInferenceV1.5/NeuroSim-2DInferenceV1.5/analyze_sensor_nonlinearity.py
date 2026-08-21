import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
THIS_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = THIS_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import functional as TF

import generate_sensor_verification_images as sensor_gen

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze ideal vs nonideal sensor outputs with difference maps and fit-based nonlinearity metrics."
    )
    parser.add_argument("--data-root", default=str(sensor_gen.REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--power-max", type=float, default=1.0)
    parser.add_argument("--params-csv", default=str(sensor_gen.DEFAULT_PARAMS_CSV))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exposure-time", type=float, default=1.0 / 30.0)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--i-thermal", type=float, default=5e-8)
    parser.add_argument("--bandwidth", type=float, default=5000.0)
    parser.add_argument("--shot-noise", type=int, default=1)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        default=str(THIS_DIR / "verification_images_nonlinearity_analysis"),
        help="Directory for analysis figure and metrics JSON.",
    )
    return parser.parse_args()


def normalize_joint(frame_a, frame_b):
    frame_a = np.asarray(frame_a, dtype=np.float64)
    frame_b = np.asarray(frame_b, dtype=np.float64)
    lo = min(float(np.min(frame_a)), float(np.min(frame_b)))
    hi = max(float(np.max(frame_a)), float(np.max(frame_b)))
    scale = max(hi - lo, 1e-12)
    return (frame_a - lo) / scale, (frame_b - lo) / scale, lo, hi


def to_display_image(frame):
    frame = np.asarray(frame, dtype=np.float64)
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = frame[..., 0]
    return frame


def diff_map_for_display(diff_raw):
    diff_raw = np.asarray(diff_raw, dtype=np.float64)
    if diff_raw.ndim == 3 and diff_raw.shape[0] in {1, 3} and diff_raw.shape[-1] not in {1, 3}:
        diff_raw = np.transpose(diff_raw, (1, 2, 0))
    if diff_raw.ndim == 3:
        return np.mean(diff_raw, axis=-1)
    return diff_raw


def fit_metrics(x, y):
    linear_coeff = np.polyfit(x, y, 1)
    quad_coeff = np.polyfit(x, y, 2)

    y_linear = np.polyval(linear_coeff, x)
    y_quad = np.polyval(quad_coeff, x)

    rmse_linear = float(np.sqrt(np.mean((y - y_linear) ** 2)))
    rmse_quadratic = float(np.sqrt(np.mean((y - y_quad) ** 2)))
    quad_gain = float((rmse_linear - rmse_quadratic) / max(rmse_linear, 1e-12))

    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    if ss_tot <= 1e-24:
        r2_linear = 1.0
        r2_quadratic = 1.0
    else:
        r2_linear = float(1.0 - np.sum((y - y_linear) ** 2) / ss_tot)
        r2_quadratic = float(1.0 - np.sum((y - y_quad) ** 2) / ss_tot)

    return {
        "linear_coeff": linear_coeff.tolist(),
        "quadratic_coeff": quad_coeff.tolist(),
        "rmse_linear": rmse_linear,
        "rmse_quadratic": rmse_quadratic,
        "quadratic_rmse_gain_ratio": quad_gain,
        "r2_linear": r2_linear,
        "r2_quadratic": r2_quadratic,
    }


def build_binned_curve(x, y, num_bins=24):
    edges = np.linspace(float(np.min(x)), float(np.max(x)), num_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_y = np.full(num_bins, np.nan, dtype=np.float64)
    std_y = np.full(num_bins, np.nan, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int32)

    for idx in range(num_bins):
        if idx == num_bins - 1:
            mask = (x >= edges[idx]) & (x <= edges[idx + 1])
        else:
            mask = (x >= edges[idx]) & (x < edges[idx + 1])
        if not np.any(mask):
            continue
        values = y[mask]
        mean_y[idx] = float(np.mean(values))
        std_y[idx] = float(np.std(values))
        counts[idx] = int(values.size)

    return centers, mean_y, std_y, counts


def format_summary(metrics):
    gain_pct = 100.0 * metrics["quadratic_rmse_gain_ratio"]
    return (
        f"linear RMSE={metrics['rmse_linear']:.3e}\n"
        f"quadratic RMSE={metrics['rmse_quadratic']:.3e}\n"
        f"quadratic gain={gain_pct:.2f}%\n"
        f"R2 linear={metrics['r2_linear']:.6f}\n"
        f"R2 quadratic={metrics['r2_quadratic']:.6f}"
    )


def render_analysis_figure(
    original_image,
    ideal_raw,
    nonideal_raw,
    diff_raw,
    scatter_x,
    scatter_y,
    binned_curve,
    metrics,
    sample_label,
    output_path,
):
    ideal_disp, nonideal_disp, _, _ = normalize_joint(ideal_raw, nonideal_raw)
    ideal_disp = to_display_image(ideal_disp)
    nonideal_disp = to_display_image(nonideal_disp)
    diff_display = diff_map_for_display(diff_raw)
    diff_abs_max = max(float(np.max(np.abs(diff_display))), 1e-12)
    residual_linear = scatter_y - np.polyval(np.asarray(metrics["linear_coeff"]), scatter_x)

    centers, mean_y, std_y, counts = binned_curve
    valid = ~np.isnan(mean_y)
    x_fit = np.linspace(float(np.min(scatter_x)), float(np.max(scatter_x)), 256)
    y_fit_linear = np.polyval(np.asarray(metrics["linear_coeff"]), x_fit)
    y_fit_quad = np.polyval(np.asarray(metrics["quadratic_coeff"]), x_fit)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    ax = axes.ravel()

    ax[0].imshow(np.asarray(original_image))
    ax[0].set_title(f"Input\n{sample_label}")
    ax[0].axis("off")

    im1 = ax[1].imshow(ideal_disp, cmap="viridis", vmin=0.0, vmax=1.0)
    ax[1].set_title("Ideal Raw\nShared Display Range")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(nonideal_disp, cmap="viridis", vmin=0.0, vmax=1.0)
    ax[2].set_title("Nonideal Raw\nShared Display Range")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(diff_display, cmap="coolwarm", vmin=-diff_abs_max, vmax=diff_abs_max)
    ax[3].set_title("Difference Map\nNonideal - Ideal")
    ax[3].axis("off")
    fig.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    max_points = min(scatter_x.size, 5000)
    sample_idx = np.linspace(0, scatter_x.size - 1, max_points, dtype=int)
    ax[4].scatter(scatter_x[sample_idx], scatter_y[sample_idx], s=8, alpha=0.18, color="#1f77b4", label="pixels")
    ax[4].plot(x_fit, x_fit, "--", color="#444444", lw=1.3, label="y=x")
    ax[4].plot(x_fit, y_fit_linear, color="#d62728", lw=2.0, label="linear fit")
    ax[4].plot(x_fit, y_fit_quad, color="#2ca02c", lw=2.0, label="quadratic fit")
    ax[4].errorbar(
        centers[valid],
        mean_y[valid],
        yerr=std_y[valid],
        fmt="o",
        ms=4,
        lw=1.0,
        color="#ff7f0e",
        ecolor="#ffbb78",
        capsize=2,
        label="bin mean ± std",
    )
    ax[4].set_title("Pixel Scatter\nIdeal vs Nonideal")
    ax[4].set_xlabel("Ideal raw")
    ax[4].set_ylabel("Nonideal raw")
    ax[4].legend(loc="best", fontsize=8)
    ax[4].grid(True, alpha=0.2)

    ax[5].scatter(scatter_x[sample_idx], residual_linear[sample_idx], s=8, alpha=0.18, color="#9467bd")
    ax[5].axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    ax[5].set_title("Residual to Linear Fit")
    ax[5].set_xlabel("Ideal raw")
    ax[5].set_ylabel("Nonideal - linear_fit")
    ax[5].grid(True, alpha=0.2)
    ax[5].text(
        0.03,
        0.97,
        format_summary(metrics),
        transform=ax[5].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#cccccc"},
    )

    fig.suptitle("Sensor Difference / Nonlinearity Analysis", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = sensor_gen.load_base_dataset(args.source_dataset, args.data_root, args.split)
    base_params = sensor_gen.resolve_base_params(args.params_csv)

    image, label = base_dataset[args.sample_index]
    label_name = base_dataset.classes[label]
    output_channels = 1 if TF.to_tensor(image).shape[0] == 1 else 3
    power_maps = sensor_gen.build_power_maps(image, args.array_size, output_channels, args.power_max)

    ideal_raw = sensor_gen.simulate_static_frame(power_maps, args, base_params, nonideal=False, seed_offset=args.sample_index * 8)
    nonideal_raw = sensor_gen.simulate_static_frame(power_maps, args, base_params, nonideal=True, seed_offset=args.sample_index * 8 + 4)

    diff_raw = nonideal_raw - ideal_raw
    scatter_x = ideal_raw.reshape(-1).astype(np.float64)
    scatter_y = nonideal_raw.reshape(-1).astype(np.float64)
    metrics = fit_metrics(scatter_x, scatter_y)
    binned_curve = build_binned_curve(scatter_x, scatter_y)

    sample_label = f"idx={args.sample_index} class={label_name}"
    figure_path = output_dir / f"sample_{args.sample_index:04d}_{label_name}_nonlinearity.png"
    metrics_path = output_dir / f"sample_{args.sample_index:04d}_{label_name}_metrics.json"

    render_analysis_figure(
        image,
        ideal_raw,
        nonideal_raw,
        diff_raw,
        scatter_x,
        scatter_y,
        binned_curve,
        metrics,
        sample_label,
        figure_path,
    )

    result = {
        "sample_index": args.sample_index,
        "label": label_name,
        "figure_path": str(figure_path),
        "ideal_min": float(np.min(ideal_raw)),
        "ideal_max": float(np.max(ideal_raw)),
        "nonideal_min": float(np.min(nonideal_raw)),
        "nonideal_max": float(np.max(nonideal_raw)),
        "difference_min": float(np.min(diff_raw)),
        "difference_max": float(np.max(diff_raw)),
        "difference_mean": float(np.mean(diff_raw)),
        "difference_abs_mean": float(np.mean(np.abs(diff_raw))),
        "nonlinearity_metrics": metrics,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    gain_pct = 100.0 * metrics["quadratic_rmse_gain_ratio"]
    print(f"figure                {figure_path}")
    print(f"metrics               {metrics_path}")
    print(f"sample                {sample_label}")
    print(f"linear_rmse           {metrics['rmse_linear']:.6e}")
    print(f"quadratic_rmse        {metrics['rmse_quadratic']:.6e}")
    print(f"quadratic_gain_pct    {gain_pct:.2f}")
    print(f"r2_linear             {metrics['r2_linear']:.8f}")
    print(f"r2_quadratic          {metrics['r2_quadratic']:.8f}")


if __name__ == "__main__":
    main()
