import argparse
import csv
import json
import math
import statistics
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MAIN_DIR = PROJECT_ROOT / "outputs" / "structured_sweep160_cifar100_nonideal_200"
DEFAULT_SUPPLEMENT1_DIR = PROJECT_ROOT / "outputs" / "structured_sweep160_cifar100_nonideal_200_supplement1"
DEFAULT_SUPPLEMENT2_DIR = PROJECT_ROOT / "outputs" / "structured_sweep160_cifar100_nonideal_200_supplement2"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "structured_sweep160_cifar100_nonideal_200_analysis"

READOUT_ORDER = [
    "integration-analog",
    "integration-adc4",
    "integration-adc8",
    "tia-analog",
    "tia-adc4",
    "tia-adc8",
]
READOUT_DISPLAY = {
    "integration-analog": "Int\nanalog",
    "integration-adc4": "Int\nADC4",
    "integration-adc8": "Int\nADC8",
    "tia-analog": "TIA\nanalog",
    "tia-adc4": "TIA\nADC4",
    "tia-adc8": "TIA\nADC8",
}
READOUT_MARKERS = {
    "integration-analog": "o",
    "integration-adc4": "s",
    "integration-adc8": "D",
    "tia-analog": "^",
    "tia-adc4": "P",
    "tia-adc8": "X",
}
BASELINE_KEYS = [
    "fps",
    "analog_readout",
    "adc_enabled",
    "adc_bits",
    "pmin_density",
    "pmax_density",
    "r_single",
    "eta_single",
    "trap_saturation_time_s",
    "trap_amplitude_pct",
    "noise_density",
    "degradation_pct",
    "spatial_pct",
]
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 80.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize the structured CIFAR-100 nonideal sweep and its supplements. "
            "The script repairs key parameters from per-scenario JSON files before plotting."
        )
    )
    parser.add_argument("--main-dir", default=str(DEFAULT_MAIN_DIR))
    parser.add_argument("--supplement1-dir", default=str(DEFAULT_SUPPLEMENT1_DIR))
    parser.add_argument("--supplement2-dir", default=str(DEFAULT_SUPPLEMENT2_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def safe_float(value):
    if value in {"", None}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value, default=0):
    if value in {"", None}:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_results_json(results_json, csv_path, scenario_name):
    path = Path(results_json)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append((PROJECT_ROOT / path).resolve())
        candidates.append((csv_path.parent / path).resolve())
        candidates.append((csv_path.parent / "scenarios" / f"{scenario_name}.json").resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve results JSON for {scenario_name}: {results_json}")


def infer_scenario_group(stage_name, row):
    group = row.get("scenario_group", "")
    if group:
        return group
    name = row["scenario_name"]
    if stage_name == "supplement2":
        if name.startswith("spatial_threshold_"):
            return "spatial_threshold"
        if name.startswith("readout_spatial5_"):
            return "readout_spatial5"
        if name.startswith("frontier2_"):
            return "frontier_refinement_round2"
    return "ungrouped"


def infer_anchor_family(name):
    if name.startswith("best_anchor_"):
        return "best_anchor"
    if name.startswith("fragile_anchor_"):
        return "fragile_anchor"
    return ""


def infer_threshold_mode(name):
    if "threshold_nominal" in name:
        return "nominal"
    if "threshold_stress" in name:
        return "stress"
    return ""


def build_readout_mode(analog_readout, adc_enabled, adc_bits):
    if adc_enabled:
        return f"{analog_readout}-adc{adc_bits}"
    return f"{analog_readout}-analog"


def format_readout_for_title(readout_mode):
    return READOUT_DISPLAY.get(readout_mode, readout_mode)


def format_scientific(value):
    if value is None:
        return "NA"
    text = f"{value:.2e}"
    text = text.replace("e-0", "e-").replace("e+0", "e+")
    return text


def format_short_float(value):
    if value is None:
        return "NA"
    if value >= 100 or (0 < value < 0.01):
        return format_scientific(value)
    return f"{value:.3g}"


def extract_accuracy(result):
    evaluation = result.get("evaluation", {})
    cases = evaluation.get("cases", {})
    nonideal = cases.get("nonideal", {})
    return safe_float(nonideal.get("accuracy"))


def normalize_record(stage_name, csv_path, row):
    result_path = resolve_results_json(row["results_json"], csv_path, row["scenario_name"])
    result = load_json(result_path)
    args = result.get("args", {})
    base_params = result.get("base_params", {})

    accuracy_nonideal = first_not_none(safe_float(row.get("accuracy_nonideal")), extract_accuracy(result))
    fps = first_not_none(safe_float(row.get("video_fps")), safe_float(args.get("video_fps")))
    analog_readout = row.get("analog_readout") or args.get("analog_readout") or args.get("readout") or "integration"
    adc_enabled = bool(safe_int(row.get("adc_enabled"), safe_int(args.get("adc_enabled"), 0)))
    adc_bits = 0
    if adc_enabled:
        adc_bits = safe_int(row.get("adc_bits"), safe_int(args.get("adc_bits"), 8))

    pmin_density = first_not_none(
        safe_float(row.get("pmin_density_w_cm2")),
        safe_float(base_params.get("pmin_density_w_cm2")),
        safe_float(args.get("pmin_density")),
    )
    pmax_density = first_not_none(
        safe_float(row.get("pmax_density_w_cm2")),
        safe_float(base_params.get("pmax_density_w_cm2")),
        safe_float(args.get("pmax_density")),
    )
    r_single = first_not_none(
        safe_float(row.get("R_single")),
        safe_float(base_params.get("R_single")),
        safe_float(args.get("single_r")),
    )
    eta_single = first_not_none(
        safe_float(row.get("eta_single")),
        safe_float(base_params.get("eta_single")),
        safe_float(args.get("single_eta")),
    )
    tau_rise_s = first_not_none(
        safe_float(row.get("tau_rise_single_s")),
        safe_float(base_params.get("tau_rise_single")),
        safe_float(args.get("single_trise")),
    )
    tau_fall_s = first_not_none(
        safe_float(row.get("tau_fall_single_s")),
        safe_float(base_params.get("tau_fall_single")),
        safe_float(args.get("single_tfall")),
    )
    trap_saturation_time_s = first_not_none(
        safe_float(row.get("trap_saturation_time_s")),
        safe_float(base_params.get("trap_saturation_time_s")),
        safe_float(args.get("trap_saturation_time")),
    )
    trap_amplitude_ratio = first_not_none(
        safe_float(base_params.get("trap_amplitude_ratio")),
        safe_float(row.get("trap_amplitude_ratio")),
    )
    trap_amplitude_pct = first_not_none(
        safe_float(args.get("trap_amplitude_pct")),
        safe_float(base_params.get("trap_amplitude_pct")),
        100.0 * trap_amplitude_ratio if trap_amplitude_ratio is not None else None,
        0.0,
    )
    if trap_amplitude_ratio is None:
        trap_amplitude_ratio = trap_amplitude_pct / 100.0

    noise_density = first_not_none(
        safe_float(row.get("noise_1f_density_1hz_a_root_hz")),
        safe_float(base_params.get("noise_1f_density_1hz_a_root_hz")),
        safe_float(args.get("noise_1f_density_1hz")),
    )

    degradation_ratio = first_not_none(
        safe_float(base_params.get("r_degradation_ratio")),
        safe_float(row.get("r_degradation_ratio")),
    )
    degradation_pct = first_not_none(
        safe_float(args.get("r_degradation_pct")),
        safe_float(base_params.get("r_degradation_pct")),
        100.0 * degradation_ratio if degradation_ratio is not None else None,
        0.0,
    )
    if degradation_ratio is None:
        degradation_ratio = degradation_pct / 100.0

    spatial_ratio = first_not_none(
        safe_float(base_params.get("spatial_variation_r_ratio")),
        safe_float(row.get("spatial_variation_r_ratio")),
    )
    spatial_pct = first_not_none(
        safe_float(args.get("spatial_variation_r_pct")),
        safe_float(base_params.get("spatial_variation_r_pct")),
        100.0 * spatial_ratio if spatial_ratio is not None else None,
        0.0,
    )
    if spatial_ratio is None:
        spatial_ratio = spatial_pct / 100.0

    readout_mode = build_readout_mode(analog_readout, adc_enabled, adc_bits)
    trap_time_multiplier = None
    if tau_rise_s and trap_saturation_time_s:
        trap_time_multiplier = trap_saturation_time_s / tau_rise_s

    return {
        "source_stage": stage_name,
        "source_dir": str(csv_path.parent),
        "scenario_name": row["scenario_name"],
        "scenario_group": infer_scenario_group(stage_name, row),
        "anchor_family": infer_anchor_family(row["scenario_name"]),
        "threshold_mode": infer_threshold_mode(row["scenario_name"]),
        "results_json": str(result_path),
        "fps": fps,
        "analog_readout": analog_readout,
        "adc_enabled": adc_enabled,
        "adc_bits": adc_bits,
        "readout_mode": readout_mode,
        "pmin_density": pmin_density,
        "pmax_density": pmax_density,
        "r_single": r_single,
        "eta_single": eta_single,
        "tau_rise_s": tau_rise_s,
        "tau_fall_s": tau_fall_s,
        "trap_saturation_time_s": trap_saturation_time_s,
        "trap_time_multiplier": trap_time_multiplier,
        "trap_amplitude_ratio": trap_amplitude_ratio,
        "trap_amplitude_pct": trap_amplitude_pct,
        "noise_density": noise_density,
        "degradation_ratio": degradation_ratio,
        "degradation_pct": degradation_pct,
        "spatial_ratio": spatial_ratio,
        "spatial_pct": spatial_pct,
        "accuracy_nonideal": accuracy_nonideal,
    }


def load_stage(stage_name, results_dir):
    results_dir = Path(results_dir)
    csv_path = results_dir / "aggregate_results.csv"
    rows = []
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(normalize_record(stage_name, csv_path, row))
    return rows


def set_plot_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfc",
            "axes.edgecolor": "#444444",
            "grid.color": "#d8dbe2",
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig, path, dpi):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def finalize_layout(fig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()


def maybe_close(actual, expected):
    if actual is None or expected is None:
        return actual == expected
    if isinstance(expected, float):
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)
    return actual == expected


def row_matches(row, constraints):
    for key, expected in constraints.items():
        if not maybe_close(row.get(key), expected):
            return False
    return True


def find_matching_rows(rows, **constraints):
    return [row for row in rows if row_matches(row, constraints)]


def find_accuracy(rows, **constraints):
    matches = find_matching_rows(rows, **constraints)
    if not matches:
        return None
    return statistics.mean(row["accuracy_nonideal"] for row in matches)


def baseline_constraints(baseline_row, exclude=None):
    exclude = set(exclude or [])
    constraints = {}
    for key in BASELINE_KEYS:
        if key in exclude:
            continue
        constraints[key] = baseline_row[key]
    return constraints


def get_baseline_row(main_rows):
    for row in main_rows:
        if row["scenario_name"] == "baseline":
            return row
    raise RuntimeError("Baseline row was not found in main sweep.")


def annotate_line(ax, xs, ys, fmt="{:.1f}"):
    for x_value, y_value in zip(xs, ys):
        if y_value is None:
            continue
        ax.annotate(fmt.format(y_value), (x_value, y_value), textcoords="offset points", xytext=(0, 7), ha="center")


def annotate_bars(ax, xs, ys, rotation=0):
    for x_value, y_value in zip(xs, ys):
        if y_value is None:
            continue
        ax.text(x_value, y_value + 0.8, f"{y_value:.1f}", ha="center", va="bottom", rotation=rotation)


def plot_heatmap(ax, matrix, x_labels, y_labels, title):
    image = ax.imshow(matrix, cmap="viridis", vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "--" if np.isnan(value) else f"{value:.1f}"
            text_color = "white" if not np.isnan(value) and value < 45.0 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color=text_color, fontsize=9)
    return image


def max_abs_delta(values, baseline_accuracy):
    candidates = [abs(value - baseline_accuracy) for value in values if value is not None]
    if not candidates:
        return 0.0
    return max(candidates)


def plot_main_single_factor(main_rows, output_dir, dpi):
    baseline_row = get_baseline_row(main_rows)
    baseline_accuracy = baseline_row["accuracy_nonideal"]

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    axes = axes.ravel()

    fps_values = [10.0, 20.0, 50.0, 100.0]
    fps_acc = [
        find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"fps"}), fps=value) for value in fps_values
    ]
    axes[0].plot(fps_values, fps_acc, marker="o", linewidth=2.2, color="#1f77b4")
    axes[0].axhline(baseline_accuracy, color="#7f7f7f", linestyle="--", linewidth=1)
    axes[0].set_title("Single Factor: FPS")
    axes[0].set_xlabel("Video FPS")
    axes[0].set_ylabel("Nonideal accuracy (%)")
    annotate_line(axes[0], fps_values, fps_acc)

    readout_acc = [
        find_accuracy(
            main_rows,
            **baseline_constraints(baseline_row, exclude={"analog_readout", "adc_enabled", "adc_bits"}),
            analog_readout=mode.split("-")[0],
            adc_enabled="adc" in mode,
            adc_bits=0 if mode.endswith("analog") else safe_int(mode.split("adc")[1]),
        )
        for mode in READOUT_ORDER
    ]
    x_positions = np.arange(len(READOUT_ORDER))
    bar_colors = ["#4c78a8", "#6b98c7", "#8cb5e0", "#f58518", "#f7a64a", "#f9c46b"]
    axes[1].bar(x_positions, readout_acc, color=bar_colors, edgecolor="#333333")
    axes[1].axhline(baseline_accuracy, color="#7f7f7f", linestyle="--", linewidth=1)
    axes[1].set_title("Single Factor: Readout")
    axes[1].set_ylabel("Nonideal accuracy (%)")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([READOUT_DISPLAY[mode] for mode in READOUT_ORDER])
    annotate_bars(axes[1], x_positions, readout_acc)

    r_values = [1e-3, 3.162277660168379e-2, 1.0, 31.622776601683793, 1e3]
    r_acc = [
        find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"r_single"}), r_single=value)
        for value in r_values
    ]
    axes[2].plot(r_values, r_acc, marker="o", linewidth=2.2, color="#d62728")
    axes[2].set_xscale("log")
    axes[2].axhline(baseline_accuracy, color="#7f7f7f", linestyle="--", linewidth=1)
    axes[2].set_title("Single Factor: R")
    axes[2].set_xlabel("R_single")
    axes[2].set_ylabel("Nonideal accuracy (%)")
    annotate_line(axes[2], r_values, r_acc)

    eta_values = [0.2, 0.5, 0.8, 1.0]
    eta_acc = [
        find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"eta_single"}), eta_single=value)
        for value in eta_values
    ]
    axes[3].plot(eta_values, eta_acc, marker="o", linewidth=2.2, color="#2ca02c")
    axes[3].axhline(baseline_accuracy, color="#7f7f7f", linestyle="--", linewidth=1)
    axes[3].set_title("Single Factor: Eta")
    axes[3].set_xlabel("Eta")
    axes[3].set_ylabel("Nonideal accuracy (%)")
    annotate_line(axes[3], eta_values, eta_acc)

    noise_values = [1e-9, 1e-8, 1e-7]
    noise_acc = [
        find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"noise_density"}), noise_density=value)
        for value in noise_values
    ]
    axes[4].plot(noise_values, noise_acc, marker="o", linewidth=2.2, color="#9467bd")
    axes[4].set_xscale("log")
    axes[4].axhline(baseline_accuracy, color="#7f7f7f", linestyle="--", linewidth=1)
    axes[4].set_title("Single Factor: 1/f Noise")
    axes[4].set_xlabel("Noise density (A/sqrt(Hz))")
    axes[4].set_ylabel("Nonideal accuracy (%)")
    annotate_line(axes[4], noise_values, noise_acc)

    trap_amp_values = [0.0, 5.0, 10.0, 20.0]
    trap_time_values = [baseline_row["tau_rise_s"] * value for value in [1.0, 10.0, 100.0, 1000.0]]
    trap_values = []
    for trap_amp in trap_amp_values:
        for trap_time in trap_time_values:
            trap_values.append(
                find_accuracy(
                    main_rows,
                    **baseline_constraints(baseline_row, exclude={"trap_amplitude_pct", "trap_saturation_time_s"}),
                    trap_amplitude_pct=trap_amp,
                    trap_saturation_time_s=trap_time,
                )
            )

    factor_deltas = {
        "R": max_abs_delta(r_acc, baseline_accuracy),
        "FPS": max_abs_delta(fps_acc, baseline_accuracy),
        "Readout": max_abs_delta(readout_acc, baseline_accuracy),
        "Eta": max_abs_delta(eta_acc, baseline_accuracy),
        "pmin": max_abs_delta(
            [
                find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"pmin_density"}), pmin_density=value)
                for value in sorted({baseline_row["pmin_density"], baseline_row["pmin_density"] * 0.25})
            ],
            baseline_accuracy,
        ),
        "pmax": max_abs_delta(
            [
                find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"pmax_density"}), pmax_density=value)
                for value in sorted({baseline_row["pmax_density"], baseline_row["pmax_density"] * 0.75})
            ],
            baseline_accuracy,
        ),
        "Noise": max_abs_delta(noise_acc, baseline_accuracy),
        "Trap grid": max_abs_delta(trap_values, baseline_accuracy),
        "Degradation": max_abs_delta(
            [
                find_accuracy(
                    main_rows,
                    **baseline_constraints(baseline_row, exclude={"degradation_pct"}),
                    degradation_pct=value,
                )
                for value in [30.0, 50.0, 80.0]
            ],
            baseline_accuracy,
        ),
        "Spatial": max_abs_delta(
            [
                find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"spatial_pct"}), spatial_pct=value)
                for value in [1.0, 5.0, 10.0]
            ],
            baseline_accuracy,
        ),
    }
    sorted_items = sorted(factor_deltas.items(), key=lambda item: item[1], reverse=True)
    labels = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    y_positions = np.arange(len(labels))
    axes[5].barh(y_positions, values, color="#72b7b2", edgecolor="#333333")
    axes[5].invert_yaxis()
    axes[5].set_yticks(y_positions)
    axes[5].set_yticklabels(labels)
    axes[5].set_title("Baseline Sensitivity Ranking")
    axes[5].set_xlabel("Max |delta accuracy| from baseline")
    for y_pos, value in zip(y_positions, values):
        axes[5].text(value + 0.3, y_pos, f"{value:.1f}", va="center")

    fig.suptitle("Main Sweep: Single-Factor Effects", fontsize=14, y=1.01)
    finalize_layout(fig)
    save_figure(fig, output_dir / "fig01_main_single_factor.png", dpi)
    return factor_deltas


def plot_main_interactions(main_rows, output_dir, dpi):
    baseline_row = get_baseline_row(main_rows)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    axes = axes.ravel()

    fps_values = [10.0, 20.0, 50.0, 100.0]
    fps_matrix = np.full((len(fps_values), len(READOUT_ORDER)), np.nan)
    for row_idx, fps in enumerate(fps_values):
        for col_idx, mode in enumerate(READOUT_ORDER):
            constraints = baseline_constraints(baseline_row, exclude={"fps", "analog_readout", "adc_enabled", "adc_bits"})
            constraints["fps"] = fps
            constraints["analog_readout"] = mode.split("-")[0]
            constraints["adc_enabled"] = "adc" in mode
            constraints["adc_bits"] = 0 if mode.endswith("analog") else safe_int(mode.split("adc")[1])
            value = find_accuracy(main_rows, **constraints)
            if value is not None:
                fps_matrix[row_idx, col_idx] = value
    image = plot_heatmap(
        axes[0],
        fps_matrix,
        [READOUT_DISPLAY[item] for item in READOUT_ORDER],
        [str(int(item)) for item in fps_values],
        "FPS x Readout",
    )
    axes[0].set_xlabel("Readout mode")
    axes[0].set_ylabel("FPS")

    noise_values = [1e-9, 1e-8, 1e-7]
    noise_matrix = np.full((len(noise_values), len(READOUT_ORDER)), np.nan)
    for row_idx, noise in enumerate(noise_values):
        for col_idx, mode in enumerate(READOUT_ORDER):
            constraints = baseline_constraints(
                baseline_row, exclude={"noise_density", "analog_readout", "adc_enabled", "adc_bits"}
            )
            constraints["noise_density"] = noise
            constraints["analog_readout"] = mode.split("-")[0]
            constraints["adc_enabled"] = "adc" in mode
            constraints["adc_bits"] = 0 if mode.endswith("analog") else safe_int(mode.split("adc")[1])
            value = find_accuracy(main_rows, **constraints)
            if value is not None:
                noise_matrix[row_idx, col_idx] = value
    plot_heatmap(
        axes[1],
        noise_matrix,
        [READOUT_DISPLAY[item] for item in READOUT_ORDER],
        [format_scientific(item) for item in noise_values],
        "Noise x Readout",
    )
    axes[1].set_xlabel("Readout mode")
    axes[1].set_ylabel("Noise density")

    trap_amp_values = [0.0, 5.0, 10.0, 20.0]
    trap_time_multipliers = [1.0, 10.0, 100.0, 1000.0]
    trap_matrix = np.full((len(trap_amp_values), len(trap_time_multipliers)), np.nan)
    for row_idx, trap_amp in enumerate(trap_amp_values):
        for col_idx, multiplier in enumerate(trap_time_multipliers):
            constraints = baseline_constraints(
                baseline_row, exclude={"trap_amplitude_pct", "trap_saturation_time_s"}
            )
            constraints["trap_amplitude_pct"] = trap_amp
            constraints["trap_saturation_time_s"] = baseline_row["tau_rise_s"] * multiplier
            value = find_accuracy(main_rows, **constraints)
            if value is not None:
                trap_matrix[row_idx, col_idx] = value
    plot_heatmap(
        axes[2],
        trap_matrix,
        [f"x{int(item)}" for item in trap_time_multipliers],
        [f"{int(item)}%" for item in trap_amp_values],
        "Trap Amplitude x Trap Time",
    )
    axes[2].set_xlabel("Trap saturation time / tau_rise")
    axes[2].set_ylabel("Trap amplitude")

    degradation_values = [30.0, 50.0, 80.0]
    spatial_values = [1.0, 5.0, 10.0]
    deg_spatial_matrix = np.full((len(spatial_values), len(degradation_values)), np.nan)
    for row_idx, spatial in enumerate(spatial_values):
        for col_idx, degradation in enumerate(degradation_values):
            constraints = baseline_constraints(baseline_row, exclude={"degradation_pct", "spatial_pct"})
            constraints["degradation_pct"] = degradation
            constraints["spatial_pct"] = spatial
            value = find_accuracy(main_rows, **constraints)
            if value is not None:
                deg_spatial_matrix[row_idx, col_idx] = value
    plot_heatmap(
        axes[3],
        deg_spatial_matrix,
        [f"{int(item)}%" for item in degradation_values],
        [f"{int(item)}%" for item in spatial_values],
        "Degradation x Spatial Variation",
    )
    axes[3].set_xlabel("Responsivity degradation")
    axes[3].set_ylabel("Spatial variation")

    cbar = fig.colorbar(image, ax=axes, shrink=0.92, location="right")
    cbar.set_label("Nonideal accuracy (%)")
    fig.suptitle("Main Sweep: Two-Factor Interactions", fontsize=14, y=1.01)
    finalize_layout(fig)
    save_figure(fig, output_dir / "fig02_main_interactions.png", dpi)


def plot_window_heatmaps(main_rows, output_dir, dpi):
    baseline_row = get_baseline_row(main_rows)
    pmin_values = sorted({row["pmin_density"] for row in main_rows if row["pmin_density"] is not None})
    pmax_values = sorted({row["pmax_density"] for row in main_rows if row["pmax_density"] is not None})
    r_values = [1e-3, 1.0, 1e3]
    eta_values = [0.2, 0.8, 1.0]

    fig, axes = plt.subplots(len(pmin_values), len(pmax_values), figsize=(12, 9))
    best_point = None
    worst_point = None
    image = None

    for row_idx, pmin in enumerate(pmin_values):
        for col_idx, pmax in enumerate(pmax_values):
            matrix = np.full((len(r_values), len(eta_values)), np.nan)
            for r_idx, r_value in enumerate(r_values):
                for eta_idx, eta_value in enumerate(eta_values):
                    constraints = baseline_constraints(
                        baseline_row, exclude={"pmin_density", "pmax_density", "r_single", "eta_single"}
                    )
                    constraints["pmin_density"] = pmin
                    constraints["pmax_density"] = pmax
                    constraints["r_single"] = r_value
                    constraints["eta_single"] = eta_value
                    value = find_accuracy(main_rows, **constraints)
                    if value is not None:
                        matrix[r_idx, eta_idx] = value
                        point = {"pmin_density": pmin, "pmax_density": pmax, "r_single": r_value, "eta_single": eta_value}
                        if best_point is None or value > best_point["accuracy_nonideal"]:
                            best_point = dict(point, accuracy_nonideal=value)
                        if worst_point is None or value < worst_point["accuracy_nonideal"]:
                            worst_point = dict(point, accuracy_nonideal=value)
            title = f"pmin={format_scientific(pmin)}, pmax={format_scientific(pmax)}"
            image = plot_heatmap(
                axes[row_idx, col_idx],
                matrix,
                [f"{item:.1f}" for item in eta_values],
                [format_short_float(item) for item in r_values],
                title,
            )
            axes[row_idx, col_idx].set_xlabel("Eta")
            axes[row_idx, col_idx].set_ylabel("R_single")

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, location="right")
    cbar.set_label("Nonideal accuracy (%)")
    fig.suptitle("Window x Nonlinearity: R and Eta Slices", fontsize=14, y=1.01)
    finalize_layout(fig)
    save_figure(fig, output_dir / "fig03_window_heatmaps.png", dpi)
    return {"best_point": best_point, "worst_point": worst_point}


def plot_global_random_landscape(main_rows, output_dir, dpi):
    global_rows = [row for row in main_rows if row["scenario_group"] == "global_random"]
    fig, ax = plt.subplots(figsize=(12, 8))

    plotted = []
    for mode in READOUT_ORDER:
        subset = [row for row in global_rows if row["readout_mode"] == mode]
        if not subset:
            continue
        scatter = ax.scatter(
            [row["r_single"] for row in subset],
            [row["eta_single"] for row in subset],
            c=[row["accuracy_nonideal"] for row in subset],
            s=85,
            marker=READOUT_MARKERS[mode],
            cmap="viridis",
            vmin=HEATMAP_VMIN,
            vmax=HEATMAP_VMAX,
            edgecolors="#222222",
            linewidths=0.5,
            alpha=0.9,
            label=format_readout_for_title(mode).replace("\n", " "),
        )
        plotted.append(scatter)

    ax.set_xscale("log")
    ax.set_xlabel("R_single")
    ax.set_ylabel("Eta")
    ax.set_title("Global Random Sweep: Accuracy Landscape in R-Eta Space")
    ax.legend(loc="lower right", ncol=2, frameon=True)

    for row in sorted(global_rows, key=lambda item: item["accuracy_nonideal"])[:3]:
        ax.annotate(
            f"{row['scenario_name']} ({row['accuracy_nonideal']:.1f})",
            (row["r_single"], row["eta_single"]),
            textcoords="offset points",
            xytext=(5, -12),
            ha="left",
            fontsize=8,
        )
    for row in sorted(global_rows, key=lambda item: item["accuracy_nonideal"], reverse=True)[:3]:
        ax.annotate(
            f"{row['scenario_name']} ({row['accuracy_nonideal']:.1f})",
            (row["r_single"], row["eta_single"]),
            textcoords="offset points",
            xytext=(5, 7),
            ha="left",
            fontsize=8,
        )

    if plotted:
        cbar = fig.colorbar(plotted[-1], ax=ax, shrink=0.9)
        cbar.set_label("Nonideal accuracy (%)")
    finalize_layout(fig)
    save_figure(fig, output_dir / "fig04_global_random_landscape.png", dpi)


def summarize_anchor_group(rows):
    summary = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["anchor_family"], row["spatial_pct"])].append(row["accuracy_nonideal"])
    for key, values in grouped.items():
        summary[key] = {
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def plot_supplement_anchor_threshold(supplement1_rows, supplement2_rows, output_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    corrected_rows = [row for row in supplement1_rows if row["scenario_group"] == "corrected_missing_dims"]
    anchor_summary = summarize_anchor_group(corrected_rows)
    for anchor_family, color in [("best_anchor", "#1f77b4"), ("fragile_anchor", "#d62728")]:
        spatial_values = sorted({row["spatial_pct"] for row in corrected_rows if row["anchor_family"] == anchor_family})
        means = [anchor_summary[(anchor_family, spatial)]["mean"] for spatial in spatial_values]
        mins = [anchor_summary[(anchor_family, spatial)]["min"] for spatial in spatial_values]
        maxs = [anchor_summary[(anchor_family, spatial)]["max"] for spatial in spatial_values]
        axes[0].plot(spatial_values, means, marker="o", linewidth=2.2, color=color, label=anchor_family.replace("_", " "))
        axes[0].fill_between(spatial_values, mins, maxs, color=color, alpha=0.15)
        annotate_line(axes[0], spatial_values, means)
    axes[0].set_title("Supplement 1: Anchor Sensitivity")
    axes[0].set_xlabel("Spatial variation (%)")
    axes[0].set_ylabel("Nonideal accuracy (%)")
    axes[0].legend()

    threshold_rows = [row for row in supplement2_rows if row["scenario_group"] == "spatial_threshold"]
    for mode in ["nominal", "stress"]:
        mode_rows = sorted([row for row in threshold_rows if row["threshold_mode"] == mode], key=lambda item: item["spatial_pct"])
        axes[1].plot(
            [row["spatial_pct"] for row in mode_rows],
            [row["accuracy_nonideal"] for row in mode_rows],
            marker="o",
            linewidth=2.2,
            label=mode,
        )
        annotate_line(
            axes[1],
            [row["spatial_pct"] for row in mode_rows],
            [row["accuracy_nonideal"] for row in mode_rows],
        )
    axes[1].set_title("Supplement 2: Spatial Threshold")
    axes[1].set_xlabel("Spatial variation (%)")
    axes[1].set_ylabel("Nonideal accuracy (%)")
    axes[1].legend()

    readout_rows = [row for row in supplement2_rows if row["scenario_group"] == "readout_spatial5"]
    threshold_stress_s05 = next(
        row for row in threshold_rows if row["threshold_mode"] == "stress" and math.isclose(row["spatial_pct"], 5.0)
    )
    comparison_rows = [threshold_stress_s05] + readout_rows
    comparison_rows = sorted(comparison_rows, key=lambda item: READOUT_ORDER.index(item["readout_mode"]))
    x_positions = np.arange(len(comparison_rows))
    heights = [row["accuracy_nonideal"] for row in comparison_rows]
    axes[2].bar(x_positions, heights, color=["#4c78a8", "#6b98c7", "#f58518", "#f9c46b"], edgecolor="#333333")
    axes[2].set_xticks(x_positions)
    axes[2].set_xticklabels([READOUT_DISPLAY[row["readout_mode"]] for row in comparison_rows])
    axes[2].set_title("Spatial=5%, Stress Case: Readout Comparison")
    axes[2].set_ylabel("Nonideal accuracy (%)")
    annotate_bars(axes[2], x_positions, heights)

    fig.suptitle("Supplement Insights: Hidden Thresholds and Robustness", fontsize=14, y=1.03)
    finalize_layout(fig)
    save_figure(fig, output_dir / "fig05_supplement_anchor_threshold.png", dpi)


def plot_frontier_refinement(supplement1_rows, supplement2_rows, output_dir, dpi):
    frontier_mode_order = [
        ("integration", False, 0, 100.0),
        ("integration", True, 4, 10.0),
        ("tia", False, 0, 100.0),
        ("tia", True, 4, 10.0),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def build_frontier_matrix(rows):
        frontier_pairs = sorted({(row["r_single"], row["eta_single"]) for row in rows})
        matrix = np.full((len(frontier_mode_order), len(frontier_pairs)), np.nan)
        for row_idx, mode in enumerate(frontier_mode_order):
            analog_readout, adc_enabled, adc_bits, fps = mode
            for col_idx, pair in enumerate(frontier_pairs):
                r_single, eta_single = pair
                value = find_accuracy(
                    rows,
                    analog_readout=analog_readout,
                    adc_enabled=adc_enabled,
                    adc_bits=adc_bits,
                    fps=fps,
                    r_single=r_single,
                    eta_single=eta_single,
                )
                if value is not None:
                    matrix[row_idx, col_idx] = value
        x_labels = [f"R={format_short_float(pair[0])}\neta={pair[1]:.2f}" for pair in frontier_pairs]
        y_labels = [
            f"{mode[0]} {'ADC' + str(mode[2]) if mode[1] else 'analog'}\n{int(mode[3])} fps" for mode in frontier_mode_order
        ]
        return matrix, x_labels, y_labels

    frontier1_rows = [row for row in supplement1_rows if row["scenario_group"] == "frontier_refinement"]
    matrix1, x_labels1, y_labels1 = build_frontier_matrix(frontier1_rows)
    image = plot_heatmap(axes[0], matrix1, x_labels1, y_labels1, "Frontier Refinement Round 1")

    frontier2_rows = [row for row in supplement2_rows if row["scenario_group"] == "frontier_refinement_round2"]
    matrix2, x_labels2, y_labels2 = build_frontier_matrix(frontier2_rows)
    plot_heatmap(axes[1], matrix2, x_labels2, y_labels2, "Frontier Refinement Round 2")

    cbar = fig.colorbar(image, ax=axes, shrink=0.92, location="right")
    cbar.set_label("Nonideal accuracy (%)")
    fig.suptitle("Low-R / Low-Eta Frontier: Readout and FPS Do Not Fully Rescue Accuracy", fontsize=14, y=1.02)
    finalize_layout(fig)
    save_figure(fig, output_dir / "fig06_frontier_refinement.png", dpi)


def write_csv(rows, path):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def convert_summary_keys(obj):
    if isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            if isinstance(key, tuple):
                new_key = " | ".join(str(item) for item in key)
            else:
                new_key = str(key)
            converted[new_key] = convert_summary_keys(value)
        return converted
    if isinstance(obj, list):
        return [convert_summary_keys(item) for item in obj]
    return obj


def compute_summary(records, factor_deltas, window_summary):
    main_rows = [row for row in records if row["source_stage"] == "main"]
    supplement1_rows = [row for row in records if row["source_stage"] == "supplement1"]
    supplement2_rows = [row for row in records if row["source_stage"] == "supplement2"]
    baseline_row = get_baseline_row(main_rows)
    baseline_accuracy = baseline_row["accuracy_nonideal"]

    fps_values = [10.0, 20.0, 50.0, 100.0]
    fps_curve = {
        str(int(value)): find_accuracy(main_rows, **baseline_constraints(baseline_row, exclude={"fps"}), fps=value)
        for value in fps_values
    }
    readout_curve = {
        mode: find_accuracy(
            main_rows,
            **baseline_constraints(baseline_row, exclude={"analog_readout", "adc_enabled", "adc_bits"}),
            analog_readout=mode.split("-")[0],
            adc_enabled="adc" in mode,
            adc_bits=0 if mode.endswith("analog") else safe_int(mode.split("adc")[1]),
        )
        for mode in READOUT_ORDER
    }

    global_rows = [row for row in main_rows if row["scenario_group"] == "global_random"]
    best_global = sorted(global_rows, key=lambda item: item["accuracy_nonideal"], reverse=True)[:5]
    worst_global = sorted(global_rows, key=lambda item: item["accuracy_nonideal"])[:5]

    corrected_rows = [row for row in supplement1_rows if row["scenario_group"] == "corrected_missing_dims"]
    anchor_summary = summarize_anchor_group(corrected_rows)

    threshold_rows = [row for row in supplement2_rows if row["scenario_group"] == "spatial_threshold"]
    threshold_curves = {}
    for mode in ["nominal", "stress"]:
        mode_rows = sorted([row for row in threshold_rows if row["threshold_mode"] == mode], key=lambda item: item["spatial_pct"])
        threshold_curves[mode] = [{"spatial_pct": row["spatial_pct"], "accuracy_nonideal": row["accuracy_nonideal"]} for row in mode_rows]

    frontier1_rows = [row for row in supplement1_rows if row["scenario_group"] == "frontier_refinement"]
    frontier2_rows = [row for row in supplement2_rows if row["scenario_group"] == "frontier_refinement_round2"]

    summary = {
        "baseline_accuracy": baseline_accuracy,
        "factor_deltas": factor_deltas,
        "fps_curve": fps_curve,
        "readout_curve": readout_curve,
        "window_summary": window_summary,
        "supplement_anchor_summary": anchor_summary,
        "spatial_threshold_curves": threshold_curves,
        "frontier_round1_best": max(frontier1_rows, key=lambda item: item["accuracy_nonideal"]) if frontier1_rows else None,
        "frontier_round2_best": max(frontier2_rows, key=lambda item: item["accuracy_nonideal"]) if frontier2_rows else None,
        "global_best": best_global,
        "global_worst": worst_global,
    }
    return summary


def write_summary_json(summary, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert_summary_keys(summary), indent=2), encoding="utf-8")


def write_report(summary, output_dir):
    factor_items = sorted(summary["factor_deltas"].items(), key=lambda item: item[1], reverse=True)
    top_factor_name, top_factor_delta = factor_items[0]
    second_factor_name, second_factor_delta = factor_items[1]

    best_point = summary["window_summary"]["best_point"]
    worst_point = summary["window_summary"]["worst_point"]
    best_anchor_1 = summary["supplement_anchor_summary"][("best_anchor", 1.0)]["mean"]
    best_anchor_10 = summary["supplement_anchor_summary"][("best_anchor", 10.0)]["mean"]
    fragile_anchor_1 = summary["supplement_anchor_summary"][("fragile_anchor", 1.0)]["mean"]
    fragile_anchor_10 = summary["supplement_anchor_summary"][("fragile_anchor", 10.0)]["mean"]
    nominal_curve = summary["spatial_threshold_curves"]["nominal"]
    stress_curve = summary["spatial_threshold_curves"]["stress"]
    frontier_round1_best = summary["frontier_round1_best"]
    frontier_round2_best = summary["frontier_round2_best"]

    lines = [
        "# Structured Sweep Analysis Report",
        "",
        "This report merges the main structured sweep and both supplement folders into a repaired analysis table.",
        "The main aggregate CSV has misleading values for trap amplitude, degradation, and spatial variation; this script reloads each `results_json` and repairs those fields before plotting.",
        "",
        "## Figure Guide",
        "",
        "- `fig01_main_single_factor.png`: single-factor sweeps for FPS, readout, R, eta, noise, plus baseline sensitivity ranking.",
        "- `fig02_main_interactions.png`: interaction heatmaps for `fps x readout`, `noise x readout`, `trap amplitude x trap time`, and `degradation x spatial`.",
        "- `fig03_window_heatmaps.png`: `pmin/pmax/R/eta` slices that show where the nonlinear window starts collapsing accuracy.",
        "- `fig04_global_random_landscape.png`: global-random points in `R-eta` space, colored by nonideal accuracy.",
        "- `fig05_supplement_anchor_threshold.png`: supplement evidence for hidden spatial-variation sensitivity and the threshold curve.",
        "- `fig06_frontier_refinement.png`: focused low-R / low-eta frontier refinements from both supplement rounds.",
        "",
        "## Key Findings",
        "",
        f"- Baseline nonideal accuracy is `{summary['baseline_accuracy']:.1f}%`.",
        f"- In the baseline-centered sweeps, the strongest lever is `{top_factor_name}` with a maximum delta of `{top_factor_delta:.1f}` points, followed by `{second_factor_name}` at `{second_factor_delta:.1f}` points.",
        f"- The FPS curve is `10 -> {summary['fps_curve']['10']:.1f}%`, `20 -> {summary['fps_curve']['20']:.1f}%`, `50 -> {summary['fps_curve']['50']:.1f}%`, `100 -> {summary['fps_curve']['100']:.1f}%`; high-speed degradation is the clearest temporal penalty.",
        f"- At the baseline operating point, the best readout is `tia-adc8` at `{summary['readout_curve']['tia-adc8']:.1f}%`, while `integration-adc4` is the weakest at `{summary['readout_curve']['integration-adc4']:.1f}%`.",
        f"- In the window slices, the best point is `R={format_short_float(best_point['r_single'])}, eta={best_point['eta_single']:.1f}, pmin={format_scientific(best_point['pmin_density'])}, pmax={format_scientific(best_point['pmax_density'])}` with `{best_point['accuracy_nonideal']:.1f}%` accuracy.",
        f"- The worst window point is `R={format_short_float(worst_point['r_single'])}, eta={worst_point['eta_single']:.1f}, pmin={format_scientific(worst_point['pmin_density'])}, pmax={format_scientific(worst_point['pmax_density'])}` with `{worst_point['accuracy_nonideal']:.1f}%` accuracy.",
        f"- Supplement 1 shows that the best anchor stays at `{best_anchor_1:.1f}%` when spatial variation is `1%`, but drops to `{best_anchor_10:.1f}%` around `10%` spatial variation. The fragile anchor is already near the floor: `{fragile_anchor_1:.1f}%` at `1%` and `{fragile_anchor_10:.1f}%` at `10%`.",
        f"- Supplement 2 places the spatial threshold in the `3%-5%` range: the nominal curve is `{nominal_curve[0]['accuracy_nonideal']:.1f}% -> {nominal_curve[1]['accuracy_nonideal']:.1f}% -> {nominal_curve[2]['accuracy_nonideal']:.1f}% -> {nominal_curve[3]['accuracy_nonideal']:.1f}%` as spatial variation rises `{int(nominal_curve[0]['spatial_pct'])}% -> {int(nominal_curve[1]['spatial_pct'])}% -> {int(nominal_curve[2]['spatial_pct'])}% -> {int(nominal_curve[3]['spatial_pct'])}%`.",
        f"- The stress curve nearly overlaps the nominal curve (`{stress_curve[0]['accuracy_nonideal']:.1f}%`, `{stress_curve[1]['accuracy_nonideal']:.1f}%`, `{stress_curve[2]['accuracy_nonideal']:.1f}%`, `{stress_curve[3]['accuracy_nonideal']:.1f}%`), so spatial variation is the dominant driver once the anchor becomes sensitive.",
        f"- The best frontier point in round 1 only reaches `{frontier_round1_best['accuracy_nonideal']:.1f}%`, and round 2 only reaches `{frontier_round2_best['accuracy_nonideal']:.1f}%`; readout and FPS tuning do not fully rescue the low-R / low-eta corner.",
        "",
        "## Output Files",
        "",
        "- `merged_results.csv`: repaired, unified table across all three result folders.",
        "- `analysis_summary.json`: machine-readable summary of the main curves and best/worst cases.",
        "- This markdown report plus the six PNG figures listed above.",
        "",
    ]
    report_path = output_dir / "analysis_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    set_plot_style()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    main_rows = load_stage("main", args.main_dir)
    supplement1_rows = load_stage("supplement1", args.supplement1_dir)
    supplement2_rows = load_stage("supplement2", args.supplement2_dir)
    all_rows = main_rows + supplement1_rows + supplement2_rows

    factor_deltas = plot_main_single_factor(main_rows, figure_dir, args.dpi)
    plot_main_interactions(main_rows, figure_dir, args.dpi)
    window_summary = plot_window_heatmaps(main_rows, figure_dir, args.dpi)
    plot_global_random_landscape(main_rows, figure_dir, args.dpi)
    plot_supplement_anchor_threshold(supplement1_rows, supplement2_rows, figure_dir, args.dpi)
    plot_frontier_refinement(supplement1_rows, supplement2_rows, figure_dir, args.dpi)

    write_csv(all_rows, output_dir / "merged_results.csv")
    summary = compute_summary(all_rows, factor_deltas, window_summary)
    write_summary_json(summary, output_dir / "analysis_summary.json")
    write_report(summary, output_dir)

    print(f"Analysis outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
