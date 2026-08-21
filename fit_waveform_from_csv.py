from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt

from photodetector_model import (
    N_CARRIER,
    NO_DRIFT_FIXED_PARAMS,
    PARAMS_TRUE,
    PARAM_BOUNDS,
    PARAM_KEYS,
    fit_parameters_subset,
    simulate,
    vec_to_params,
)

DEFAULT_FIT_KEYS = [
    "R_fast",
    "tau_rise_fast",
    "tau_fall_fast",
    "R_slow",
    "tau_rise_slow",
    "tau_fall_slow",
    "alpha",
    "beta",
    "delta",
]

DEFAULT_FIXED_PARAMS = {
    "eta_fast": 1.0,
    "eta_slow": 1.0,
    **NO_DRIFT_FIXED_PARAMS,
}
DEFAULT_FIXED_PARAM_TEXT = ",".join(f"{key}={value}" for key, value in DEFAULT_FIXED_PARAMS.items())

IMAGE_FIT_PARAM_BOUNDS = dict(PARAM_BOUNDS)
IMAGE_FIT_PARAM_BOUNDS.update(
    {
        "tau_rise_fast": (1e-8, 5e-3, "log"),
        "tau_fall_fast": (1e-8, 1e-2, "log"),
        "tau_rise_slow": (1e-7, 1e-1, "log"),
        "tau_fall_slow": (1e-7, 5e-1, "log"),
        "alpha": (1e-1, 1e4, "log"),
        "beta": (1e-4, 2e1, "log"),
        "delta": (1e-4, 5e-1, "log"),
    }
)

PARAM_UNITS = {
    "R_fast": "a.u.",
    "eta_fast": "1",
    "tau_rise_fast": "s",
    "tau_fall_fast": "s",
    "R_slow": "a.u.",
    "eta_slow": "1",
    "tau_rise_slow": "s",
    "tau_fall_slow": "s",
    "alpha": "1/s",
    "beta": "1/s",
    "delta": "a.u.",
    "gamma": "a.u.",
    "tau_drift": "s",
    "drift_scale": "1",
}

PRESENTATION_STYLE = {
    "title_size": 17,
    "label_size": 15,
    "tick_size": 13,
    "legend_size": 12,
    "spine_width": 1.8,
    "tick_width": 1.6,
    "tick_length": 6,
    "line_width": 2.8,
    "secondary_line_width": 2.4,
    "table_font_size": 12,
    "save_dpi": 300,
}


def apply_axis_style(
    ax: plt.Axes,
    *,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    if title is not None:
        ax.set_title(title, fontsize=PRESENTATION_STYLE["title_size"], pad=10, fontweight="semibold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=PRESENTATION_STYLE["label_size"], labelpad=8)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=PRESENTATION_STYLE["label_size"], labelpad=8)
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=PRESENTATION_STYLE["tick_size"],
        width=PRESENTATION_STYLE["tick_width"],
        length=PRESENTATION_STYLE["tick_length"],
    )
    for spine in ax.spines.values():
        spine.set_linewidth(PRESENTATION_STYLE["spine_width"])
        spine.set_color("#111827")


def style_legend(legend: plt.Legend | None) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_linewidth(1.2)
    frame.set_edgecolor("#111827")
    frame.set_facecolor("white")
    frame.set_alpha(0.95)


def parse_name_list(csv_text: str) -> list[str]:
    names = [item.strip() for item in csv_text.split(",") if item.strip()]
    invalid = [name for name in names if name not in PARAM_KEYS]
    if invalid:
        raise ValueError(f"Unsupported parameter names: {invalid}")
    return names


def parse_fixed_params(text: str) -> dict[str, float]:
    if not text.strip():
        return {}
    parsed: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid fixed param assignment: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        if key not in PARAM_KEYS:
            raise ValueError(f"Unsupported fixed parameter: {key}")
        parsed[key] = float(value)
    return parsed


def parse_pulse_indices(text: str) -> list[int]:
    if not text.strip():
        return []
    indices: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Pulse indices must be positive 1-based integers, got: {value}")
        indices.append(value)
    unique_sorted = sorted(set(indices))
    if unique_sorted != indices:
        indices = unique_sorted
    if any(b != a + 1 for a, b in zip(indices, indices[1:])):
        raise ValueError(f"Pulse indices must be consecutive, got: {indices}")
    return indices


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def smooth_signal(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 2:
        return y
    window = min(window, len(y) - (1 - len(y) % 2))
    window = int(window) | 1
    if window < 5 or window >= len(y):
        return y
    polyorder = min(3, window - 2)
    return savgol_filter(y, window_length=window, polyorder=polyorder, mode="interp")


def resample_uniform(
    t: np.ndarray,
    response: np.ndarray,
    on_mask: np.ndarray,
    num_points: int,
    p_on: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_fit = np.linspace(float(t[0]), float(t[-1]), num_points)
    response_fit = np.interp(t_fit, t, response)
    on_numeric = np.interp(t_fit, t, on_mask.astype(float))
    on_fit = on_numeric >= 0.5
    power_fit = on_fit.astype(float) * p_on
    return t_fit, response_fit, power_fit


def remove_short_binary_runs(states: np.ndarray, min_run_length: int = 3) -> np.ndarray:
    cleaned = np.asarray(states, dtype=bool).copy()
    if min_run_length <= 1 or len(cleaned) < min_run_length + 2:
        return cleaned

    changed = True
    while changed:
        changed = False
        run_start = 0
        while run_start < len(cleaned):
            run_end = run_start + 1
            while run_end < len(cleaned) and cleaned[run_end] == cleaned[run_start]:
                run_end += 1
            run_length = run_end - run_start
            left_state = cleaned[run_start - 1] if run_start > 0 else None
            right_state = cleaned[run_end] if run_end < len(cleaned) else None
            if (
                run_length < min_run_length
                and left_state is not None
                and right_state is not None
                and left_state == right_state
                and left_state != cleaned[run_start]
            ):
                cleaned[run_start:run_end] = left_state
                changed = True
                break
            run_start = run_end
    return cleaned


def find_on_segments(on_mask: np.ndarray) -> list[tuple[int, int]]:
    on_mask = np.asarray(on_mask, dtype=bool)
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_on in enumerate(on_mask):
        if is_on and start is None:
            start = idx
        elif not is_on and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(on_mask) - 1))
    return segments


def select_pulse_subset(
    t: np.ndarray,
    response: np.ndarray,
    on_mask: np.ndarray,
    pulse_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], list[tuple[int, int]]]:
    if not pulse_indices:
        return t, response, on_mask, (float(t[0]), float(t[-1])), []

    segments = find_on_segments(on_mask)
    if not segments:
        raise RuntimeError("No ON pulse segments found in the waveform.")
    if pulse_indices[-1] > len(segments):
        raise ValueError(f"Requested pulse {pulse_indices[-1]}, but only {len(segments)} pulses were detected.")

    first_idx = pulse_indices[0] - 1
    last_idx = pulse_indices[-1] - 1

    if first_idx == 0:
        left = 0
    else:
        prev_end = segments[first_idx - 1][1]
        curr_start = segments[first_idx][0]
        left = (prev_end + curr_start + 1) // 2

    if last_idx == len(segments) - 1:
        right = len(t)
    else:
        curr_end = segments[last_idx][1]
        next_start = segments[last_idx + 1][0]
        right = (curr_end + next_start + 1) // 2

    selection = slice(left, right)
    t_selected = np.asarray(t[selection], dtype=float)
    response_selected = np.asarray(response[selection], dtype=float)
    on_selected = np.asarray(on_mask[selection], dtype=bool)

    if len(t_selected) < 3:
        raise RuntimeError("Selected pulse window is too short for fitting.")

    t_selected = t_selected - float(t_selected[0])
    selected_window = (float(t[left]), float(t[right - 1]))
    return t_selected, response_selected, on_selected, selected_window, segments


def select_time_subset(
    t: np.ndarray,
    response: np.ndarray,
    on_mask: np.ndarray,
    time_start: float | None,
    time_end: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    if time_start is None and time_end is None:
        return t, response, on_mask, (float(t[0]), float(t[-1]))
    if time_start is None or time_end is None:
        raise ValueError("Both --fit-time-start and --fit-time-end must be provided together.")
    if not time_end > time_start:
        raise ValueError(f"Expected fit-time-end > fit-time-start, got {time_start} and {time_end}.")

    mask = (t >= time_start) & (t <= time_end)
    if int(np.count_nonzero(mask)) < 3:
        raise RuntimeError(
            f"Selected time window [{time_start}, {time_end}] does not contain enough samples for fitting."
        )

    t_selected = np.asarray(t[mask], dtype=float)
    response_selected = np.asarray(response[mask], dtype=float)
    on_selected = np.asarray(on_mask[mask], dtype=bool)
    t_selected = t_selected - float(t_selected[0])
    return t_selected, response_selected, on_selected, (float(time_start), float(time_end))


def build_initial_guess(response_fit: np.ndarray, dark_current: float, fixed_params: dict[str, float]) -> dict[str, float]:
    amplitude = max(float(np.max(response_fit) - dark_current), 0.05)
    initial = dict(PARAMS_TRUE)
    initial["R_fast"] = float(np.clip(0.72 * amplitude, 0.1, 1.0))
    initial["R_slow"] = float(np.clip(0.22 * amplitude, 0.01, 0.5))
    delta_lo, delta_hi, _ = IMAGE_FIT_PARAM_BOUNDS["delta"]
    initial["alpha"] = 20.0
    initial["beta"] = 1.0
    initial["delta"] = float(np.clip(0.05 * amplitude, max(delta_lo, 5e-3), min(delta_hi, 0.2)))
    initial["gamma"] = float(np.clip(0.03 * amplitude, 0.002, 0.05))
    initial.update(fixed_params)
    return initial


def save_digitized_csv(
    output_path: Path,
    t_fit: np.ndarray,
    measured: np.ndarray,
    power_fit: np.ndarray,
    fitted: np.ndarray,
) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "response_au", "power_norm", "fitted_response_au"])
        for t_value, measured_value, power_value, fit_value in zip(t_fit, measured, power_fit, fitted):
            writer.writerow([f"{t_value:.9e}", f"{measured_value:.9e}", f"{power_value:.9e}", f"{fit_value:.9e}"])


def save_parameter_csv(
    output_path: Path,
    fitted_params: dict[str, float],
    fit_keys: list[str],
    fixed_params: dict[str, float],
) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "unit", "status"])
        for key in PARAM_KEYS:
            status = "fitted" if key in fit_keys else "fixed" if key in fixed_params else "default"
            writer.writerow([key, f"{fitted_params[key]:.9e}", PARAM_UNITS.get(key, ""), status])


def save_parameter_figure(
    output_path: Path,
    fitted_params: dict[str, float],
    fit_keys: list[str],
    fixed_params: dict[str, float],
    fit_stats: dict[str, float],
) -> None:
    ensure_parent_dir(output_path)
    fig, ax = plt.subplots(figsize=(11.6, 7.4))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    rows = []
    for key in PARAM_KEYS:
        status = "fitted" if key in fit_keys else "fixed" if key in fixed_params else "default"
        rows.append([key, PARAM_UNITS.get(key, ""), f"{fitted_params[key]:.4e}", status])

    table = ax.table(
        cellText=rows,
        colLabels=["Parameter", "Unit", "Value", "Status"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(PRESENTATION_STYLE["table_font_size"])
    table.scale(1.18, 1.65)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#111827")
        cell.set_linewidth(1.2 if row == 0 else 0.9)
        if row == 0:
            cell.set_text_props(fontweight="semibold")
    ax.set_title(
        "Extracted parameters from waveform CSV\n"
        f"success={bool(fit_stats['success'])}  cost={fit_stats['cost']:.3e}  nfev={fit_stats['nfev']:.0f}",
        fontsize=16,
        pad=16,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit photodetector_model.py parameters from input/output waveform CSV files."
    )
    parser.add_argument("--input-csv", type=str, required=True, help="CSV of the input square waveform.")
    parser.add_argument("--output-csv", type=str, required=True, help="CSV of the output response waveform.")
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1e-6,
        help="Multiply CSV x-axis values by this factor to convert to seconds. Default assumes us.",
    )
    parser.add_argument(
        "--fit-keys",
        type=str,
        default=",".join(DEFAULT_FIT_KEYS),
        help="Comma-separated parameter names to fit.",
    )
    parser.add_argument(
        "--fixed-params",
        type=str,
        default=DEFAULT_FIXED_PARAM_TEXT,
        help="Comma-separated fixed parameter assignments, e.g. eta_fast=1,eta_slow=1.",
    )
    parser.add_argument(
        "--pulse-indices",
        type=str,
        default="",
        help="Optional comma-separated 1-based ON-pulse indices to fit, e.g. 2,3. Selected pulses must be consecutive.",
    )
    parser.add_argument(
        "--fit-time-start",
        type=float,
        default=None,
        help="Optional absolute start time in seconds of the waveform window to fit.",
    )
    parser.add_argument(
        "--fit-time-end",
        type=float,
        default=None,
        help="Optional absolute end time in seconds of the waveform window to fit.",
    )
    parser.add_argument(
        "--assumed-p-on",
        type=float,
        default=1.0,
        help="Assigned optical power level for detected ON samples.",
    )
    parser.add_argument(
        "--input-threshold",
        type=float,
        default=None,
        help="Optional threshold in normalized input units for ON/OFF detection. Default auto-infers midpoint.",
    )
    parser.add_argument(
        "--dark-percentile",
        type=float,
        default=2.0,
        help="Percentile used to estimate output baseline.",
    )
    parser.add_argument(
        "--input-smooth-window",
        type=int,
        default=7,
        help="Odd Savitzky-Golay window for the input trace. Use 0 to disable.",
    )
    parser.add_argument(
        "--output-smooth-window",
        type=int,
        default=11,
        help="Odd Savitzky-Golay window for the output trace. Use 0 to disable.",
    )
    parser.add_argument("--resample-points", type=int, default=480, help="Uniform samples used for fitting.")
    parser.add_argument("--max-nfev", type=int, default=500, help="Maximum least-squares evaluations.")
    parser.add_argument("--output-prefix", type=str, default="outputs/csv_waveform_fit", help="Output prefix.")
    return parser.parse_args()


def load_waveform_csv(path: Path, time_scale: float) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                t_value = float(row[0]) * time_scale
                y_value = float(row[1])
            except ValueError:
                continue
            rows.append((t_value, y_value))

    if len(rows) < 3:
        raise ValueError(f"Expected at least 3 numeric samples in {path}, got {len(rows)}")

    data = np.asarray(rows, dtype=float)
    order = np.argsort(data[:, 0], kind="mergesort")
    t_sorted = data[order, 0]
    y_sorted = data[order, 1]

    unique_t, inverse = np.unique(t_sorted, return_inverse=True)
    y_accum = np.zeros_like(unique_t)
    counts = np.zeros_like(unique_t)
    np.add.at(y_accum, inverse, y_sorted)
    np.add.at(counts, inverse, 1.0)
    y_unique = y_accum / np.maximum(counts, 1.0)
    return unique_t, y_unique


def load_waveform_points(path: Path, time_scale: float) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    with path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                t_value = float(row[0]) * time_scale
                y_value = float(row[1])
            except ValueError:
                continue
            rows.append((t_value, y_value))
    if len(rows) < 3:
        raise ValueError(f"Expected at least 3 numeric samples in {path}, got {len(rows)}")
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1]


def normalize_trace(y: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    lo = float(np.percentile(y, 5.0))
    hi = float(np.percentile(y, 95.0))
    if hi <= lo:
        lo = float(np.min(y))
        hi = float(np.max(y))
    scale = hi - lo
    if scale <= 1e-12:
        return np.zeros_like(y), (lo, hi)
    normalized = np.clip((y - lo) / scale, 0.0, 1.0)
    return normalized, (lo, hi)


def reconstruct_square_input(
    t_points: np.ndarray,
    y_points: np.ndarray,
    *,
    threshold: float | None,
    min_run_length: int = 3,
) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(t_points, kind="mergesort")
    t_sorted = np.asarray(t_points[order], dtype=float)
    y_sorted = np.asarray(y_points[order], dtype=float)

    unique_t, inverse = np.unique(t_sorted, return_inverse=True)
    y_median = np.zeros_like(unique_t)
    for idx in range(len(unique_t)):
        y_median[idx] = float(np.median(y_sorted[inverse == idx]))

    y_norm, _ = normalize_trace(y_median)
    auto_threshold = 0.5 * (
        float(np.percentile(y_norm, 10.0)) + float(np.percentile(y_norm, 90.0))
    )
    threshold_value = float(auto_threshold if threshold is None else threshold)
    states = y_norm >= threshold_value

    if len(unique_t) >= 3:
        cleaned = states.copy()
        for idx in range(1, len(states) - 1):
            if states[idx - 1] == states[idx + 1] != states[idx]:
                cleaned[idx] = states[idx - 1]
        states = cleaned

    states = remove_short_binary_runs(states, min_run_length=min_run_length)

    square_norm = states.astype(float)
    return unique_t, square_norm, threshold_value


def build_power_from_input(
    t_input: np.ndarray,
    input_trace: np.ndarray,
    t_output: np.ndarray,
    threshold: float | None,
    p_on: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    t_square, square_norm, threshold_value = reconstruct_square_input(
        t_input,
        input_trace,
        threshold=threshold,
    )
    power_interp = np.interp(t_output, t_square, square_norm)
    on_mask_output = remove_short_binary_runs(power_interp >= 0.5, min_run_length=3)
    power_output = on_mask_output.astype(float) * p_on
    return on_mask_output, power_output, threshold_value


def save_fit_figure(
    output_path: Path,
    t_input: np.ndarray,
    input_norm: np.ndarray,
    threshold_value: float,
    t_output_raw: np.ndarray,
    measured_output_raw: np.ndarray,
    t_fit: np.ndarray,
    measured_fit: np.ndarray,
    fitted_fit: np.ndarray,
    power_fit: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    ensure_parent_dir(output_path)
    fig, axes = plt.subplots(3, 1, figsize=(12.4, 9.6), gridspec_kw={"height_ratios": [1.3, 2.1, 1.0]})
    fig.patch.set_facecolor("white")

    axes[0].step(t_input, input_norm, where="post", color="#2563eb", linewidth=2.2, label="reconstructed input")
    axes[0].axhline(threshold_value, color="#dc2626", linestyle="--", linewidth=1.6, label="ON threshold")
    apply_axis_style(axes[0], title="Input square waveform", ylabel="Normalized input")
    style_legend(axes[0].legend(loc="upper right"))

    axes[1].scatter(
        t_output_raw,
        measured_output_raw,
        color="#93c5fd",
        s=18,
        alpha=0.75,
        edgecolors="none",
        label="output2",
    )
    axes[1].plot(t_fit, fitted_fit, color="#dc2626", linewidth=2.3, label="model fit")
    axes[1].fill_between(
        t_fit,
        np.min(measured_fit),
        np.max(measured_fit),
        where=power_fit > 0,
        color="#d4d84b",
        alpha=0.20,
        label="ON region",
    )
    apply_axis_style(axes[1], title="Output waveform and fitted result", ylabel="Response (a.u.)")
    style_legend(axes[1].legend(loc="upper right"))

    residual = fitted_fit - measured_fit
    axes[2].plot(t_fit, residual, color="#111827", linewidth=2.0)
    axes[2].axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.4)
    apply_axis_style(axes[2], title="Fit residual", xlabel="Time (s)", ylabel="Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_prefix = Path(args.output_prefix)
    ensure_parent_dir(output_prefix.with_suffix(".tmp"))

    fit_keys = parse_name_list(args.fit_keys)
    fixed_params = parse_fixed_params(args.fixed_params)
    pulse_indices = parse_pulse_indices(args.pulse_indices)
    if pulse_indices and (args.fit_time_start is not None or args.fit_time_end is not None):
        raise ValueError("Use either --pulse-indices or --fit-time-start/--fit-time-end, not both.")

    t_input_points, y_input_points = load_waveform_points(input_csv, args.time_scale)
    t_output, y_output = load_waveform_csv(output_csv, args.time_scale)
    y_output = smooth_signal(y_output, args.output_smooth_window)

    if args.input_smooth_window > 2:
        t_input_pre, y_input_pre = load_waveform_csv(input_csv, args.time_scale)
        y_input_for_fit = smooth_signal(y_input_pre, args.input_smooth_window)
        t_input_for_fit, input_norm, threshold_value = reconstruct_square_input(
            t_input_pre,
            y_input_for_fit,
            threshold=args.input_threshold,
        )
    else:
        t_input_for_fit, input_norm, threshold_value = reconstruct_square_input(
            t_input_points,
            y_input_points,
            threshold=args.input_threshold,
        )
    on_mask_output, _, threshold_value = build_power_from_input(
        t_input=t_input_points,
        input_trace=y_input_points,
        t_output=t_output,
        threshold=args.input_threshold,
        p_on=args.assumed_p_on,
    )

    detected_segments = find_on_segments(on_mask_output)
    if args.fit_time_start is not None or args.fit_time_end is not None:
        t_selected_raw, response_selected_raw, on_mask_selected_raw, selected_window = select_time_subset(
            t=t_output,
            response=y_output,
            on_mask=on_mask_output,
            time_start=args.fit_time_start,
            time_end=args.fit_time_end,
        )
    else:
        t_selected_raw, response_selected_raw, on_mask_selected_raw, selected_window, _ = select_pulse_subset(
            t=t_output,
            response=y_output,
            on_mask=on_mask_output,
            pulse_indices=pulse_indices,
        )

    t_fit, measured_fit, power_fit = resample_uniform(
        t=t_selected_raw,
        response=response_selected_raw,
        on_mask=on_mask_selected_raw,
        num_points=args.resample_points,
        p_on=args.assumed_p_on,
    )

    dark_current = float(np.percentile(measured_fit, args.dark_percentile))
    initial_params = build_initial_guess(measured_fit, dark_current, fixed_params)

    fitted_vec, fit_stats = fit_parameters_subset(
        t=t_fit,
        P=power_fit,
        I_data=measured_fit,
        fit_keys=fit_keys,
        initial_params=initial_params,
        fixed_params=fixed_params,
        param_bounds=IMAGE_FIT_PARAM_BOUNDS,
        dark_current=dark_current,
        max_nfev=args.max_nfev,
    )
    _, fitted_fit, _, _, _ = simulate(
        t_fit,
        power_fit,
        fitted_vec,
        N_CARRIER,
        noise_fn=None,
        dark_current=dark_current,
    )
    fitted_params = vec_to_params(fitted_vec)

    fit_plot_path = output_prefix.with_name(output_prefix.name + "_fit.png")
    param_plot_path = output_prefix.with_name(output_prefix.name + "_params.png")
    data_csv_path = output_prefix.with_name(output_prefix.name + "_digitized.csv")
    param_csv_path = output_prefix.with_name(output_prefix.name + "_params.csv")

    save_fit_figure(
        output_path=fit_plot_path,
        t_input=t_input_for_fit,
        input_norm=input_norm,
        threshold_value=threshold_value,
        t_output_raw=t_output,
        measured_output_raw=y_output,
        t_fit=t_fit,
        measured_fit=measured_fit,
        fitted_fit=fitted_fit,
        power_fit=power_fit,
    )
    save_parameter_figure(
        output_path=param_plot_path,
        fitted_params=fitted_params,
        fit_keys=fit_keys,
        fixed_params=fixed_params,
        fit_stats=fit_stats,
    )
    save_digitized_csv(
        output_path=data_csv_path,
        t_fit=t_fit,
        measured=measured_fit,
        power_fit=power_fit,
        fitted=fitted_fit,
    )
    save_parameter_csv(
        output_path=param_csv_path,
        fitted_params=fitted_params,
        fit_keys=fit_keys,
        fixed_params=fixed_params,
    )

    print(f"input_csv              {input_csv}")
    print(f"output_csv             {output_csv}")
    print(f"time_scale_s_per_unit  {args.time_scale:.6e}")
    print(f"input_threshold_norm   {threshold_value:.6f}")
    print(f"dark_current_au        {dark_current:.6e}")
    print(f"detected_pulses        {len(detected_segments)}")
    if pulse_indices:
        print(f"selected_pulses        {pulse_indices}")
        print(f"selected_window_s      ({selected_window[0]:.6e}, {selected_window[1]:.6e})")
    elif args.fit_time_start is not None or args.fit_time_end is not None:
        print("selected_mode          absolute-time-window")
        print(f"selected_window_s      ({selected_window[0]:.6e}, {selected_window[1]:.6e})")
    print(f"optimizer_success      {fit_stats['success']}")
    print(f"optimizer_cost         {fit_stats['cost']:.6e}")
    print(f"optimizer_nfev         {fit_stats['nfev']:.0f}")
    for key in PARAM_KEYS:
        print(f"{key:<22} {fitted_params[key]:.6e}")
    print(f"fit_plot               {fit_plot_path}")
    print(f"param_plot             {param_plot_path}")
    print(f"digitized_csv          {data_csv_path}")
    print(f"params_csv             {param_csv_path}")


if __name__ == "__main__":
    main()
