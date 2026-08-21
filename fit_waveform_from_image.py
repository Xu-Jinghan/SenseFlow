from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from photodetector_model import (
    N_CARRIER,
    PARAM_KEYS,
    PARAM_BOUNDS,
    NO_DRIFT_FIXED_PARAMS,
    PARAMS_TRUE,
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
    # Disable the explicit baseline-drift branch unless explicitly overridden.
    **NO_DRIFT_FIXED_PARAMS,
}
DEFAULT_FIXED_PARAM_TEXT = ",".join(f"{key}={value}" for key, value in DEFAULT_FIXED_PARAMS.items())

# Looser bounds for digitized-paper waveforms in arbitrary units. These
# fits are driven by a single normalized ON/OFF trace rather than
# calibrated A/W data, so the x2 branch needs more headroom than the
# physical-simulation defaults.
IMAGE_FIT_PARAM_BOUNDS = dict(PARAM_BOUNDS)
IMAGE_FIT_PARAM_BOUNDS.update(
    {
        "tau_rise_fast": (1e-4, 5e-3, "log"),
        "tau_fall_fast": (1e-4, 1e-2, "log"),
        "tau_rise_slow": (1e-3, 1e-1, "log"),
        "tau_fall_slow": (5e-3, 5e-1, "log"),
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
    "box_line_width": 3.0,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Digitize a time-Iout waveform from an image and fit it with photodetector_model.py."
    )
    parser.add_argument("--image", type=str, required=True, help="Path to the waveform image.")
    parser.add_argument("--time-min", type=float, default=0.0, help="Left x-axis value in seconds.")
    parser.add_argument("--time-max", type=float, default=1.2, help="Right x-axis value in seconds.")
    parser.add_argument("--response-min", type=float, default=0.0, help="Bottom y-axis value in arbitrary units.")
    parser.add_argument("--response-max", type=float, default=1.0, help="Top y-axis value in arbitrary units.")
    parser.add_argument(
        "--plot-bbox",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=None,
        help="Optional manual plot bounding box in image pixels.",
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
        help="Optional absolute start time (s) of the waveform window to fit.",
    )
    parser.add_argument(
        "--fit-time-end",
        type=float,
        default=None,
        help="Optional absolute end time (s) of the waveform window to fit.",
    )
    parser.add_argument("--assumed-p-on", type=float, default=1.0, help="Assumed ON optical power level.")
    parser.add_argument("--dark-percentile", type=float, default=2.0, help="Percentile used to estimate baseline.")
    parser.add_argument("--resample-points", type=int, default=480, help="Uniform samples used for fitting.")
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=11,
        help="Odd Savitzky-Golay window for the digitized trace. Use 0 to disable.",
    )
    parser.add_argument("--max-nfev", type=int, default=500, help="Maximum least-squares evaluations.")
    parser.add_argument("--output-prefix", type=str, default="outputs/image_waveform_fit", help="Output prefix.")
    return parser.parse_args()


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


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def _smooth_1d(values: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = np.ones(kernel_size, dtype=float) / kernel_size
    return np.convolve(values, kernel, mode="same")


def _peak_band(score: np.ndarray, peak_idx: int, frac: float = 0.6) -> tuple[int, int]:
    threshold = max(0.02, frac * float(score[peak_idx]))
    left = peak_idx
    right = peak_idx
    while left > 0 and score[left - 1] >= threshold:
        left -= 1
    while right < len(score) - 1 and score[right + 1] >= threshold:
        right += 1
    return left, right


def auto_detect_plot_bbox(image_bgr: np.ndarray) -> tuple[int, int, int, int]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark = gray < 80

    row_score = _smooth_1d(dark.mean(axis=1).astype(float), kernel_size=11)
    col_score = _smooth_1d(dark.mean(axis=0).astype(float), kernel_size=11)

    half_h = len(row_score) // 2
    half_w = len(col_score) // 2
    top_peak = int(np.argmax(row_score[:half_h]))
    bottom_peak = int(half_h + np.argmax(row_score[half_h:]))
    left_peak = int(np.argmax(col_score[:half_w]))
    right_peak = int(half_w + np.argmax(col_score[half_w:]))

    top_band = _peak_band(row_score, top_peak)
    bottom_band = _peak_band(row_score, bottom_peak)
    left_band = _peak_band(col_score, left_peak)
    right_band = _peak_band(col_score, right_peak)

    peak_bbox = (
        int(round((left_band[0] + left_band[1]) / 2)),
        int(round((top_band[0] + top_band[1]) / 2)),
        int(round((right_band[0] + right_band[1]) / 2)),
        int(round((bottom_band[0] + bottom_band[1]) / 2)),
    )
    if (
        peak_bbox[2] - peak_bbox[0] > 0.3 * image_bgr.shape[1]
        and peak_bbox[3] - peak_bbox[1] > 0.3 * image_bgr.shape[0]
    ):
        return peak_bbox

    row_thresh = max(0.05, 0.42 * float(row_score.max()))
    col_thresh = max(0.05, 0.42 * float(col_score.max()))
    rows = np.where(row_score >= row_thresh)[0]
    cols = np.where(col_score >= col_thresh)[0]

    if len(rows) >= 2 and len(cols) >= 2:
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])
        if (x1 - x0) > 0.3 * image_bgr.shape[1] and (y1 - y0) > 0.3 * image_bgr.shape[0]:
            return x0, y0, x1, y1

    binary = (dark.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    best_bbox = None
    best_area = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < 0.15 * image_area:
            continue
        if area > best_area:
            best_area = area
            best_bbox = (x, y, x + w - 1, y + h - 1)

    if best_bbox is None:
        raise RuntimeError("Failed to auto-detect the plot area. Please pass --plot-bbox.")
    return best_bbox


def shrink_bbox(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0 + 1
    height = y1 - y0 + 1
    pad_x = max(2, int(round(width * 0.01)))
    pad_y = max(2, int(round(height * 0.01)))
    x0 = max(0, x0 + pad_x)
    y0 = max(0, y0 + pad_y)
    x1 = min(image_shape[1] - 1, x1 - pad_x)
    y1 = min(image_shape[0] - 1, y1 - pad_y)
    if x1 <= x0 or y1 <= y0:
        raise RuntimeError("Plot box became invalid after removing the border.")
    return x0, y0, x1, y1


def build_blue_mask(plot_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    blue_hsv = cv2.inRange(hsv, (85, 40, 30), (140, 255, 255))
    b, g, r = cv2.split(plot_bgr)
    blue_rgb = ((b > g + 15) & (b > r + 15) & (b > 50)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(blue_hsv, blue_rgb)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask > 0


def build_yellow_mask(plot_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2HSV)
    yellow_hsv = cv2.inRange(hsv, (18, 25, 80), (45, 255, 255))
    b, g, r = cv2.split(plot_bgr)
    yellow_rgb = (
        (r > 90)
        & (g > 90)
        & (r > b + 15)
        & (g > b + 10)
    ).astype(np.uint8) * 255
    mask = cv2.bitwise_and(yellow_hsv, yellow_rgb)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask > 0


def extract_trace_pixels(trace_mask: np.ndarray) -> np.ndarray:
    height, width = trace_mask.shape
    y_pixels = np.full(width, np.nan, dtype=float)
    for x in range(width):
        ys = np.flatnonzero(trace_mask[:, x])
        if ys.size:
            y_pixels[x] = float(np.median(ys))

    valid = np.isfinite(y_pixels)
    if valid.sum() < max(20, width // 8):
        raise RuntimeError("Failed to extract enough blue trace pixels from the image.")
    x_axis = np.arange(width, dtype=float)
    y_pixels = np.interp(x_axis, x_axis[valid], y_pixels[valid])
    return y_pixels


def extract_on_mask(yellow_mask: np.ndarray, response_trace: np.ndarray) -> tuple[np.ndarray, str]:
    score = yellow_mask.mean(axis=0).astype(float)
    if float(score.max()) > 0.02:
        scaled = np.clip(score / max(float(score.max()), 1e-9) * 255.0, 0, 255).astype(np.uint8)[None, :]
        _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary.astype(np.uint8)
        kernel_w = max(5, int(round(yellow_mask.shape[1] * 0.02)) | 1)
        kernel = np.ones((1, kernel_w), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        on_mask = binary[0] > 0
        if on_mask.any() and (~on_mask).any():
            return on_mask, "yellow-band"

    threshold = 0.5 * (float(np.nanmin(response_trace)) + float(np.nanmax(response_trace)))
    fallback = response_trace >= threshold
    return fallback, "response-threshold"


def map_pixels_to_axes(
    y_pixels: np.ndarray,
    width: int,
    height: int,
    time_min: float,
    time_max: float,
    response_min: float,
    response_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_pixels = np.arange(width, dtype=float)
    t = time_min + (time_max - time_min) * x_pixels / max(width - 1, 1)
    response = response_max - (response_max - response_min) * y_pixels / max(height - 1, 1)
    return t, response


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


def save_fit_figure(
    output_path: Path,
    image_bgr: np.ndarray,
    plot_bbox: tuple[int, int, int, int],
    trace_y_pixels: np.ndarray,
    t_fit: np.ndarray,
    measured_fit: np.ndarray,
    fitted_fit: np.ndarray,
    power_fit: np.ndarray,
    detection_source: str,
) -> None:
    ensure_parent_dir(output_path)
    x0, y0, _, _ = plot_bbox
    overlay_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    x_coords = x0 + np.arange(len(trace_y_pixels))
    y_coords = y0 + trace_y_pixels

    fig, axes = plt.subplots(3, 1, figsize=(13.2, 10.8), gridspec_kw={"height_ratios": [3, 2, 1]}, sharex=False)
    fig.patch.set_facecolor("white")

    axes[0].imshow(overlay_rgb)
    axes[0].plot(
        x_coords,
        y_coords,
        color="#ef4444",
        linewidth=PRESENTATION_STYLE["secondary_line_width"],
        label="digitized trace",
    )
    axes[0].add_patch(
        plt.Rectangle(
            (plot_bbox[0], plot_bbox[1]),
            plot_bbox[2] - plot_bbox[0],
            plot_bbox[3] - plot_bbox[1],
            fill=False,
            edgecolor="#22c55e",
            linewidth=PRESENTATION_STYLE["box_line_width"],
        )
    )
    axes[0].set_title(
        "Detected plot area and digitized waveform",
        fontsize=PRESENTATION_STYLE["title_size"],
        pad=12,
        fontweight="semibold",
    )
    style_legend(axes[0].legend(loc="upper right", fontsize=PRESENTATION_STYLE["legend_size"]))
    axes[0].axis("off")

    axes[1].plot(
        t_fit,
        measured_fit,
        color="#2563eb",
        linewidth=PRESENTATION_STYLE["line_width"],
        label="digitized response",
    )
    axes[1].plot(
        t_fit,
        fitted_fit,
        color="#dc2626",
        linewidth=PRESENTATION_STYLE["secondary_line_width"],
        label="model fit",
    )
    axes[1].fill_between(
        t_fit,
        np.min(measured_fit),
        np.max(measured_fit),
        where=power_fit > 0,
        color="#d4d84b",
        alpha=0.20,
        label=f"ON region ({detection_source})",
    )
    apply_axis_style(axes[1], title="Digitized waveform and fitted result", ylabel="Response (a.u.)")
    style_legend(axes[1].legend(loc="upper right", fontsize=PRESENTATION_STYLE["legend_size"]))

    residual = fitted_fit - measured_fit
    axes[2].plot(t_fit, residual, color="#111827", linewidth=2.2)
    axes[2].axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.5)
    apply_axis_style(axes[2], title="Fit residual", xlabel="Time (s)", ylabel="Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches="tight")
    plt.close(fig)


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
        "Extracted parameters from digitized waveform\n"
        f"success={bool(fit_stats['success'])}  cost={fit_stats['cost']:.3e}  nfev={fit_stats['nfev']:.0f}",
        fontsize=16,
        pad=16,
        fontweight="semibold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).expanduser().resolve()
    output_prefix = Path(args.output_prefix)
    ensure_parent_dir(output_prefix.with_suffix(".tmp"))

    fit_keys = parse_name_list(args.fit_keys)
    fixed_params = parse_fixed_params(args.fixed_params)
    pulse_indices = parse_pulse_indices(args.pulse_indices)
    if pulse_indices and (args.fit_time_start is not None or args.fit_time_end is not None):
        raise ValueError("Use either --pulse-indices or --fit-time-start/--fit-time-end, not both.")

    image_bgr = load_image(image_path)
    raw_bbox = tuple(args.plot_bbox) if args.plot_bbox is not None else auto_detect_plot_bbox(image_bgr)
    plot_bbox = shrink_bbox(raw_bbox, image_bgr.shape)
    x0, y0, x1, y1 = plot_bbox
    plot_bgr = image_bgr[y0 : y1 + 1, x0 : x1 + 1]

    trace_mask = build_blue_mask(plot_bgr)
    yellow_mask = build_yellow_mask(plot_bgr)
    trace_y_pixels = extract_trace_pixels(trace_mask)

    t_raw, response_raw = map_pixels_to_axes(
        y_pixels=trace_y_pixels,
        width=plot_bgr.shape[1],
        height=plot_bgr.shape[0],
        time_min=args.time_min,
        time_max=args.time_max,
        response_min=args.response_min,
        response_max=args.response_max,
    )
    response_raw = smooth_signal(response_raw, args.smooth_window)
    on_mask_raw, detection_source = extract_on_mask(yellow_mask, response_raw)
    detected_segments = find_on_segments(on_mask_raw)
    if args.fit_time_start is not None or args.fit_time_end is not None:
        t_selected_raw, response_selected_raw, on_mask_selected_raw, selected_window = select_time_subset(
            t=t_raw,
            response=response_raw,
            on_mask=on_mask_raw,
            time_start=args.fit_time_start,
            time_end=args.fit_time_end,
        )
    else:
        t_selected_raw, response_selected_raw, on_mask_selected_raw, selected_window, _ = select_pulse_subset(
            t=t_raw,
            response=response_raw,
            on_mask=on_mask_raw,
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
        image_bgr=image_bgr,
        plot_bbox=plot_bbox,
        trace_y_pixels=trace_y_pixels,
        t_fit=t_fit,
        measured_fit=measured_fit,
        fitted_fit=fitted_fit,
        power_fit=power_fit,
        detection_source=detection_source,
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

    print(f"image                 {image_path}")
    print(f"plot_bbox             {plot_bbox}")
    print(f"dark_current_au       {dark_current:.6e}")
    print(f"on_detection          {detection_source}")
    print(f"detected_pulses       {len(detected_segments)}")
    if pulse_indices:
        print(f"selected_pulses       {pulse_indices}")
        print(f"selected_window_s     ({selected_window[0]:.6f}, {selected_window[1]:.6f})")
    elif args.fit_time_start is not None or args.fit_time_end is not None:
        print("selected_mode         absolute-time-window")
        print(f"selected_window_s     ({selected_window[0]:.6f}, {selected_window[1]:.6f})")
    print(f"optimizer_success     {fit_stats['success']}")
    print(f"optimizer_cost        {fit_stats['cost']:.6e}")
    print(f"optimizer_nfev        {fit_stats['nfev']:.0f}")
    for key in PARAM_KEYS:
        print(f"{key:<18} {fitted_params[key]:.6e}")
    print(f"fit_plot              {fit_plot_path}")
    print(f"param_plot            {param_plot_path}")
    print(f"digitized_csv         {data_csv_path}")
    print(f"params_csv            {param_csv_path}")


if __name__ == "__main__":
    main()
