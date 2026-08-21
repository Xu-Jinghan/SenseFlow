"""
Analyze photodetector paper dataset by major material and structure classes.

This script focuses on four paper-level figures of merit:
- Response (responsivity)
- LDR
- Response time
- D*

The source CSV contains a few rows with shifted metric columns, especially in the
paper-comparison table entries. To keep the original dataset untouched, this script
parses the file defensively and extracts the target metrics with lightweight
heuristics that match the table layout.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATASET_PATH = Path("data") / "photodetector_paper_dataset.csv"
OUTPUT_DIR = Path("outputs") / "photodetector_dataset_analysis"

HEADER = [
    "entry_id",
    "material",
    "material_subcategory",
    "specific_material",
    "structure",
    "paper_year",
    "paper_title",
    "doi",
    "notes",
    "substrate",
    "waveband",
    "responsivity_bias_V",
    "stability",
    "ref_id",
    "D_star_jones",
    "D_note",
    "LDR_dB",
    "LDR_note",
    "EQE_pct",
    "f3db_Hz",
    "R_fast",
    "eta_fast",
    "tau_rise_fast",
    "tau_fall_fast",
    "R_slow",
    "eta_slow",
    "tau_rise_slow",
    "tau_fall_slow",
    "alpha",
    "beta",
    "delta",
    "gamma",
    "tau_drift",
    "drift_scale",
    "white_sigma",
    "flicker_sigma",
    "kappa",
    "low_freq_amp",
    "low_freq_hz",
]

MAJOR_STRUCTURES = {
    "photoconductor": "Photoconductor",
    "photodiode": "Photodiode",
    "phototransistor": "Phototransistor",
    "photovoltaic": "Photovoltaic",
}

MAJOR_MATERIALS = {
    "perovskite": "Perovskite",
    "2d semiconductor": "2D semiconductor",
    "oxide semiconductor": "Oxide semiconductor",
    "organic semiconductor": "Organic semiconductor",
}

TARGET_METRICS = {
    "response_a_per_w": "Response (A/W)",
    "ldr_db": "LDR (dB)",
    "response_time_s": "Response time (s)",
    "d_star_jones": "D* (Jones)",
}

FOCUS_STRUCTURES = ("Photoconductor", "Photodiode", "Phototransistor")
FOCUS_STRUCTURE_METRICS = ("response_a_per_w", "response_time_s", "d_star_jones")
RADAR_NOISE_SCORES = {
    "Photoconductor": 0.93,
    "Photodiode": 0.08,
    "Phototransistor": 0.97,
}
RADAR_DISPLAY_MIN = 0.0
RADAR_DISPLAY_MAX = 1.0
SPEED_RADAR_PERCENTILE_LOW = 0.20
SPEED_RADAR_PERCENTILE_HIGH = 0.80
RADAR_LABEL_PAD = 22
RADAR_LEGEND_BBOX_X = 1.20
RADAR_LEGEND_BBOX_Y = 0.16
STRUCTURE_TICK_LABEL_SIZE = 26
STRUCTURE_AXIS_LABEL_SIZE = 28
STRUCTURE_PANEL_TITLE_SIZE = 32
STRUCTURE_LOG_NOTE_SIZE = 21
STRUCTURE_SUPTITLE_SIZE = 38
STRUCTURE_XLABEL_ROTATION = 28

GROUP_COLORS = {
    "Perovskite": "#1f77b4",
    "2D semiconductor": "#ff7f0e",
    "Oxide semiconductor": "#2ca02c",
    "Photoconductor": "#2ca02c",
    "Photodiode": "#d62728",
    "Phototransistor": "#1f77b4",
    "Photovoltaic": "#17becf",
}


def normalize_text(value: str) -> Optional[str]:
    text = str(value).strip()
    return text or None


def parse_float(value: str) -> Optional[float]:
    text = normalize_text(value)
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def normalize_material(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.strip().lower()
    return MAJOR_MATERIALS.get(key)


def normalize_structure(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    key = value.strip().lower()
    return MAJOR_STRUCTURES.get(key)


def keep_major_entry(entry_id: Optional[str], material: Optional[str], structure: Optional[str]) -> bool:
    if not entry_id or entry_id.startswith("default_") or entry_id.startswith("paper_stub_"):
        return False
    if material is None or structure is None:
        return False
    return True


def geometric_mean(values: Sequence[float]) -> Optional[float]:
    positives = [value for value in values if value > 0]
    if not positives:
        return None
    return math.exp(sum(math.log(value) for value in positives) / len(positives))


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def summarize_values(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "geomean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "geomean": geometric_mean(values),
        "min": min(values),
        "max": max(values),
    }


def read_rows(path: Path) -> List[List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        expected_prefix = HEADER[:18]
        if header[: len(expected_prefix)] != expected_prefix:
            raise ValueError(f"Unexpected CSV header in {path}")
        return list(reader)


def extract_metric_triplet(row: Sequence[str]) -> Dict[str, Optional[float]]:
    # The paper tables are inconsistent around columns 18-22. We only need
    # responsivity and response time, so we compress the non-empty numeric
    # entries in that local window and interpret them as:
    # [response, rise_time, fall_time]
    numeric_values: List[float] = []
    response_index = HEADER.index("R_fast")
    slow_fall_index = HEADER.index("tau_fall_slow")
    for raw_value in row[response_index : slow_fall_index + 1]:
        value = parse_float(raw_value)
        if value is not None:
            numeric_values.append(value)

    response = numeric_values[0] if len(numeric_values) >= 1 else None
    rise_time = numeric_values[1] if len(numeric_values) >= 2 else None
    fall_time = numeric_values[2] if len(numeric_values) >= 3 else None
    response_time_candidates = [value for value in (rise_time, fall_time) if value is not None]
    response_time = max(response_time_candidates) if response_time_candidates else None

    return {
        "response_a_per_w": response,
        "response_rise_s": rise_time,
        "response_fall_s": fall_time,
        "response_time_s": response_time,
    }


def parse_entry(row: Sequence[str]) -> Dict[str, Optional[object]]:
    padded = list(row) + [""] * max(0, len(HEADER) - len(row))
    base = {name: normalize_text(padded[index]) for index, name in enumerate(HEADER)}
    metrics = extract_metric_triplet(padded)

    return {
        "entry_id": base["entry_id"],
        "material_major": normalize_material(base["material"]),
        "material_raw": base["material"],
        "material_subcategory": base["material_subcategory"],
        "specific_material": base["specific_material"],
        "structure_major": normalize_structure(base["structure"]),
        "structure_raw": base["structure"],
        "notes": base["notes"],
        "ref_id": base["ref_id"],
        "d_star_jones": parse_float(base["D_star_jones"] or ""),
        "ldr_db": parse_float(base["LDR_dB"] or ""),
        **metrics,
    }


def collect_entries(rows: Iterable[Sequence[str]]) -> List[Dict[str, Optional[object]]]:
    parsed_entries = [parse_entry(row) for row in rows]
    return [
        entry
        for entry in parsed_entries
        if keep_major_entry(
            entry_id=entry["entry_id"],
            material=entry["material_major"],
            structure=entry["structure_major"],
        )
    ]


def build_group_rows(
    entries: Sequence[Dict[str, Optional[object]]],
    group_key: str,
    group_label: str,
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, Optional[object]]]] = defaultdict(list)
    for entry in entries:
        group_value = entry[group_key]
        if group_value is None:
            continue
        grouped[str(group_value)].append(entry)

    summary_rows: List[Dict[str, object]] = []
    for value in sorted(grouped):
        group_entries = grouped[value]
        base_row: Dict[str, object] = {
            group_label: value,
            "device_count": len(group_entries),
        }
        for metric_key in TARGET_METRICS:
            metric_values = [
                float(entry[metric_key])  # type: ignore[arg-type]
                for entry in group_entries
                if entry[metric_key] is not None
            ]
            summary = summarize_values(metric_values)
            base_row[f"{metric_key}_count"] = summary["count"]
            base_row[f"{metric_key}_median"] = summary["median"]
            base_row[f"{metric_key}_mean"] = summary["mean"]
            base_row[f"{metric_key}_geomean"] = summary["geomean"]
            base_row[f"{metric_key}_min"] = summary["min"]
            base_row[f"{metric_key}_max"] = summary["max"]
        summary_rows.append(base_row)
    return summary_rows


def build_cross_rows(entries: Sequence[Dict[str, Optional[object]]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, Optional[object]]]] = defaultdict(list)
    for entry in entries:
        material = entry["material_major"]
        structure = entry["structure_major"]
        if material is None or structure is None:
            continue
        grouped[(str(material), str(structure))].append(entry)

    summary_rows: List[Dict[str, object]] = []
    for material, structure in sorted(grouped):
        group_entries = grouped[(material, structure)]
        row: Dict[str, object] = {
            "material_major": material,
            "structure_major": structure,
            "device_count": len(group_entries),
        }
        for metric_key in TARGET_METRICS:
            metric_values = [
                float(entry[metric_key])  # type: ignore[arg-type]
                for entry in group_entries
                if entry[metric_key] is not None
            ]
            summary = summarize_values(metric_values)
            row[f"{metric_key}_count"] = summary["count"]
            row[f"{metric_key}_median"] = summary["median"]
            row[f"{metric_key}_geomean"] = summary["geomean"]
        summary_rows.append(row)
    return summary_rows


def sanitize_filename(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_")


def metric_values_by_group(
    entries: Sequence[Dict[str, Optional[object]]],
    group_key: str,
    metric_key: str,
) -> Dict[str, List[float]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for entry in entries:
        group_value = entry[group_key]
        metric_value = entry[metric_key]
        if group_value is None or metric_value is None:
            continue
        grouped[str(group_value)].append(float(metric_value))
    return dict(sorted(grouped.items()))


def make_log_metric(values: Sequence[float], metric_key: str) -> List[float]:
    if metric_key in {"response_a_per_w", "response_time_s", "d_star_jones"}:
        return [math.log10(value) for value in values if value > 0]
    return list(values)


def metric_axis_label(metric_key: str) -> str:
    label = TARGET_METRICS[metric_key]
    if metric_key in {"response_a_per_w", "response_time_s", "d_star_jones"}:
        return f"log10 {label}"
    return label


def style_axis_text(axis: plt.Axes) -> None:
    axis.tick_params(axis="both", which="major", labelsize=STRUCTURE_TICK_LABEL_SIZE, width=2.0, length=7)
    axis.tick_params(axis="both", which="minor", width=1.4, length=4)
    for spine in axis.spines.values():
        spine.set_linewidth(2.2)
        spine.set_color("#303030")
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontweight("bold")


def swarm_offsets_for_log_values(
    values: Sequence[float],
    x_step: float = 0.10,
    y_threshold: float = 0.14,
    max_layers: int = 5,
) -> List[float]:
    if not values:
        return []

    sorted_pairs = sorted(enumerate(values), key=lambda item: math.log10(item[1]))
    offsets = [0.0] * len(values)
    placed: List[tuple[float, float]] = []
    candidate_offsets = [0.0]
    for layer in range(1, max_layers + 1):
        candidate_offsets.extend([-layer * x_step, layer * x_step])

    for original_index, value in sorted_pairs:
        log_value = math.log10(value)
        chosen_offset = 0.0
        for candidate in candidate_offsets:
            collision = any(
                abs(candidate - placed_x) < x_step and abs(log_value - placed_y) < y_threshold
                for placed_x, placed_y in placed
            )
            if not collision:
                chosen_offset = candidate
                break
        offsets[original_index] = chosen_offset
        placed.append((chosen_offset, log_value))

    return offsets


def plot_group_metric_panels(
    entries: Sequence[Dict[str, Optional[object]]],
    group_key: str,
    group_title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    axes_flat = axes.flatten()

    for axis, metric_key in zip(axes_flat, TARGET_METRICS):
        grouped = metric_values_by_group(entries, group_key=group_key, metric_key=metric_key)
        labels = list(grouped.keys())
        if not labels:
            axis.set_visible(False)
            continue

        for idx, label in enumerate(labels):
            raw_values = grouped[label]
            plot_values = make_log_metric(raw_values, metric_key=metric_key)
            if not plot_values:
                continue
            x_values = [idx] * len(plot_values)
            color = GROUP_COLORS.get(label, "#4c4c4c")
            axis.scatter(x_values, plot_values, alpha=0.65, s=35, color=color, edgecolors="none")
            axis.hlines(
                statistics.median(plot_values),
                idx - 0.25,
                idx + 0.25,
                colors="black",
                linewidth=2.0,
            )

        axis.set_title(TARGET_METRICS[metric_key])
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=20, ha="right")
        axis.set_ylabel(metric_axis_label(metric_key))
        axis.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.suptitle(f"Photodetector metrics by {group_title}", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_structure_focus_panels(
    entries: Sequence[Dict[str, Optional[object]]],
    output_path: Path,
) -> None:
    focus_entries = [entry for entry in entries if entry["structure_major"] in FOCUS_STRUCTURES]
    fig, axes = plt.subplots(1, len(FOCUS_STRUCTURE_METRICS), figsize=(18.2, 7.0), constrained_layout=True)
    axes_flat = list(axes.flatten()) if hasattr(axes, "flatten") else [axes]

    for axis, metric_key in zip(axes_flat, FOCUS_STRUCTURE_METRICS):
        grouped = metric_values_by_group(focus_entries, group_key="structure_major", metric_key=metric_key)
        labels = [label for label in FOCUS_STRUCTURES if label in grouped]
        if not labels:
            axis.set_visible(False)
            continue

        positive_values = [value for label in labels for value in grouped[label] if value > 0]
        if not positive_values:
            axis.set_visible(False)
            continue

        log_min = math.log10(min(positive_values))
        log_max = math.log10(max(positive_values))
        log_padding = max(0.25, (log_max - log_min) * 0.10)

        axis.set_facecolor("#fbfcfe")
        axis.set_axisbelow(True)
        axis.set_yscale("log")
        axis.set_ylim(10 ** (log_min - log_padding), 10 ** (log_max + log_padding))
        axis.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        axis.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=tuple(range(2, 10))))
        axis.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
        axis.grid(True, which="major", axis="y", linestyle="--", linewidth=1.1, alpha=0.38)
        axis.grid(True, which="minor", axis="y", linestyle=":", linewidth=0.8, alpha=0.22)

        group_centers = [index * 1.45 for index in range(len(labels))]
        x_tick_labels: List[str] = []
        for idx, label in enumerate(labels):
            center_x = group_centers[idx]
            raw_values = sorted(value for value in grouped[label] if value > 0)
            if not raw_values:
                x_tick_labels.append(label)
                continue

            axis.axvspan(center_x - 0.54, center_x + 0.54, color=GROUP_COLORS.get(label, "#4c4c4c"), alpha=0.05, zorder=0)
            x_offsets = swarm_offsets_for_log_values(raw_values)
            x_values = [center_x + offset for offset in x_offsets]
            color = GROUP_COLORS.get(label, "#4c4c4c")
            median_value = statistics.median(raw_values)

            axis.scatter(
                x_values,
                raw_values,
                s=140,
                color=color,
                alpha=0.86,
                edgecolors="white",
                linewidths=1.1,
                zorder=3,
            )
            axis.hlines(median_value, center_x - 0.24, center_x + 0.24, colors="#111111", linewidth=3.4, zorder=5)
            axis.scatter(
                [center_x],
                [median_value],
                s=58,
                color="white",
                edgecolors="#111111",
                linewidths=1.3,
                zorder=6,
            )
            x_tick_labels.append(label)

        if group_centers:
            axis.set_xlim(group_centers[0] - 0.8, group_centers[-1] + 0.8)
        axis.set_xticks(group_centers)
        axis.set_xticklabels(x_tick_labels, rotation=STRUCTURE_XLABEL_ROTATION, ha="right")
        axis.set_title(TARGET_METRICS[metric_key], fontsize=STRUCTURE_PANEL_TITLE_SIZE, fontweight="bold", pad=12)
        axis.set_ylabel(TARGET_METRICS[metric_key], fontsize=STRUCTURE_AXIS_LABEL_SIZE, fontweight="bold")
        axis.text(
            0.02,
            0.98,
            "log scale",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=STRUCTURE_LOG_NOTE_SIZE,
            fontweight="bold",
            color="#505050",
        )
        style_axis_text(axis)

    fig.suptitle("Photodetector Metrics by Major Structure", fontsize=STRUCTURE_SUPTITLE_SIZE, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def scale_value_to_range(
    value: Optional[float],
    source_min: float,
    source_max: float,
    target_min: float = RADAR_DISPLAY_MIN,
    target_max: float = RADAR_DISPLAY_MAX,
) -> float:
    if value is None:
        return target_min
    if math.isclose(source_min, source_max):
        return 0.5 * (target_min + target_max)
    normalized = (value - source_min) / (source_max - source_min)
    clipped = max(0.0, min(1.0, normalized))
    return target_min + clipped * (target_max - target_min)


def build_structure_radar_rows(
    entries: Sequence[Dict[str, Optional[object]]],
    structure_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    structure_map = {
        str(row["structure_major"]): row
        for row in structure_rows
        if str(row["structure_major"]) in FOCUS_STRUCTURES
    }

    response_dataset_logs = [
        math.log10(float(entry["response_a_per_w"]))
        for entry in entries
        if entry["response_a_per_w"] is not None and float(entry["response_a_per_w"]) > 0
    ]
    speed_dataset_logs = [
        -math.log10(float(entry["response_time_s"]))
        for entry in entries
        if entry["response_time_s"] is not None and float(entry["response_time_s"]) > 0
    ]
    dstar_dataset_logs = [
        math.log10(float(entry["d_star_jones"]))
        for entry in entries
        if entry["d_star_jones"] is not None and float(entry["d_star_jones"]) > 0
    ]

    response_source_min = min(response_dataset_logs)
    response_source_max = max(response_dataset_logs)
    speed_source_min = percentile(speed_dataset_logs, SPEED_RADAR_PERCENTILE_LOW)
    speed_source_max = percentile(speed_dataset_logs, SPEED_RADAR_PERCENTILE_HIGH)
    dstar_source_min = min(dstar_dataset_logs)
    dstar_source_max = max(dstar_dataset_logs)

    if speed_source_min is None or speed_source_max is None:
        speed_source_min = min(speed_dataset_logs)
        speed_source_max = max(speed_dataset_logs)

    response_logs: Dict[str, Optional[float]] = {}
    speed_logs: Dict[str, Optional[float]] = {}
    dstar_logs: Dict[str, Optional[float]] = {}
    for structure in FOCUS_STRUCTURES:
        row = structure_map.get(structure)
        response_median = None if row is None else row.get("response_a_per_w_median")
        response_time_median = None if row is None else row.get("response_time_s_median")
        dstar_median = None if row is None else row.get("d_star_jones_median")

        response_logs[structure] = (
            math.log10(float(response_median))
            if response_median is not None and float(response_median) > 0
            else None
        )
        speed_logs[structure] = (
            -math.log10(float(response_time_median))
            if response_time_median is not None and float(response_time_median) > 0
            else None
        )
        dstar_logs[structure] = (
            math.log10(float(dstar_median))
            if dstar_median is not None and float(dstar_median) > 0
            else None
        )

    radar_rows: List[Dict[str, object]] = []
    for structure in FOCUS_STRUCTURES:
        row = structure_map.get(structure)
        response_log_value = response_logs[structure]
        speed_log_value = speed_logs[structure]
        dstar_log_value = dstar_logs[structure]
        noise_manual = RADAR_NOISE_SCORES[structure]
        radar_rows.append(
            {
                "structure_major": structure,
                "response_median": None if row is None else row.get("response_a_per_w_median"),
                "response_log10_median": response_log_value,
                "response_dataset_log10_min": response_source_min,
                "response_dataset_log10_max": response_source_max,
                "response_radar_value": scale_value_to_range(
                    response_log_value,
                    response_source_min,
                    response_source_max,
                ),
                "response_time_median_s": None if row is None else row.get("response_time_s_median"),
                "speed_log10_median": speed_log_value,
                "speed_dataset_log10_min": speed_source_min,
                "speed_dataset_log10_max": speed_source_max,
                "speed_radar_value": scale_value_to_range(
                    speed_log_value,
                    speed_source_min,
                    speed_source_max,
                ),
                "d_star_median": None if row is None else row.get("d_star_jones_median"),
                "d_star_log10_median": dstar_log_value,
                "d_star_dataset_log10_min": dstar_source_min,
                "d_star_dataset_log10_max": dstar_source_max,
                "d_star_radar_value": scale_value_to_range(
                    dstar_log_value,
                    dstar_source_min,
                    dstar_source_max,
                ),
                "noise_score_manual_0_1": noise_manual,
                "noise_radar_value": noise_manual,
            }
        )
    return radar_rows


def plot_structure_radar_chart(
    radar_rows: Sequence[Dict[str, object]],
    output_path: Path,
) -> None:
    metric_labels = ["Response", "Speed", "D*", "Noise"]
    angles = [index / len(metric_labels) * 2.0 * math.pi for index in range(len(metric_labels))]
    angles.append(angles[0])

    fig, axis = plt.subplots(figsize=(7.0, 6.4), subplot_kw={"projection": "polar"}, constrained_layout=True)
    axis.set_theta_offset(math.pi / 2.0)
    axis.set_theta_direction(-1)
    axis.set_facecolor("#fbfcfe")
    axis.set_ylim(RADAR_DISPLAY_MIN, RADAR_DISPLAY_MAX)
    axis.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=12, fontweight="bold")
    axis.set_rlabel_position(90)
    axis.grid(True, which="major", axis="both", linestyle="--", linewidth=1.0, alpha=0.35)
    axis.spines["polar"].set_linewidth(2.2)
    axis.spines["polar"].set_color("#303030")
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(metric_labels, fontsize=14, fontweight="bold")
    axis.tick_params(axis="x", pad=RADAR_LABEL_PAD)

    for row in radar_rows:
        structure = str(row["structure_major"])
        values = [
            float(row["response_radar_value"]),
            float(row["speed_radar_value"]),
            float(row["d_star_radar_value"]),
            float(row["noise_radar_value"]),
        ]
        values.append(values[0])
        color = GROUP_COLORS.get(structure, "#4c4c4c")
        axis.plot(angles, values, color=color, linewidth=3.0, marker="o", markersize=8, label=structure)
        axis.fill(angles, values, color=color, alpha=0.13)

    axis.legend(
        loc="center left",
        bbox_to_anchor=(RADAR_LEGEND_BBOX_X, RADAR_LEGEND_BBOX_Y),
        frameon=False,
        fontsize=12,
        borderaxespad=0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_cross_heatmaps(
    cross_rows: Sequence[Dict[str, object]],
    output_dir: Path,
) -> None:
    materials = sorted({str(row["material_major"]) for row in cross_rows})
    structures = sorted({str(row["structure_major"]) for row in cross_rows})
    if not materials or not structures:
        return

    for metric_key in TARGET_METRICS:
        matrix: List[List[float]] = []
        for material in materials:
            line: List[float] = []
            for structure in structures:
                match = next(
                    (
                        row
                        for row in cross_rows
                        if row["material_major"] == material and row["structure_major"] == structure
                    ),
                    None,
                )
                value = None if match is None else match.get(f"{metric_key}_median")
                if value is None:
                    line.append(float("nan"))
                else:
                    numeric_value = float(value)
                    if metric_key in {"response_a_per_w", "response_time_s", "d_star_jones"} and numeric_value > 0:
                        numeric_value = math.log10(numeric_value)
                    line.append(numeric_value)
            matrix.append(line)

        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(structures)))
        ax.set_xticklabels(structures, rotation=20, ha="right")
        ax.set_yticks(range(len(materials)))
        ax.set_yticklabels(materials)
        ax.set_title(f"{TARGET_METRICS[metric_key]} median by material and structure")
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(metric_axis_label(metric_key))

        for row_idx, material in enumerate(materials):
            for col_idx, structure in enumerate(structures):
                match = next(
                    (
                        row
                        for row in cross_rows
                        if row["material_major"] == material and row["structure_major"] == structure
                    ),
                    None,
                )
                count = 0 if match is None else int(match.get("device_count", 0))
                ax.text(col_idx, row_idx, f"n={count}", ha="center", va="center", color="white", fontsize=9)

        output_path = output_dir / f"heatmap_{sanitize_filename(metric_key)}.png"
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(
    entries: Sequence[Dict[str, Optional[object]]],
    material_rows: Sequence[Dict[str, object]],
    structure_rows: Sequence[Dict[str, object]],
    cross_rows: Sequence[Dict[str, object]],
    radar_rows: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append("# Photodetector paper dataset analysis")
    lines.append("")
    lines.append("Scope: major material classes and major device structures only.")
    lines.append(f"Valid entries: {len(entries)}")
    lines.append("")

    material_counter = Counter(str(entry["material_major"]) for entry in entries if entry["material_major"] is not None)
    structure_counter = Counter(str(entry["structure_major"]) for entry in entries if entry["structure_major"] is not None)

    lines.append("## Material counts")
    for name, count in sorted(material_counter.items()):
        lines.append(f"- {name}: {count}")
    lines.append("")

    lines.append("## Structure counts")
    for name, count in sorted(structure_counter.items()):
        lines.append(f"- {name}: {count}")
    lines.append("")

    lines.append("## Material summary")
    for row in material_rows:
        lines.append(
            "- {name}: N={n}, Response median={response}, LDR median={ldr}, "
            "ResponseTime median={rt}, D* median={dstar}".format(
                name=row["material_major"],
                n=row["device_count"],
                response=format_value(row["response_a_per_w_median"]),
                ldr=format_value(row["ldr_db_median"]),
                rt=format_value(row["response_time_s_median"]),
                dstar=format_value(row["d_star_jones_median"]),
            )
        )
    lines.append("")

    lines.append("## Structure summary")
    for row in structure_rows:
        lines.append(
            "- {name}: N={n}, Response median={response}, LDR median={ldr}, "
            "ResponseTime median={rt}, D* median={dstar}".format(
                name=row["structure_major"],
                n=row["device_count"],
                response=format_value(row["response_a_per_w_median"]),
                ldr=format_value(row["ldr_db_median"]),
                rt=format_value(row["response_time_s_median"]),
                dstar=format_value(row["d_star_jones_median"]),
            )
        )
    lines.append("")

    lines.append("## Material x Structure coverage")
    for row in cross_rows:
        lines.append(
            "- {material} / {structure}: N={n}, Response count={response_n}, "
            "LDR count={ldr_n}, ResponseTime count={rt_n}, D* count={dstar_n}".format(
                material=row["material_major"],
                structure=row["structure_major"],
                n=row["device_count"],
                response_n=row["response_a_per_w_count"],
                ldr_n=row["ldr_db_count"],
                rt_n=row["response_time_s_count"],
                dstar_n=row["d_star_jones_count"],
            )
        )

    lines.append("")
    lines.append("## Structure radar scores")
    lines.append("Radar values use whole-paper dataset normalization in log domain and stay on a 0-1 axis.")
    lines.append("Speed = -log10(response time) and uses paper-dataset P20-P80 normalization to reduce outlier compression. Noise stays manual 0-1.")
    for row in radar_rows:
        lines.append(
            "- {structure}: Response={response:.3f}, Speed={speed:.3f}, D*={dstar:.3f}, Noise={noise:.3f}".format(
                structure=row["structure_major"],
                response=float(row["response_radar_value"]),
                speed=float(row["speed_radar_value"]),
                dstar=float(row["d_star_radar_value"]),
                noise=float(row["noise_radar_value"]),
            )
        )

    return "\n".join(lines) + "\n"


def format_value(value: object) -> str:
    if value is None:
        return "NA"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) >= 1e3 or abs(number) < 1e-2:
        return f"{number:.3e}"
    return f"{number:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the photodetector paper dataset by major classes.")
    parser.add_argument("--input", type=Path, default=DATASET_PATH, help="Path to photodetector_paper_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for analysis outputs")
    args = parser.parse_args()

    rows = read_rows(args.input)
    entries = collect_entries(rows)

    material_rows = build_group_rows(entries, group_key="material_major", group_label="material_major")
    structure_rows = build_group_rows(entries, group_key="structure_major", group_label="structure_major")
    cross_rows = build_cross_rows(entries)

    cleaned_rows: List[Dict[str, object]] = []
    for entry in entries:
        cleaned_rows.append(
            {
                "entry_id": entry["entry_id"],
                "material_major": entry["material_major"],
                "material_raw": entry["material_raw"],
                "material_subcategory": entry["material_subcategory"],
                "specific_material": entry["specific_material"],
                "structure_major": entry["structure_major"],
                "structure_raw": entry["structure_raw"],
                "response_a_per_w": entry["response_a_per_w"],
                "response_rise_s": entry["response_rise_s"],
                "response_fall_s": entry["response_fall_s"],
                "response_time_s": entry["response_time_s"],
                "ldr_db": entry["ldr_db"],
                "d_star_jones": entry["d_star_jones"],
                "ref_id": entry["ref_id"],
                "notes": entry["notes"],
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cleaned_major_class_entries.csv", cleaned_rows)
    write_csv(args.output_dir / "summary_by_material_major.csv", material_rows)
    write_csv(args.output_dir / "summary_by_structure_major.csv", structure_rows)
    write_csv(args.output_dir / "summary_by_material_structure_major.csv", cross_rows)
    radar_rows = build_structure_radar_rows(entries, structure_rows)
    write_csv(args.output_dir / "radar_by_structure_major.csv", radar_rows)

    plot_group_metric_panels(
        entries,
        group_key="material_major",
        group_title="major material class",
        output_path=args.output_dir / "metrics_by_material_major.png",
    )
    plot_structure_focus_panels(
        entries,
        output_path=args.output_dir / "metrics_by_structure_major.png",
    )
    plot_structure_radar_chart(radar_rows, args.output_dir / "radar_by_structure_major.png")
    plot_cross_heatmaps(cross_rows, args.output_dir)

    report = build_report(entries, material_rows, structure_rows, cross_rows, radar_rows)
    (args.output_dir / "analysis_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
