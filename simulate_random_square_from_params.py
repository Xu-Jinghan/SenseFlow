from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
MPLCONFIGDIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt

from photodetector_model import N_CARRIER, PARAM_KEYS, params_to_vec, prepare_model_config, simulate


PRESENTATION_STYLE = {
    "title_size": 17,
    "label_size": 15,
    "tick_size": 13,
    "legend_size": 12,
    "spine_width": 1.8,
    "tick_width": 1.6,
    "tick_length": 6,
    "primary_line_width": 2.8,
    "secondary_line_width": 2.4,
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
        description="Generate a random multi-level square-wave optical input and simulate the output with fitted photodetector parameters."
    )
    parser.add_argument("--params-csv", type=str, required=True, help="Parameter CSV from fit_waveform_from_image.py")
    parser.add_argument(
        "--digitized-csv",
        type=str,
        default=None,
        help="Optional digitized waveform CSV. If omitted, the script will try to infer it from the params CSV name.",
    )
    parser.add_argument(
        "--dark-current",
        type=float,
        default=None,
        help="Optional dark current baseline in the same arbitrary unit as the fitted waveform.",
    )
    parser.add_argument(
        "--dark-percentile",
        type=float,
        default=2.0,
        help="Percentile used to estimate dark current from the digitized waveform when --dark-current is omitted.",
    )
    parser.add_argument("--duration", type=float, default=2.0, help="Total waveform duration in seconds.")
    parser.add_argument("--dt", type=float, default=1e-3, help="Simulation time step in seconds.")
    parser.add_argument("--min-on", type=float, default=0.05, help="Minimum ON duration in seconds.")
    parser.add_argument("--max-on", type=float, default=0.20, help="Maximum ON duration in seconds.")
    parser.add_argument("--min-off", type=float, default=0.03, help="Minimum OFF duration in seconds.")
    parser.add_argument("--max-off", type=float, default=0.15, help="Maximum OFF duration in seconds.")
    parser.add_argument("--min-power", type=float, default=0.15, help="Minimum ON power level.")
    parser.add_argument("--max-power", type=float, default=1.00, help="Maximum ON power level.")
    parser.add_argument(
        "--num-levels",
        type=int,
        default=6,
        help="Number of discrete ON amplitude levels. Use 0 or 1 for continuous uniform levels.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the input waveform.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Output prefix. Default: sibling of params CSV with suffix _random_square.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_params_csv(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    params: dict[str, float] = {}
    for row in rows:
        key = row["parameter"].strip()
        if key not in PARAM_KEYS:
            continue
        params[key] = float(row["value"])
    missing = [key for key in PARAM_KEYS if key not in params]
    if missing:
        raise ValueError(f"Parameter CSV is missing fields: {missing}")
    return params


def infer_digitized_csv(params_csv: Path) -> Path | None:
    name = params_csv.name
    if not name.endswith("_params.csv"):
        return None
    candidate = params_csv.with_name(name[:-11] + "_digitized.csv")
    return candidate if candidate.exists() else None


def estimate_dark_current_from_digitized(path: Path, percentile: float) -> float:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Digitized CSV is empty: {path}")
    values = np.asarray([float(row["response_au"]) for row in rows], dtype=float)
    return float(np.percentile(values, percentile))


def generate_random_square_waveform(
    duration_s: float,
    dt_s: float,
    min_on_s: float,
    max_on_s: float,
    min_off_s: float,
    max_off_s: float,
    min_power: float,
    max_power: float,
    num_levels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    if dt_s <= 0:
        raise ValueError("dt must be positive")
    if duration_s <= 0:
        raise ValueError("duration must be positive")
    if min_on_s <= 0 or max_on_s < min_on_s:
        raise ValueError("Invalid ON duration range")
    if min_off_s < 0 or max_off_s < min_off_s:
        raise ValueError("Invalid OFF duration range")
    if min_power < 0 or max_power < min_power:
        raise ValueError("Invalid power range")

    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s, dtype=float)
    p = np.zeros_like(t)

    if num_levels and num_levels > 1:
        levels = np.linspace(min_power, max_power, num_levels)
        sample_level = lambda: float(rng.choice(levels))
    else:
        sample_level = lambda: float(rng.uniform(min_power, max_power))

    cursor = 0.0
    segments: list[dict[str, float]] = []
    pulse_index = 0
    while cursor < duration_s:
        off_s = float(rng.uniform(min_off_s, max_off_s))
        off_end = min(cursor + off_s, duration_s)
        segments.append(
            {"kind": "off", "start_s": cursor, "end_s": off_end, "power": 0.0, "pulse_index": float(pulse_index)}
        )
        cursor = off_end
        if cursor >= duration_s:
            break

        on_s = float(rng.uniform(min_on_s, max_on_s))
        on_power = sample_level()
        on_end = min(cursor + on_s, duration_s)
        mask = (t >= cursor) & (t < on_end)
        p[mask] = on_power
        pulse_index += 1
        segments.append(
            {
                "kind": "on",
                "start_s": cursor,
                "end_s": on_end,
                "power": on_power,
                "pulse_index": float(pulse_index),
            }
        )
        cursor = on_end

    return t, p, segments


def compute_ideal_nonlinear_current(
    p: np.ndarray,
    params: dict[str, float],
    dark_current: float,
) -> np.ndarray:
    model_config = prepare_model_config(params_to_vec(params), n_carrier=N_CARRIER)
    ideal_photo_current = np.zeros_like(p, dtype=float)
    p_eps = np.maximum(p, 1e-20)
    for responsivity, eta_i, _, _ in model_config["carrier_params"]:
        component = responsivity * np.power(p_eps, eta_i)
        component = np.where(p > 0, component, 0.0)
        ideal_photo_current += component
    return dark_current + ideal_photo_current


def save_waveform_csv(
    output_path: Path,
    t: np.ndarray,
    p: np.ndarray,
    i_ideal: np.ndarray,
    i_out: np.ndarray,
    i_det: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    params: dict[str, float],
) -> None:
    ensure_parent_dir(output_path)
    delta = float(params["delta"])
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "power_norm",
                "ideal_current_a",
                "output_current_a",
                "deterministic_current_a",
                "x1_fast_current_a",
                "x1_slow_current_a",
                "trap_term_current_a",
                "drift_term_current_a",
            ]
        )
        for idx in range(len(t)):
            writer.writerow(
                [
                    f"{t[idx]:.9e}",
                    f"{p[idx]:.9e}",
                    f"{i_ideal[idx]:.9e}",
                    f"{i_out[idx]:.9e}",
                    f"{i_det[idx]:.9e}",
                    f"{x1[0, idx]:.9e}",
                    f"{x1[1, idx]:.9e}",
                    f"{(delta * x2[idx]):.9e}",
                    f"{x3[idx]:.9e}",
                ]
            )


def save_segment_csv(output_path: Path, segments: list[dict[str, float]]) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["kind", "start_s", "end_s", "power", "pulse_index"])
        writer.writeheader()
        writer.writerows(segments)


def save_plot(
    output_path: Path,
    t: np.ndarray,
    p: np.ndarray,
    i_ideal: np.ndarray,
    i_det: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    params: dict[str, float],
) -> None:
    ensure_parent_dir(output_path)
    delta = float(params["delta"])
    fig, axes = plt.subplots(3, 1, figsize=(14.2, 9.8), sharex=True, gridspec_kw={"height_ratios": [1, 2, 1.3]})
    fig.patch.set_facecolor("white")

    axes[0].step(t, p, where="post", color="#65a30d", linewidth=PRESENTATION_STYLE["primary_line_width"])
    apply_axis_style(axes[0], title="Random multi-level square-wave optical input", ylabel="P_in")

    axes[1].plot(
        t,
        i_ideal,
        color="#1d4ed8",
        linestyle="--",
        linewidth=PRESENTATION_STYLE["secondary_line_width"],
        alpha=0.9,
        label="ideal nonlinear current",
    )
    axes[1].plot(
        t,
        i_det,
        color="#dc2626",
        linewidth=PRESENTATION_STYLE["primary_line_width"],
        label="simulated current",
    )
    apply_axis_style(axes[1], title="Simulated output current with extracted parameters", ylabel="Current (A)")
    style_legend(axes[1].legend(loc="upper right", fontsize=PRESENTATION_STYLE["legend_size"]))

    axes[2].plot(t, x1[0], color="#ea580c", linewidth=PRESENTATION_STYLE["secondary_line_width"], label="fast carrier")
    axes[2].plot(t, x1[1], color="#7c3aed", linewidth=PRESENTATION_STYLE["secondary_line_width"], label="slow carrier")
    axes[2].plot(t, delta * x2, color="#db2777", linewidth=PRESENTATION_STYLE["secondary_line_width"], label="trap term")
    axes[2].plot(t, x3, color="#0891b2", linewidth=PRESENTATION_STYLE["secondary_line_width"], label="drift term")
    apply_axis_style(axes[2], xlabel="Time (s)", ylabel="Current (A)")
    style_legend(axes[2].legend(loc="upper right", ncol=2, fontsize=PRESENTATION_STYLE["legend_size"]))

    for ax in axes:
        ax.margins(x=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=PRESENTATION_STYLE["save_dpi"], bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    params_csv = Path(args.params_csv).expanduser().resolve()
    params = load_params_csv(params_csv)

    if args.output_prefix:
        output_prefix = Path(args.output_prefix)
    else:
        stem = params_csv.name[:-11] if params_csv.name.endswith("_params.csv") else params_csv.stem
        output_prefix = params_csv.with_name(stem + "_random_square")

    if args.dark_current is not None:
        dark_current = float(args.dark_current)
        dark_source = "cli"
    else:
        digitized_csv = Path(args.digitized_csv).expanduser().resolve() if args.digitized_csv else infer_digitized_csv(params_csv)
        if digitized_csv is not None and digitized_csv.exists():
            dark_current = estimate_dark_current_from_digitized(digitized_csv, args.dark_percentile)
            dark_source = str(digitized_csv)
        else:
            dark_current = 0.0
            dark_source = "default_zero"

    t, p, segments = generate_random_square_waveform(
        duration_s=args.duration,
        dt_s=args.dt,
        min_on_s=args.min_on,
        max_on_s=args.max_on,
        min_off_s=args.min_off,
        max_off_s=args.max_off,
        min_power=args.min_power,
        max_power=args.max_power,
        num_levels=args.num_levels,
        seed=args.seed,
    )

    i_out, i_det, x1, x2, x3 = simulate(
        t=t,
        P=p,
        params=params_to_vec(params),
        n_carrier=N_CARRIER,
        noise_fn=None,
        dark_current=dark_current,
    )
    i_ideal = compute_ideal_nonlinear_current(p, params, dark_current)

    plot_path = output_prefix.with_name(output_prefix.name + "_plot.png")
    waveform_csv = output_prefix.with_name(output_prefix.name + "_waveform.csv")
    segments_csv = output_prefix.with_name(output_prefix.name + "_segments.csv")

    save_plot(plot_path, t, p, i_ideal, i_det, x1, x2, x3, params)
    save_waveform_csv(waveform_csv, t, p, i_ideal, i_out, i_det, x1, x2, x3, params)
    save_segment_csv(segments_csv, segments)

    print(f"params_csv             {params_csv}")
    print(f"dark_current_a         {dark_current:.6e}")
    print(f"dark_current_source    {dark_source}")
    print(f"duration_s             {args.duration:.6f}")
    print(f"dt_s                   {args.dt:.6e}")
    print(f"num_samples            {len(t)}")
    print(f"num_segments           {len(segments)}")
    print(f"plot                   {plot_path}")
    print(f"waveform_csv           {waveform_csv}")
    print(f"segments_csv           {segments_csv}")


if __name__ == "__main__":
    main()
