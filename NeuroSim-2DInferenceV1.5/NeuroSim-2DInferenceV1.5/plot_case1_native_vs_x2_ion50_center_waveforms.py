import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_sensor_verification_images as base_pipeline  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402
import photodetector_model as pm  # noqa: E402


FPS_VALUES = [20.0, 50.0, 100.0, 200.0]
COLORS = ["tab:red", "tab:green", "tab:blue"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot center-pixel current waveforms across 10 consecutive CIFAR-10 samples "
            "for native case1 versus the x2~50%Ion configuration."
        )
    )
    parser.add_argument(
        "--native-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence"),
        help="Directory containing native case1 parameter_summary.json",
    )
    parser.add_argument(
        "--native-params-csv",
        default=None,
        help="Optional direct path to the native parameter CSV. If set, this overrides --native-results-dir.",
    )
    parser.add_argument(
        "--x2-results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_x2_ion50_video_sequence"),
        help="Directory containing x2-ion50 parameter summary",
    )
    parser.add_argument(
        "--x2-params-csv",
        default=None,
        help="Optional direct path to the x2 parameter CSV. If set, this overrides --x2-results-dir.",
    )
    parser.add_argument(
        "--x2-label",
        default="x2 total",
        help="Legend label used for the non-native waveform.",
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--start-index", type=int, default=23)
    parser.add_argument("--num-images", type=int, default=10)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--output-path", default="auto")
    return parser.parse_args()


def resolve_output_path(args):
    if args.output_path not in {None, "", "auto"}:
        return Path(args.output_path).expanduser()
    return Path(args.x2_results_dir) / f"center_pixel_waveforms_native_vs_x2ion50_{args.start_index:04d}_{args.start_index + args.num_images - 1:04d}.png"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_params_csvs(native_results_dir, x2_results_dir):
    native_summary = load_json(Path(native_results_dir) / "parameter_summary.json")
    x2_summary = load_json(Path(x2_results_dir) / "case1_x2_ion50_summary.json")
    return Path(native_summary["native_csv"]).resolve(), Path(x2_summary["parameter_summary"]["params_csv"]).resolve()


def build_model_config(base_params):
    trap_saturation_time_s = base_params.get("trap_saturation_time_s")
    trap_amplitude_ratio = base_params.get("trap_amplitude_ratio")
    if trap_saturation_time_s is not None:
        trap_saturation_time_s = float(trap_saturation_time_s)
    if trap_amplitude_ratio is not None:
        trap_amplitude_ratio = float(trap_amplitude_ratio)
        if trap_amplitude_ratio <= 0.0:
            trap_amplitude_ratio = None
    return pm.prepare_model_config(
        pm.params_to_vec(base_params),
        n_carrier=pm.infer_n_carrier_from_params(base_params),
        trap_mode=str(base_params.get("trap_mode", "power")),
        trap_threshold_w=float(base_params.get("trap_threshold_w", base_params.get("pmin_w", 0.0))),
        trap_output_mode=str(base_params.get("trap_output_mode", "always")),
        power_min_w=float(base_params.get("pmin_w", 0.0)),
        power_max_w=float(base_params.get("pmax_w", float("inf"))),
        trap_saturation_time_s=trap_saturation_time_s,
        trap_amplitude_ratio=trap_amplitude_ratio,
    )


def extract_center_power_sequence(args):
    raise RuntimeError("extract_center_power_sequence requires a power-mapping parameter set")


def extract_center_power_sequence_with_params(args, power_base_params):
    dataset = base_pipeline.load_base_dataset(args.source_dataset, args.data_root, args.split)
    power_sequence = []
    labels = []
    dummy = argparse.Namespace(array_size=32, output_channels=3)
    for dataset_index in range(args.start_index, args.start_index + args.num_images):
        image, label = dataset[dataset_index]
        power_maps = pipeline.build_sequence_power_maps(image, dummy, power_base_params)
        center_power = np.asarray(power_maps[:, power_maps.shape[1] // 2, power_maps.shape[2] // 2], dtype=np.float64)
        power_sequence.append(center_power)
        labels.append(dataset.classes[label])
    return np.asarray(power_sequence, dtype=np.float64), labels


def simulate_sequence(center_power_sequence, base_params, fps, fps_sim):
    model_config = build_model_config(base_params)
    n_carrier = pm.infer_n_carrier_from_params(base_params)
    frame_s = 1.0 / float(fps)
    n_steps = max(1, int(round(frame_s * float(fps_sim))))
    dt = frame_s / n_steps
    n_frames, n_channels = center_power_sequence.shape

    x1 = np.zeros((n_channels, n_carrier), dtype=np.float64)
    x2 = np.zeros(n_channels, dtype=np.float64)
    x3 = np.zeros(n_channels, dtype=np.float64)

    total_trace = []
    main_trace = []
    x2_trace = []
    time_ms = []
    for frame_idx in range(n_frames):
        power_ch = center_power_sequence[frame_idx]
        for step_idx in range(n_steps):
            current_time_ms = (frame_idx * frame_s + (step_idx + 1) * dt) * 1e3
            step_total = np.zeros(n_channels, dtype=np.float64)
            step_main = np.zeros(n_channels, dtype=np.float64)
            step_x2 = np.zeros(n_channels, dtype=np.float64)
            for ch in range(n_channels):
                x1[ch], x2[ch], x3[ch] = pm.step_model_state(power_ch[ch], dt, model_config, x1[ch], x2[ch], x3[ch])
                drift = max(1.0 + x3[ch], 0.0)
                main_current = float(np.sum(x1[ch]) * drift)
                x2_current = float(base_params["delta"] * x2[ch] * drift)
                step_main[ch] = main_current
                step_x2[ch] = x2_current
                step_total[ch] = main_current + x2_current
            total_trace.append(step_total)
            main_trace.append(step_main)
            x2_trace.append(step_x2)
            time_ms.append(current_time_ms)

    return {
        "time_ms": np.asarray(time_ms, dtype=np.float64),
        "total_trace_a": np.asarray(total_trace, dtype=np.float64),
        "main_trace_a": np.asarray(main_trace, dtype=np.float64),
        "x2_trace_a": np.asarray(x2_trace, dtype=np.float64),
        "n_steps_per_frame": n_steps,
        "frame_duration_ms": frame_s * 1e3,
    }


def annotate_boundaries(ax, labels, frame_duration_ms):
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.08 * (ymax - ymin)
    for idx, label in enumerate(labels):
        x_left = idx * frame_duration_ms
        x_mid = x_left + 0.5 * frame_duration_ms
        if idx > 0:
            ax.axvline(x_left, color="#888888", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(x_mid, y_text, f"{idx+1}:{label}", ha="center", va="top", fontsize=8, color="#333333")


def render_plot(center_power_sequence, labels, native_data, x2_data_by_fps, output_path, x2_label):
    fig, axes = plt.subplots(len(FPS_VALUES) + 1, 1, figsize=(16.5, 12.5), constrained_layout=True)

    power_time_ms = []
    power_value = []
    for frame_idx in range(center_power_sequence.shape[0]):
        for step_idx in range(native_data[20.0]["n_steps_per_frame"]):
            power_time_ms.append(
                (frame_idx * native_data[20.0]["frame_duration_ms"] / 1000.0
                 + (step_idx + 1) * (native_data[20.0]["frame_duration_ms"] / native_data[20.0]["n_steps_per_frame"]) / 1000.0)
                * 1e3
            )
            power_value.append(center_power_sequence[frame_idx])
    power_time_ms = np.asarray(power_time_ms, dtype=np.float64)
    power_value = np.asarray(power_value, dtype=np.float64)

    axes[0].set_title("Center Pixel Power Sequence", fontsize=13, fontweight="bold")
    for ch_idx in range(center_power_sequence.shape[1]):
        axes[0].step(power_time_ms, power_value[:, ch_idx] * 1e6, where="post", color=COLORS[ch_idx], linewidth=2.0)
    axes[0].set_ylabel("Power (uW)", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.25)
    annotate_boundaries(axes[0], labels, native_data[20.0]["frame_duration_ms"])

    for row_idx, fps in enumerate(FPS_VALUES, start=1):
        ax = axes[row_idx]
        native = native_data[fps]
        x2 = x2_data_by_fps[fps]
        for ch_idx in range(center_power_sequence.shape[1]):
            color = COLORS[ch_idx]
            ax.plot(native["time_ms"], native["total_trace_a"][:, ch_idx] * 1e6, linestyle="--", color=color, linewidth=1.6)
            ax.plot(x2["time_ms"], x2["total_trace_a"][:, ch_idx] * 1e6, color=color, linewidth=2.1)
            ax.plot(x2["time_ms"], x2["x2_trace_a"][:, ch_idx] * 1e6, linestyle=":", color=color, linewidth=1.2, alpha=0.9)
        ax.set_title(f"FPS {int(fps)} | dashed=native, solid={x2_label}, dotted=x2 contribution", fontsize=12, fontweight="bold")
        ax.set_ylabel("Current (uA)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Continuous time across 10 images (ms)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        annotate_boundaries(ax, labels, native["frame_duration_ms"])
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)
        ax.tick_params(labelsize=10, width=1.1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    output_path = resolve_output_path(args)
    if args.native_params_csv:
        native_csv = Path(args.native_params_csv).expanduser().resolve()
    elif args.x2_params_csv:
        native_summary = load_json(Path(args.native_results_dir) / "parameter_summary.json")
        native_csv = Path(native_summary["native_csv"]).resolve()
    else:
        native_csv, x2_csv = resolve_params_csvs(args.native_results_dir, args.x2_results_dir)

    if args.x2_params_csv:
        x2_csv = Path(args.x2_params_csv).expanduser().resolve()
    elif not args.native_params_csv:
        pass
    else:
        _native_csv_from_summary, x2_csv = resolve_params_csvs(args.native_results_dir, args.x2_results_dir)

    class A:
        pass

    a = A()
    a.params_csv = str(native_csv)
    for name in ["device_area_cm2", "prange1_density", "prange2_density", "pmin_density", "pmax_density", "single_r", "single_eta", "single_trise", "single_tfall", "trap_saturation_time", "trap_amplitude_pct", "noise_1f_density_1hz", "aging_tau_hours", "r_degradation_pct", "spatial_variation_r_pct", "tia_gain_ohm", "integration_gain_v_per_c"]:
        setattr(a, name, None)
    a.force_single_carrier = 0
    native_params = pipeline.resolve_sequence_base_params(a)
    a.params_csv = str(x2_csv)
    x2_params = pipeline.resolve_sequence_base_params(a)

    center_power_sequence, labels = extract_center_power_sequence_with_params(args, x2_params)
    native_data = {}
    x2_data = {}
    for fps in FPS_VALUES:
        native_data[fps] = simulate_sequence(center_power_sequence, native_params, fps=fps, fps_sim=args.fps_sim)
        x2_data[fps] = simulate_sequence(center_power_sequence, x2_params, fps=fps, fps_sim=args.fps_sim)

    render_plot(center_power_sequence, labels, native_data, x2_data, output_path, x2_label=args.x2_label)

    summary = {
        "native_params_csv": str(native_csv),
        "x2_params_csv": str(x2_csv),
        "start_index": int(args.start_index),
        "num_images": int(args.num_images),
        "fps_values": FPS_VALUES,
        "output_plot": str(output_path),
        "sample_labels": labels,
        "fps_summaries": {},
    }
    for fps in FPS_VALUES:
        x2_total_mean = np.mean(x2_data[fps]["total_trace_a"], axis=0)
        x2_main_mean = np.mean(x2_data[fps]["main_trace_a"], axis=0)
        x2_branch_mean = np.mean(x2_data[fps]["x2_trace_a"], axis=0)
        summary["fps_summaries"][str(int(fps))] = {
            "native_mean_current_uA": [float(v * 1e6) for v in np.mean(native_data[fps]["total_trace_a"], axis=0)],
            "x2_total_mean_current_uA": [float(v * 1e6) for v in x2_total_mean],
            "x2_branch_mean_current_uA": [float(v * 1e6) for v in x2_branch_mean],
            "x2_branch_pct_of_total_mean": [
                float(100.0 * x2_branch_mean[idx] / max(x2_total_mean[idx], 1e-30))
                for idx in range(len(x2_total_mean))
            ],
            "native_vs_x2_mean_delta_uA": [
                float((x2_total_mean[idx] - np.mean(native_data[fps]["total_trace_a"], axis=0)[idx]) * 1e6)
                for idx in range(len(x2_total_mean))
            ],
        }

    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Waveform plot: {output_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
