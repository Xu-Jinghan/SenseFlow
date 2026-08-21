import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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
from sensor_video_sequence_backend import StatefulNonidealVideoSensor, frame_duration, steps_per_frame  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot center-pixel waveforms for the noisy case1 native/x2 comparison and "
            "compute noise-current ratios relative to lowest/highest optical-power photocurrent."
        )
    )
    parser.add_argument(
        "--results-dir",
        default=str(THIS_DIR / "artifacts" / "case1_native_vs_x2_video_sequence_noise1153p69"),
        help="Results directory created by evaluate_case1_native_vs_x2_video_sequence.py",
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10"])
    parser.add_argument("--split", default="test", choices=["test"])
    parser.add_argument("--sample-index", type=int, default=23)
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--output-path", default="auto")
    return parser.parse_args()


def resolve_output_path(args):
    if args.output_path not in {None, "", "auto"}:
        return Path(args.output_path).expanduser()
    return Path(args.results_dir) / f"center_pixel_noise_waveform_fps{int(args.video_fps):03d}_sample{args.sample_index:04d}.png"


def build_args_template(args, params_csv, use_noise_fn):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        split=args.split,
        generate_images=False,
        run_eval=False,
        eval_cases=["nonideal"],
        batch_size=1,
        num_workers=0,
        max_eval_batches=0,
        model_path=None,
        num_classes=0,
        results_json=None,
        seed=1234,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=32,
        output_channels=3,
        post_norm="auto",
        num_images=0,
        start_index=0,
        array_size=32,
        tile_size=256,
        readout="integration",
        analog_readout="integration",
        adc_enabled=0,
        power_max=base_pipeline.DEFAULT_POWER_MAX_W,
        params_csv=str(params_csv),
        normalization_mode="calibration",
        prange1_density=None,
        prange2_density=None,
        pmin_density=None,
        pmax_density=None,
        device_area_cm2=None,
        force_single_carrier=0,
        single_r=None,
        single_eta=None,
        single_trise=None,
        single_tfall=None,
        trap_saturation_time=None,
        trap_amplitude_pct=None,
        noise_1f_density_1hz=None,
        aging_tau_hours=None,
        r_degradation_pct=None,
        spatial_variation_r_pct=None,
        tia_gain_ohm=None,
        integration_gain_v_per_c=None,
        video_fps=args.video_fps,
        fps_sim=args.fps_sim,
        adc_bits=8,
        adc_full_scale=None,
        range_mode="auto",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=1024,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=0,
        use_noise_fn=int(bool(use_noise_fn)),
        startup_dark_frames=0,
        output_dir=None,
        analyze_center_pixel=0,
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def extract_center_values(array):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray([arr[arr.shape[0] // 2, arr.shape[1] // 2]], dtype=np.float64)
    if arr.ndim == 3:
        return np.asarray(arr[:, arr.shape[1] // 2, arr.shape[2] // 2], dtype=np.float64)
    raise ValueError(f"Unsupported array rank: {arr.ndim}")


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


def frame_end_photocurrent_from_power(base_params, power_w, video_fps, fps_sim):
    model_config = build_model_config(base_params)
    n_carrier = pm.infer_n_carrier_from_params(base_params)
    frame_s = 1.0 / float(video_fps)
    n_steps = max(1, int(round(frame_s * float(fps_sim))))
    dt = frame_s / n_steps

    x1 = np.zeros((n_carrier,), dtype=np.float64)
    x2 = 0.0
    x3 = 0.0
    for _ in range(n_steps):
        x1, x2, x3 = pm.step_model_state(power_w, dt, model_config, x1, x2, x3)
    _total_current, photo_current = pm.current_from_state(model_config, x1, x2, x3, P=power_w, dark_current=0.0)
    return float(max(photo_current, 0.0))


def simulate_center_waveform(args, params_csv):
    base_dataset = base_pipeline.load_base_dataset(args.source_dataset, args.data_root, args.split)
    noisy_args = build_args_template(args, params_csv, use_noise_fn=True)
    noisy_args = pipeline.resolve_runtime_defaults(noisy_args)
    det_args = build_args_template(args, params_csv, use_noise_fn=False)
    det_args = pipeline.resolve_runtime_defaults(det_args)

    base_params = pipeline.resolve_sequence_base_params(noisy_args)
    if getattr(noisy_args, "tia_gain_ohm", None) is None:
        noisy_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if getattr(noisy_args, "integration_gain_v_per_c", None) is None:
        noisy_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    if getattr(det_args, "tia_gain_ohm", None) is None:
        det_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if getattr(det_args, "integration_gain_v_per_c", None) is None:
        det_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    noisy_sensor = StatefulNonidealVideoSensor(args=noisy_args, base_params=base_params)
    det_sensor = StatefulNonidealVideoSensor(args=det_args, base_params=base_params)

    max_needed_index = int(args.sample_index)
    noisy_args.total_sequence_frames = max_needed_index + 1
    det_args.total_sequence_frames = max_needed_index + 1

    selected_label = None
    center_power = None
    det_center_trace = None
    noisy_center_trace = None
    for dataset_index in range(max_needed_index + 1):
        image, label = base_dataset[dataset_index]
        power_maps = pipeline.build_sequence_power_maps(image, noisy_args, base_params)
        det_frame, det_center = det_sensor.simulate_frame(power_maps, record_center_trace=True)
        noisy_frame, noisy_center = noisy_sensor.simulate_frame(power_maps, record_center_trace=True)
        del det_frame, noisy_frame
        if dataset_index == args.sample_index:
            selected_label = base_dataset.classes[label]
            center_power = extract_center_values(power_maps)
            det_center_trace = np.asarray(det_center, dtype=np.float64)
            noisy_center_trace = np.asarray(noisy_center, dtype=np.float64)
            break

    if det_center_trace is None or noisy_center_trace is None:
        raise RuntimeError(f"Failed to collect sample {args.sample_index}")

    noise_trace = noisy_center_trace - det_center_trace
    time_ms = (
        (np.arange(steps_per_frame(noisy_args), dtype=np.float64) + 1.0)
        * (frame_duration(noisy_args) / steps_per_frame(noisy_args))
        * 1e3
    )
    return {
        "label": selected_label,
        "center_power_w": center_power,
        "det_trace_a": det_center_trace,
        "noisy_trace_a": noisy_center_trace,
        "noise_trace_a": noise_trace,
        "time_ms": time_ms,
        "base_params": base_params,
    }


def compute_noise_ratios(base_params, noise_trace_a, video_fps, fps_sim):
    noise_trace_a = np.asarray(noise_trace_a, dtype=np.float64)
    noise_rms_a = float(np.sqrt(np.mean(noise_trace_a ** 2)))
    noise_abs_max_a = float(np.max(np.abs(noise_trace_a)))
    photo_low_a = frame_end_photocurrent_from_power(
        base_params,
        float(base_params["pmin_w"]),
        video_fps=video_fps,
        fps_sim=fps_sim,
    )
    photo_high_a = frame_end_photocurrent_from_power(
        base_params,
        float(base_params["pmax_w"]),
        video_fps=video_fps,
        fps_sim=fps_sim,
    )

    return {
        "noise_rms_a": noise_rms_a,
        "noise_abs_max_a": noise_abs_max_a,
        "photo_current_low_power_a": photo_low_a,
        "photo_current_high_power_a": photo_high_a,
        "rms_vs_low_pct": 100.0 * noise_rms_a / max(photo_low_a, 1e-30),
        "rms_vs_high_pct": 100.0 * noise_rms_a / max(photo_high_a, 1e-30),
        "peak_vs_low_pct": 100.0 * noise_abs_max_a / max(photo_low_a, 1e-30),
        "peak_vs_high_pct": 100.0 * noise_abs_max_a / max(photo_high_a, 1e-30),
    }


def render_plot(results_by_group, output_path, fps, sample_index):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.0), constrained_layout=True, sharex="col")
    colors = ["tab:red", "tab:green", "tab:blue"]

    for col, (group_key, payload) in enumerate(results_by_group.items()):
        top_ax = axes[0, col]
        bottom_ax = axes[1, col]
        for ch_idx in range(payload["det_trace_a"].shape[1]):
            color = colors[ch_idx % len(colors)]
            top_ax.plot(payload["time_ms"], payload["det_trace_a"][:, ch_idx] * 1e3, "--", color=color, linewidth=1.6)
            top_ax.plot(payload["time_ms"], payload["noisy_trace_a"][:, ch_idx] * 1e3, color=color, linewidth=2.0)
            bottom_ax.plot(payload["time_ms"], payload["noise_trace_a"][:, ch_idx] * 1e6, color=color, linewidth=1.8)

        top_ax.set_title(
            f"{payload['display']} | sample {sample_index}: {payload['label']}",
            fontsize=12,
            fontweight="bold",
        )
        top_ax.set_ylabel("Center current (mA)", fontsize=11, fontweight="bold")
        top_ax.grid(True, alpha=0.25)
        bottom_ax.set_ylabel("Noise current (uA)", fontsize=11, fontweight="bold")
        bottom_ax.set_xlabel("Time within frame (ms)", fontsize=11, fontweight="bold")
        bottom_ax.grid(True, alpha=0.25)

        ratio = payload["ratios"]
        note = (
            f"RMS={ratio['noise_rms_a']*1e6:.3f} uA | "
            f"peak={ratio['noise_abs_max_a']*1e6:.3f} uA\n"
            f"RMS/Ilow={ratio['rms_vs_low_pct']:.4f}% | RMS/Ihigh={ratio['rms_vs_high_pct']:.4f}%"
        )
        bottom_ax.text(
            0.02,
            0.96,
            note,
            transform=bottom_ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#999999"},
        )

        for ax in (top_ax, bottom_ax):
            for spine in ax.spines.values():
                spine.set_linewidth(1.4)
            ax.tick_params(labelsize=10, width=1.2)

    fig.suptitle(f"Center-pixel waveform with noise | FPS {int(fps)}", fontsize=14, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    parameter_summary_path = results_dir / "parameter_summary.json"
    if not parameter_summary_path.is_file():
        raise FileNotFoundError(f"parameter_summary.json not found in {results_dir}")
    parameter_summary = json.loads(parameter_summary_path.read_text(encoding="utf-8"))

    params_by_group = {
        "native": Path(parameter_summary["native_csv"]).resolve(),
        "x2_enhanced": Path(parameter_summary["x2_enhanced_csv"]).resolve(),
    }

    output_path = resolve_output_path(args)
    results_by_group = {}
    for group_key, params_csv in params_by_group.items():
        waveform = simulate_center_waveform(args, params_csv=params_csv)
        ratios = compute_noise_ratios(
            waveform["base_params"],
            waveform["noise_trace_a"],
            video_fps=args.video_fps,
            fps_sim=args.fps_sim,
        )
        results_by_group[group_key] = {
            "display": "Case1 Native" if group_key == "native" else "Case1 + Enhanced x2",
            **waveform,
            "ratios": ratios,
            "params_csv": str(params_csv),
        }

    render_plot(results_by_group, output_path, fps=args.video_fps, sample_index=args.sample_index)

    summary = {
        "results_dir": str(results_dir),
        "sample_index": int(args.sample_index),
        "video_fps": float(args.video_fps),
        "fps_sim": float(args.fps_sim),
        "output_plot": str(output_path),
        "groups": {
            group_key: {
                "display": payload["display"],
                "label": payload["label"],
                "params_csv": payload["params_csv"],
                "center_power_w": [float(value) for value in payload["center_power_w"]],
                **payload["ratios"],
            }
            for group_key, payload in results_by_group.items()
        },
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Waveform plot: {output_path}", flush=True)
    print(f"Summary JSON: {summary_path}", flush=True)
    for group_key, payload in results_by_group.items():
        ratios = payload["ratios"]
        print(
            (
                f"{payload['display']}: "
                f"noise_rms={ratios['noise_rms_a']*1e6:.6f} uA, "
                f"noise_abs_max={ratios['noise_abs_max_a']*1e6:.6f} uA, "
                f"RMS/Ilow={ratios['rms_vs_low_pct']:.6f}%, "
                f"RMS/Ihigh={ratios['rms_vs_high_pct']:.6f}%, "
                f"Peak/Ilow={ratios['peak_vs_low_pct']:.6f}%, "
                f"Peak/Ihigh={ratios['peak_vs_high_pct']:.6f}%"
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
