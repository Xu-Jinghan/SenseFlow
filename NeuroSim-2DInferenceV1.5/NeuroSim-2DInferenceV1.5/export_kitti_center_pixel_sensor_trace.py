import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent.parent
LOCAL_DATA_PATH = REPO_ROOT / ".datasets"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eval_yolo11n_kitti_tracking_sensor as kitti_eval  # noqa: E402


def _default_kitti_root():
    return str((LOCAL_DATA_PATH / "kitti_tracking") if LOCAL_DATA_PATH.exists() else REPO_ROOT / ".datasets" / "kitti_tracking")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export center-pixel optical power and nonideal Iout traces for KITTI sensor video frames."
    )
    parser.add_argument("--kitti-root", default=_default_kitti_root())
    parser.add_argument("--sequence", default="0019")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=500)
    parser.add_argument("--output-width", type=int, default=640)
    parser.add_argument("--output-height", type=int, default=192)
    parser.add_argument("--params-csv", default=str(kitti_eval.DEFAULT_PARAMS_CSV))
    parser.add_argument("--normalization-mode", default="calibration", choices=["physical", "calibration", "per_frame", "none"])
    parser.add_argument("--range-mode", default="minmax", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument("--range-calibration-samples", type=int, default=500)
    parser.add_argument("--range-calibration-max-values-per-frame", type=int, default=20000)
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--prange1-density", type=float, default=None)
    parser.add_argument("--prange2-density", type=float, default=None)
    parser.add_argument("--pmin-density", type=float, default=None)
    parser.add_argument("--pmax-density", type=float, default=None)
    parser.add_argument("--device-area-cm2", type=float, default=None)
    parser.add_argument("--force-single-carrier", type=int, default=0)
    parser.add_argument("--single-r", type=float, default=None)
    parser.add_argument("--single-eta", type=float, default=None)
    parser.add_argument("--single-trise", type=float, default=None)
    parser.add_argument("--single-tfall", type=float, default=None)
    parser.add_argument("--trap-saturation-time", type=float, default=None)
    parser.add_argument("--trap-amplitude-pct", type=float, default=None)
    parser.add_argument("--x2-trap-mode", default="power", choices=["binary", "power"])
    parser.add_argument("--x2-trap-output-mode", default="always", choices=["always", "illumination_gated"])
    parser.add_argument("--x2-alpha", type=float, default=None, help="Override x2 alpha; default uses the case2 CSV value.")
    parser.add_argument("--x2-beta", type=float, default=None, help="Override x2 beta; default uses the CSV value.")
    parser.add_argument(
        "--detection-dark-current-a",
        type=float,
        default=0.0,
        help="Dark-current baseline used for the exported detection sensor trace. Default disables dark current.",
    )
    parser.add_argument(
        "--noise-1f-density-1hz",
        type=float,
        default=None,
        help=(
            "Manual 1/f noise density at 1 Hz in A/Hz^0.5. If omitted, derive it from "
            "the case1 PSD using --case1-noise-scale-vs-case1."
        ),
    )
    parser.add_argument(
        "--case1-noise-scale-vs-case1",
        type=float,
        default=kitti_eval.DEFAULT_CASE1_NOISE_SCALE_VS_CASE1,
        help="Scale factor applied to the measured case1 PSD before extracting the 1 Hz density.",
    )
    parser.add_argument("--aging-tau-hours", type=float, default=None)
    parser.add_argument("--r-degradation-pct", type=float, default=None)
    parser.add_argument("--spatial-variation-r-pct", type=float, default=0.0)
    parser.add_argument("--tia-gain-ohm", type=float, default=None)
    parser.add_argument("--integration-gain-v-per-c", type=float, default=None)
    parser.add_argument("--readout", default="tia", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog-readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc-enabled", type=int, default=0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--adc-calibration-low", type=float, default=None)
    parser.add_argument("--adc-calibration-high", type=float, default=None)
    parser.add_argument("--video-fps", type=float, default=10.0)
    parser.add_argument("--fps-sim", type=float, default=200.0)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--shot-noise", type=int, default=0)
    parser.add_argument("--use-noise-fn", type=int, default=1)
    parser.add_argument(
        "--temporal-noise-mode",
        default="pixel_buffered",
        choices=["pixel_buffered", "pixel_repeated_window", "global_full_sequence", "global_repeated_window"],
        help=(
            "Use pixel_repeated_window for independent per-pixel reusable traces, "
            "global_full_sequence for a full-duration 1/f trace per channel, "
            "or global_repeated_window for a shorter per-channel reusable trace."
        ),
    )
    parser.add_argument(
        "--temporal-noise-window-frames",
        type=int,
        default=10,
        help="Number of frames in the reusable temporal-noise window for repeated-window modes.",
    )
    parser.add_argument("--target-channel", type=int, default=1, help="Center pixel channel to plot: 0=R, 1=G, 2=B.")
    parser.add_argument("--max-plot-frames", type=int, default=120)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "detection_runs" / "yolo11n_kitti_tracking_case2_sensor_calibrated_subset_0019_500" / "center_pixel_trace"),
    )
    return parser.parse_args()


def make_sensor_args(args, base_pipeline, total_sequence_frames):
    return kitti_eval.make_sensor_args(args, base_pipeline, total_sequence_frames=total_sequence_frames)


def main():
    args = parse_args()
    requested_noise_config = kitti_eval.resolve_detection_noise_config(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_pipeline, video_pipeline = kitti_eval.import_sensor_pipeline()
    kitti_root = Path(args.kitti_root).expanduser().resolve()
    frame_paths = kitti_eval.select_consecutive_frames(kitti_root, args.sequence, args.start_frame, args.num_frames)

    sensor_args = make_sensor_args(
        args,
        base_pipeline,
        total_sequence_frames=len(frame_paths) + max(0, int(args.startup_dark_frames)),
    )
    if sensor_args.analog_readout is None:
        sensor_args.analog_readout = video_pipeline.resolve_analog_readout_mode(sensor_args)
    sensor_args.frame_range_mode_override = video_pipeline.effective_frame_range_mode(sensor_args)

    base_params = video_pipeline.resolve_sequence_base_params(sensor_args)
    base_params = kitti_eval.apply_detection_x2_overrides(args, base_params)
    if sensor_args.tia_gain_ohm is None:
        sensor_args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if sensor_args.integration_gain_v_per_c is None:
        sensor_args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))

    range_cases = ["ideal", "nonideal"]
    case_range_bounds = kitti_eval.compute_dataset_range_bounds(
        args,
        frame_paths,
        sensor_args,
        base_params,
        range_cases,
        base_pipeline,
        video_pipeline,
    )

    sensor = video_pipeline.StatefulNonidealVideoSensor(args=sensor_args, base_params=base_params)
    zero_power = np.zeros((3, args.output_height, args.output_width), dtype=np.float64)
    for _ in range(max(0, int(args.startup_dark_frames))):
        sensor.simulate_frame(zero_power)

    center_y = args.output_height // 2
    center_x = args.output_width // 2
    target_channel = int(np.clip(args.target_channel, 0, 2))
    steps_per_frame = int(sensor.n_steps)
    dt = float(sensor.dt)

    rows = []
    frame_rows = []
    global_step = 0
    for frame_offset, image_path in enumerate(frame_paths):
        source = Image.open(image_path).convert("RGB")
        power_maps = kitti_eval.build_power_maps_rect(source, args.output_width, args.output_height, 3, base_params)
        output_frame, center_trace = sensor.simulate_frame(power_maps, record_center_trace=True)
        nonideal_unit = kitti_eval.scale_sensor_frame(
            output_frame,
            "nonideal",
            sensor_args,
            case_range_bounds,
            base_pipeline,
            video_pipeline,
        )
        nonideal_rgb = kitti_eval.frame_to_rgb_array(nonideal_unit)

        center_power = np.asarray(power_maps[:, center_y, center_x], dtype=np.float64)
        center_iout = np.asarray(center_trace, dtype=np.float64)
        center_scaled = np.asarray(nonideal_rgb[center_y, center_x, :], dtype=np.float64)
        frame_mean = np.asarray(nonideal_rgb, dtype=np.float64).mean(axis=(0, 1))
        frame_rows.append(
            {
                "frame_offset": frame_offset,
                "frame_name": image_path.name,
                "frame_time_s": frame_offset / float(args.video_fps),
                "center_power_w_ch0": center_power[0],
                "center_power_w_ch1": center_power[1],
                "center_power_w_ch2": center_power[2],
                "center_scaled_ch0": center_scaled[0],
                "center_scaled_ch1": center_scaled[1],
                "center_scaled_ch2": center_scaled[2],
                "nonideal_mean_ch0": frame_mean[0],
                "nonideal_mean_ch1": frame_mean[1],
                "nonideal_mean_ch2": frame_mean[2],
                "nonideal_mean_rgb": float(np.mean(frame_mean)),
            }
        )
        for step_idx in range(steps_per_frame):
            t_s = global_step * dt
            rows.append(
                {
                    "global_step": global_step,
                    "frame_offset": frame_offset,
                    "frame_name": image_path.name,
                    "step_in_frame": step_idx,
                    "time_s": t_s,
                    "power_w_ch0": center_power[0],
                    "power_w_ch1": center_power[1],
                    "power_w_ch2": center_power[2],
                    "iout_a_ch0": center_iout[step_idx, 0],
                    "iout_a_ch1": center_iout[step_idx, 1],
                    "iout_a_ch2": center_iout[step_idx, 2],
                    "power_w_target": center_power[target_channel],
                    "iout_a_target": center_iout[step_idx, target_channel],
                }
            )
            global_step += 1

        processed = frame_offset + 1
        if processed % 100 == 0 or processed == len(frame_paths):
            print(f"trace progress: frames={processed}/{len(frame_paths)}", flush=True)

    waveform_csv = output_dir / "center_pixel_waveform.csv"
    with waveform_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    frame_csv = output_dir / "center_pixel_frame_summary.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    plot_frames = min(len(frame_paths), int(args.max_plot_frames))
    plot_steps = plot_frames * steps_per_frame
    time_axis = np.asarray([row["time_s"] for row in rows[:plot_steps]], dtype=np.float64)
    power_axis = np.asarray([row["power_w_target"] for row in rows[:plot_steps]], dtype=np.float64)
    iout_axis = np.asarray([row["iout_a_target"] for row in rows[:plot_steps]], dtype=np.float64)
    mean_axis = np.asarray([row["nonideal_mean_rgb"] for row in frame_rows[:plot_frames]], dtype=np.float64)
    frame_time = np.asarray([row["frame_time_s"] for row in frame_rows[:plot_frames]], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
    axes[0].plot(time_axis, power_axis, lw=1.3)
    axes[0].set_ylabel("P center (W)")
    axes[0].set_title(f"KITTI {args.sequence} center pixel ch{target_channel}: optical input and nonideal Iout")
    axes[1].plot(time_axis, iout_axis, lw=1.3, color="#b03030")
    axes[1].set_ylabel("Iout center (A)")
    axes[2].plot(frame_time, mean_axis, lw=1.3, color="#2f6f4e")
    axes[2].set_ylabel("nonideal mean")
    axes[2].set_xlabel("time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    waveform_png = output_dir / "center_pixel_waveform.png"
    fig.savefig(waveform_png, dpi=180)
    plt.close(fig)

    summary = {
        "sequence": args.sequence,
        "start_frame": args.start_frame,
        "num_frames": len(frame_paths),
        "video_fps": args.video_fps,
        "fps_sim": args.fps_sim,
        "steps_per_frame": steps_per_frame,
        "dt_s": dt,
        "center_xy": [center_x, center_y],
        "target_channel": target_channel,
        "params_csv": args.params_csv,
        "shot_noise": int(args.shot_noise),
        "use_noise_fn": int(args.use_noise_fn),
        "temporal_noise_mode": str(args.temporal_noise_mode),
        "temporal_noise_window_frames": int(args.temporal_noise_window_frames),
        "spatial_variation_r_pct": float(args.spatial_variation_r_pct),
        "x2_trap_mode": args.x2_trap_mode,
        "x2_trap_output_mode": args.x2_trap_output_mode,
        "x2_alpha": args.x2_alpha,
        "x2_beta": args.x2_beta,
        "noise_config": requested_noise_config,
        "normalization_mode": args.normalization_mode,
        "range_mode": args.range_mode,
        "range_calibration_samples": args.range_calibration_samples,
        "range_calibration_max_values_per_frame": args.range_calibration_max_values_per_frame,
        "case_range_bounds": case_range_bounds,
        "noise_trace": kitti_eval.summarize_actual_noise_trace(sensor_args, base_params),
        "base_params": {
            key: (None if value is None else float(value) if isinstance(value, (int, float, np.floating)) else value)
            for key, value in base_params.items()
            if key not in {"spatial_variation_cache_dir"}
        },
        "outputs": {
            "waveform_csv": str(waveform_csv),
            "frame_summary_csv": str(frame_csv),
            "waveform_png": str(waveform_png),
        },
        "center_power_target_w_min": float(np.min([row["power_w_target"] for row in rows])),
        "center_power_target_w_max": float(np.max([row["power_w_target"] for row in rows])),
        "center_iout_target_a_min": float(np.min([row["iout_a_target"] for row in rows])),
        "center_iout_target_a_max": float(np.max([row["iout_a_target"] for row in rows])),
        "nonideal_mean_rgb_first": float(frame_rows[0]["nonideal_mean_rgb"]),
        "nonideal_mean_rgb_last": float(frame_rows[-1]["nonideal_mean_rgb"]),
    }
    summary_path = output_dir / "center_pixel_trace_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Waveform CSV: {waveform_csv}")
    print(f"Frame summary CSV: {frame_csv}")
    print(f"Plot: {waveform_png}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
