import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
MPLCONFIGDIR = THIS_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import generate_fig3_arbitrary_case1_on_state_drift as fig3_case1  # noqa: E402
import generate_sensor_verification_images as base_pipeline  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402
import photodetector_model as pm  # noqa: E402
import sensor_video_sequence_backend as video_backend  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "case1_fig3_video_sequence_montage"
DEFAULT_FALLBACK_PARAMS_CSV = REPO_ROOT / "outputs" / "case1_single_equiv_video_params.csv"
FONT_SIZE_LABEL = 16
FONT_SIZE_TICK = 13
FONT_SIZE_IMAGE_TEXT = 16
FONT_SIZE_LEGEND = 14
SPINE_WIDTH = 2.3
TRACE_LINE_WIDTH = 2.6
TRACE_BASELINE_WIDTH = 1.8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a video-sequence montage that matches the current fig3 case1 setup "
            "while allowing stronger x2 and visible noise overlays."
        )
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument(
        "--param-source",
        default="fig3_updated_case1",
        choices=["fig3_updated_case1", "csv"],
    )
    parser.add_argument("--params-csv", default=str(DEFAULT_FALLBACK_PARAMS_CSV))
    parser.add_argument("--output-path", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog-readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc-enabled", type=int, default=0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--video-fps", type=float, default=50.0)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument(
        "--normalization-mode",
        default="calibration",
        choices=["physical", "calibration", "per_frame", "none"],
    )
    parser.add_argument("--range-mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument("--range-scope", default="calibration", choices=["per_frame", "calibration"])
    parser.add_argument("--range-calibration-split", default="train", choices=["train", "test"])
    parser.add_argument("--range-calibration-samples", type=int, default=1024)
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument("--shot-noise", type=int, default=0)
    parser.add_argument("--use-noise-fn", type=int, default=1)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--fig3-x2-delta-scale", type=float, default=1.0)
    parser.add_argument("--fig3-noise-scale-multiplier", type=float, default=1.0)
    parser.add_argument("--plot-noise-scale", type=float, default=1.0)
    return parser.parse_args()


def resolve_output_path(cli_args):
    if cli_args.output_path not in {None, "", "auto"}:
        return Path(cli_args.output_path).expanduser()
    filename = (
        f"case1_fig3_{cli_args.source_dataset}_{cli_args.split}_start{cli_args.start_index}"
        f"_first{cli_args.num_images}_continuous_center_pixel_traces.png"
    )
    return DEFAULT_OUTPUT_DIR / filename


def build_runtime_args(cli_args, output_path):
    return argparse.Namespace(
        data_root=cli_args.data_root,
        source_dataset=cli_args.source_dataset,
        split=cli_args.split,
        generate_images=0,
        run_eval=0,
        eval_cases=["nonideal"],
        batch_size=1,
        num_workers=0,
        max_eval_batches=0,
        model_path=None,
        num_classes=0,
        results_json=None,
        seed=cli_args.seed,
        sensor_rng_seed=cli_args.sensor_rng_seed,
        target_size=cli_args.target_size,
        output_channels=cli_args.output_channels,
        post_norm="auto",
        num_images=cli_args.num_images,
        start_index=cli_args.start_index,
        array_size=cli_args.array_size,
        tile_size=cli_args.tile_size,
        readout=cli_args.readout,
        analog_readout=cli_args.analog_readout,
        adc_enabled=cli_args.adc_enabled,
        power_max=base_pipeline.DEFAULT_POWER_MAX_W,
        params_csv=cli_args.params_csv,
        normalization_mode=cli_args.normalization_mode,
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
        video_fps=cli_args.video_fps,
        fps_sim=cli_args.fps_sim,
        adc_bits=cli_args.adc_bits,
        adc_full_scale=cli_args.adc_full_scale,
        range_mode=cli_args.range_mode,
        range_scope=cli_args.range_scope,
        percentile_low=cli_args.percentile_low,
        percentile_high=cli_args.percentile_high,
        range_calibration_split=cli_args.range_calibration_split,
        range_calibration_samples=cli_args.range_calibration_samples,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=cli_args.shot_noise,
        use_noise_fn=cli_args.use_noise_fn,
        startup_dark_frames=cli_args.startup_dark_frames,
        output_dir=str(output_path.parent),
        analyze_center_pixel=0,
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def to_display_image(array):
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 2:
        return np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"Unsupported array rank for image display: {arr.ndim}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return np.clip(arr, 0.0, 1.0)


def extract_center_pixel_values(array):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray([arr[arr.shape[0] // 2, arr.shape[1] // 2]], dtype=np.float64)
    if arr.ndim == 3:
        if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            return np.asarray(arr[:, arr.shape[1] // 2, arr.shape[2] // 2], dtype=np.float64)
        return np.asarray(arr[arr.shape[0] // 2, arr.shape[1] // 2, :], dtype=np.float64)
    raise ValueError(f"Unsupported array rank for center-pixel extraction: {arr.ndim}")


def channel_labels_for(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        return ["mono"]
    if arr.ndim == 3:
        if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            n_channels = arr.shape[0]
        else:
            n_channels = arr.shape[-1]
        return [f"ch{i}" for i in range(n_channels)]
    raise ValueError(f"Unsupported array rank for channel labels: {arr.ndim}")


def center_xy_for(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        return (int(arr.shape[0] // 2), int(arr.shape[1] // 2))
    if arr.ndim == 3:
        if arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            return (int(arr.shape[1] // 2), int(arr.shape[2] // 2))
        return (int(arr.shape[0] // 2), int(arr.shape[1] // 2))
    raise ValueError(f"Unsupported array rank for center coordinate: {arr.ndim}")


def compute_unit_interval_psnr(reference, compared):
    reference = np.asarray(reference, dtype=np.float64)
    compared = np.asarray(compared, dtype=np.float64)
    if reference.shape != compared.shape:
        raise ValueError(f"PSNR inputs must match in shape, got {reference.shape} and {compared.shape}")
    mse = float(np.mean((np.clip(reference, 0.0, 1.0) - np.clip(compared, 0.0, 1.0)) ** 2))
    if mse <= 1e-24:
        return float("inf")
    return float(20.0 * np.log10(1.0 / np.sqrt(mse)))


def write_parameter_csv(csv_path, rows):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key, value in rows:
            writer.writerow([key, value])


def build_updated_fig3_case1_param_artifacts(output_dir, delta_scale=1.0, noise_scale_multiplier=1.0):
    case1_results, params_with_x2, _ = fig3_case1.build_case1_parameter_sets()
    waveform_t_s, waveform_power_w = fig3_case1.build_small_range_waveform()
    _scaled_noise_fn, noise_scale, base_noise_rms_a = fig3_case1.calibrate_noise(
        case1_results,
        waveform_t_s,
        waveform_power_w,
    )

    delta_scale = float(delta_scale)
    noise_scale_multiplier = float(noise_scale_multiplier)
    effective_noise_scale = float(noise_scale * noise_scale_multiplier)

    scaled_density = pm.scale_noise_density_components(
        case1_results["dataset"]["noise_freq_hz"],
        case1_results["dataset"]["noise_density"],
        white_scale=effective_noise_scale,
        flicker_scale=effective_noise_scale,
    )
    noise_density_1hz = float(
        pm._resample_noise_density(
            np.asarray([1.0], dtype=np.float64),
            case1_results["dataset"]["noise_freq_hz"],
            scaled_density,
        )[0]
    )

    derived_params = dict(params_with_x2)
    derived_params["delta"] = float(derived_params["delta"]) * delta_scale
    derived_params.update(
        {
            "device_area_cm2": float(pm.DEVICE_AREA_CM2),
            "power_ref_w": float(case1_results["power_ref_w"]),
            "noise_1f_density_1hz_a_root_hz": noise_density_1hz,
            "noise_scale_vs_case1": float(effective_noise_scale),
        }
    )

    csv_rows = [
        ("R_single", derived_params["R_single"]),
        ("eta_single", derived_params["eta_single"]),
        ("tau_rise_single", derived_params["tau_rise_single"]),
        ("tau_fall_single", derived_params["tau_fall_single"]),
        ("alpha", derived_params["alpha"]),
        ("beta", derived_params["beta"]),
        ("delta", derived_params["delta"]),
        ("gamma", derived_params["gamma"]),
        ("tau_drift", derived_params["tau_drift"]),
        ("drift_scale", derived_params["drift_scale"]),
        ("device_area_cm2", derived_params["device_area_cm2"]),
        ("power_ref_w", derived_params["power_ref_w"]),
        ("noise_1f_density_1hz_a_root_hz", derived_params["noise_1f_density_1hz_a_root_hz"]),
        ("noise_scale_vs_case1", derived_params["noise_scale_vs_case1"]),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "case1_fig3_updated_video_params.csv"
    write_parameter_csv(csv_path, csv_rows)

    exact_noise_fn = pm.make_scaled_case1_noise_function(
        case1_results,
        amplitude_scale=effective_noise_scale,
        flicker_scale=1.0,
    )
    summary = {
        "derived_params_csv": str(csv_path),
        "power_ref_w": float(case1_results["power_ref_w"]),
        "device_area_cm2": float(pm.DEVICE_AREA_CM2),
        "alpha": float(derived_params["alpha"]),
        "beta": float(derived_params["beta"]),
        "delta_a": float(derived_params["delta"]),
        "gamma": float(derived_params["gamma"]),
        "tau_drift_s": float(derived_params["tau_drift"]),
        "delta_scale": float(delta_scale),
        "base_noise_scale_vs_case1": float(noise_scale),
        "noise_scale_multiplier": float(noise_scale_multiplier),
        "noise_scale_vs_case1": float(effective_noise_scale),
        "noise_1f_density_1hz_a_root_hz": float(noise_density_1hz),
        "base_noise_rms_ua": float(base_noise_rms_a * 1e6),
        "target_noise_rms_ua": float(fig3_case1.TARGET_NOISE_RMS_UA),
    }
    summary_path = output_dir / "case1_fig3_updated_video_params.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return csv_path, exact_noise_fn, summary


def resolve_param_source(cli_args, output_path):
    if cli_args.param_source == "csv":
        return {
            "params_csv": Path(cli_args.params_csv).expanduser(),
            "exact_noise_fn": None,
            "summary": {
                "param_source": "csv",
                "params_csv": str(Path(cli_args.params_csv).expanduser()),
            },
        }

    derived_csv_path, exact_noise_fn, derived_summary = build_updated_fig3_case1_param_artifacts(
        output_path.parent,
        delta_scale=cli_args.fig3_x2_delta_scale,
        noise_scale_multiplier=cli_args.fig3_noise_scale_multiplier,
    )
    return {
        "params_csv": derived_csv_path,
        "exact_noise_fn": exact_noise_fn,
        "summary": {
            "param_source": "fig3_updated_case1",
            **derived_summary,
        },
    }


def simulate_frame_with_custom_noise(sequence_sensor, power_maps, exact_noise_fn=None):
    power_maps = np.asarray(power_maps, dtype=np.float64)
    sequence_sensor._ensure_state(power_maps.shape)

    current_det_trace = []
    for _ in range(sequence_sensor.n_steps):
        sequence_sensor.x1, sequence_sensor.x2, sequence_sensor.x3 = pm.step_model_state(
            power_maps,
            sequence_sensor.dt,
            sequence_sensor.model_config,
            sequence_sensor.x1,
            sequence_sensor.x2,
            sequence_sensor.x3,
        )
        current_det, _ = pm.current_from_state(
            sequence_sensor.model_config,
            sequence_sensor.x1,
            sequence_sensor.x2,
            sequence_sensor.x3,
            P=power_maps,
            dark_current=sequence_sensor.dark_current,
        )
        current_det_trace.append(np.asarray(current_det, dtype=np.float64))

    current_det_trace = np.stack(current_det_trace, axis=0)
    step_time_s = sequence_sensor.elapsed_time_s + (
        (np.arange(sequence_sensor.n_steps, dtype=np.float64) + 1.0) * sequence_sensor.dt
    )
    if exact_noise_fn is None:
        noise_trace = sequence_sensor._sample_noise_trace_chunk(power_maps.shape)
    else:
        noise_trace = pm.sample_time_noise_fn_trace(
            exact_noise_fn,
            step_time_s,
            current_det_trace,
            power_maps,
            sequence_sensor.rng,
        )

    current_out_trace = current_det_trace + np.asarray(noise_trace, dtype=np.float64)
    readout_frame = video_backend.apply_readout_to_current_trace(
        current_out_trace,
        sequence_sensor.args,
        sequence_sensor.dt,
    )
    sequence_sensor.elapsed_time_s += sequence_sensor.frame_duration
    return (
        np.asarray(readout_frame, dtype=np.float32),
        np.asarray(current_det_trace, dtype=np.float64),
        np.asarray(current_out_trace, dtype=np.float64),
    )


def collect_samples(args, base_params, exact_noise_fn=None):
    base_dataset = base_pipeline.load_base_dataset(args.source_dataset, args.data_root, args.split)
    if len(base_dataset) == 0:
        raise ValueError("Dataset is empty.")

    if args.start_index < 0 or args.start_index >= len(base_dataset):
        raise IndexError(f"start_index {args.start_index} is outside dataset length {len(base_dataset)}")

    stop_index = min(len(base_dataset), args.start_index + max(0, args.num_images))
    if stop_index <= args.start_index:
        raise ValueError("num_images must be positive.")

    calibration_dataset = pipeline._load_calibration_dataset(args, base_dataset)
    case_range_bounds = pipeline.compute_case_range_bounds(
        args=args,
        calibration_dataset=calibration_dataset,
        base_params=base_params,
        cases=pipeline.RANGE_CASES,
    )

    max_needed_index = stop_index - 1
    args.total_sequence_frames = max_needed_index + 1 + max(0, int(args.startup_dark_frames))
    sequence_sensor = video_backend.StatefulNonidealVideoSensor(args=args, base_params=base_params)

    samples = []
    for dataset_index in range(stop_index):
        image, label = base_dataset[dataset_index]
        power_maps = pipeline.build_sequence_power_maps(image, args, base_params)
        readout_frame, current_det_trace, current_out_trace = simulate_frame_with_custom_noise(
            sequence_sensor,
            power_maps,
            exact_noise_fn=exact_noise_fn,
        )
        nonideal_scaled = pipeline.scale_case_frame(readout_frame, "nonideal", args, case_range_bounds)

        if dataset_index < args.start_index:
            continue

        center_power = extract_center_pixel_values(power_maps)
        center_power_trace = np.repeat(center_power[None, :], sequence_sensor.n_steps, axis=0)
        center_det_trace = np.stack(
            [extract_center_pixel_values(current_det_trace[step_idx]) for step_idx in range(sequence_sensor.n_steps)],
            axis=0,
        )
        center_iout_trace = np.stack(
            [extract_center_pixel_values(current_out_trace[step_idx]) for step_idx in range(sequence_sensor.n_steps)],
            axis=0,
        )
        center_noise_trace = center_iout_trace - center_det_trace
        local_time_ms = (
            (np.arange(sequence_sensor.n_steps, dtype=np.float64) + 1.0)
            * sequence_sensor.dt
            * 1e3
        )
        unit_interval_nonideal = base_pipeline.scaled_frame_to_unit_interval(
            nonideal_scaled,
            args.readout,
            pipeline.effective_frame_range_mode(args),
        )

        samples.append(
            {
                "dataset_index": dataset_index,
                "label_index": int(label),
                "label_name": base_dataset.classes[label],
                "input_image": np.asarray(image).astype(np.float32) / 255.0,
                "power_map": np.asarray(power_maps, dtype=np.float64),
                "center_power": center_power,
                "center_power_trace": center_power_trace,
                "center_det_trace": center_det_trace,
                "center_iout_trace": center_iout_trace,
                "center_noise_trace": center_noise_trace,
                "local_time_ms": local_time_ms,
                "readout_frame": np.asarray(readout_frame, dtype=np.float64),
                "nonideal_unit_interval": np.asarray(unit_interval_nonideal, dtype=np.float32),
                "channel_labels": channel_labels_for(power_maps),
            }
        )

    return samples, case_range_bounds


def build_continuous_trace(samples, value_key, frame_duration_ms):
    all_times = []
    all_values = []
    for sample_idx, sample in enumerate(samples):
        offset_ms = sample_idx * frame_duration_ms
        all_times.append(offset_ms + sample["local_time_ms"])
        all_values.append(np.asarray(sample[value_key], dtype=np.float64))
    return np.concatenate(all_times, axis=0), np.concatenate(all_values, axis=0)


def _padded_limits(values, lower_pad_ratio=0.05, upper_pad_ratio=0.12, min_span=1e-12):
    values = np.asarray(values, dtype=np.float64)
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = max(vmax - vmin, min_span)
    return vmin - lower_pad_ratio * span, vmax + upper_pad_ratio * span


def annotate_frame_regions(ax, samples, frame_duration_ms):
    ymax = ax.get_ylim()[1]
    ymin = ax.get_ylim()[0]
    y_text = ymax - 0.08 * (ymax - ymin)
    for sample_idx, sample in enumerate(samples):
        x_left = sample_idx * frame_duration_ms
        x_mid = x_left + 0.5 * frame_duration_ms
        if sample_idx > 0:
            ax.axvline(x_left, color="#888888", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.text(
            x_mid,
            y_text,
            f"{sample_idx + 1}:{sample['label_name']}",
            ha="center",
            va="top",
            fontsize=FONT_SIZE_TICK,
            fontweight="bold",
            color="#333333",
        )


def style_plot_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#222222")
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE_TICK, width=SPINE_WIDTH, length=6)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def style_image_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#222222")


def render_montage(samples, args, case_range_bounds, output_path, param_source_summary, plot_noise_scale=1.0):
    n_cols = len(samples)
    frame_duration_ms = float(video_backend.frame_duration(args) * 1e3)
    center_xy = center_xy_for(samples[0]["power_map"])
    channel_labels = samples[0]["channel_labels"]
    analog_mode = video_backend.resolve_analog_readout_mode(args)
    colors = ["tab:red", "tab:green", "tab:blue", "tab:purple"]
    plot_noise_scale = float(plot_noise_scale)

    power_time_ms, power_trace = build_continuous_trace(samples, "center_power_trace", frame_duration_ms)
    iout_time_ms, det_trace = build_continuous_trace(samples, "center_det_trace", frame_duration_ms)
    _noise_time_ms, noise_trace = build_continuous_trace(samples, "center_noise_trace", frame_duration_ms)
    visual_iout_trace = det_trace + plot_noise_scale * noise_trace

    power_trace_uw = power_trace * 1e6
    det_trace_ma = det_trace * 1e3
    visual_iout_ma = visual_iout_trace * 1e3

    power_min, power_max = _padded_limits(power_trace_uw, lower_pad_ratio=0.03, upper_pad_ratio=0.12)
    iout_min, iout_max = _padded_limits(
        np.concatenate([det_trace_ma.reshape(-1), visual_iout_ma.reshape(-1)], axis=0),
        lower_pad_ratio=0.05,
        upper_pad_ratio=0.15,
    )

    fig = plt.figure(figsize=(4.6 * n_cols, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(4, n_cols, height_ratios=[1.0, 0.625, 0.7, 1.0])

    top_axes = [fig.add_subplot(grid[0, col]) for col in range(n_cols)]
    power_ax = fig.add_subplot(grid[1, :])
    iout_ax = fig.add_subplot(grid[2, :])
    bottom_axes = [fig.add_subplot(grid[3, col]) for col in range(n_cols)]

    for col, sample in enumerate(samples):
        original_display = to_display_image(sample["input_image"])
        nonideal_display = to_display_image(sample["nonideal_unit_interval"])
        psnr_db = compute_unit_interval_psnr(original_display, nonideal_display)

        ax = top_axes[col]
        ax.imshow(original_display)
        ax.set_xticks([])
        ax.set_yticks([])
        style_image_axis(ax)
        ax.text(
            0.5,
            1.04,
            sample["label_name"],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_IMAGE_TEXT,
            fontweight="bold",
            color="#111111",
        )

        ax = bottom_axes[col]
        ax.imshow(nonideal_display)
        ax.set_xticks([])
        ax.set_yticks([])
        style_image_axis(ax)
        psnr_text = "PSNR=inf dB" if np.isinf(psnr_db) else f"PSNR={psnr_db:.2f} dB"
        ax.text(
            0.5,
            1.04,
            psnr_text,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_IMAGE_TEXT - 1,
            fontweight="bold",
            color="#111111",
        )

    for channel_idx, channel_label in enumerate(channel_labels):
        color = colors[channel_idx % len(colors)]
        power_ax.step(
            power_time_ms,
            power_trace_uw[:, channel_idx],
            where="post",
            linewidth=TRACE_LINE_WIDTH,
            color=color,
            label=channel_label,
        )
        iout_ax.plot(
            iout_time_ms,
            det_trace_ma[:, channel_idx],
            linestyle="--",
            linewidth=TRACE_BASELINE_WIDTH,
            color=color,
            alpha=0.68,
        )
        iout_ax.plot(
            iout_time_ms,
            visual_iout_ma[:, channel_idx],
            linewidth=TRACE_LINE_WIDTH - 0.2,
            color=color,
            label=channel_label,
        )

    power_ax.set_xlim(0.0, n_cols * frame_duration_ms)
    power_ax.set_ylim(power_min, power_max)
    power_ax.grid(True, alpha=0.28)
    power_ax.set_ylabel("Center Pixel Popt (uW)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    power_ax.set_xlabel("Continuous Time (ms)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    power_ax.legend(loc="upper right", ncol=len(channel_labels), frameon=False, fontsize=FONT_SIZE_LEGEND)
    style_plot_axis(power_ax)
    annotate_frame_regions(power_ax, samples, frame_duration_ms)

    iout_ax.set_xlim(0.0, n_cols * frame_duration_ms)
    iout_ax.set_ylim(iout_min, iout_max)
    iout_ax.grid(True, alpha=0.28)
    iout_ax.set_ylabel("Center Pixel Iout (mA)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    iout_ax.set_xlabel("Continuous Time (ms)", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    style_plot_axis(iout_ax)
    annotate_frame_regions(iout_ax, samples, frame_duration_ms)

    top_axes[0].set_ylabel("Dataset Image", fontsize=FONT_SIZE_LABEL, fontweight="bold")
    bottom_axes[0].set_ylabel("Non-Ideal Image", fontsize=FONT_SIZE_LABEL, fontweight="bold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "output_path": str(output_path),
        "source_dataset": args.source_dataset,
        "split": args.split,
        "start_index": args.start_index,
        "num_images": len(samples),
        "readout": args.readout,
        "analog_readout": analog_mode,
        "video_fps": args.video_fps,
        "fps_sim": args.fps_sim,
        "frame_duration_ms": frame_duration_ms,
        "normalization_mode": args.normalization_mode,
        "range_scope": args.range_scope,
        "range_calibration_split": args.range_calibration_split,
        "range_calibration_samples": args.range_calibration_samples,
        "center_xy": list(center_xy),
        "channel_labels": channel_labels,
        "param_source": param_source_summary,
        "case_range_bounds": case_range_bounds,
        "continuous_trace": {
            "duration_ms": float(n_cols * frame_duration_ms),
            "num_steps": int(len(power_time_ms)),
            "power_min_uw": power_min,
            "power_max_uw": power_max,
            "iout_min_ma": iout_min,
            "iout_max_ma": iout_max,
            "plot_noise_scale": float(plot_noise_scale),
        },
        "samples": [
            {
                "dataset_index": sample["dataset_index"],
                "label_index": sample["label_index"],
                "label_name": sample["label_name"],
                "center_power_w": [float(value) for value in sample["center_power"]],
                "center_det_min_a": float(np.min(sample["center_det_trace"])),
                "center_det_max_a": float(np.max(sample["center_det_trace"])),
                "center_noise_min_a": float(np.min(sample["center_noise_trace"])),
                "center_noise_max_a": float(np.max(sample["center_noise_trace"])),
                "center_iout_min_a": float(np.min(sample["center_iout_trace"])),
                "center_iout_max_a": float(np.max(sample["center_iout_trace"])),
                "readout_min": float(np.min(sample["readout_frame"])),
                "readout_max": float(np.max(sample["readout_frame"])),
            }
            for sample in samples
        ],
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def main():
    cli_args = parse_args()
    output_path = resolve_output_path(cli_args)
    args = build_runtime_args(cli_args, output_path)
    param_source = resolve_param_source(cli_args, output_path)
    args.params_csv = str(param_source["params_csv"])
    args = pipeline.resolve_runtime_defaults(args)

    pipeline.seed_everything(args.seed)
    base_params = pipeline.resolve_sequence_base_params(args)
    if getattr(args, "tia_gain_ohm", None) is None:
        args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if getattr(args, "integration_gain_v_per_c", None) is None:
        args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))

    print(f"Parameter source: {param_source['summary']['param_source']}", flush=True)
    print(f"Using params csv: {args.params_csv}", flush=True)
    print(
        f"Dataset: {args.source_dataset} {args.split}, start={args.start_index}, num_images={args.num_images}",
        flush=True,
    )
    print(
        f"Simulation: video_fps={args.video_fps:.1f}, fps_sim={args.fps_sim:.1f}, "
        f"readout={args.readout}, shot_noise={bool(args.shot_noise)}, use_noise_fn={bool(args.use_noise_fn)}",
        flush=True,
    )
    if param_source["summary"]["param_source"] == "fig3_updated_case1":
        print(
            "Updated fig3 case1 params: "
            f"alpha={param_source['summary']['alpha']:.3e}, "
            f"beta={param_source['summary']['beta']:.3e}, "
            f"delta={param_source['summary']['delta_a']:.3e} A, "
            f"delta_scale={param_source['summary']['delta_scale']:.2f}, "
            f"noise_scale={param_source['summary']['noise_scale_vs_case1']:.2f}",
            flush=True,
        )

    samples, case_range_bounds = collect_samples(
        args,
        base_params,
        exact_noise_fn=param_source["exact_noise_fn"],
    )
    summary_path = render_montage(
        samples,
        args,
        case_range_bounds,
        output_path,
        param_source["summary"],
        plot_noise_scale=cli_args.plot_noise_scale,
    )
    print(f"Saved montage to {output_path}", flush=True)
    print(f"Saved summary to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
