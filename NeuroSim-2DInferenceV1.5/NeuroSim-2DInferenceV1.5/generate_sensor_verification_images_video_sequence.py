import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
THIS_DIR = Path(__file__).resolve().parent
MPLCONFIGDIR = THIS_DIR / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import generate_sensor_verification_images as base_pipeline  # noqa: E402
from sensor_video_sequence_backend import (  # noqa: E402
    StatefulNonidealVideoSensor,
    adc_enabled,
    frame_duration as _frame_duration,
    ideal_readout_from_power,
    nonideal_readout_from_power,
    resolve_analog_readout_mode,
    responsivity_total_from_params,
    simulate_ideal_video_frame,
    steps_per_frame as _steps_per_frame,
)

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
})


DEFAULT_MODEL_PATHS = {
    "cifar10": THIS_DIR / "models" / "resnet18_cifar10.pth",
    "cifar100": THIS_DIR / "models" / "resnet18_cifar100.pth",
}
DEFAULT_OUTPUT_DIRS = {
    "cifar10": THIS_DIR / "verification_images_video_sequence",
    "cifar100": THIS_DIR / "verification_images_video_sequence_cifar100",
}
DEFAULT_RESULTS_JSONS = {
    "cifar10": THIS_DIR / "artifacts" / "eval_runs" / "video_sequence_eval.json",
    "cifar100": THIS_DIR / "artifacts" / "eval_runs" / "video_sequence_eval_cifar100.json",
}
EVAL_CASES = ("raw", "ideal", "nonideal")
RANGE_CASES = ("ideal", "nonideal")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Stateful video-sequence sensor pipeline: stitch dataset samples into a video "
            "stream at a given frame rate, run pixel-wise temporal sensor simulation, "
            "export input/ideal/nonideal images, compute PSNR, and evaluate accuracy."
        )
    )
    parser.add_argument("--data-root", default=str(base_pipeline.REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--generate-images", type=int, default=1)
    parser.add_argument("--run-eval", type=int, default=1)
    parser.add_argument("--eval-cases", nargs="+", default=["nonideal"], choices=EVAL_CASES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Model checkpoint path. Defaults to models/resnet18_<source_dataset>.pth",
    )
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument(
        "--results-json",
        default=None,
        help="Optional path to save JSON results. Defaults to an artifacts/eval_runs path for the selected dataset.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument(
        "--post-norm",
        default="auto",
        choices=["none", "auto", "cifar10", "cifar100", "imagenet"],
    )
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument("--analog-readout", default=None, choices=["tia", "integration"])
    parser.add_argument("--adc-enabled", type=int, default=0)
    parser.add_argument(
        "--power-max",
        type=float,
        default=base_pipeline.DEFAULT_POWER_MAX_W,
        help=(
            "Deprecated legacy total-power upper bound. The updated video flow uses fixed "
            "Prange1~Prange2 power-density mapping and keeps this argument only for compatibility."
        ),
    )
    parser.add_argument("--params-csv", default=str(base_pipeline.DEFAULT_PARAMS_CSV))
    parser.add_argument("--normalization-mode", default="physical", choices=["physical", "calibration", "per_frame", "none"])
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
    parser.add_argument("--noise-1f-density-1hz", type=float, default=None)
    parser.add_argument("--aging-tau-hours", type=float, default=None)
    parser.add_argument("--r-degradation-pct", type=float, default=None)
    parser.add_argument("--spatial-variation-r-pct", type=float, default=None)
    parser.add_argument("--tia-gain-ohm", type=float, default=None)
    parser.add_argument("--integration-gain-v-per-c", type=float, default=None)
    parser.add_argument(
        "--video-fps",
        type=float,
        default=50.0,
        help="Dataset samples are stitched as video frames at this frame rate.",
    )
    parser.add_argument(
        "--fps-sim",
        type=float,
        default=1000.0,
        help="Internal state-update rate used inside each video frame.",
    )
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--adc-calibration-low", type=float, default=None)
    parser.add_argument("--adc-calibration-high", type=float, default=None)
    parser.add_argument("--range-mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument(
        "--range-scope",
        default="calibration",
        choices=["per_frame", "calibration"],
        help="Whether range scaling is computed per frame or fixed from a calibration set.",
    )
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument(
        "--range-calibration-split",
        default="train",
        choices=["train", "test"],
        help="Dataset split used to estimate fixed low/high range bounds.",
    )
    parser.add_argument(
        "--range-calibration-samples",
        type=int,
        default=1024,
        help="Number of calibration samples used to estimate fixed low/high bounds; 0 uses the whole split.",
    )
    parser.add_argument(
        "--i-thermal",
        type=float,
        default=0.0,
        help="Thermal-noise density for optional fast stepwise shot/thermal noise.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=5000.0,
        help="Bandwidth for optional fast steise shot/thermal noise.",
    )
    parser.add_argument(
        "--shot-noise",
        type=int,
        default=1,
        help="Enable the fixed Pmax-based shot-noise component inside the precomputed temporal noise trace.",
    )
    parser.add_argument(
        "--use-noise-fn",
        type=int,
        default=1,
        help="Enable the precomputed shot+1/f temporal noise trace during stateful video simulation.",
    )
    parser.add_argument(
        "--temporal-noise-mode",
        default="pixel_buffered",
        choices=["pixel_buffered", "pixel_repeated_window", "global_full_sequence", "global_repeated_window"],
        help=(
            "Temporal-noise synthesis mode. pixel_repeated_window uses independent "
            "per-pixel reusable traces; global modes share each trace spatially per channel."
        ),
    )
    parser.add_argument(
        "--temporal-noise-window-frames",
        type=int,
        default=10,
        help="Number of frames in the reusable temporal-noise window for repeated-window modes.",
    )
    parser.add_argument(
        "--startup-dark-frames",
        type=int,
        default=0,
        help="Optional number of dark video frames before sample 0 to precondition the state.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save generated images. Defaults to a dataset-specific verification_images_video_sequence folder.",
    )
    parser.add_argument(
        "--analyze-center-pixel",
        type=int,
        default=1,
        help="Save center-pixel ideal/nonideal waveform analysis across the stitched video sequence.",
    )
    parser.add_argument(
        "--drift-hours",
        nargs="+",
        type=float,
        default=[0.0],
        help="Long-term drift aging times in hours. Example: --drift-hours 0 100 500",
    )
    parser.add_argument(
        "--drift-aging-power-w",
        type=float,
        default=None,
        help=(
            "Deprecated legacy argument from the old additive drift model. "
            "The current long-term drift model uses power-independent response attenuation."
        ),
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_num_classes(args):
    if args.num_classes > 0:
        return args.num_classes
    return 10 if args.source_dataset == "cifar10" else 100


def resolve_default_model_path(source_dataset):
    return DEFAULT_MODEL_PATHS[source_dataset]


def resolve_default_output_dir(source_dataset):
    return DEFAULT_OUTPUT_DIRS[source_dataset]


def resolve_default_results_json(source_dataset):
    return DEFAULT_RESULTS_JSONS[source_dataset]


def _is_auto_path(value):
    return value in {None, "", "auto"}


def resolve_runtime_defaults(args):
    source_dataset = args.source_dataset.lower()
    if _is_auto_path(args.model_path):
        args.model_path = str(resolve_default_model_path(source_dataset))
    if _is_auto_path(args.output_dir):
        args.output_dir = str(resolve_default_output_dir(source_dataset))
    if _is_auto_path(args.results_json):
        args.results_json = str(resolve_default_results_json(source_dataset))
    if getattr(args, "analog_readout", None) is None:
        args.analog_readout = resolve_analog_readout_mode(args)
    args.frame_range_mode_override = effective_frame_range_mode(args)
    return args


def effective_frame_range_mode(args):
    normalization_mode = getattr(args, "normalization_mode", "physical")
    if normalization_mode == "physical":
        return "minmax"
    if normalization_mode == "none":
        return "none"
    return getattr(args, "range_mode", "auto")


def _single_carrier_requested(args, base_params):
    if "R_single" in base_params:
        return True
    if bool(getattr(args, "force_single_carrier", 0)):
        return True
    return any(
        getattr(args, name) is not None
        for name in ["single_r", "single_eta", "single_trise", "single_tfall"]
    )


def resolve_sequence_base_params(args):
    base_params = base_pipeline.resolve_base_params(args.params_csv)
    if _single_carrier_requested(args, base_params) and "R_single" not in base_params:
        base_params = {
            **base_pipeline.PARAMS_SINGLE_CARRIER,
            **{
                key: value
                for key, value in base_params.items()
                if key not in base_pipeline.PARAMS_TRUE
            },
        }

    overrides = {}
    if args.device_area_cm2 is not None:
        overrides["device_area_cm2"] = args.device_area_cm2
    if args.prange1_density is not None:
        overrides["prange1_density_w_cm2"] = args.prange1_density
    if args.prange2_density is not None:
        overrides["prange2_density_w_cm2"] = args.prange2_density
    if args.pmin_density is not None:
        overrides["pmin_density_w_cm2"] = args.pmin_density
    if args.pmax_density is not None:
        overrides["pmax_density_w_cm2"] = args.pmax_density
    if args.single_r is not None:
        overrides["R_single"] = args.single_r
    if args.single_eta is not None:
        overrides["eta_single"] = args.single_eta
    if args.single_trise is not None:
        overrides["tau_rise_single"] = args.single_trise
    if args.single_tfall is not None:
        overrides["tau_fall_single"] = args.single_tfall
    if args.trap_saturation_time is not None:
        overrides["trap_saturation_time_s"] = args.trap_saturation_time
    if args.trap_amplitude_pct is not None:
        overrides["trap_amplitude_pct"] = args.trap_amplitude_pct
    if args.noise_1f_density_1hz is not None:
        overrides["noise_1f_density_1hz_a_root_hz"] = args.noise_1f_density_1hz
    if args.aging_tau_hours is not None:
        overrides["aging_tau_hours"] = args.aging_tau_hours
    if args.r_degradation_pct is not None:
        overrides["r_degradation_pct"] = args.r_degradation_pct
    if args.spatial_variation_r_pct is not None:
        overrides["spatial_variation_r_pct"] = args.spatial_variation_r_pct
    if args.tia_gain_ohm is not None:
        overrides["tia_gain_ohm"] = args.tia_gain_ohm
    if args.integration_gain_v_per_c is not None:
        overrides["integration_gain_v_per_c"] = args.integration_gain_v_per_c
    return base_pipeline.apply_sensor_param_overrides(base_params, overrides)


def build_sequence_power_maps(image, args, base_params):
    return base_pipeline.build_power_maps(
        image,
        args.array_size,
        base_pipeline._resolve_output_channels(image, args.output_channels),
        base_params["prange2_w"],
        power_floor_w=base_params["prange1_w"],
    )


def compute_physical_case_range_bounds(args, base_params, cases):
    requested_cases = [case for case in RANGE_CASES if case in set(cases)]
    bounds = {}
    for case_name in requested_cases:
        if case_name == "ideal":
            low = float(np.min(ideal_readout_from_power(base_params["prange1_w"], args, base_params)))
            high = float(np.max(ideal_readout_from_power(base_params["prange2_w"], args, base_params)))
            power_low_density = float(base_params["prange1_density_w_cm2"])
            power_high_density = float(base_params["prange2_density_w_cm2"])
        else:
            low = float(np.min(nonideal_readout_from_power(base_params["pmin_w"], args, base_params)))
            high = float(np.max(nonideal_readout_from_power(base_params["pmax_w"], args, base_params)))
            power_low_density = float(base_params["pmin_density_w_cm2"])
            power_high_density = float(base_params["pmax_density_w_cm2"])

        if high - low <= 1e-12:
            high = low + 1e-12
        bounds[case_name] = {
            "low": low,
            "high": high,
            "mode": "physical_power_window",
            "power_low_density_w_cm2": power_low_density,
            "power_high_density_w_cm2": power_high_density,
        }
    return bounds


def _resolve_eval_sample_limit(args, dataset_length):
    if not args.run_eval:
        return 0
    if args.max_eval_batches > 0:
        return min(dataset_length, args.max_eval_batches * args.batch_size)
    return dataset_length


def _resolve_export_indices(args, dataset_length):
    if not args.generate_images:
        return set()
    start = max(0, args.start_index)
    stop = min(dataset_length, start + max(0, args.num_images))
    return set(range(start, stop))


def _prepare_output_dirs(args):
    output_dir = Path(args.output_dir)
    input_dir = output_dir / "input"
    ideal_dir = output_dir / "sensor_ideal"
    nonideal_dir = output_dir / "sensor_nonideal"
    compare_dir = output_dir / "comparison"
    for folder in [input_dir, ideal_dir, nonideal_dir, compare_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    return output_dir, input_dir, ideal_dir, nonideal_dir, compare_dir


def _prepare_analysis_dir(args):
    analysis_dir = Path(args.output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_dir


def _extract_center_pixel_values(array):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray([arr[arr.shape[0] // 2, arr.shape[1] // 2]], dtype=np.float64)
    if arr.ndim == 3:
        return np.asarray(arr[:, arr.shape[1] // 2, arr.shape[2] // 2], dtype=np.float64)
    raise ValueError(f"Unsupported array rank for center-pixel extraction: {arr.ndim}")


def _center_pixel_coordinate(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        return [int(arr.shape[0] // 2), int(arr.shape[1] // 2)]
    if arr.ndim == 3:
        return [int(arr.shape[1] // 2), int(arr.shape[2] // 2)]
    raise ValueError(f"Unsupported array rank for center-pixel coordinate: {arr.ndim}")


def _channel_labels_from_array(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        return ["mono"]
    if arr.ndim == 3:
        return [f"ch{i}" for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported array rank for channel labels: {arr.ndim}")


def export_center_pixel_analysis(analysis_dir, analysis_payload, readout_mode):
    png_path = analysis_dir / "center_pixel_waveform.png"
    npz_path = analysis_dir / "center_pixel_waveform.npz"
    json_path = analysis_dir / "center_pixel_waveform_summary.json"

    step_time_s = np.asarray(analysis_payload["step_time_s"], dtype=np.float64)
    frame_time_s = np.asarray(analysis_payload["frame_time_s"], dtype=np.float64)
    ideal_step_trace = np.asarray(analysis_payload["ideal_step_trace"], dtype=np.float64)
    nonideal_step_trace = np.asarray(analysis_payload["nonideal_step_trace"], dtype=np.float64)
    ideal_frame_trace = np.asarray(analysis_payload["ideal_frame_trace"], dtype=np.float64)
    nonideal_frame_trace = np.asarray(analysis_payload["nonideal_frame_trace"], dtype=np.float64)
    power_center_trace = np.asarray(analysis_payload["power_center_trace"], dtype=np.float64)
    dataset_indices = np.asarray(analysis_payload["dataset_indices"], dtype=np.int64)
    channel_labels = list(analysis_payload["channel_labels"])

    np.savez_compressed(
        npz_path,
        step_time_s=step_time_s,
        frame_time_s=frame_time_s,
        ideal_step_trace=ideal_step_trace,
        nonideal_step_trace=nonideal_step_trace,
        ideal_frame_trace=ideal_frame_trace,
        nonideal_frame_trace=nonideal_frame_trace,
        power_center_trace=power_center_trace,
        dataset_indices=dataset_indices,
        center_xy=np.asarray(analysis_payload["center_xy"], dtype=np.int64),
        channel_labels=np.asarray(channel_labels, dtype="<U16"),
    )

    if len(frame_time_s) > 0 and len(channel_labels) > 0:
        steps_per_frame = max(1, int(round(len(step_time_s) / len(frame_time_s))))
        power_step_trace = np.repeat(power_center_trace, steps_per_frame, axis=0)
        if len(power_step_trace) != len(step_time_s):
            min_len = min(len(power_step_trace), len(step_time_s))
            power_step_trace = power_step_trace[:min_len]
            step_time_s = step_time_s[:min_len]
            ideal_step_trace = ideal_step_trace[:min_len]
            nonideal_step_trace = nonideal_step_trace[:min_len]
    else:
        power_step_trace = np.empty((0, len(channel_labels)), dtype=np.float64)

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]

    for channel_idx, channel_label in enumerate(channel_labels):
        color = colors[channel_idx % len(colors)]
        axes[0].plot(
            step_time_s,
            power_step_trace[:, channel_idx],
            linestyle="-",
            linewidth=1.5,
            color=color,
            alpha=0.95,
            label=f"Input power {channel_label}",
        )
        axes[1].plot(
            step_time_s,
            ideal_step_trace[:, channel_idx],
            linestyle="--",
            linewidth=1.6,
            color=color,
            alpha=0.9,
            label=f"Ideal {channel_label}",
        )
        axes[2].plot(
            step_time_s,
            nonideal_step_trace[:, channel_idx],
            linestyle="-",
            linewidth=1.3,
            color=color,
            alpha=0.95,
            label=f"Nonideal {channel_label}",
        )
        axes[3].plot(
            frame_time_s,
            ideal_frame_trace[:, channel_idx],
            linestyle="--",
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=color,
            alpha=0.9,
            label=f"Ideal {channel_label}",
        )
        axes[4].plot(
            frame_time_s,
            nonideal_frame_trace[:, channel_idx],
            linestyle="-",
            marker="o",
            markersize=3,
            linewidth=1.2,
            color=color,
            alpha=0.95,
            label=f"Nonideal {channel_label}",
        )

    axes[0].set_title(
        f"Center Pixel Input Power Used For Simulation (xy={tuple(analysis_payload['center_xy'])})"
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Power (W)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=2, fontsize=9)

    axes[1].set_title(f"Center Pixel Ideal Step-Level Waveform (xy={tuple(analysis_payload['center_xy'])})")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Current (A)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2, fontsize=9)

    axes[2].set_title(f"Center Pixel Nonideal Step-Level Waveform (xy={tuple(analysis_payload['center_xy'])})")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Current (A)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2, fontsize=9)

    axes[3].set_title(f"Center Pixel Ideal Frame Readout Trace ({readout_mode})")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Frame output")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(ncol=2, fontsize=9)

    axes[4].set_title(f"Center Pixel Nonideal Frame Readout Trace ({readout_mode})")
    axes[4].set_xlabel("Time (s)")
    axes[4].set_ylabel("Frame output")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(ncol=2, fontsize=9)

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "plot_path": str(png_path),
        "data_path": str(npz_path),
        "center_xy": list(analysis_payload["center_xy"]),
        "channel_labels": channel_labels,
        "num_frames": int(len(frame_time_s)),
        "num_steps": int(len(step_time_s)),
        "readout_mode": readout_mode,
        "frame_time_start_s": float(frame_time_s[0]) if len(frame_time_s) else None,
        "frame_time_end_s": float(frame_time_s[-1]) if len(frame_time_s) else None,
        "power_frame_min_w": float(np.min(power_center_trace)) if power_center_trace.size else None,
        "power_frame_max_w": float(np.max(power_center_trace)) if power_center_trace.size else None,
        "ideal_frame_min": float(np.min(ideal_frame_trace)) if ideal_frame_trace.size else None,
        "ideal_frame_max": float(np.max(ideal_frame_trace)) if ideal_frame_trace.size else None,
        "nonideal_frame_min": float(np.min(nonideal_frame_trace)) if nonideal_frame_trace.size else None,
        "nonideal_frame_max": float(np.max(nonideal_frame_trace)) if nonideal_frame_trace.size else None,
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary_path"] = str(json_path)
    return summary


class EvalAccumulator:
    def __init__(self, model, device, cases):
        self.model = model
        self.device = device
        self.cases = list(cases)
        self.criterion = nn.CrossEntropyLoss()
        self.pending_labels = []
        self.pending_tensors = {case: [] for case in self.cases}
        self.results = {
            "device": str(device),
            "model_path": None,
            "cases": {
                case: {
                    "loss": 0.0,
                    "correct": 0,
                    "samples": 0,
                }
                for case in self.cases
            },
        }
        self.start_time = time.time()

    def set_model_path(self, model_path):
        self.results["model_path"] = model_path

    def add(self, case_tensors, label):
        self.pending_labels.append(int(label))
        for case in self.cases:
            self.pending_tensors[case].append(case_tensors[case])

    def has_pending(self):
        return bool(self.pending_labels)

    def pending_size(self):
        return len(self.pending_labels)

    def flush(self):
        if not self.pending_labels:
            return

        labels = torch.tensor(self.pending_labels, dtype=torch.long, device=self.device)
        with torch.no_grad():
            self.model.eval()
            for case in self.cases:
                images = torch.stack(self.pending_tensors[case], dim=0).to(self.device, non_blocking=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                preds = logits.argmax(dim=1)
                batch_size = labels.size(0)

                self.results["cases"][case]["loss"] += float(loss.item()) * batch_size
                self.results["cases"][case]["correct"] += int(preds.eq(labels).sum().item())
                self.results["cases"][case]["samples"] += int(batch_size)

        self.pending_labels.clear()
        for case in self.cases:
            self.pending_tensors[case].clear()

    def finalize(self):
        self.flush()
        elapsed = time.time() - self.start_time
        for case in self.cases:
            total_samples = max(self.results["cases"][case]["samples"], 1)
            total_loss = self.results["cases"][case].pop("loss")
            total_correct = self.results["cases"][case].pop("correct")
            self.results["cases"][case]["loss"] = total_loss / total_samples
            self.results["cases"][case]["accuracy"] = 100.0 * total_correct / total_samples
            self.results["cases"][case]["elapsed_sec"] = elapsed
        return self.results


def uses_calibrated_range(args):
    return (
        getattr(args, "normalization_mode", "physical") == "calibration"
        and getattr(args, "range_scope", "per_frame") == "calibration"
        and effective_frame_range_mode(args) != "none"
    )


def resolve_range_calibration_sample_limit(args, dataset_length):
    requested = int(getattr(args, "range_calibration_samples", 0))
    if requested > 0:
        return min(dataset_length, requested)
    return dataset_length


def scale_case_frame(frame, case_name, args, case_range_bounds=None):
    bounds = None
    if case_range_bounds is not None:
        bounds = case_range_bounds.get(case_name)
    return base_pipeline.scale_frame(
        frame,
        args.readout,
        effective_frame_range_mode(args),
        args.percentile_low,
        args.percentile_high,
        bounds=bounds,
    )


def compute_case_range_bounds(args, calibration_dataset, base_params, cases):
    if getattr(args, "normalization_mode", "physical") == "physical":
        return compute_physical_case_range_bounds(args, base_params, cases)
    requested_cases = [case for case in RANGE_CASES if case in set(cases)]
    if not requested_cases or not uses_calibrated_range(args):
        return {}
    if len(calibration_dataset) == 0:
        raise ValueError("Calibration dataset is empty; cannot estimate fixed range bounds.")

    first_image, _ = calibration_dataset[0]
    output_channels = base_pipeline._resolve_output_channels(first_image, args.output_channels)
    skip_frames = max(0, int(getattr(args, "range_calibration_skip_frames", 0)))
    if len(calibration_dataset) <= skip_frames:
        raise ValueError(
            f"Calibration dataset has {len(calibration_dataset)} samples, "
            f"cannot skip the first {skip_frames} frames."
        )
    requested_samples = int(getattr(args, "range_calibration_samples", 0))
    if skip_frames > 0 and requested_samples > 0:
        sample_limit = min(len(calibration_dataset), skip_frames + requested_samples)
    else:
        sample_limit = resolve_range_calibration_sample_limit(args, len(calibration_dataset))
    collected = {case: [] for case in requested_cases}

    sequence_sensor = None
    if "nonideal" in requested_cases:
        sequence_sensor = StatefulNonidealVideoSensor(args=args, base_params=base_params)
        for _ in range(max(0, args.startup_dark_frames)):
            zero_power = np.zeros((output_channels, args.array_size, args.array_size), dtype=np.float64)
            if output_channels == 1:
                zero_power = zero_power[0]
            sequence_sensor.simulate_frame(zero_power)

    effective_mode = base_pipeline.resolve_effective_range_mode(args.readout, effective_frame_range_mode(args))
    if skip_frames > 0:
        print(
            f"Estimating fixed {effective_mode} range bounds from frames "
            f"{skip_frames}..{sample_limit - 1} "
            f"(skip={skip_frames}, split={args.range_calibration_split}, cases={requested_cases})",
            flush=True,
        )
    else:
        print(
            f"Estimating fixed {effective_mode} range bounds from {sample_limit} calibration samples "
            f"(split={args.range_calibration_split}, cases={requested_cases})",
            flush=True,
        )

    for dataset_index in range(sample_limit):
        image, _ = calibration_dataset[dataset_index]
        power_maps = build_sequence_power_maps(image, args, base_params)

        include_frame = dataset_index >= skip_frames

        if "ideal" in requested_cases:
            ideal_raw = simulate_ideal_video_frame(power_maps, args, base_params)
            if include_frame:
                collected["ideal"].append(np.asarray(ideal_raw, dtype=np.float32).reshape(-1))

        if "nonideal" in requested_cases:
            if sequence_sensor is None:
                raise RuntimeError("Nonideal calibration requested without a sequence sensor.")
            nonideal_raw = sequence_sensor.simulate_frame(power_maps)
            if include_frame:
                collected["nonideal"].append(np.asarray(nonideal_raw, dtype=np.float32).reshape(-1))

        processed = dataset_index + 1
        if processed % 200 == 0 or processed == sample_limit:
            if skip_frames > 0:
                kept = max(0, processed - skip_frames)
                print(
                    f"  calibration progress: processed={processed}/{sample_limit}, kept={kept}",
                    flush=True,
                )
            else:
                print(f"  calibration progress: samples={processed}/{sample_limit}", flush=True)

    bounds = {}
    for case_name, arrays in collected.items():
        if not arrays:
            raise ValueError(f"No calibration frames collected for case {case_name!r}.")
        values = np.concatenate(arrays, axis=0)
        low = float(np.percentile(values, args.percentile_low))
        high = float(np.percentile(values, args.percentile_high))
        if high - low <= 1e-12:
            high = low + 1e-12
        bounds[case_name] = {
            "low": low,
            "high": high,
            "mode": effective_mode,
            "num_samples": int(len(arrays)),
            "num_skipped_frames": int(skip_frames),
            "num_values": int(values.size),
        }
        print(
            f"  fixed range {case_name}: low={low:.6e} high={high:.6e} "
            f"span={high - low:.6e}",
            flush=True,
        )

    return bounds


def resolve_drift_hours(args):
    raw_values = getattr(args, "drift_hours", None)
    if not raw_values:
        return [0.0]

    drift_hours = []
    for raw_value in raw_values:
        value = float(raw_value)
        if value < 0:
            raise ValueError(f"drift_hours must be non-negative, got {value}")
        if not any(abs(value - existing) <= 1e-12 for existing in drift_hours):
            drift_hours.append(value)
    return drift_hours or [0.0]


def format_drift_scenario_tag(drift_hours):
    drift_hours = float(drift_hours)
    if abs(drift_hours - round(drift_hours)) <= 1e-9:
        return f"drift_{int(round(drift_hours)):04d}h"
    return f"drift_{drift_hours:.1f}h".replace(".", "p")


def resolve_scenario_output_dir(base_output_dir, drift_hours, multi_scenario):
    output_dir = Path(base_output_dir)
    if not multi_scenario:
        return output_dir
    return output_dir / format_drift_scenario_tag(drift_hours)


def _load_calibration_dataset(args, base_dataset):
    calibration_split = getattr(args, "range_calibration_split", args.split)
    if calibration_split == args.split:
        return base_dataset
    return base_pipeline.load_base_dataset(
        args.source_dataset,
        args.data_root,
        calibration_split,
    )


def compute_drift_aging_power_reference(args, calibration_dataset, base_params):
    del calibration_dataset
    del args
    return None, {
        "source": "power_independent_response_attenuation",
        "power_dependent": False,
        "gamma": float(base_params["gamma"]),
        "tau_drift": float(base_params["tau_drift"]),
        "drift_scale": float(base_params["drift_scale"]),
    }


def _run_single_sequence_pipeline(args):
    if not args.generate_images and not args.run_eval:
        raise ValueError("At least one of --generate-images or --run-eval must be enabled.")

    args = resolve_runtime_defaults(args)
    if bool(getattr(args, "suppress_results_json", False)):
        args.results_json = None
    seed_everything(args.seed)
    base_dataset = base_pipeline.load_base_dataset(args.source_dataset, args.data_root, args.split)
    base_params = resolve_sequence_base_params(args)
    if getattr(args, "tia_gain_ohm", None) is None:
        args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if getattr(args, "integration_gain_v_per_c", None) is None:
        args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    case_range_bounds = dict(getattr(args, "precomputed_case_range_bounds", {}))

    if getattr(args, "normalization_mode", "physical") == "physical" and not case_range_bounds:
        case_range_bounds = compute_case_range_bounds(
            args=args,
            calibration_dataset=None,
            base_params=base_params,
            cases=RANGE_CASES,
        )
    elif uses_calibrated_range(args) and not case_range_bounds:
        if args.range_calibration_split == args.split:
            calibration_dataset = base_dataset
        else:
            calibration_dataset = base_pipeline.load_base_dataset(
                args.source_dataset,
                args.data_root,
                args.range_calibration_split,
            )
        case_range_bounds = compute_case_range_bounds(
            args=args,
            calibration_dataset=calibration_dataset,
            base_params=base_params,
            cases=RANGE_CASES,
        )

    dataset_length = len(base_dataset)
    eval_limit = _resolve_eval_sample_limit(args, dataset_length)
    export_indices = _resolve_export_indices(args, dataset_length)

    if args.generate_images and args.start_index + args.num_images > dataset_length:
        available = max(0, dataset_length - args.start_index)
        print(
            f"Warning: requested {args.num_images} images from start_index={args.start_index}, "
            f"but only {available} are available.",
            flush=True,
        )

    max_needed_index = -1
    if eval_limit > 0:
        max_needed_index = max(max_needed_index, eval_limit - 1)
    if export_indices:
        max_needed_index = max(max_needed_index, max(export_indices))
    if max_needed_index < 0:
        max_needed_index = -1
    args.total_sequence_frames = max_needed_index + 1 + max(0, int(args.startup_dark_frames))

    output_dir = None
    input_dir = None
    ideal_dir = None
    nonideal_dir = None
    compare_dir = None
    analysis_dir = None
    if args.generate_images:
        output_dir, input_dir, ideal_dir, nonideal_dir, compare_dir = _prepare_output_dirs(args)
    if args.analyze_center_pixel:
        analysis_dir = _prepare_analysis_dir(args)

    model = None
    device = None
    eval_accumulator = None
    if args.run_eval or args.generate_images:
        device = base_pipeline.select_device()
    if args.run_eval:
        model = base_pipeline.load_model(args.model_path, device, infer_num_classes(args))
        eval_accumulator = EvalAccumulator(model=model, device=device, cases=args.eval_cases)
        eval_accumulator.set_model_path(args.model_path)

    scenario_tag = format_drift_scenario_tag(getattr(args, "drift_hours", 0.0))
    sequence_sensor = StatefulNonidealVideoSensor(args=args, base_params=base_params)
    drift_power_reference = getattr(args, "drift_power_reference", None)
    drift_power_summary = getattr(args, "drift_power_summary", None)
    drift_hours_value = float(getattr(args, "drift_hours", 0.0))
    args_json = dict(vars(args))
    args_json.pop("drift_power_reference", None)
    args_json.pop("drift_power_summary", None)
    args_json.pop("precomputed_case_range_bounds", None)
    args_json.pop("suppress_results_json", None)
    args_json.pop("frame_range_mode_override", None)
    args_json.pop("total_sequence_frames", None)
    if drift_hours_value > 0:
        sequence_sensor.set_long_term_drift_state(drift_hours_value, drift_power_reference)
    manifest = []
    psnr_values = []
    center_pixel_analysis = None

    sequence_output_channels = None
    if dataset_length > 0:
        first_image, _ = base_dataset[0]
        sequence_output_channels = base_pipeline._resolve_output_channels(first_image, args.output_channels)

    if sequence_output_channels is None:
        raise ValueError("Empty dataset: no samples available for sequence simulation.")

    print("\n" + "=" * 72, flush=True)
    print(f"Drift scenario: {scenario_tag} ({drift_hours_value:.1f} h)", flush=True)
    print(f"Using device: {device}", flush=True)
    print(f"Dataset length: {dataset_length}", flush=True)
    print(
        f"Video stitching: fps={args.video_fps:.4f} frame_duration={_frame_duration(args):.6f}s "
        f"n_steps_per_frame={_steps_per_frame(args)}",
        flush=True,
    )
    print(
        f"Sensor backend: stateful sample-and-hold sequence, analog_readout={resolve_analog_readout_mode(args)}, "
        f"adc_enabled={adc_enabled(args)}, adc_bits={args.adc_bits}, "
        f"adc_window={None if getattr(args, 'adc_calibration_low', None) is None or getattr(args, 'adc_calibration_high', None) is None else f'[{float(args.adc_calibration_low):.3e}, {float(args.adc_calibration_high):.3e}]'}, "
        f"noise_trace={'enabled' if bool(args.use_noise_fn) else 'disabled'}, "
        f"shot_noise={bool(args.shot_noise)}, flicker_1Hz={base_params['noise_1f_density_1hz_a_root_hz']:.3e} A/Hz^0.5",
        flush=True,
    )
    print(
        f"Optical ranges: Prange=[{base_params['prange1_density_w_cm2']:.3e}, {base_params['prange2_density_w_cm2']:.3e}] W/cm^2 "
        f"Pwindow=[{base_params['pmin_density_w_cm2']:.3e}, {base_params['pmax_density_w_cm2']:.3e}] W/cm^2 "
        f"area={base_params['device_area_cm2']:.3e} cm^2",
        flush=True,
    )
    if drift_power_summary is not None:
        if drift_power_summary.get("power_dependent", True):
            print(
                f"Drift aging power: source={drift_power_summary['source']} "
                f"min={drift_power_summary['power_min_w']:.6e} W "
                f"max={drift_power_summary['power_max_w']:.6e} W",
                flush=True,
            )
        else:
            print(
                f"Drift aging model: source={drift_power_summary['source']} "
                f"gamma={drift_power_summary['gamma']:.6e} "
                f"tau_drift={drift_power_summary['tau_drift']:.6e} s",
                flush=True,
            )

    for _ in range(max(0, args.startup_dark_frames)):
        zero_power = np.zeros((sequence_output_channels, args.array_size, args.array_size), dtype=np.float64)
        if sequence_output_channels == 1:
            zero_power = zero_power[0]
        sequence_sensor.simulate_frame(zero_power)

    if args.analyze_center_pixel:
        center_pixel_analysis = {
            "dataset_indices": [],
            "frame_time_s": [],
            "step_time_s": [],
            "ideal_step_trace": [],
            "nonideal_step_trace": [],
            "ideal_frame_trace": [],
            "nonideal_frame_trace": [],
            "power_center_trace": [],
            "center_xy": None,
            "channel_labels": None,
        }

    def build_export_item(dataset_index, sample, image, label, raw_tensor, ideal_tensor, nonideal_tensor):
        input_img = base_pipeline.to_rgb_image(np.asarray(image).astype(np.float32) / 255.0, args.tile_size)
        ideal_img = base_pipeline.to_rgb_image(
            base_pipeline.scaled_frame_to_unit_interval(
                sample["ideal_scaled"],
                args.readout,
                effective_frame_range_mode(args),
            ),
            args.tile_size,
        )
        nonideal_img = base_pipeline.to_rgb_image(
            base_pipeline.scaled_frame_to_unit_interval(
                sample["nonideal_scaled"],
                args.readout,
                effective_frame_range_mode(args),
            ),
            args.tile_size,
        )
        triptych = base_pipeline.compose_triptych(
            input_img,
            ideal_img,
            nonideal_img,
            f"idx={dataset_index} class={sample['label_name']}",
            sample["psnr_db"],
        )

        stem = f"sample_{dataset_index:04d}_{sample['label_name']}"
        input_path = input_dir / f"{stem}_input.png"
        ideal_path = ideal_dir / f"{stem}_sensor_ideal.png"
        nonideal_path = nonideal_dir / f"{stem}_sensor_nonideal.png"
        compare_path = compare_dir / f"{stem}_triptych.png"

        input_img.save(input_path)
        ideal_img.save(ideal_path)
        nonideal_img.save(nonideal_path)
        triptych.save(compare_path)

        item = {
            "dataset_index": dataset_index,
            "label_index": int(label),
            "label": sample["label_name"],
            "input_path": str(input_path),
            "ideal_path": str(ideal_path),
            "nonideal_path": str(nonideal_path),
            "compare_path": str(compare_path),
            "power_map_min": float(np.min(sample["power_maps"])),
            "power_map_max": float(np.max(sample["power_maps"])),
            "psnr_db": sample["psnr_db"],
            "ideal_raw_min": float(np.min(sample["ideal_raw"])),
            "ideal_raw_max": float(np.max(sample["ideal_raw"])),
            "nonideal_raw_min": float(np.min(sample["nonideal_raw"])),
            "nonideal_raw_max": float(np.max(sample["nonideal_raw"])),
        }

        if model is not None and device is not None:
            item["predictions"] = {
                "raw": base_pipeline.predict_tensor(model, raw_tensor, device, base_dataset.classes),
                "ideal": base_pipeline.predict_tensor(model, ideal_tensor, device, base_dataset.classes),
                "nonideal": base_pipeline.predict_tensor(model, nonideal_tensor, device, base_dataset.classes),
            }
            item["correct"] = {
                case: (pred["pred_index"] == int(label))
                for case, pred in item["predictions"].items()
            }
        return item

    start_time = time.time()
    for dataset_index in range(max_needed_index + 1):
        image, label = base_dataset[dataset_index]
        label_name = base_dataset.classes[label]
        power_maps = build_sequence_power_maps(image, args, base_params)

        ideal_raw = simulate_ideal_video_frame(power_maps, args, base_params)
        center_trace = None
        if args.analyze_center_pixel:
            nonideal_raw, center_trace = sequence_sensor.simulate_frame(power_maps, record_center_trace=True)
        else:
            nonideal_raw = sequence_sensor.simulate_frame(power_maps)
        ideal_scaled = scale_case_frame(ideal_raw, "ideal", args, case_range_bounds)
        nonideal_scaled = scale_case_frame(nonideal_raw, "nonideal", args, case_range_bounds)

        if args.analyze_center_pixel:
            if center_pixel_analysis["center_xy"] is None:
                center_pixel_analysis["center_xy"] = _center_pixel_coordinate(ideal_raw)
                center_pixel_analysis["channel_labels"] = _channel_labels_from_array(ideal_raw)

            frame_time_s = (dataset_index + 1) * _frame_duration(args)
            step_time_s = (
                dataset_index * _frame_duration(args)
                + (np.arange(sequence_sensor.n_steps, dtype=np.float64) + 1.0) * sequence_sensor.dt
            )
            ideal_center_current = (
                responsivity_total_from_params(base_params) * _extract_center_pixel_values(power_maps)
            )

            center_pixel_analysis["dataset_indices"].append(dataset_index)
            center_pixel_analysis["frame_time_s"].append(frame_time_s)
            center_pixel_analysis["ideal_step_trace"].append(
                np.repeat(ideal_center_current[None, :], sequence_sensor.n_steps, axis=0)
            )
            center_pixel_analysis["nonideal_step_trace"].append(np.asarray(center_trace, dtype=np.float64))
            center_pixel_analysis["ideal_frame_trace"].append(_extract_center_pixel_values(ideal_raw))
            center_pixel_analysis["nonideal_frame_trace"].append(_extract_center_pixel_values(nonideal_raw))
            center_pixel_analysis["power_center_trace"].append(_extract_center_pixel_values(power_maps))
            center_pixel_analysis["step_time_s"].append(step_time_s)

        raw_tensor = None
        ideal_tensor = None
        nonideal_tensor = None

        if args.run_eval or dataset_index in export_indices:
            raw_tensor = base_pipeline._raw_image_to_model_tensor(image, args)
            ideal_tensor = base_pipeline._frame_to_model_tensor(ideal_scaled, args)
            nonideal_tensor = base_pipeline._frame_to_model_tensor(nonideal_scaled, args)

        if args.run_eval and dataset_index < eval_limit:
            case_tensors = {
                "raw": raw_tensor,
                "ideal": ideal_tensor,
                "nonideal": nonideal_tensor,
            }
            eval_accumulator.add(case_tensors, label)
            if eval_accumulator.pending_size() >= args.batch_size:
                eval_accumulator.flush()

            processed = dataset_index + 1
            if processed % max(args.batch_size * 10, 10) == 0:
                elapsed = time.time() - start_time
                print(
                    f"  eval progress: samples={processed}/{eval_limit} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if dataset_index in export_indices:
            psnr_db = base_pipeline.compute_psnr(ideal_raw, nonideal_raw)
            sample = {
                "label_name": label_name,
                "power_maps": power_maps,
                "ideal_raw": ideal_raw,
                "nonideal_raw": nonideal_raw,
                "ideal_scaled": ideal_scaled,
                "nonideal_scaled": nonideal_scaled,
                "psnr_db": psnr_db,
            }
            item = build_export_item(
                dataset_index=dataset_index,
                sample=sample,
                image=image,
                label=label,
                raw_tensor=raw_tensor,
                ideal_tensor=ideal_tensor,
                nonideal_tensor=nonideal_tensor,
            )
            manifest.append(item)
            psnr_values.append(psnr_db)
            print(
                f"  export sample: idx={dataset_index} class={label_name} psnr={base_pipeline.format_psnr(psnr_db)}",
                flush=True,
            )

    evaluation = None
    if eval_accumulator is not None:
        evaluation = eval_accumulator.finalize()
        for case, case_result in evaluation["cases"].items():
            print(
                f"{case}: acc={case_result['accuracy']:.2f}% "
                f"loss={case_result['loss']:.4f} "
                f"samples={case_result['samples']} "
                f"time={case_result['elapsed_sec']:.1f}s",
                flush=True,
            )

    center_pixel_analysis_result = None
    if args.analyze_center_pixel and center_pixel_analysis["dataset_indices"]:
        analysis_payload = {
            "dataset_indices": center_pixel_analysis["dataset_indices"],
            "frame_time_s": center_pixel_analysis["frame_time_s"],
            "step_time_s": np.concatenate(center_pixel_analysis["step_time_s"], axis=0),
            "ideal_step_trace": np.concatenate(center_pixel_analysis["ideal_step_trace"], axis=0),
            "nonideal_step_trace": np.concatenate(center_pixel_analysis["nonideal_step_trace"], axis=0),
            "ideal_frame_trace": np.stack(center_pixel_analysis["ideal_frame_trace"], axis=0),
            "nonideal_frame_trace": np.stack(center_pixel_analysis["nonideal_frame_trace"], axis=0),
            "power_center_trace": np.stack(center_pixel_analysis["power_center_trace"], axis=0),
            "center_xy": center_pixel_analysis["center_xy"],
            "channel_labels": center_pixel_analysis["channel_labels"],
        }
        center_pixel_analysis_result = export_center_pixel_analysis(
            analysis_dir=analysis_dir,
            analysis_payload=analysis_payload,
            readout_mode=(
                f"{resolve_analog_readout_mode(args)}{'_adc' if adc_enabled(args) else ''}"
            ),
        )
        print(f"center_pixel_plot      {center_pixel_analysis_result['plot_path']}")
        print(f"center_pixel_data      {center_pixel_analysis_result['data_path']}")

    image_generation = None
    if args.generate_images:
        manifest_path = output_dir / "manifest.json"
        manifest_payload = {
            "args": args_json,
            "base_params": base_params,
            "simulation_backend": "photodetector_model_stateful_video_sequence",
            "video_mode": "sample_and_hold",
            "scenario_tag": scenario_tag,
            "drift_hours": drift_hours_value,
            "drift_aging_power_reference": drift_power_summary,
            "frame_duration_sec": _frame_duration(args),
            "n_steps_per_frame": _steps_per_frame(args),
            "range_scope": args.range_scope,
            "range_calibration_split": args.range_calibration_split,
            "range_calibration_samples": args.range_calibration_samples,
            "case_range_bounds": case_range_bounds,
            "nonideal_model_noise": (
                "precomputed_pmax_shot_plus_1f_trace" if bool(args.use_noise_fn) else None
            ),
            "nonideal_dark_current": sequence_sensor.dark_current,
            "center_pixel_analysis": center_pixel_analysis_result,
            "num_images_exported": len(manifest),
            "mean_psnr_db": float(np.mean(psnr_values)) if psnr_values else None,
            "samples": manifest,
        }
        manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

        print(f"output_dir            {output_dir}")
        print(f"manifest              {manifest_path}")

        image_generation = {
            "output_dir": str(output_dir),
            "manifest_path": str(manifest_path),
            "num_images": len(manifest),
            "mean_psnr_db": manifest_payload["mean_psnr_db"],
        }

    combined_results = {
        "args": args_json,
        "base_params": base_params,
        "model_path": args.model_path if args.run_eval else None,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "simulation_backend": "photodetector_model_stateful_video_sequence",
        "video_mode": "sample_and_hold",
        "scenario_tag": scenario_tag,
        "drift_hours": drift_hours_value,
        "drift_aging_seconds": drift_hours_value * 3600.0,
        "drift_aging_power_reference": drift_power_summary,
        "frame_duration_sec": _frame_duration(args),
        "n_steps_per_frame": _steps_per_frame(args),
        "analog_readout": resolve_analog_readout_mode(args),
        "adc_enabled": adc_enabled(args),
        "range_scope": args.range_scope,
        "normalization_mode": args.normalization_mode,
        "range_calibration_split": args.range_calibration_split,
        "range_calibration_samples": args.range_calibration_samples,
        "case_range_bounds": case_range_bounds,
        "center_pixel_analysis": center_pixel_analysis_result,
        "evaluation": evaluation,
        "image_generation": image_generation,
    }

    if args.results_json:
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}", flush=True)

    return combined_results


def run_sequence_pipeline(args):
    if not args.generate_images and not args.run_eval:
        raise ValueError("At least one of --generate-images or --run-eval must be enabled.")

    args = resolve_runtime_defaults(args)
    base_params = resolve_sequence_base_params(args)
    if getattr(args, "tia_gain_ohm", None) is None:
        args.tia_gain_ohm = float(base_params.get("tia_gain_ohm", 1.0))
    if getattr(args, "integration_gain_v_per_c", None) is None:
        args.integration_gain_v_per_c = float(base_params.get("integration_gain_v_per_c", 1.0))
    drift_hours_list = resolve_drift_hours(args)
    multi_scenario = len(drift_hours_list) > 1
    calibration_dataset = None

    if uses_calibrated_range(args) or any(hours > 0 for hours in drift_hours_list):
        seed_everything(args.seed)
        base_dataset = base_pipeline.load_base_dataset(args.source_dataset, args.data_root, args.split)
        calibration_dataset = _load_calibration_dataset(args, base_dataset)
    else:
        base_dataset = None

    case_range_bounds = {}
    if getattr(args, "normalization_mode", "physical") == "physical":
        case_range_bounds = compute_case_range_bounds(
            args=args,
            calibration_dataset=None,
            base_params=base_params,
            cases=RANGE_CASES,
        )
    elif uses_calibrated_range(args):
        if base_dataset is None:
            seed_everything(args.seed)
            base_dataset = base_pipeline.load_base_dataset(args.source_dataset, args.data_root, args.split)
        case_range_bounds = compute_case_range_bounds(
            args=args,
            calibration_dataset=calibration_dataset,
            base_params=base_params,
            cases=RANGE_CASES,
        )

    drift_power_reference = None
    drift_power_summary = None
    if any(hours > 0 for hours in drift_hours_list):
        drift_power_reference, drift_power_summary = compute_drift_aging_power_reference(
            args=args,
            calibration_dataset=calibration_dataset,
            base_params=base_params,
        )

    scenario_results = []
    for drift_hours in drift_hours_list:
        scenario_args = argparse.Namespace(**vars(args))
        scenario_args.results_json = None
        scenario_args.suppress_results_json = True
        scenario_args.drift_hours = float(drift_hours)
        scenario_args.drift_power_reference = drift_power_reference
        scenario_args.drift_power_summary = drift_power_summary
        scenario_args.precomputed_case_range_bounds = case_range_bounds
        if multi_scenario:
            scenario_args.output_dir = str(resolve_scenario_output_dir(args.output_dir, drift_hours, multi_scenario))
        scenario_result = _run_single_sequence_pipeline(scenario_args)
        scenario_results.append(scenario_result)

    if len(scenario_results) == 1:
        combined_results = scenario_results[0]
        combined_args = dict(vars(args))
        combined_args.pop("frame_range_mode_override", None)
        combined_args.pop("total_sequence_frames", None)
        combined_results["args"] = combined_args
    else:
        combined_results = {
            "args": vars(args),
            "model_path": args.model_path if args.run_eval else None,
            "source_dataset": args.source_dataset,
            "split": args.split,
            "simulation_backend": "photodetector_model_stateful_video_sequence",
            "video_mode": "sample_and_hold",
            "frame_duration_sec": _frame_duration(args),
            "n_steps_per_frame": _steps_per_frame(args),
            "analog_readout": resolve_analog_readout_mode(args),
            "adc_enabled": adc_enabled(args),
            "normalization_mode": args.normalization_mode,
            "range_scope": args.range_scope,
            "range_calibration_split": args.range_calibration_split,
            "range_calibration_samples": args.range_calibration_samples,
            "case_range_bounds": case_range_bounds,
            "drift_hours": drift_hours_list,
            "drift_aging_power_reference": drift_power_summary,
            "scenarios": scenario_results,
        }

    if args.results_json:
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}", flush=True)

    return combined_results


def main():
    args = parse_args()
    run_sequence_pipeline(args)


if __name__ == "__main__":
    main()
