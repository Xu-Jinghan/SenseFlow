import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import functional as TF


THIS_DIR = Path(__file__).resolve().parent


def _find_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "photodetector_array.py").exists():
            return candidate
    return start_dir.parent


REPO_ROOT = _find_repo_root(THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from photodetector_model import (  # noqa: E402
    ANALYSIS_CONFIG,
    DARK_CURRENT_MEASURED,
    DEFAULT_DEVICE_AREA_CM2,
    DEFAULT_NOISE_1F_DENSITY_1HZ_AHZ05,
    DEFAULT_PRANGE1_DENSITY_W_CM2,
    DEFAULT_PRANGE2_DENSITY_W_CM2,
    NOISE_FN,
    NONLINEAR_POWER_REF_W,
    PARAMS_SINGLE_CARRIER,
    PARAMS_TRUE,
    compute_total_power_from_density_w_cm2,
    params_to_vec,
    set_device_context,
    simulate,
)
from models import resnet  # noqa: E402


DEFAULT_PARAMS_CSV = REPO_ROOT / "outputs" / "synthetic_image_fit_v3_params.csv"
DEFAULT_MODEL_PATH = THIS_DIR / "models" / "resnet18_cifar10.pth"
DEFAULT_OUTPUT_DIR = THIS_DIR / "verification_images_unified_50samples_eval_default"
DEFAULT_RESULTS_JSON = THIS_DIR / "artifacts" / "eval_runs" / "unified_50samples_eval_default.json"
DEFAULT_POWER_MAX_W = float(ANALYSIS_CONFIG["arbitrary_pmax"])
EVAL_CASES = ("raw", "ideal", "nonideal")
IGNORED_SENSOR_ARGS = [
    "sensor_pixel_var_resp",
    "sensor_pixel_var_eta",
    "sensor_pixel_var_tau",
    "sensor_pixel_var_dark",
    "sensor_pixel_var_noise",
]
_POST_NORM_STATS = {
    "cifar10": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    "cifar100": (
        (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        (0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
    ),
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
_MISSING_DEFAULT_PARAMS_WARNED = False
_SENSOR_EXTRA_ALIASES = {
    "Prange1_density_w_cm2": "prange1_density_w_cm2",
    "Prange2_density_w_cm2": "prange2_density_w_cm2",
    "Pmin_density_w_cm2": "pmin_density_w_cm2",
    "Pmax_density_w_cm2": "pmax_density_w_cm2",
    "power_min_density_w_cm2": "pmin_density_w_cm2",
    "power_max_density_w_cm2": "pmax_density_w_cm2",
    "trap_amplitude_pct": "trap_amplitude_pct",
    "trap_amplitude_pct_ion": "trap_amplitude_pct",
    "noise_1f_1hz_a_root_hz": "noise_1f_density_1hz_a_root_hz",
    "aging_time_hours": "aging_tau_hours",
    "aging_tau_h": "aging_tau_hours",
    "r_degradation_pct": "r_degradation_pct",
    "spatial_variation_r_pct": "spatial_variation_r_pct",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Unified ResNet18 sensor pipeline: generate input/ideal/nonideal verification "
            "images, compute PSNR, and optionally evaluate classification accuracy."
        )
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--generate-images", type=int, default=1)
    parser.add_argument("--run-eval", type=int, default=1)
    parser.add_argument("--eval-cases", nargs="+", default=list(EVAL_CASES), choices=EVAL_CASES)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-eval-batches", type=int, default=20)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--results-json", default=str(DEFAULT_RESULTS_JSON))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument(
        "--post-norm",
        default="cifar10",
        choices=["none", "auto", "cifar10", "cifar100", "imagenet"],
    )
    parser.add_argument("--num-images", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--readout", default="integration", choices=["tia", "integration", "adc"])
    parser.add_argument(
        "--power-max",
        type=float,
        default=DEFAULT_POWER_MAX_W,
        help=(
            "Max optical power in W. Image intensities are mapped to "
            f"[{NONLINEAR_POWER_REF_W:.3e}, power_max]. Defaults to photodetector_model.ANALYSIS_CONFIG['arbitrary_pmax'] "
            f"= {DEFAULT_POWER_MAX_W:.3e} W."
        ),
    )
    parser.add_argument("--params-csv", default=str(DEFAULT_PARAMS_CSV))
    parser.add_argument("--exposure-time", type=float, default=1.0 / 30.0)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--adc-bits", type=int, default=8)
    parser.add_argument("--adc-full-scale", type=float, default=None)
    parser.add_argument("--range-mode", default="auto", choices=["auto", "minmax", "signed", "none"])
    parser.add_argument("--percentile-low", type=float, default=1.0)
    parser.add_argument("--percentile-high", type=float, default=99.0)
    parser.add_argument(
        "--i-thermal",
        type=float,
        default=5e-8,
        help="Thermal-noise density used by photodetector_model shot/thermal sampling.",
    )
    parser.add_argument(
        "--bandwidth",
        type=float,
        default=5000.0,
        help="Bandwidth used by photodetector_model shot/thermal sampling.",
    )
    parser.add_argument(
        "--shot-noise",
        type=int,
        default=1,
        help="Whether to add shot-noise/thermal-noise terms when enabled in the model path.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save generated images.",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_num_classes(args):
    if args.num_classes > 0:
        return args.num_classes
    return 10 if args.source_dataset == "cifar10" else 100


def resolve_sensor_seed(args):
    return args.sensor_rng_seed if args.sensor_rng_seed is not None else args.seed


def load_model(model_path, device, num_classes):
    model = resnet.resnet18(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value

    model.load_state_dict(cleaned)
    return model.to(device)


def load_base_dataset(source_dataset, data_root, split):
    data_root = os.path.expanduser(data_root)
    train = split == "train"
    if source_dataset == "cifar10":
        data_root = os.path.join(data_root, "cifar10-data")
        return datasets.CIFAR10(root=data_root, train=train, download=True, transform=None)
    if source_dataset == "cifar100":
        data_root = os.path.join(data_root, "cifar100-data")
        return datasets.CIFAR100(root=data_root, train=train, download=True, transform=None)
    raise ValueError(f"Unsupported source dataset: {source_dataset}")


def load_params_csv(csv_path):
    raw_values = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = row["parameter"].strip()
            if not key:
                continue
            try:
                raw_values[key] = float(row["value"])
            except (TypeError, ValueError):
                continue

    single_carrier_markers = {
        "R_single",
        "eta_single",
        "tau_rise_single",
        "tau_fall_single",
        "R_single_transient",
        "main_branch_ceiling_ua",
    }
    is_single_carrier = any(key in raw_values for key in single_carrier_markers)

    if is_single_carrier:
        params = dict(PARAMS_SINGLE_CARRIER)
        compatible_keys = set(PARAMS_SINGLE_CARRIER)
    else:
        params = dict(PARAMS_TRUE)
        compatible_keys = set(PARAMS_TRUE)

    found_compatible = False
    for key, value in raw_values.items():
        if key in compatible_keys:
            params[key] = value
            found_compatible = True

    for key, value in raw_values.items():
        normalized_key = _SENSOR_EXTRA_ALIASES.get(key, key)
        if normalized_key in {
            "device_area_cm2",
            "prange1_density_w_cm2",
            "prange2_density_w_cm2",
            "pmin_density_w_cm2",
            "pmax_density_w_cm2",
            "trap_saturation_time_s",
            "trap_amplitude_pct",
            "trap_delta_r_ratio",
            "trap_x1_ratio",
            "trap_x2_reference_state",
            "trap_x2_tau_on_s",
            "trap_x2_tau_off_s",
            "noise_1f_density_1hz_a_root_hz",
            "aging_tau_hours",
            "r_degradation_pct",
            "spatial_variation_r_pct",
            "tia_gain_ohm",
            "integration_gain_v_per_c",
            "rise_fast_weight",
            "fall_fast_weight",
        }:
            params[normalized_key] = float(value)

    if is_single_carrier and "R_single_transient" in raw_values:
        params["R_single"] = float(raw_values["R_single_transient"])
        found_compatible = True

    if is_single_carrier:
        params["case_label"] = "case2" if "R_single_transient" in raw_values else "single_carrier"
        if params["case_label"] == "case2":
            params["trap_mode"] = "binary"
            params["trap_output_mode"] = "illumination_gated"
            params["power_floor_w"] = float(raw_values.get("power_ref_w", NONLINEAR_POWER_REF_W))
        if "dark_current_a" in raw_values:
            params["dark_current_a"] = float(raw_values["dark_current_a"])
        if "noise_scale_vs_case1" in raw_values:
            params["noise_scale_vs_case1"] = float(raw_values["noise_scale_vs_case1"])
        if "power_ref_w" in raw_values:
            params["power_ref_w"] = float(raw_values["power_ref_w"])

    if not found_compatible:
        raise ValueError(f"No compatible photodetector parameters found in {csv_path}")
    return normalize_sensor_params(params)


def normalize_sensor_params(params):
    params = dict(params)

    for src_key, dst_key in _SENSOR_EXTRA_ALIASES.items():
        if src_key in params and dst_key not in params:
            params[dst_key] = params[src_key]

    params.setdefault("device_area_cm2", DEFAULT_DEVICE_AREA_CM2)
    params.setdefault("prange1_density_w_cm2", DEFAULT_PRANGE1_DENSITY_W_CM2)
    params.setdefault("prange2_density_w_cm2", DEFAULT_PRANGE2_DENSITY_W_CM2)
    params.setdefault("pmin_density_w_cm2", params["prange1_density_w_cm2"])
    params.setdefault("pmax_density_w_cm2", params["prange2_density_w_cm2"])
    params.setdefault("noise_1f_density_1hz_a_root_hz", DEFAULT_NOISE_1F_DENSITY_1HZ_AHZ05)
    params.setdefault("tia_gain_ohm", 1.0)
    params.setdefault("integration_gain_v_per_c", 1.0)
    params.setdefault("spatial_variation_r_pct", 0.0)
    params.setdefault("trap_delta_r_ratio", 0.0)
    params.setdefault("trap_x1_ratio", 0.0)
    params.setdefault("trap_x2_reference_state", 0.0)
    params.setdefault("trap_x2_tau_on_s", 0.0)
    params.setdefault("trap_x2_tau_off_s", 0.0)

    if "trap_amplitude_pct" in params and params["trap_amplitude_pct"] is not None:
        params["trap_amplitude_ratio"] = float(params.get("trap_amplitude_pct", 0.0)) / 100.0
    elif "trap_amplitude_ratio" not in params:
        params["trap_amplitude_ratio"] = float(params.get("trap_amplitude_pct", 0.0)) / 100.0
    if "r_degradation_pct" in params and params["r_degradation_pct"] is not None:
        params["r_degradation_ratio"] = float(params["r_degradation_pct"]) / 100.0
    elif "r_degradation_ratio" not in params:
        params["r_degradation_ratio"] = max(0.0, -float(params.get("gamma", 0.0)))
    if "spatial_variation_r_pct" in params and params["spatial_variation_r_pct"] is not None:
        params["spatial_variation_r_ratio"] = float(params.get("spatial_variation_r_pct", 0.0)) / 100.0
    elif "spatial_variation_r_ratio" not in params:
        params["spatial_variation_r_ratio"] = float(params.get("spatial_variation_r_pct", 0.0)) / 100.0
    if "aging_tau_hours" not in params or params["aging_tau_hours"] is None:
        params["aging_tau_hours"] = float(params.get("tau_drift", 0.0)) / 3600.0
    if "trap_saturation_time_s" not in params or params["trap_saturation_time_s"] is None:
        params["trap_saturation_time_s"] = None
    if "rise_fast_weight" in params:
        params["rise_fast_weight"] = np.clip(float(params["rise_fast_weight"]), 0.0, 1.0)
    if "fall_fast_weight" in params:
        params["fall_fast_weight"] = np.clip(float(params["fall_fast_weight"]), 0.0, 1.0)

    params["device_area_cm2"] = float(params["device_area_cm2"])
    params["prange1_density_w_cm2"] = float(params["prange1_density_w_cm2"])
    params["prange2_density_w_cm2"] = float(params["prange2_density_w_cm2"])
    params["pmin_density_w_cm2"] = float(params["pmin_density_w_cm2"])
    params["pmax_density_w_cm2"] = float(params["pmax_density_w_cm2"])
    params["noise_1f_density_1hz_a_root_hz"] = float(params["noise_1f_density_1hz_a_root_hz"])
    params["tia_gain_ohm"] = float(params["tia_gain_ohm"])
    params["integration_gain_v_per_c"] = float(params["integration_gain_v_per_c"])
    params["trap_delta_r_ratio"] = max(0.0, float(params["trap_delta_r_ratio"]))
    params["trap_x1_ratio"] = max(0.0, float(params["trap_x1_ratio"]))
    params["trap_x2_reference_state"] = max(0.0, float(params["trap_x2_reference_state"]))
    params["trap_x2_tau_on_s"] = max(0.0, float(params["trap_x2_tau_on_s"]))
    params["trap_x2_tau_off_s"] = max(0.0, float(params["trap_x2_tau_off_s"]))
    params["trap_amplitude_ratio"] = max(0.0, float(params["trap_amplitude_ratio"]))
    params["r_degradation_ratio"] = np.clip(float(params["r_degradation_ratio"]), 0.0, 1.0)
    params["spatial_variation_r_ratio"] = max(0.0, float(params["spatial_variation_r_ratio"]))
    params["aging_tau_hours"] = max(0.0, float(params["aging_tau_hours"]))

    if params["prange2_density_w_cm2"] < params["prange1_density_w_cm2"]:
        raise ValueError("prange2_density_w_cm2 must be >= prange1_density_w_cm2")
    if params["pmax_density_w_cm2"] < params["pmin_density_w_cm2"]:
        raise ValueError("pmax_density_w_cm2 must be >= pmin_density_w_cm2")

    params["prange1_w"] = float(
        compute_total_power_from_density_w_cm2(
            params["prange1_density_w_cm2"], params["device_area_cm2"]
        )
    )
    params["prange2_w"] = float(
        compute_total_power_from_density_w_cm2(
            params["prange2_density_w_cm2"], params["device_area_cm2"]
        )
    )
    params["pmin_w"] = float(
        compute_total_power_from_density_w_cm2(
            params["pmin_density_w_cm2"], params["device_area_cm2"]
        )
    )
    params["pmax_w"] = float(
        compute_total_power_from_density_w_cm2(
            params["pmax_density_w_cm2"], params["device_area_cm2"]
        )
    )

    # Keep the physical case-specific power reference unchanged. Clipping it into the
    # active pmin/pmax window changes the fitted case1 I-P law and can inflate the
    # resulting photocurrent by orders of magnitude.
    params["power_ref_w"] = float(params.get("power_ref_w", NONLINEAR_POWER_REF_W))
    params["gamma"] = -params["r_degradation_ratio"]
    if params["aging_tau_hours"] > 0:
        params["tau_drift"] = params["aging_tau_hours"] * 3600.0

    set_device_context(
        device_area_cm2=params["device_area_cm2"],
        power_ref_w=params["power_ref_w"],
    )
    return params


def apply_sensor_param_overrides(base_params, overrides):
    merged = dict(base_params)
    for key, value in overrides.items():
        if value is None:
            continue
        merged[key] = value
    return normalize_sensor_params(merged)


def resolve_base_params(csv_path):
    global _MISSING_DEFAULT_PARAMS_WARNED
    csv_path = Path(csv_path).expanduser()
    if csv_path.is_file():
        return load_params_csv(csv_path)

    if csv_path.resolve(strict=False) == DEFAULT_PARAMS_CSV.resolve(strict=False):
        if not _MISSING_DEFAULT_PARAMS_WARNED:
            print(
                f"Warning: params csv not found at {csv_path}; using photodetector_model.PARAMS_TRUE instead.",
                file=sys.stderr,
            )
            _MISSING_DEFAULT_PARAMS_WARNED = True
        return normalize_sensor_params(dict(PARAMS_TRUE))

    raise FileNotFoundError(f"Sensor params csv not found: {csv_path}")


def build_power_maps(image, array_size, output_channels, power_max, power_floor_w=None):
    power_floor_w = float(NONLINEAR_POWER_REF_W if power_floor_w is None else power_floor_w)
    if power_max < power_floor_w:
        raise ValueError(
            f"power_max must be >= power_floor_w ({power_floor_w:.3e} W), got {power_max:.3e} W"
        )

    image_tensor = TF.to_tensor(image)
    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(array_size, array_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    if resized.shape[0] == 1:
        resized = resized[:1]
    elif output_channels == 1:
        resized = TF.rgb_to_grayscale(resized, num_output_channels=1)
    elif resized.shape[0] > output_channels:
        resized = resized[:output_channels]

    normalized = np.clip(resized.cpu().numpy().astype(np.float64), 0.0, 1.0)
    return power_floor_w + normalized * (power_max - power_floor_w)


def _simulation_grid(args):
    n_steps = max(1, int(round(args.exposure_time * args.fps_sim)))
    dt = args.exposure_time / n_steps
    t = np.linspace(0.0, args.exposure_time, n_steps + 1, dtype=np.float64)
    return t, dt


def _apply_readout(current_trace, readout_mode, dt, adc_bits, adc_full_scale):
    if readout_mode == "tia":
        return current_trace[-1].astype(np.float32)

    frame = np.sum(current_trace[1:], axis=0, dtype=np.float64) * dt
    if readout_mode == "integration":
        return frame.astype(np.float32)

    if readout_mode == "adc":
        full_scale = adc_full_scale
        if full_scale is None:
            full_scale = np.max(frame) * 1.2 if np.max(frame) > 0 else 1e-6
        normalized = np.clip(frame / full_scale, 0.0, 1.0)
        n_levels = 2 ** adc_bits
        quantized = np.floor(normalized * (n_levels - 1)).astype(np.float64)
        return (quantized / (n_levels - 1) * full_scale).astype(np.float32)

    raise ValueError(f"Unsupported readout mode: {readout_mode}")


def resolve_effective_range_mode(readout_mode, range_mode):
    if range_mode == "auto":
        return "signed" if readout_mode == "tia" else "minmax"
    return range_mode


def simulate_ideal_linear_frame_channel(power_map, args, base_params):
    responsivity_total = float(base_params["R_fast"] + base_params["R_slow"])

    if args.readout == "tia":
        return (responsivity_total * power_map).astype(np.float32)

    frame = responsivity_total * power_map * args.exposure_time
    if args.readout == "integration":
        return frame.astype(np.float32)

    if args.readout == "adc":
        full_scale = args.adc_full_scale
        if full_scale is None:
            full_scale = np.max(frame) * 1.2 if np.max(frame) > 0 else 1e-6
        normalized = np.clip(frame / full_scale, 0.0, 1.0)
        n_levels = 2 ** args.adc_bits
        quantized = np.floor(normalized * (n_levels - 1)).astype(np.float64)
        return (quantized / (n_levels - 1) * full_scale).astype(np.float32)

    raise ValueError(f"Unsupported readout mode: {args.readout}")


def simulate_nonideal_model_frame_channel(power_map, args, base_params, seed_offset):
    t, dt = _simulation_grid(args)
    params_vec = params_to_vec(base_params)
    flat_power = power_map.reshape(-1)
    traces = np.empty((len(t), flat_power.size), dtype=np.float64)

    seed_rng = np.random.default_rng(resolve_sensor_seed(args) + seed_offset)
    pixel_seeds = seed_rng.integers(0, np.iinfo(np.int32).max, size=flat_power.size, dtype=np.int64)

    for pixel_idx, (pixel_power, pixel_seed) in enumerate(zip(flat_power, pixel_seeds)):
        pixel_rng = np.random.default_rng(int(pixel_seed))
        power_trace = np.full_like(t, float(pixel_power), dtype=np.float64)
        current_trace, _, _, _, _ = simulate(
            t,
            power_trace,
            params_vec,
            noise_fn=NOISE_FN,
            dark_current=DARK_CURRENT_MEASURED,
            rng=pixel_rng,
        )
        current_trace = np.asarray(current_trace, dtype=np.float64)
        traces[:, pixel_idx] = current_trace

    current_trace = traces.reshape((len(t),) + power_map.shape)
    return _apply_readout(current_trace, args.readout, dt, args.adc_bits, args.adc_full_scale)


def simulate_static_frame(power_maps, args, base_params, nonideal, seed_offset):
    if power_maps.ndim == 2:
        if nonideal:
            return simulate_nonideal_model_frame_channel(power_maps, args, base_params, seed_offset)
        return simulate_ideal_linear_frame_channel(power_maps, args, base_params)

    frames = [
        (
            simulate_nonideal_model_frame_channel(channel_map, args, base_params, seed_offset + channel_idx)
            if nonideal
            else simulate_ideal_linear_frame_channel(channel_map, args, base_params)
        )
        for channel_idx, channel_map in enumerate(power_maps)
    ]
    return np.stack(frames, axis=0)


def scale_frame(frame, readout_mode, range_mode, low_pct, high_pct, bounds=None):
    if range_mode == "none":
        return frame.astype(np.float32)

    mode = resolve_effective_range_mode(readout_mode, range_mode)
    frame = np.asarray(frame, dtype=np.float32)

    if bounds is not None:
        low = float(bounds["low"])
        high = float(bounds["high"])
        denom = max(high - low, 1e-12)

        if mode == "minmax":
            return np.clip((frame - low) / denom, 0.0, 1.0).astype(np.float32)

        if mode == "signed":
            normalized = np.clip((frame - low) / denom, 0.0, 1.0)
            return (normalized * 2.0 - 1.0).astype(np.float32)

        raise ValueError(f"Unsupported range mode: {range_mode}")

    if mode == "minmax":
        low = np.percentile(frame, low_pct)
        high = np.percentile(frame, high_pct)
        denom = max(high - low, 1e-12)
        return np.clip((frame - low) / denom, 0.0, 1.0).astype(np.float32)

    if mode == "signed":
        center = np.median(frame)
        radius = np.percentile(np.abs(frame - center), high_pct)
        radius = max(radius, 1e-8)
        return np.clip((frame - center) / radius, -1.0, 1.0).astype(np.float32)

    raise ValueError(f"Unsupported range mode: {range_mode}")


def scaled_frame_to_unit_interval(frame, readout_mode, range_mode):
    frame = np.asarray(frame, dtype=np.float32)
    if resolve_effective_range_mode(readout_mode, range_mode) == "signed":
        return np.clip((frame + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    return frame


def normalize_for_psnr(frame):
    frame = np.asarray(frame, dtype=np.float64)
    frame_min = float(np.min(frame))
    frame_max = float(np.max(frame))
    if frame_max - frame_min <= 1e-12:
        return np.ones_like(frame, dtype=np.float64)
    normalized = (frame - frame_min) / (frame_max - frame_min)
    return 1.0 + normalized * 254.0


def compute_psnr(reference, compared):
    reference_norm = normalize_for_psnr(reference)
    compared_norm = normalize_for_psnr(compared)
    mse = float(np.mean((reference_norm - compared_norm) ** 2))
    if mse <= 1e-24:
        return float("inf")
    return float(20.0 * np.log10(255.0) - 10.0 * np.log10(mse))


def format_psnr(psnr_db):
    if np.isinf(psnr_db):
        return "inf dB"
    return f"{psnr_db:.2f} dB"


def to_rgb_image(array_2d_or_3d, tile_size):
    if array_2d_or_3d.ndim == 2:
        img = np.clip(array_2d_or_3d, 0.0, 1.0)
        uint8 = (img * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(uint8, mode="L").convert("RGB")
    else:
        img = np.asarray(array_2d_or_3d)
        if img.shape[0] in {1, 3} and img.shape[-1] not in {1, 3}:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0.0, 1.0)
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        uint8 = (img * 255.0).round().astype(np.uint8)
        pil = Image.fromarray(uint8, mode="RGB")
    return pil.resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)


def compose_triptych(input_img, ideal_img, nonideal_img, label_text, psnr_db):
    title_h = 28
    tile_w, tile_h = input_img.size
    canvas = Image.new("RGB", (tile_w * 3, tile_h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    labels = [
        f"Input | {label_text}",
        "Ideal linear sensor",
        f"Nonideal sensor | PSNR={format_psnr(psnr_db)}",
    ]
    for idx, (img, text) in enumerate(zip([input_img, ideal_img, nonideal_img], labels)):
        x0 = idx * tile_w
        canvas.paste(img, (x0, title_h))
        draw.text((x0 + 8, 6), text, fill=(17, 24, 39))
    return canvas


def _resolve_output_channels(image, requested_output_channels):
    image_channels = len(image.getbands()) if hasattr(image, "getbands") else TF.to_tensor(image).shape[0]
    return 1 if image_channels == 1 else requested_output_channels


def _apply_post_norm(tensor, post_norm, source_dataset):
    if post_norm == "none":
        return tensor

    norm_key = source_dataset if post_norm == "auto" else post_norm
    if norm_key not in _POST_NORM_STATS:
        raise ValueError(f"Unsupported post norm: {post_norm}")

    mean, std = _POST_NORM_STATS[norm_key]
    if tensor.shape[0] == 1:
        mean = (mean[0],)
        std = (std[0],)
    return TF.normalize(tensor, mean=mean, std=std)


def _frame_to_model_tensor(frame, args):
    range_mode = getattr(args, "frame_range_mode_override", args.range_mode)
    frame = scaled_frame_to_unit_interval(frame, args.readout, range_mode)
    tensor = torch.from_numpy(np.asarray(frame)).float()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(args.target_size, args.target_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if args.output_channels == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    return _apply_post_norm(tensor, args.post_norm, args.source_dataset)


def _raw_image_to_model_tensor(image, args):
    tensor = TF.to_tensor(image).float()
    if tensor.shape[0] == 1:
        tensor = tensor[:1]
    elif args.output_channels == 1:
        tensor = TF.rgb_to_grayscale(tensor, num_output_channels=1)
    elif tensor.shape[0] > args.output_channels:
        tensor = tensor[:args.output_channels]

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(args.target_size, args.target_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    if args.output_channels == 3 and tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)

    return _apply_post_norm(tensor, args.post_norm, args.source_dataset)


def _simulate_sensor_case(image, dataset_index, args, base_params, nonideal):
    output_channels = _resolve_output_channels(image, args.output_channels)
    power_maps = build_power_maps(image, args.array_size, output_channels, args.power_max)
    seed_offset = dataset_index * 8 + (4 if nonideal else 0)
    sensor_raw = simulate_static_frame(
        power_maps,
        args,
        base_params,
        nonideal=nonideal,
        seed_offset=seed_offset,
    )
    sensor_scaled = scale_frame(
        sensor_raw,
        args.readout,
        args.range_mode,
        args.percentile_low,
        args.percentile_high,
    )
    return power_maps, sensor_raw, sensor_scaled


def prepare_visual_sample(base_dataset, dataset_index, args, base_params):
    image, label = base_dataset[dataset_index]
    label_name = base_dataset.classes[label]
    output_channels = _resolve_output_channels(image, args.output_channels)
    power_maps = build_power_maps(image, args.array_size, output_channels, args.power_max)
    ideal_raw = simulate_static_frame(power_maps, args, base_params, nonideal=False, seed_offset=dataset_index * 8)
    nonideal_raw = simulate_static_frame(power_maps, args, base_params, nonideal=True, seed_offset=dataset_index * 8 + 4)
    ideal_scaled = scale_frame(
        ideal_raw,
        args.readout,
        args.range_mode,
        args.percentile_low,
        args.percentile_high,
    )
    nonideal_scaled = scale_frame(
        nonideal_raw,
        args.readout,
        args.range_mode,
        args.percentile_low,
        args.percentile_high,
    )
    psnr_db = compute_psnr(ideal_raw, nonideal_raw)
    return {
        "dataset_index": dataset_index,
        "image": image,
        "label": label,
        "label_name": label_name,
        "power_maps": power_maps,
        "ideal_raw": ideal_raw,
        "nonideal_raw": nonideal_raw,
        "ideal_scaled": ideal_scaled,
        "nonideal_scaled": nonideal_scaled,
        "psnr_db": psnr_db,
    }


class RawEvalDataset(Dataset):
    def __init__(self, base_dataset, args):
        self.base_dataset = base_dataset
        self.args = args

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        return _raw_image_to_model_tensor(image, self.args), label


class SensorEvalDataset(Dataset):
    def __init__(self, base_dataset, args, base_params, nonideal):
        self.base_dataset = base_dataset
        self.args = args
        self.base_params = base_params
        self.nonideal = nonideal

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        image, label = self.base_dataset[index]
        _, _, sensor_scaled = _simulate_sensor_case(
            image=image,
            dataset_index=index,
            args=self.args,
            base_params=self.base_params,
            nonideal=self.nonideal,
        )
        return _frame_to_model_tensor(sensor_scaled, self.args), label


def build_eval_loader(case, base_dataset, args, base_params):
    if case == "raw":
        dataset = RawEvalDataset(base_dataset, args)
    elif case == "ideal":
        dataset = SensorEvalDataset(base_dataset, args, base_params, nonideal=False)
    elif case == "nonideal":
        dataset = SensorEvalDataset(base_dataset, args, base_params, nonideal=True)
    else:
        raise ValueError(f"Unsupported evaluation case: {case}")

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )


def evaluate(model, loader, criterion, device, max_batches=0):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_correct += preds.eq(labels).sum().item()
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if batch_idx % 10 == 0:
                elapsed = time.time() - start
                acc = 100.0 * total_correct / max(total_samples, 1)
                print(
                    f"  batch {batch_idx}: samples={total_samples} acc={acc:.2f}% elapsed={elapsed:.1f}s",
                    flush=True,
                )

    elapsed = time.time() - start
    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = 100.0 * total_correct / max(total_samples, 1)
    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
        "samples": total_samples,
        "elapsed_sec": elapsed,
    }


def predict_tensor(model, tensor, device, class_names):
    model.eval()
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
    pred_idx = int(pred.item())
    return {
        "pred_index": pred_idx,
        "pred_label": class_names[pred_idx],
        "confidence": float(conf.item()),
    }


def run_evaluation(args, base_dataset, base_params, model, device):
    criterion = nn.CrossEntropyLoss()
    results = {
        "device": str(device),
        "model_path": args.model_path,
        "cases": {},
    }

    print(f"Using device: {device}", flush=True)
    print(f"Model checkpoint: {args.model_path}", flush=True)
    print(f"Cases: {args.eval_cases}", flush=True)
    print("Sensor backend: generate_sensor_verification_images photodetector_model_only", flush=True)

    for case in args.eval_cases:
        print(f"\n=== Evaluating {case} ===", flush=True)
        loader = build_eval_loader(case, base_dataset, args, base_params)
        case_result = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            max_batches=args.max_eval_batches,
        )
        results["cases"][case] = case_result
        print(
            f"{case}: acc={case_result['accuracy']:.2f}% "
            f"loss={case_result['loss']:.4f} "
            f"samples={case_result['samples']} "
            f"time={case_result['elapsed_sec']:.1f}s",
            flush=True,
        )

    return results


def export_verification_images(args, base_dataset, base_params, model=None, device=None):
    output_dir = Path(args.output_dir)
    input_dir = output_dir / "input"
    ideal_dir = output_dir / "sensor_ideal"
    nonideal_dir = output_dir / "sensor_nonideal"
    compare_dir = output_dir / "comparison"
    for folder in [input_dir, ideal_dir, nonideal_dir, compare_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    manifest = []
    max_images = max(0, args.num_images)
    available = max(0, len(base_dataset) - args.start_index)
    export_count = min(max_images, available)
    if export_count < max_images:
        print(
            f"Warning: requested {max_images} images from start_index={args.start_index}, "
            f"but only {available} are available.",
            flush=True,
        )

    for sample_offset in range(export_count):
        dataset_index = args.start_index + sample_offset
        sample = prepare_visual_sample(base_dataset, dataset_index, args, base_params)
        image = sample["image"]

        input_img = to_rgb_image(np.asarray(image).astype(np.float32) / 255.0, args.tile_size)
        ideal_img = to_rgb_image(
            scaled_frame_to_unit_interval(sample["ideal_scaled"], args.readout, args.range_mode),
            args.tile_size,
        )
        nonideal_img = to_rgb_image(
            scaled_frame_to_unit_interval(sample["nonideal_scaled"], args.readout, args.range_mode),
            args.tile_size,
        )
        triptych = compose_triptych(
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
            "label_index": int(sample["label"]),
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
            raw_tensor = _raw_image_to_model_tensor(image, args)
            ideal_tensor = _frame_to_model_tensor(sample["ideal_scaled"], args)
            nonideal_tensor = _frame_to_model_tensor(sample["nonideal_scaled"], args)
            item["predictions"] = {
                "raw": predict_tensor(model, raw_tensor, device, base_dataset.classes),
                "ideal": predict_tensor(model, ideal_tensor, device, base_dataset.classes),
                "nonideal": predict_tensor(model, nonideal_tensor, device, base_dataset.classes),
            }
            item["correct"] = {
                case: (pred["pred_index"] == int(sample["label"]))
                for case, pred in item["predictions"].items()
            }

        manifest.append(item)

    manifest_path = output_dir / "manifest.json"
    manifest_payload = {
        "args": vars(args),
        "base_params": base_params,
        "simulation_backend": "photodetector_model_only",
        "array_variation_enabled": False,
        "array_noise_enabled": False,
        "nonideal_model_noise": getattr(NOISE_FN, "__name__", str(NOISE_FN)),
        "nonideal_dark_current": DARK_CURRENT_MEASURED,
        "num_images_exported": len(manifest),
        "mean_psnr_db": (
            float(np.mean([item["psnr_db"] for item in manifest]))
            if manifest
            else None
        ),
        "samples": manifest,
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")

    print(f"output_dir            {output_dir}")
    print(f"manifest              {manifest_path}")
    for item in manifest:
        print(
            "sample                "
            f"idx={item['dataset_index']} class={item['label']} "
            f"psnr={format_psnr(item['psnr_db'])} compare={item['compare_path']}"
        )

    return {
        "output_dir": str(output_dir),
        "manifest_path": str(manifest_path),
        "num_images": len(manifest),
        "mean_psnr_db": manifest_payload["mean_psnr_db"],
    }


def run_pipeline(args):
    if not args.generate_images and not args.run_eval:
        raise ValueError("At least one of --generate-images or --run-eval must be enabled.")

    if any(hasattr(args, name) and getattr(args, name) != 0.0 for name in IGNORED_SENSOR_ARGS):
        print(
            "Note: sensor_pixel_var_* arguments are ignored to match the photodetector_model_only "
            "pipeline exactly.",
            flush=True,
        )

    seed_everything(args.seed)
    base_dataset = load_base_dataset(args.source_dataset, args.data_root, args.split)
    base_params = resolve_base_params(args.params_csv)

    model = None
    device = None
    if args.run_eval or args.generate_images:
        device = select_device()
    if args.run_eval:
        model = load_model(args.model_path, device, infer_num_classes(args))

    evaluation = None
    if args.run_eval:
        evaluation = run_evaluation(args, base_dataset, base_params, model, device)

    image_generation = None
    if args.generate_images:
        image_generation = export_verification_images(
            args=args,
            base_dataset=base_dataset,
            base_params=base_params,
            model=model,
            device=device,
        )

    combined_results = {
        "args": vars(args),
        "model_path": args.model_path if args.run_eval else None,
        "source_dataset": args.source_dataset,
        "split": args.split,
        "simulation_backend": "photodetector_model_only",
        "ignored_sensor_args": IGNORED_SENSOR_ARGS,
        "evaluation": evaluation,
        "image_generation": image_generation,
    }

    if args.results_json:
        output_path = Path(args.results_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(combined_results, indent=2), encoding="utf-8")
        print(f"\nSaved results to {output_path}", flush=True)

    return combined_results


def main():
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
