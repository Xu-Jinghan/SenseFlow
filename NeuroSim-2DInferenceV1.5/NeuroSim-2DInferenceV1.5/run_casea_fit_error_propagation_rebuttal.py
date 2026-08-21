"""
CaseA comparison rebuttal experiment for model-fit error propagation.

CaseA is a deliberately constructed, case1-near full sensor model:
nonlinear I-P response, two rise/fall time constants, on-state drift, and a
small temporal noise term. CaseB keeps every non-transient parameter identical
but replaces the two-tau transient with one fitted rise tau and one fitted fall
tau. The script then visualizes the device-level fit residual and propagates
the selected models through the existing CIFAR-10 video-sequence evaluation pipeline.
Optionally, CaseC keeps the CaseA transient model unchanged but perturbs the
nonlinear I-P slope eta to isolate nonlinearity-fit error propagation.

Outputs are written to a fresh artifacts directory by default. Existing case1
data, paper figures, and parameter files are not modified.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import least_squares


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import evaluate_case1_native_vs_x2_video_sequence as eval_cmp  # noqa: E402
import generate_fig3_arbitrary_case1_on_state_drift as fig3_case1  # noqa: E402
import generate_sensor_verification_images_video_sequence as pipeline  # noqa: E402
import photodetector_model as pm  # noqa: E402
from sensor_video_sequence_backend import apply_transient_weight_overrides  # noqa: E402


FPS_VALUES_DEFAULT = [20.0, 50.0, 100.0, 200.0]
BASE_MODEL_KEYS = ["caseA_two_tau", "caseB_single_tau_fit"]
MODEL_DISPLAY = {
    "caseA_two_tau": "CaseA full two-tau model",
    "caseB_single_tau_fit": "CaseB single-tau fit",
    "caseC_eta_error": "CaseC eta-slope error",
}
MODEL_COLORS = {
    "caseA_two_tau": "#0072B2",
    "caseB_single_tau_fit": "#D55E00",
    "caseC_eta_error": "#009E73",
}
READOUT_COLORS = {
    "tia": "#F6BD60",
    "integration": "#84A59D",
    "adc4": "#A8DADC",
    "adc8": "#CDB4DB",
}


DUAL_PARAM_KEYS = [
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
    "device_area_cm2",
    "power_ref_w",
    "prange1_density_w_cm2",
    "prange2_density_w_cm2",
    "pmin_density_w_cm2",
    "pmax_density_w_cm2",
    "trap_saturation_time_s",
    "trap_amplitude_pct",
    "noise_1f_density_1hz_a_root_hz",
    "noise_scale_vs_case1",
    "rise_fast_weight",
    "fall_fast_weight",
]
SINGLE_PARAM_KEYS = [
    "R_single",
    "eta_single",
    "tau_rise_single",
    "tau_fall_single",
    "alpha",
    "beta",
    "delta",
    "gamma",
    "tau_drift",
    "drift_scale",
    "device_area_cm2",
    "power_ref_w",
    "prange1_density_w_cm2",
    "prange2_density_w_cm2",
    "pmin_density_w_cm2",
    "pmax_density_w_cm2",
    "trap_saturation_time_s",
    "trap_amplitude_pct",
    "noise_1f_density_1hz_a_root_hz",
    "noise_scale_vs_case1",
]


def active_model_keys(args):
    model_keys = list(BASE_MODEL_KEYS)
    if bool(getattr(args, "include_model_c", 0)):
        model_keys.append("caseC_eta_error")
    return model_keys


def parameter_keys_for_model(model_key):
    if model_key == "caseB_single_tau_fit":
        return SINGLE_PARAM_KEYS
    return DUAL_PARAM_KEYS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and run CaseA comparison model-fit error propagation rebuttal experiments."
    )
    parser.add_argument("--data-root", default=str(REPO_ROOT / ".datasets"))
    parser.add_argument(
        "--output-dir",
        default="auto",
        help="Output directory. Default creates artifacts/caseA_fit_error_propagation_rebuttal_<timestamp>.",
    )
    parser.add_argument("--run-eval", type=int, default=1)
    parser.add_argument("--resume", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=10, help="10 * 20 = 200 CIFAR-10 test samples.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--fps-values", nargs="+", type=float, default=FPS_VALUES_DEFAULT)
    parser.add_argument("--range-calibration-samples", type=int, default=100)
    parser.add_argument("--post-norm", default="auto", choices=["none", "auto", "cifar10", "cifar100", "imagenet"])
    parser.add_argument("--generate-eval-images", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--visual-start-index", type=int, default=23)
    parser.add_argument(
        "--generate-abc-visuals",
        type=int,
        default=0,
        help="Export aligned CaseA/CaseB/CaseC image comparison panels and PSNR tables.",
    )
    parser.add_argument("--visual-num-images", type=int, default=4)

    parser.add_argument("--casea-eta", type=float, default=0.0, help="If <=0, use extracted case1 eta.")
    parser.add_argument("--rise-fast-ms", type=float, default=0.8)
    parser.add_argument("--rise-slow-ms", type=float, default=5.0)
    parser.add_argument("--rise-fast-weight", type=float, default=0.80)
    parser.add_argument("--fall-fast-ms", type=float, default=1.8)
    parser.add_argument("--fall-slow-ms", type=float, default=10.0)
    parser.add_argument("--fall-fast-weight", type=float, default=0.68)
    parser.add_argument("--on-drift-amplitude-pct", type=float, default=40.0)
    parser.add_argument("--on-drift-tau-ms", type=float, default=30.0)
    parser.add_argument("--noise-1f-density-1hz", type=float, default=2.0e-10)
    parser.add_argument("--noise-scale-label", type=float, default=1.0)
    parser.add_argument("--use-noise-fn", type=int, default=1)
    parser.add_argument("--shot-noise", type=int, default=1)
    parser.add_argument(
        "--include-model-c",
        type=int,
        default=0,
        help="Also build/evaluate CaseC, which keeps CaseA tau but perturbs nonlinear eta.",
    )
    parser.add_argument(
        "--casec-eta-scale",
        type=float,
        default=1.05,
        help="Multiplicative eta slope error for CaseC relative to CaseA.",
    )

    parser.add_argument("--fit-rise-window-ms", type=float, default=80.0)
    parser.add_argument("--fit-fall-window-ms", type=float, default=120.0)
    parser.add_argument("--fit-num-points", type=int, default=48)
    parser.add_argument(
        "--fit-weight-mode",
        default="uniform",
        choices=["uniform", "early-window"],
        help="Weighting used when compressing CaseA two-tau transients into CaseB single tau.",
    )
    parser.add_argument(
        "--fit-early-fraction",
        type=float,
        default=0.1,
        help="Fraction of each fit time window given elevated point weight.",
    )
    parser.add_argument(
        "--fit-early-weight",
        type=float,
        default=10.0,
        help="Point weight applied to the early transient when --fit-weight-mode=early-window.",
    )
    return parser.parse_args()


def resolve_output_dir(raw):
    if raw not in {None, "", "auto"}:
        path = Path(raw).expanduser()
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = THIS_DIR / "artifacts" / f"caseA_fit_error_propagation_rebuttal_{stamp}"
    path.mkdir(parents=True, exist_ok=False if raw in {None, "", "auto"} else True)
    return path


def residual_summary(measured, predicted, num_params):
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - measured
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    value_range = max(float(np.max(measured) - np.min(measured)), 1e-30)
    ss_tot = max(float(np.sum((measured - float(np.mean(measured))) ** 2)), 1e-30)
    return {
        "count": int(measured.size),
        "num_params": int(num_params),
        "sse": sse,
        "rmse": rmse,
        "mae": mae,
        "nrmse_range": float(rmse / value_range),
        "max_abs": float(np.max(np.abs(residual))),
        "r2": float(1.0 - sse / ss_tot),
    }


def weighted_residual_summary(measured, predicted, weights, num_params):
    measured = np.asarray(measured, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if measured.shape != predicted.shape or measured.shape != weights.shape:
        raise ValueError("measured, predicted, and weights must have the same shape")
    weights = np.maximum(weights, 0.0)
    weight_sum = max(float(np.sum(weights)), 1e-30)
    residual = predicted - measured
    weighted_sse = float(np.sum(weights * residual ** 2))
    rmse = float(np.sqrt(weighted_sse / weight_sum))
    mae = float(np.sum(weights * np.abs(residual)) / weight_sum)
    value_range = max(float(np.max(measured) - np.min(measured)), 1e-30)
    weighted_mean = float(np.sum(weights * measured) / weight_sum)
    ss_tot = max(float(np.sum(weights * (measured - weighted_mean) ** 2)), 1e-30)
    return {
        "count": int(measured.size),
        "num_params": int(num_params),
        "weight_sum": weight_sum,
        "weighted_sse": weighted_sse,
        "weighted_rmse": rmse,
        "weighted_mae": mae,
        "weighted_nrmse_range": float(rmse / value_range),
        "max_abs": float(np.max(np.abs(residual))),
        "weighted_r2": float(1.0 - weighted_sse / ss_tot),
    }


def build_fit_weights(t_s, mode, early_fraction, early_weight):
    t_s = np.asarray(t_s, dtype=np.float64)
    weights = np.ones_like(t_s, dtype=np.float64)
    fraction = float(np.clip(early_fraction, 0.0, 1.0))
    cutoff = float(np.min(t_s) + (np.max(t_s) - np.min(t_s)) * fraction)
    early_mask = t_s <= cutoff + 1e-15
    if mode == "early-window":
        weights[early_mask] = max(float(early_weight), 1e-12)
    elif mode != "uniform":
        raise ValueError(f"Unsupported fit weight mode: {mode}")
    return weights, early_mask, cutoff


def two_tau_rise(t_s, tau_fast, tau_slow, fast_weight):
    t_s = np.asarray(t_s, dtype=np.float64)
    w = float(np.clip(fast_weight, 0.0, 1.0))
    return 1.0 - (
        w * np.exp(-t_s / max(float(tau_fast), 1e-30))
        + (1.0 - w) * np.exp(-t_s / max(float(tau_slow), 1e-30))
    )


def two_tau_fall(t_s, tau_fast, tau_slow, fast_weight):
    t_s = np.asarray(t_s, dtype=np.float64)
    w = float(np.clip(fast_weight, 0.0, 1.0))
    return (
        w * np.exp(-t_s / max(float(tau_fast), 1e-30))
        + (1.0 - w) * np.exp(-t_s / max(float(tau_slow), 1e-30))
    )


def single_rise(t_s, tau):
    return 1.0 - np.exp(-np.asarray(t_s, dtype=np.float64) / max(float(tau), 1e-30))


def single_fall(t_s, tau):
    return np.exp(-np.asarray(t_s, dtype=np.float64) / max(float(tau), 1e-30))


def fit_single_tau_to_two_tau(args):
    rise_t = np.linspace(0.0, args.fit_rise_window_ms * 1e-3, int(args.fit_num_points))
    fall_t = np.linspace(0.0, args.fit_fall_window_ms * 1e-3, int(args.fit_num_points))
    rise_weights, rise_early_mask, rise_early_cutoff = build_fit_weights(
        rise_t,
        args.fit_weight_mode,
        args.fit_early_fraction,
        args.fit_early_weight,
    )
    fall_weights, fall_early_mask, fall_early_cutoff = build_fit_weights(
        fall_t,
        args.fit_weight_mode,
        args.fit_early_fraction,
        args.fit_early_weight,
    )
    rise_true = two_tau_rise(
        rise_t,
        args.rise_fast_ms * 1e-3,
        args.rise_slow_ms * 1e-3,
        args.rise_fast_weight,
    )
    fall_true = two_tau_fall(
        fall_t,
        args.fall_fast_ms * 1e-3,
        args.fall_slow_ms * 1e-3,
        args.fall_fast_weight,
    )

    def fit_tau(t_s, y, weights, branch):
        model = single_rise if branch == "rise" else single_fall
        sqrt_weights = np.sqrt(weights)
        starts = [0.5e-3, 1.5e-3, 3e-3, 8e-3, 20e-3, 60e-3]
        best = None
        for start in starts:
            result = least_squares(
                lambda x: (model(t_s, math.exp(float(x[0]))) - y) * sqrt_weights,
                np.asarray([math.log(start)], dtype=np.float64),
                bounds=([math.log(1e-5)], [math.log(1.0)]),
                max_nfev=50000,
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
            )
            tau = math.exp(float(result.x[0]))
            pred = model(t_s, tau)
            sse = float(np.sum((pred - y) ** 2))
            weighted_sse = float(np.sum(weights * (pred - y) ** 2))
            if best is None or weighted_sse < best["weighted_sse"]:
                best = {
                    "tau_s": tau,
                    "predicted": pred,
                    "sse": sse,
                    "weighted_sse": weighted_sse,
                    "success": bool(result.success),
                }
        return best

    rise_fit = fit_tau(rise_t, rise_true, rise_weights, "rise")
    fall_fit = fit_tau(fall_t, fall_true, fall_weights, "fall")
    rise_late_mask = ~rise_early_mask
    fall_late_mask = ~fall_early_mask

    def segment_metrics(rise_mask, fall_mask):
        return {
            "rise": residual_summary(rise_true[rise_mask], rise_fit["predicted"][rise_mask], num_params=1),
            "fall": residual_summary(fall_true[fall_mask], fall_fit["predicted"][fall_mask], num_params=1),
            "combined": residual_summary(
                np.concatenate([rise_true[rise_mask], fall_true[fall_mask]]),
                np.concatenate([rise_fit["predicted"][rise_mask], fall_fit["predicted"][fall_mask]]),
                num_params=2,
            ),
        }

    dense_rise_t = np.linspace(0.0, args.fit_rise_window_ms * 1e-3, 1200)
    dense_fall_t = np.linspace(0.0, args.fit_fall_window_ms * 1e-3, 1200)
    dense = {
        "rise_t_s": dense_rise_t,
        "fall_t_s": dense_fall_t,
        "rise_caseA": two_tau_rise(
            dense_rise_t,
            args.rise_fast_ms * 1e-3,
            args.rise_slow_ms * 1e-3,
            args.rise_fast_weight,
        ),
        "fall_caseA": two_tau_fall(
            dense_fall_t,
            args.fall_fast_ms * 1e-3,
            args.fall_slow_ms * 1e-3,
            args.fall_fast_weight,
        ),
        "rise_caseB": single_rise(dense_rise_t, rise_fit["tau_s"]),
        "fall_caseB": single_fall(dense_fall_t, fall_fit["tau_s"]),
    }
    return {
        "fit_grid": {
            "rise_t_s": rise_t,
            "fall_t_s": fall_t,
            "rise_caseA": rise_true,
            "fall_caseA": fall_true,
            "rise_caseB": rise_fit["predicted"],
            "fall_caseB": fall_fit["predicted"],
            "rise_weight": rise_weights,
            "fall_weight": fall_weights,
        },
        "dense": dense,
        "caseB_tau_rise_s": float(rise_fit["tau_s"]),
        "caseB_tau_fall_s": float(fall_fit["tau_s"]),
        "weighting": {
            "mode": str(args.fit_weight_mode),
            "early_fraction": float(np.clip(args.fit_early_fraction, 0.0, 1.0)),
            "early_weight": float(max(args.fit_early_weight, 1e-12)),
            "rise_early_cutoff_s": float(rise_early_cutoff),
            "fall_early_cutoff_s": float(fall_early_cutoff),
            "rise_early_points": int(np.sum(rise_early_mask)),
            "fall_early_points": int(np.sum(fall_early_mask)),
            "rise_total_points": int(rise_t.size),
            "fall_total_points": int(fall_t.size),
        },
        "metrics": {
            "rise": residual_summary(rise_true, rise_fit["predicted"], num_params=1),
            "fall": residual_summary(fall_true, fall_fit["predicted"], num_params=1),
            "combined": residual_summary(
                np.concatenate([rise_true, fall_true]),
                np.concatenate([rise_fit["predicted"], fall_fit["predicted"]]),
                num_params=2,
            ),
        },
        "weighted_metrics": {
            "rise": weighted_residual_summary(rise_true, rise_fit["predicted"], rise_weights, num_params=1),
            "fall": weighted_residual_summary(fall_true, fall_fit["predicted"], fall_weights, num_params=1),
            "combined": weighted_residual_summary(
                np.concatenate([rise_true, fall_true]),
                np.concatenate([rise_fit["predicted"], fall_fit["predicted"]]),
                np.concatenate([rise_weights, fall_weights]),
                num_params=2,
            ),
        },
        "segment_metrics": {
            "early_window": segment_metrics(rise_early_mask, fall_early_mask),
            "late_window": segment_metrics(rise_late_mask, fall_late_mask),
        },
    }


def compare_to_uniform_fit(args, weighted_fit):
    if args.fit_weight_mode == "uniform":
        return None
    uniform_args = copy.copy(args)
    uniform_args.fit_weight_mode = "uniform"
    uniform_fit = fit_single_tau_to_two_tau(uniform_args)
    return {
        "uniform_tau_rise_s": uniform_fit["caseB_tau_rise_s"],
        "uniform_tau_fall_s": uniform_fit["caseB_tau_fall_s"],
        "weighted_tau_rise_s": weighted_fit["caseB_tau_rise_s"],
        "weighted_tau_fall_s": weighted_fit["caseB_tau_fall_s"],
        "uniform_segment_metrics": uniform_fit["segment_metrics"],
        "weighted_segment_metrics": weighted_fit["segment_metrics"],
    }


def write_parameter_csv(csv_path, params, keys):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value"])
        for key in keys:
            writer.writerow([key, params[key]])


def build_case_params(args, case1_results, fit):
    base = case1_results["params"]
    r_total = float(base["R_single"])
    eta = float(base["eta_single"] if args.casea_eta <= 0 else args.casea_eta)
    shared = {
        "alpha": 0.0,
        "beta": 1.0,
        "delta": 0.0,
        "gamma": float(base["gamma"]),
        "tau_drift": float(base["tau_drift"]),
        "drift_scale": float(base["drift_scale"]),
        "device_area_cm2": float(pm.DEVICE_AREA_CM2),
        "power_ref_w": float(case1_results["power_ref_w"]),
        "prange1_density_w_cm2": float(pm.DEFAULT_PRANGE1_DENSITY_W_CM2),
        "prange2_density_w_cm2": float(pm.DEFAULT_PRANGE2_DENSITY_W_CM2),
        "pmin_density_w_cm2": float(pm.DEFAULT_PRANGE1_DENSITY_W_CM2),
        "pmax_density_w_cm2": float(pm.DEFAULT_PRANGE2_DENSITY_W_CM2),
        "trap_saturation_time_s": float(args.on_drift_tau_ms) * 1e-3,
        "trap_amplitude_pct": float(args.on_drift_amplitude_pct),
        "noise_1f_density_1hz_a_root_hz": float(args.noise_1f_density_1hz),
        "noise_scale_vs_case1": float(args.noise_scale_label),
    }
    case_a = {
        "R_fast": 0.5 * r_total,
        "eta_fast": eta,
        "tau_rise_fast": float(args.rise_fast_ms) * 1e-3,
        "tau_fall_fast": float(args.fall_fast_ms) * 1e-3,
        "R_slow": 0.5 * r_total,
        "eta_slow": eta,
        "tau_rise_slow": float(args.rise_slow_ms) * 1e-3,
        "tau_fall_slow": float(args.fall_slow_ms) * 1e-3,
        "rise_fast_weight": float(np.clip(args.rise_fast_weight, 0.0, 1.0)),
        "fall_fast_weight": float(np.clip(args.fall_fast_weight, 0.0, 1.0)),
        **shared,
    }
    case_b = {
        "R_single": r_total,
        "eta_single": eta,
        "tau_rise_single": fit["caseB_tau_rise_s"],
        "tau_fall_single": fit["caseB_tau_fall_s"],
        **shared,
    }
    case_c = dict(case_a)
    case_c["eta_fast"] = float(case_a["eta_fast"]) * float(args.casec_eta_scale)
    case_c["eta_slow"] = float(case_a["eta_slow"]) * float(args.casec_eta_scale)
    return {
        "caseA_two_tau": case_a,
        "caseB_single_tau_fit": case_b,
        "caseC_eta_error": case_c,
        "shared_design": {
            "source": "Synthetic case1-near rebuttal benchmark. Non-transient parameters are shared.",
            "R_total_A_per_W": r_total,
            "eta": eta,
            "caseC_eta_scale": float(args.casec_eta_scale),
            **shared,
        },
    }


def model_config_from_params(params):
    pm.set_device_context(
        device_area_cm2=float(params["device_area_cm2"]),
        power_ref_w=float(params["power_ref_w"]),
    )
    n_carrier = pm.infer_n_carrier_from_params(params)
    cfg = pm.prepare_model_config(
        pm.params_to_vec(params),
        n_carrier=n_carrier,
        power_min_w=pm.compute_total_power_from_density_w_cm2(
            params["pmin_density_w_cm2"], params["device_area_cm2"]
        ),
        power_max_w=pm.compute_total_power_from_density_w_cm2(
            params["pmax_density_w_cm2"], params["device_area_cm2"]
        ),
        trap_saturation_time_s=params.get("trap_saturation_time_s"),
        trap_amplitude_ratio=float(params.get("trap_amplitude_pct", 0.0)) / 100.0,
    )
    if "rise_fast_weight" in params and "fall_fast_weight" in params:
        cfg = apply_transient_weight_overrides(cfg, params)
    return cfg


def simulate_current_trace(params, power_w, t_s, include_noise=False, rng_seed=1234):
    power_w = np.asarray(power_w, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    dt = float(np.median(np.diff(t_s)))
    cfg = model_config_from_params(params)
    n_carrier = pm.infer_n_carrier_from_params(params)
    x1, x2, x3 = pm.init_state_arrays(power_w.shape[1:] if power_w.ndim > 1 else (), n_carrier=n_carrier)
    current = []
    for p in power_w:
        x1, x2, x3 = pm.step_model_state(p, dt, cfg, x1, x2, x3)
        det, _ = pm.current_from_state(cfg, x1, x2, x3, P=p, dark_current=pm.DARK_CURRENT_MEASURED)
        current.append(float(np.asarray(det)))
    current = np.asarray(current, dtype=np.float64)
    if include_noise:
        rng = np.random.default_rng(rng_seed)
        pmax = pm.compute_total_power_from_density_w_cm2(params["pmax_density_w_cm2"], params["device_area_cm2"])
        pmax_current = float(
            np.max(pm.steady_state_current_from_power(pmax, cfg, dark_current=pm.DARK_CURRENT_MEASURED))
        )
        shot_density = math.sqrt(max(0.0, 2.0 * 1.602e-19 * pmax_current))
        noise = pm.sample_combined_psd_noise_signal_trace(
            t_s,
            current.shape,
            shot_noise_density_ahz05=shot_density,
            flicker_noise_density_1hz_ahz05=float(params["noise_1f_density_1hz_a_root_hz"]),
            rng=rng,
        )
        current = current + noise
    return current


def simulate_case_trace_with_states(params, power_w, t_s, include_noise=False, rng_seed=1234, initial_state=None):
    power_w = np.asarray(power_w, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    dt = float(np.median(np.diff(t_s)))
    cfg = model_config_from_params(params)
    n_carrier = pm.infer_n_carrier_from_params(params)
    if initial_state is None:
        x1, x2, x3 = pm.init_state_arrays((), n_carrier=n_carrier)
    else:
        x1, x2, x3 = initial_state
        x1 = np.asarray(x1, dtype=np.float64).copy()
        x2 = np.asarray(x2, dtype=np.float64).copy()
        x3 = np.asarray(x3, dtype=np.float64).copy()

    i_det = []
    i_photo = []
    main_current = []
    drift_current = []
    x1_trace = []
    x2_trace = []
    x3_trace = []
    for p in power_w:
        x1, x2, x3 = pm.step_model_state(p, dt, cfg, x1, x2, x3)
        det, photo = pm.current_from_state(cfg, x1, x2, x3, P=p, dark_current=pm.DARK_CURRENT_MEASURED)
        main_photo = float(np.sum(x1))
        if cfg.get("trap_ratio_mode"):
            trap_photo = float(cfg["trap_amplitude_ratio"] * main_photo * np.asarray(x2))
        else:
            trap_photo = float(cfg["delta"] * np.asarray(x2))
        drift_multiplier = float(np.clip(1.0 + np.asarray(x3), 0.0, None))
        i_det.append(float(np.asarray(det)))
        i_photo.append(float(np.asarray(photo)))
        main_current.append(main_photo * drift_multiplier)
        drift_current.append(trap_photo * drift_multiplier)
        x1_trace.append(np.asarray(x1, dtype=np.float64).copy())
        x2_trace.append(float(np.asarray(x2)))
        x3_trace.append(float(np.asarray(x3)))

    i_det = np.asarray(i_det, dtype=np.float64)
    if include_noise:
        rng = np.random.default_rng(rng_seed)
        pmax = pm.compute_total_power_from_density_w_cm2(params["pmax_density_w_cm2"], params["device_area_cm2"])
        pmax_current = float(
            np.max(pm.steady_state_current_from_power(pmax, cfg, dark_current=pm.DARK_CURRENT_MEASURED))
        )
        shot_density = math.sqrt(max(0.0, 2.0 * 1.602e-19 * pmax_current))
        noise = pm.sample_combined_psd_noise_signal_trace(
            t_s,
            i_det.shape,
            shot_noise_density_ahz05=shot_density,
            flicker_noise_density_1hz_ahz05=float(params["noise_1f_density_1hz_a_root_hz"]),
            rng=rng,
        )
    else:
        noise = np.zeros_like(i_det)

    return {
        "config": cfg,
        "i_det": i_det,
        "i_out": i_det + noise,
        "i_photo": np.asarray(i_photo, dtype=np.float64),
        "noise": np.asarray(noise, dtype=np.float64),
        "main_current": np.asarray(main_current, dtype=np.float64),
        "drift_current": np.asarray(drift_current, dtype=np.float64),
        "x1": np.asarray(x1_trace, dtype=np.float64),
        "x2": np.asarray(x2_trace, dtype=np.float64),
        "x3": np.asarray(x3_trace, dtype=np.float64),
        "final_state": (np.asarray(x1, dtype=np.float64).copy(), np.asarray(x2, dtype=np.float64).copy(), np.asarray(x3, dtype=np.float64).copy()),
    }


def advance_state_without_illumination(params, initial_state, duration_s):
    cfg = model_config_from_params(params)
    x1, x2, x3 = initial_state
    x1, x2, x3 = pm.step_model_state(
        0.0,
        float(duration_s),
        cfg,
        np.asarray(x1, dtype=np.float64).copy(),
        np.asarray(x2, dtype=np.float64).copy(),
        np.asarray(x3, dtype=np.float64).copy(),
    )
    return x1, x2, x3


def build_fig3_reference_waveform_for_params(params):
    t_s, source_power_w = fig3_case1.build_small_range_waveform()
    source_active = source_power_w > 0.0
    pmin = pm.compute_total_power_from_density_w_cm2(params["pmin_density_w_cm2"], params["device_area_cm2"])
    pmax = pm.compute_total_power_from_density_w_cm2(params["pmax_density_w_cm2"], params["device_area_cm2"])
    if not np.any(source_active):
        return t_s, np.zeros_like(source_power_w, dtype=np.float64)
    src_min = float(np.min(source_power_w[source_active]))
    src_max = float(np.max(source_power_w[source_active]))
    normalized = np.divide(
        source_power_w - src_min,
        max(src_max - src_min, 1e-30),
        out=np.zeros_like(source_power_w, dtype=np.float64),
        where=source_power_w > 0.0,
    )
    power_w = np.where(source_active, pmin + normalized * (pmax - pmin), 0.0)
    return t_s, power_w


def render_fig3_style_casea_caseb_waveform(fig_dir, params_by_model):
    case_a = params_by_model["caseA_two_tau"]
    case_b = params_by_model["caseB_single_tau_fit"]
    t_s, power_w = build_fig3_reference_waveform_for_params(case_a)
    gap_s = fig3_case1.LATE_START_S - float(t_s[-1])
    if gap_s <= 0:
        raise ValueError("late segment must start after the early waveform segment")

    early_a = simulate_case_trace_with_states(case_a, power_w, t_s, include_noise=True, rng_seed=fig3_case1.DEFAULT_EARLY_NOISE_SEED)
    early_b = simulate_case_trace_with_states(case_b, power_w, t_s, include_noise=True, rng_seed=fig3_case1.DEFAULT_EARLY_NOISE_SEED)
    late_a = simulate_case_trace_with_states(
        case_a,
        power_w,
        t_s,
        include_noise=True,
        rng_seed=fig3_case1.DEFAULT_LATE_NOISE_SEED,
        initial_state=advance_state_without_illumination(case_a, early_a["final_state"], gap_s),
    )
    late_b = simulate_case_trace_with_states(
        case_b,
        power_w,
        t_s,
        include_noise=True,
        rng_seed=fig3_case1.DEFAULT_LATE_NOISE_SEED,
        initial_state=advance_state_without_illumination(case_b, early_b["final_state"], gap_s),
    )

    display_early_ms = t_s * 1e3
    display_late_ms = fig3_case1.BREAK_RIGHT_MS + t_s * 1e3
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.6, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.65, 1.25]},
    )

    axes[0].plot(display_early_ms, power_w / case_a["device_area_cm2"] * 1e3, color="tab:green", lw=2.4)
    axes[0].plot(display_late_ms, power_w / case_a["device_area_cm2"] * 1e3, color="tab:green", lw=2.4)
    axes[0].set_ylabel("P opt\n(mW/cm^2)")
    axes[0].text(0.015, 0.70, "P_in", transform=axes[0].transAxes, fontsize=16, fontweight="bold")

    axes[1].plot(display_early_ms, early_a["i_out"] * 1e6, color="#0072B2", lw=2.1, label="CaseA two-tau")
    axes[1].plot(display_late_ms, late_a["i_out"] * 1e6, color="#0072B2", lw=2.1)
    axes[1].plot(display_early_ms, early_b["i_out"] * 1e6, color="#D55E00", lw=2.0, ls="--", label="CaseB fitted single-tau")
    axes[1].plot(display_late_ms, late_b["i_out"] * 1e6, color="#D55E00", lw=2.0, ls="--")
    axes[1].set_ylabel("I_out (uA)")
    axes[1].text(0.015, 0.78, "I_out", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    axes[1].annotate(
        "500 hours later",
        xy=(fig3_case1.BREAK_RIGHT_MS - 8.0, 0.60),
        xytext=(fig3_case1.BREAK_LEFT_MS - 42.0, 0.60),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": "#AA0000"},
        ha="right",
        va="center",
        color="#AA0000",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(loc="upper right", frameon=True, fontsize=10)

    axes[2].plot(display_early_ms, early_a["main_current"] * 1e6, color="#0072B2", lw=2.0, label="CaseA main branch")
    axes[2].plot(display_late_ms, late_a["main_current"] * 1e6, color="#0072B2", lw=2.0)
    axes[2].plot(display_early_ms, early_b["main_current"] * 1e6, color="#D55E00", lw=1.8, ls="--", label="CaseB main branch")
    axes[2].plot(display_late_ms, late_b["main_current"] * 1e6, color="#D55E00", lw=1.8, ls="--")
    axes[2].plot(display_early_ms, early_a["drift_current"] * 1e6, color="#AA4499", lw=1.8, label="CaseA on-state drift current")
    axes[2].plot(display_late_ms, late_a["drift_current"] * 1e6, color="#AA4499", lw=1.8)
    axes[2].set_ylabel("Photo current (uA)")
    axes[2].set_xlabel("Time (0-500 ms, then jump to 500 h)")
    axes[2].legend(loc="upper right", frameon=True, fontsize=10, ncol=2)

    for ax in axes:
        ax.axvspan(
            fig3_case1.BREAK_LEFT_MS,
            fig3_case1.BREAK_RIGHT_MS,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            hatch="////",
            zorder=0,
        )
        ax.grid(True, which="major", linestyle="--", linewidth=0.75, color="#777777", alpha=0.22)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, color="#999999", alpha=0.11)
        ax.set_xlim(0.0, display_late_ms[-1])

    axes[-1].set_xticks(
        [0, 100, 200, 300, 400, 500, fig3_case1.BREAK_RIGHT_MS, fig3_case1.BREAK_RIGHT_MS + 200, fig3_case1.BREAK_RIGHT_MS + 400]
    )
    axes[-1].set_xticklabels(["0", "100", "200", "300", "400", "500", "500 h", "+200 ms", "+400 ms"])

    fig.tight_layout(pad=0.8)
    waveform_fig = fig_dir / "fig_caseA_caseB_fig3_style_arbitrary_waveform.png"
    fig.savefig(waveform_fig, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return waveform_fig


def render_fig3_style_casea_casec_waveform(fig_dir, params_by_model):
    if "caseC_eta_error" not in params_by_model:
        return None
    case_a = params_by_model["caseA_two_tau"]
    case_c = params_by_model["caseC_eta_error"]
    t_s, power_w = build_fig3_reference_waveform_for_params(case_a)
    gap_s = fig3_case1.LATE_START_S - float(t_s[-1])
    if gap_s <= 0:
        raise ValueError("late segment must start after the early waveform segment")

    early_a = simulate_case_trace_with_states(case_a, power_w, t_s, include_noise=True, rng_seed=fig3_case1.DEFAULT_EARLY_NOISE_SEED)
    early_c = simulate_case_trace_with_states(case_c, power_w, t_s, include_noise=True, rng_seed=fig3_case1.DEFAULT_EARLY_NOISE_SEED)
    late_a = simulate_case_trace_with_states(
        case_a,
        power_w,
        t_s,
        include_noise=True,
        rng_seed=fig3_case1.DEFAULT_LATE_NOISE_SEED,
        initial_state=advance_state_without_illumination(case_a, early_a["final_state"], gap_s),
    )
    late_c = simulate_case_trace_with_states(
        case_c,
        power_w,
        t_s,
        include_noise=True,
        rng_seed=fig3_case1.DEFAULT_LATE_NOISE_SEED,
        initial_state=advance_state_without_illumination(case_c, early_c["final_state"], gap_s),
    )

    display_early_ms = t_s * 1e3
    display_late_ms = fig3_case1.BREAK_RIGHT_MS + t_s * 1e3
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15.6, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [0.85, 1.65, 1.25]},
    )

    axes[0].plot(display_early_ms, power_w / case_a["device_area_cm2"] * 1e3, color="tab:green", lw=2.4)
    axes[0].plot(display_late_ms, power_w / case_a["device_area_cm2"] * 1e3, color="tab:green", lw=2.4)
    axes[0].set_ylabel("P opt\n(mW/cm^2)")
    axes[0].text(0.015, 0.70, "P_in", transform=axes[0].transAxes, fontsize=16, fontweight="bold")

    axes[1].plot(display_early_ms, early_a["i_out"] * 1e6, color=MODEL_COLORS["caseA_two_tau"], lw=2.1, label="CaseA reference")
    axes[1].plot(display_late_ms, late_a["i_out"] * 1e6, color=MODEL_COLORS["caseA_two_tau"], lw=2.1)
    axes[1].plot(display_early_ms, early_c["i_out"] * 1e6, color=MODEL_COLORS["caseC_eta_error"], lw=2.0, ls="--", label="CaseC eta error")
    axes[1].plot(display_late_ms, late_c["i_out"] * 1e6, color=MODEL_COLORS["caseC_eta_error"], lw=2.0, ls="--")
    axes[1].set_ylabel("I_out (uA)")
    axes[1].text(0.015, 0.78, "I_out", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
    axes[1].annotate(
        "500 hours later",
        xy=(fig3_case1.BREAK_RIGHT_MS - 8.0, 0.60),
        xytext=(fig3_case1.BREAK_LEFT_MS - 42.0, 0.60),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        arrowprops={"arrowstyle": "-|>", "lw": 2.6, "color": "#AA0000"},
        ha="right",
        va="center",
        color="#AA0000",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(loc="upper right", frameon=True, fontsize=10)

    axes[2].plot(display_early_ms, early_a["main_current"] * 1e6, color=MODEL_COLORS["caseA_two_tau"], lw=2.0, label="CaseA main branch")
    axes[2].plot(display_late_ms, late_a["main_current"] * 1e6, color=MODEL_COLORS["caseA_two_tau"], lw=2.0)
    axes[2].plot(display_early_ms, early_c["main_current"] * 1e6, color=MODEL_COLORS["caseC_eta_error"], lw=1.8, ls="--", label="CaseC main branch")
    axes[2].plot(display_late_ms, late_c["main_current"] * 1e6, color=MODEL_COLORS["caseC_eta_error"], lw=1.8, ls="--")
    axes[2].plot(display_early_ms, early_a["drift_current"] * 1e6, color="#AA4499", lw=1.8, label="CaseA on-state drift current")
    axes[2].plot(display_late_ms, late_a["drift_current"] * 1e6, color="#AA4499", lw=1.8)
    axes[2].set_ylabel("Photo current (uA)")
    axes[2].set_xlabel("Time (0-500 ms, then jump to 500 h)")
    axes[2].legend(loc="upper right", frameon=True, fontsize=10, ncol=2)

    for ax in axes:
        ax.axvspan(
            fig3_case1.BREAK_LEFT_MS,
            fig3_case1.BREAK_RIGHT_MS,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            hatch="////",
            zorder=0,
        )
        ax.grid(True, which="major", linestyle="--", linewidth=0.75, color="#777777", alpha=0.22)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.45, color="#999999", alpha=0.11)
        ax.set_xlim(0.0, display_late_ms[-1])

    axes[-1].set_xticks(
        [0, 100, 200, 300, 400, 500, fig3_case1.BREAK_RIGHT_MS, fig3_case1.BREAK_RIGHT_MS + 200, fig3_case1.BREAK_RIGHT_MS + 400]
    )
    axes[-1].set_xticklabels(["0", "100", "200", "300", "400", "500", "500 h", "+200 ms", "+400 ms"])

    fig.tight_layout(pad=0.8)
    waveform_fig = fig_dir / "fig_caseA_caseC_fig3_style_eta_error_waveform.png"
    fig.savefig(waveform_fig, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return waveform_fig


def render_device_figures(output_dir, params_by_model, fit):
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    case_a = params_by_model["caseA_two_tau"]
    case_b = params_by_model["caseB_single_tau_fit"]
    case_c = params_by_model.get("caseC_eta_error")
    cfg_a = model_config_from_params(case_a)
    cfg_b = model_config_from_params(case_b)
    cfg_c = model_config_from_params(case_c) if case_c else None

    p_density = np.logspace(
        math.log10(case_a["prange1_density_w_cm2"]),
        math.log10(case_a["prange2_density_w_cm2"]),
        160,
    )
    p_w = pm.compute_total_power_from_density_w_cm2(p_density, case_a["device_area_cm2"])
    ideal_current = float(case_b["R_single"]) * p_w + pm.DARK_CURRENT_MEASURED
    ip_a = pm.steady_state_current_from_power(p_w, cfg_a, dark_current=pm.DARK_CURRENT_MEASURED)
    ip_b = pm.steady_state_current_from_power(p_w, cfg_b, dark_current=pm.DARK_CURRENT_MEASURED)
    ip_c = pm.steady_state_current_from_power(p_w, cfg_c, dark_current=pm.DARK_CURRENT_MEASURED) if cfg_c else None

    dense = fit["dense"]
    t_noise = np.arange(0.0, 0.2, 2.0e-4)
    p_mid = pm.compute_total_power_from_density_w_cm2(
        0.5 * (case_a["prange1_density_w_cm2"] + case_a["prange2_density_w_cm2"]),
        case_a["device_area_cm2"],
    )
    p_const = np.full_like(t_noise, p_mid, dtype=np.float64)
    clean_a = simulate_current_trace(case_a, p_const, t_noise, include_noise=False)
    noisy_a = simulate_current_trace(case_a, p_const, t_noise, include_noise=True, rng_seed=case_a.get("seed", 1234))

    t_arb = np.arange(0.0, 0.24, 2.0e-4)
    phase = t_arb / max(float(t_arb[-1]), 1e-30)
    pmin = pm.compute_total_power_from_density_w_cm2(case_a["pmin_density_w_cm2"], case_a["device_area_cm2"])
    pmax = pm.compute_total_power_from_density_w_cm2(case_a["pmax_density_w_cm2"], case_a["device_area_cm2"])
    waveform = 0.25 + 0.55 * np.sin(2.0 * np.pi * (2.2 * phase + 0.1)) ** 2
    waveform += 0.18 * ((phase > 0.32) & (phase < 0.52)).astype(float)
    waveform += 0.12 * ((phase > 0.70) & (phase < 0.83)).astype(float)
    waveform = np.clip(waveform, 0.0, 1.0)
    p_arb = pmin + (pmax - pmin) * waveform
    ideal_arb = float(case_b["R_single"]) * p_arb + pm.DARK_CURRENT_MEASURED
    arb_a = simulate_current_trace(case_a, p_arb, t_arb, include_noise=True, rng_seed=5678)
    arb_b = simulate_current_trace(case_b, p_arb, t_arb, include_noise=True, rng_seed=5678)

    plt.rcParams.update({"font.family": ["DejaVu Sans", "sans-serif"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.4), constrained_layout=True)
    axes = axes.ravel()

    ax = axes[0]
    ax.loglog(p_density, ideal_current * 1e6, color="#444444", lw=2.0, ls="--", label="Ideal linear")
    ax.loglog(p_density, ip_a * 1e6, color="#0072B2", lw=2.4, label="CaseA nonlinear steady state")
    ax.loglog(p_density, ip_b * 1e6, color="#D55E00", lw=1.7, ls=":", label="CaseB overlap")
    if ip_c is not None:
        ax.loglog(p_density, ip_c * 1e6, color=MODEL_COLORS["caseC_eta_error"], lw=1.9, ls="-.", label="CaseC eta error")
    ax.set_xlabel("Optical power density (W/cm^2)")
    ax.set_ylabel("Current (uA)")
    ax.set_title("I-P response")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(dense["rise_t_s"] * 1e3, dense["rise_caseA"], color="#0072B2", lw=2.2, label="CaseA rise")
    ax.plot(dense["rise_t_s"] * 1e3, dense["rise_caseB"], color="#D55E00", lw=2.0, ls="--", label="CaseB fitted rise")
    ax.plot(dense["fall_t_s"] * 1e3, dense["fall_caseA"], color="#009E73", lw=2.2, label="CaseA fall")
    ax.plot(dense["fall_t_s"] * 1e3, dense["fall_caseB"], color="#CC79A7", lw=2.0, ls="--", label="CaseB fitted fall")
    ax.set_xlabel("Time after edge (ms)")
    ax.set_ylabel("Normalized transient")
    ax.set_title("Two-tau transient compressed to one tau")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[2]
    ax.plot(t_noise * 1e3, (clean_a - np.mean(clean_a)) * 1e9, color="#0072B2", lw=1.8, label="Deterministic")
    ax.plot(t_noise * 1e3, (noisy_a - clean_a) * 1e9, color="#222222", lw=0.9, alpha=0.75, label="Noise trace")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current deviation (nA)")
    ax.set_title("Temporal noise at constant illumination")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[3]
    ax2 = ax.twinx()
    ax2.fill_between(t_arb * 1e3, p_arb / case_a["device_area_cm2"] * 1e3, color="#BDBDBD", alpha=0.25, label="Power")
    ax.plot(t_arb * 1e3, ideal_arb * 1e6, color="#444444", lw=1.8, ls="--", label="Ideal")
    ax.plot(t_arb * 1e3, arb_a * 1e6, color="#0072B2", lw=2.0, label="CaseA nonideal")
    ax.plot(t_arb * 1e3, arb_b * 1e6, color="#D55E00", lw=1.8, ls="--", label="CaseB nonideal")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (uA)")
    ax2.set_ylabel("Power density (mW/cm^2)")
    ax.set_title("Arbitrary waveform output")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False, fontsize=9, loc="upper left")

    device_fig = fig_dir / "fig_caseA_caseB_device_model_summary.png"
    fig.savefig(device_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig3_waveform_fig = render_fig3_style_casea_caseb_waveform(fig_dir, params_by_model)
    fig3_casec_waveform_fig = render_fig3_style_casea_casec_waveform(fig_dir, params_by_model)

    grid = fit["fit_grid"]
    weighting = fit["weighting"]
    show_weight_window = weighting["mode"] != "uniform"
    rise_cutoff_ms = weighting["rise_early_cutoff_s"] * 1e3
    fall_cutoff_ms = weighting["fall_early_cutoff_s"] * 1e3
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), constrained_layout=True)
    if show_weight_window:
        axes[0].axvspan(0.0, rise_cutoff_ms, color="#F0E442", alpha=0.16, label="Weighted window")
    axes[0].plot(grid["rise_t_s"] * 1e3, grid["rise_caseA"], "o", color="#0072B2", label="CaseA samples")
    axes[0].plot(grid["rise_t_s"] * 1e3, grid["rise_caseB"], "-", color="#D55E00", label="CaseB fit")
    axes[0].set_xlabel("Time after rising edge (ms)")
    axes[0].set_ylabel("Normalized current")
    axes[0].set_title("Rise fit")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False)

    if show_weight_window:
        axes[1].axvspan(0.0, fall_cutoff_ms, color="#F0E442", alpha=0.16, label="Weighted window")
    axes[1].plot(grid["fall_t_s"] * 1e3, grid["fall_caseA"], "o", color="#009E73", label="CaseA samples")
    axes[1].plot(grid["fall_t_s"] * 1e3, grid["fall_caseB"], "-", color="#CC79A7", label="CaseB fit")
    axes[1].set_xlabel("Time after falling edge (ms)")
    axes[1].set_ylabel("Normalized current")
    axes[1].set_title("Fall fit")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(frameon=False)
    fit_fig = fig_dir / "fig_caseA_caseB_single_tau_fit.png"
    fig.savefig(fit_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), constrained_layout=True)
    axes[0].axhline(0.0, color="#333333", lw=1.0)
    if show_weight_window:
        axes[0].axvspan(0.0, rise_cutoff_ms, color="#F0E442", alpha=0.16)
    axes[0].plot(
        grid["rise_t_s"] * 1e3,
        (grid["rise_caseB"] - grid["rise_caseA"]) * 100.0,
        color="#D55E00",
        lw=2.0,
    )
    axes[0].set_xlabel("Time after rising edge (ms)")
    axes[0].set_ylabel("CaseB - CaseA residual (% FS)")
    axes[0].set_title("Rise residual")
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(0.0, color="#333333", lw=1.0)
    if show_weight_window:
        axes[1].axvspan(0.0, fall_cutoff_ms, color="#F0E442", alpha=0.16)
    axes[1].plot(
        grid["fall_t_s"] * 1e3,
        (grid["fall_caseB"] - grid["fall_caseA"]) * 100.0,
        color="#CC79A7",
        lw=2.0,
    )
    axes[1].set_xlabel("Time after falling edge (ms)")
    axes[1].set_ylabel("CaseB - CaseA residual (% FS)")
    axes[1].set_title("Fall residual")
    axes[1].grid(True, alpha=0.25)
    residual_fig = fig_dir / "fig_caseA_caseB_fit_residual.png"
    fig.savefig(residual_fig, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "device_summary": str(device_fig),
        "fig3_style_arbitrary_waveform": str(fig3_waveform_fig),
        "single_tau_fit": str(fit_fig),
        "fit_residual": str(residual_fig),
        **({"fig3_style_caseA_caseC_eta_error_waveform": str(fig3_casec_waveform_fig)} if fig3_casec_waveform_fig else {}),
    }


def build_eval_args(args, model_key, params_csv, scenario_dir, fps, readout_cfg):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset="cifar10",
        split="test",
        generate_images=bool(args.generate_eval_images),
        run_eval=True,
        eval_cases=["nonideal"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=None,
        num_classes=0,
        results_json=str(scenario_dir / "results.json"),
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        use_noise_fn=int(bool(args.use_noise_fn)),
        target_size=32,
        output_channels=3,
        post_norm=args.post_norm,
        num_images=args.num_images,
        start_index=args.visual_start_index,
        array_size=32,
        tile_size=256,
        readout=readout_cfg["readout"],
        analog_readout=readout_cfg["analog_readout"],
        adc_enabled=int(readout_cfg["adc_enabled"]),
        power_max=pipeline.base_pipeline.DEFAULT_POWER_MAX_W,
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
        video_fps=float(fps),
        fps_sim=float(args.fps_sim),
        adc_bits=int(readout_cfg["adc_bits"]),
        adc_full_scale=None,
        adc_calibration_low=None,
        adc_calibration_high=None,
        range_mode="auto",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=args.range_calibration_samples,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=int(bool(args.shot_noise)),
        startup_dark_frames=0,
        output_dir=str(scenario_dir),
        analyze_center_pixel=0,
        drift_hours=[0.0],
        drift_aging_power_w=None,
        model_key=model_key,
    )


def run_or_load_eval(cfg, resume):
    results_path = Path(cfg.results_json)
    if bool(resume) and results_path.is_file():
        return json.loads(results_path.read_text(encoding="utf-8"))
    return pipeline.run_sequence_pipeline(cfg)


def run_accuracy_sweep(args, output_dir, params_csvs):
    records = []
    eval_dir = output_dir / "eval"
    adc_window_cache = {}
    model_keys = active_model_keys(args)
    total = len(model_keys) * len(args.fps_values) * len(eval_cmp.READOUT_CONFIGS)
    counter = 0
    for model_key in model_keys:
        for fps in args.fps_values:
            for readout_cfg in eval_cmp.READOUT_CONFIGS:
                counter += 1
                scenario = f"{model_key}_fps{int(fps):03d}_{readout_cfg['label']}"
                scenario_dir = eval_dir / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                cfg = build_eval_args(args, model_key, params_csvs[model_key], scenario_dir, fps, readout_cfg)
                if cfg.adc_enabled:
                    adc_low, adc_high = eval_cmp.estimate_adc_quantization_bounds(cfg, adc_window_cache)
                    cfg.adc_calibration_low = adc_low
                    cfg.adc_calibration_high = adc_high
                    cfg.adc_full_scale = None
                print(
                    f"[{counter}/{total}] {MODEL_DISPLAY[model_key]} | FPS={int(fps)} | {readout_cfg['display']}",
                    flush=True,
                )
                payload = run_or_load_eval(cfg, args.resume)
                accuracy = eval_cmp.accuracy_from_result(payload)
                records.append(
                    {
                        "model_key": model_key,
                        "model_display": MODEL_DISPLAY[model_key],
                        "video_fps": float(fps),
                        "readout_label": readout_cfg["label"],
                        "readout_display": readout_cfg["display"],
                        "accuracy_nonideal": float(accuracy),
                        "results_json": str(Path(cfg.results_json).resolve()),
                        "adc_calibration_low": cfg.adc_calibration_low,
                        "adc_calibration_high": cfg.adc_calibration_high,
                    }
                )
    return records


def write_accuracy_records(output_dir, records):
    csv_path = output_dir / "caseA_comparison_accuracy_records.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return str(csv_path)


def render_accuracy_chart(output_dir, records, fps_values, model_keys):
    lookup = {
        (r["model_key"], float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"])
        for r in records
    }
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_width = max(11.2 * len(model_keys), 11.2)
    fig, axes = plt.subplots(1, len(model_keys), figsize=(fig_width, 7.2), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(len(fps_values), dtype=np.float64)
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * width

    for ax, model_key in zip(axes, model_keys):
        for idx, readout_cfg in enumerate(eval_cmp.READOUT_CONFIGS):
            label = readout_cfg["label"]
            values = [lookup[(model_key, float(fps), label)] for fps in fps_values]
            bars = ax.bar(
                x + offsets[idx],
                values,
                width=width,
                color=READOUT_COLORS[label],
                edgecolor="#222222",
                linewidth=1.4,
                label=readout_cfg["display"],
            )
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() * 0.5,
                    value + 1.2,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=18,
                    fontweight="bold",
                )
        ax.set_title(MODEL_DISPLAY[model_key], fontsize=28, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"FPS {int(fps)}" for fps in fps_values], fontsize=20, fontweight="bold")
        ax.set_xlabel("Video FPS", fontsize=24, fontweight="bold")
        ax.set_ylim(0.0, 112.0)
        ax.grid(axis="y", alpha=0.18)
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
        ax.tick_params(axis="both", width=2.2, length=6, labelsize=20)
    axes[0].set_ylabel("CIFAR-10 Accuracy", fontsize=24, fontweight="bold")
    axes[-1].legend(loc="lower center", bbox_to_anchor=(-0.05, -0.22), ncol=4, frameon=False, fontsize=18)
    chart = fig_dir / "fig_caseA_comparison_accuracy_bars.png"
    fig.savefig(chart, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(11.5, 5.2), constrained_layout=True)
    for target_model in [key for key in model_keys if key != "caseA_two_tau"]:
        for readout_cfg in eval_cmp.READOUT_CONFIGS:
            label = readout_cfg["label"]
            deltas = [
                lookup[("caseA_two_tau", float(fps), label)]
                - lookup[(target_model, float(fps), label)]
                for fps in fps_values
            ]
            ax.plot(
                fps_values,
                deltas,
                marker="o",
                lw=2.2,
                color=READOUT_COLORS[label],
                ls="-" if target_model == "caseB_single_tau_fit" else "--",
                label=f"A - {MODEL_DISPLAY[target_model].replace('Case', '')} {readout_cfg['display']}",
            )
    ax.axhline(0.0, color="#222222", lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks(fps_values)
    ax.set_xticklabels([str(int(v)) for v in fps_values])
    ax.set_xlabel("Video FPS")
    ax.set_ylabel("Accuracy gain: CaseA - comparison (pp)")
    ax.set_title("Error propagation is amplified at higher FPS")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    delta_chart = fig_dir / "fig_caseA_minus_comparison_accuracy_delta.png"
    fig.savefig(delta_chart, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "accuracy_bars": str(chart),
        "accuracy_delta": str(delta_chart),
    }


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rgb_array_from_path(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def psnr_uint8(reference_path, compared_path):
    reference = rgb_array_from_path(reference_path)
    compared = rgb_array_from_path(compared_path)
    mse = float(np.mean((reference - compared) ** 2))
    if mse <= 1e-24:
        return float("inf")
    return float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def format_db(value):
    return "inf" if math.isinf(float(value)) else f"{float(value):.4f}"


def scenario_name(model_key, fps, readout_label):
    return f"{model_key}_fps{int(fps):03d}_{readout_label}"


def run_or_load_visual_export(cfg, resume):
    results_path = Path(cfg.results_json)
    manifest_path = Path(cfg.output_dir) / "manifest.json"
    if bool(resume) and results_path.is_file() and manifest_path.is_file():
        return load_json(results_path), load_json(manifest_path)
    result_payload = pipeline.run_sequence_pipeline(cfg)
    return result_payload, load_json(manifest_path)


def collect_abc_visuals(args, output_dir, params_csvs, model_keys):
    if len(model_keys) < 2:
        return {}
    visual_dir = output_dir / "abc_visuals"
    export_dir = visual_dir / "exports"
    panel_dir = visual_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    adc_window_cache = {}
    scenario_payloads = {}
    visual_rows = []
    mean_rows = []
    panel_paths = []
    total = len(model_keys) * len(args.fps_values) * len(eval_cmp.READOUT_CONFIGS)
    counter = 0
    for model_key in model_keys:
        for fps in args.fps_values:
            for readout_cfg in eval_cmp.READOUT_CONFIGS:
                counter += 1
                scenario = scenario_name(model_key, fps, readout_cfg["label"])
                scenario_dir = export_dir / scenario
                scenario_dir.mkdir(parents=True, exist_ok=True)
                cfg = build_eval_args(args, model_key, params_csvs[model_key], scenario_dir, fps, readout_cfg)
                cfg.generate_images = True
                cfg.run_eval = False
                cfg.eval_cases = ["nonideal"]
                cfg.results_json = str(scenario_dir / "visual_results.json")
                cfg.output_dir = str(scenario_dir)
                cfg.num_images = int(args.visual_num_images)
                cfg.start_index = int(args.visual_start_index)
                cfg.analyze_center_pixel = 0
                if cfg.adc_enabled:
                    adc_low, adc_high = eval_cmp.estimate_adc_quantization_bounds(cfg, adc_window_cache)
                    cfg.adc_calibration_low = adc_low
                    cfg.adc_calibration_high = adc_high
                    cfg.adc_full_scale = None
                print(
                    f"[visual {counter}/{total}] {MODEL_DISPLAY[model_key]} | FPS={int(fps)} | {readout_cfg['display']}",
                    flush=True,
                )
                _, manifest_payload = run_or_load_visual_export(cfg, args.resume)
                scenario_payloads[(model_key, float(fps), readout_cfg["label"])] = manifest_payload

    for fps in args.fps_values:
        for readout_cfg in eval_cmp.READOUT_CONFIGS:
            label = readout_cfg["label"]
            manifests = {
                model_key: scenario_payloads[(model_key, float(fps), label)]
                for model_key in model_keys
            }
            samples_by_model = {
                model_key: {
                    int(sample["dataset_index"]): sample
                    for sample in (manifest.get("samples") or [])
                }
                for model_key, manifest in manifests.items()
            }
            common_indices = sorted(set.intersection(*(set(samples.keys()) for samples in samples_by_model.values())))
            for dataset_index in common_indices:
                sample_a = samples_by_model["caseA_two_tau"][dataset_index]
                panel_entries = []
                for model_key in model_keys:
                    sample = samples_by_model[model_key][dataset_index]
                    row = {
                        "dataset_index": int(dataset_index),
                        "label": sample["label"],
                        "video_fps": float(fps),
                        "readout_label": label,
                        "readout_display": readout_cfg["display"],
                        "model_key": model_key,
                        "model_display": MODEL_DISPLAY[model_key],
                        "psnr_vs_ideal_db": float(sample["psnr_db"]),
                        "psnr_vs_caseA_image_db": (
                            float("inf")
                            if model_key == "caseA_two_tau"
                            else psnr_uint8(sample_a["nonideal_path"], sample["nonideal_path"])
                        ),
                        "nonideal_path": sample["nonideal_path"],
                    }
                    visual_rows.append(row)
                    panel_entries.append((model_key, sample, row))

                panel_path = render_abc_image_panel(
                    panel_dir=panel_dir,
                    fps=fps,
                    readout_cfg=readout_cfg,
                    dataset_index=dataset_index,
                    label_name=sample_a["label"],
                    input_path=sample_a["input_path"],
                    ideal_path=sample_a["ideal_path"],
                    panel_entries=panel_entries,
                )
                panel_paths.append(str(panel_path))

            for model_key in model_keys:
                rows = [
                    row
                    for row in visual_rows
                    if row["model_key"] == model_key
                    and float(row["video_fps"]) == float(fps)
                    and row["readout_label"] == label
                ]
                if rows:
                    mean_rows.append(
                        {
                            "video_fps": float(fps),
                            "readout_label": label,
                            "readout_display": readout_cfg["display"],
                            "model_key": model_key,
                            "model_display": MODEL_DISPLAY[model_key],
                            "mean_psnr_vs_ideal_db": float(np.mean([r["psnr_vs_ideal_db"] for r in rows])),
                            "mean_psnr_vs_caseA_image_db": (
                                float("inf")
                                if model_key == "caseA_two_tau"
                                else float(np.mean([r["psnr_vs_caseA_image_db"] for r in rows]))
                            ),
                            "num_images": len(rows),
                        }
                    )

    psnr_csv = visual_dir / "abc_image_psnr_records.csv"
    if visual_rows:
        with psnr_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(visual_rows[0].keys()))
            writer.writeheader()
            writer.writerows(visual_rows)

    mean_csv = visual_dir / "abc_image_psnr_summary.csv"
    if mean_rows:
        with mean_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(mean_rows[0].keys()))
            writer.writeheader()
            writer.writerows(mean_rows)

    md_path = visual_dir / "abc_image_visual_psnr_summary.md"
    write_abc_visual_summary(md_path, visual_rows, mean_rows, panel_paths)
    return {
        "abc_visual_dir": str(visual_dir),
        "abc_image_psnr_records": str(psnr_csv),
        "abc_image_psnr_summary": str(mean_csv),
        "abc_image_visual_psnr_summary": str(md_path),
        "abc_image_panels": panel_paths,
    }


def render_abc_image_panel(panel_dir, fps, readout_cfg, dataset_index, label_name, input_path, ideal_path, panel_entries):
    title_h = 46
    tile_images = [Image.open(input_path).convert("RGB"), Image.open(ideal_path).convert("RGB")]
    labels = [f"Input idx={dataset_index} {label_name}", "Ideal sensor"]
    for model_key, sample, row in panel_entries:
        tile_images.append(Image.open(sample["nonideal_path"]).convert("RGB"))
        labels.append(
            f"{MODEL_DISPLAY[model_key]}\nPSNRi {format_db(row['psnr_vs_ideal_db'])} dB"
            + (
                ""
                if model_key == "caseA_two_tau"
                else f" | PSNR_A {format_db(row['psnr_vs_caseA_image_db'])} dB"
            )
        )

    tile_w, tile_h = tile_images[0].size
    canvas = Image.new("RGB", (tile_w * len(tile_images), tile_h + title_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for idx, (img, text) in enumerate(zip(tile_images, labels)):
        x0 = idx * tile_w
        canvas.paste(img, (x0, title_h))
        draw.text((x0 + 6, 5), text, fill=(17, 24, 39))
    panel_path = panel_dir / f"abc_visual_idx{int(dataset_index):04d}_fps{int(fps):03d}_{readout_cfg['label']}.png"
    canvas.save(panel_path)
    return panel_path


def write_abc_visual_summary(md_path, visual_rows, mean_rows, panel_paths):
    lines = [
        "# ABC Image Visual and PSNR Summary",
        "",
        "PSNRi means PSNR between each model's nonideal raw sensor output and its own ideal sensor output.",
        "PSNR_A means PSNR between the exported nonideal image and CaseA's exported nonideal image for the same sample/FPS/readout.",
        "",
        "## Mean PSNR",
        "",
        "| FPS | Readout | Model | Images | Mean PSNRi (dB) | Mean PSNR_A image (dB) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in mean_rows:
        lines.append(
            f"| {row['video_fps']:.0f} | {row['readout_display']} | {row['model_display']} | "
            f"{row['num_images']} | {format_db(row['mean_psnr_vs_ideal_db'])} | "
            f"{format_db(row['mean_psnr_vs_caseA_image_db'])} |"
        )
    lines.extend(["", "## Visual Panels", ""])
    for path in panel_paths:
        lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "## Per-Image PSNR",
            "",
            "| Index | Label | FPS | Readout | Model | PSNRi (dB) | PSNR_A image (dB) |",
            "|---:|---|---:|---|---|---:|---:|",
        ]
    )
    for row in visual_rows:
        lines.append(
            f"| {row['dataset_index']} | {row['label']} | {row['video_fps']:.0f} | "
            f"{row['readout_display']} | {row['model_display']} | "
            f"{format_db(row['psnr_vs_ideal_db'])} | {format_db(row['psnr_vs_caseA_image_db'])} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_accuracy(records, fps_values, model_keys):
    lookup = {
        (r["model_key"], float(r["video_fps"]), r["readout_label"]): float(r["accuracy_nonideal"])
        for r in records
    }
    rows = []
    mean_by_fps = []
    for fps in fps_values:
        for target_model in [key for key in model_keys if key != "caseA_two_tau"]:
            deltas = []
            for readout_cfg in eval_cmp.READOUT_CONFIGS:
                label = readout_cfg["label"]
                acc_a = lookup[("caseA_two_tau", float(fps), label)]
                acc_target = lookup[(target_model, float(fps), label)]
                delta = acc_a - acc_target
                deltas.append(delta)
                rows.append(
                    {
                        "video_fps": float(fps),
                        "comparison_model_key": target_model,
                        "comparison_model_display": MODEL_DISPLAY[target_model],
                        "readout_label": label,
                        "readout_display": readout_cfg["display"],
                        "caseA_accuracy_percent": acc_a,
                        "comparison_accuracy_percent": acc_target,
                        "caseA_minus_comparison_pp": delta,
                    }
                )
            mean_by_fps.append(
                {
                    "video_fps": float(fps),
                    "comparison_model_key": target_model,
                    "comparison_model_display": MODEL_DISPLAY[target_model],
                    "mean_caseA_minus_comparison_pp": float(np.mean(deltas)),
                    "max_caseA_minus_comparison_pp": float(np.max(deltas)),
                    "min_caseA_minus_comparison_pp": float(np.min(deltas)),
                }
            )
    return rows, mean_by_fps


def write_rebuttal_summary(
    output_dir,
    args,
    params_by_model,
    fit,
    fit_comparison,
    figures,
    accuracy_records,
    accuracy_figures,
):
    summary_json = output_dir / "caseA_model_fit_error_propagation_summary.json"
    model_keys = active_model_keys(args)
    accuracy_rows = []
    mean_by_fps = []
    if accuracy_records:
        accuracy_rows, mean_by_fps = summarize_accuracy(accuracy_records, args.fps_values, model_keys)
    if fit["weighting"]["mode"] == "uniform":
        fit_weighting_line = "- Single-tau fitting uses ordinary uniform least squares over the sampled transient windows."
    else:
        fit_weighting_line = (
            f"- Single-tau fitting uses `{fit['weighting']['mode']}` weighted least squares; "
            f"the first {fit['weighting']['early_fraction'] * 100.0:.1f}% of each transient window "
            f"({fit['weighting']['rise_early_cutoff_s'] * 1e3:.3g} ms rise, "
            f"{fit['weighting']['fall_early_cutoff_s'] * 1e3:.3g} ms fall) has "
            f"{fit['weighting']['early_weight']:.3g}x point weight."
        )
    has_casec = "caseC_eta_error" in params_by_model
    casec = params_by_model.get("caseC_eta_error")
    param_rows = [
        f"| eta | {params_by_model['caseA_two_tau']['eta_fast']:.6g} | {params_by_model['caseB_single_tau_fit']['eta_single']:.6g} | "
        f"{casec['eta_fast']:.6g} |" if has_casec else f"| eta | {params_by_model['caseA_two_tau']['eta_fast']:.6g} | {params_by_model['caseB_single_tau_fit']['eta_single']:.6g} | - |",
        f"| rise tau fast/slow (ms) | {params_by_model['caseA_two_tau']['tau_rise_fast'] * 1e3:.6g} / {params_by_model['caseA_two_tau']['tau_rise_slow'] * 1e3:.6g} | {params_by_model['caseB_single_tau_fit']['tau_rise_single'] * 1e3:.6g} | "
        f"{casec['tau_rise_fast'] * 1e3:.6g} / {casec['tau_rise_slow'] * 1e3:.6g} |" if has_casec else f"| rise tau fast/slow (ms) | {params_by_model['caseA_two_tau']['tau_rise_fast'] * 1e3:.6g} / {params_by_model['caseA_two_tau']['tau_rise_slow'] * 1e3:.6g} | {params_by_model['caseB_single_tau_fit']['tau_rise_single'] * 1e3:.6g} | - |",
        f"| fall tau fast/slow (ms) | {params_by_model['caseA_two_tau']['tau_fall_fast'] * 1e3:.6g} / {params_by_model['caseA_two_tau']['tau_fall_slow'] * 1e3:.6g} | {params_by_model['caseB_single_tau_fit']['tau_fall_single'] * 1e3:.6g} | "
        f"{casec['tau_fall_fast'] * 1e3:.6g} / {casec['tau_fall_slow'] * 1e3:.6g} |" if has_casec else f"| fall tau fast/slow (ms) | {params_by_model['caseA_two_tau']['tau_fall_fast'] * 1e3:.6g} / {params_by_model['caseA_two_tau']['tau_fall_slow'] * 1e3:.6g} | {params_by_model['caseB_single_tau_fit']['tau_fall_single'] * 1e3:.6g} | - |",
        f"| rise/fall fast weight | {params_by_model['caseA_two_tau']['rise_fast_weight']:.6g} / {params_by_model['caseA_two_tau']['fall_fast_weight']:.6g} | - | "
        f"{casec['rise_fast_weight']:.6g} / {casec['fall_fast_weight']:.6g} |" if has_casec else f"| rise/fall fast weight | {params_by_model['caseA_two_tau']['rise_fast_weight']:.6g} / {params_by_model['caseA_two_tau']['fall_fast_weight']:.6g} | - | - |",
        f"| on-state drift amplitude | {params_by_model['caseA_two_tau']['trap_amplitude_pct']:.6g}% | same | {'same' if has_casec else '-'} |",
        f"| on-state drift tau | {params_by_model['caseA_two_tau']['trap_saturation_time_s'] * 1e3:.6g} ms | same | {'same' if has_casec else '-'} |",
        f"| noise density at 1 Hz | {params_by_model['caseA_two_tau']['noise_1f_density_1hz_a_root_hz']:.6e} A/Hz^0.5 | same | {'same' if has_casec else '-'} |",
    ]

    payload = {
        "task": "CaseA comparison model-fit error propagation rebuttal benchmark",
        "important_note": (
            "CaseA is a synthetic case1-near benchmark, not a new experimental extraction. "
            "It is designed to isolate how replacing a two-tau transient with a single-tau estimator "
            "propagates into sensor-video neural-network accuracy."
        ),
        "args": vars(args),
        "params": {
            **{
                key: {k: float(v) for k, v in params_by_model[key].items()}
                for key in model_keys
            },
            "shared_design": {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in params_by_model["shared_design"].items()
            },
        },
        "fit": {
            "caseB_tau_rise_s": fit["caseB_tau_rise_s"],
            "caseB_tau_fall_s": fit["caseB_tau_fall_s"],
            "weighting": fit["weighting"],
            "metrics": fit["metrics"],
            "weighted_metrics": fit["weighted_metrics"],
            "segment_metrics": fit["segment_metrics"],
            "comparison_to_uniform_fit": fit_comparison,
        },
        "figures": {**figures, **accuracy_figures},
        "reference_waveform_source": {
            "script": str((REPO_ROOT / "generate_fig3_arbitrary_case1_on_state_drift.py").resolve()),
            "default_output": str((REPO_ROOT / "fig3_arbitrary.png").resolve()),
            "note": "The rebuttal waveform reuses the paper Fig.3 arbitrary-waveform timing but writes separate rebuttal artifacts.",
        },
        "accuracy_table": accuracy_rows,
        "mean_delta_by_fps": mean_by_fps,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_path = output_dir / "caseA_model_fit_error_propagation_summary.md"
    lines = [
        "# Rebuttal Experiment: Fit Error and Neural-Network Propagation",
        "",
        "This experiment is intentionally saved as a new artifact directory and does not overwrite the current paper data/code.",
        "",
        "## Design",
        "",
        "- `CaseA`: synthetic case1-near full model with nonlinear I-P, two-tau rise/fall, temporal noise, and on-state drift.",
        "- `CaseB`: same non-transient model, but the two-tau transient is compressed into one fitted rise tau and one fitted fall tau.",
        *(
            [
                f"- `CaseC`: same two-tau transient/noise/drift model as CaseA, but eta is scaled by {args.casec_eta_scale:.6g} to represent nonlinear-slope fit error.",
            ]
            if "caseC_eta_error" in model_keys
            else []
        ),
        "- CaseA/CaseB isolates transient fitting error; CaseA/CaseC isolates nonlinear I-P slope error.",
        fit_weighting_line,
        "",
        "## Key Parameters",
        "",
        "| Parameter | CaseA | CaseB | CaseC |",
        "|---|---:|---:|---:|",
        *param_rows,
        "",
        "## Single-Tau Fit Residual (Unweighted Evaluation)",
        "",
        "| Component | RMSE (% FS) | MAE (% FS) | Max abs (% FS) | R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["rise", "fall", "combined"]:
        metric = fit["metrics"][key]
        lines.append(
            f"| {key} | {metric['rmse'] * 100.0:.4f} | {metric['mae'] * 100.0:.4f} | "
            f"{metric['max_abs'] * 100.0:.4f} | {metric['r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Weighted Fit Objective Residual",
            "",
            "| Component | Weighted RMSE (% FS) | Weighted MAE (% FS) | Max abs (% FS) | Weighted R2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for key in ["rise", "fall", "combined"]:
        metric = fit["weighted_metrics"][key]
        lines.append(
            f"| {key} | {metric['weighted_rmse'] * 100.0:.4f} | "
            f"{metric['weighted_mae'] * 100.0:.4f} | {metric['max_abs'] * 100.0:.4f} | "
            f"{metric['weighted_r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Early/Late Residual Split",
            "",
            "| Window | Component | RMSE (% FS) | MAE (% FS) | Max abs (% FS) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for window_key, window_name in [("early_window", "weighted early"), ("late_window", "remaining late")]:
        for key in ["rise", "fall", "combined"]:
            metric = fit["segment_metrics"][window_key][key]
            lines.append(
                f"| {window_name} | {key} | {metric['rmse'] * 100.0:.4f} | "
                f"{metric['mae'] * 100.0:.4f} | {metric['max_abs'] * 100.0:.4f} |"
            )
    if fit_comparison:
        lines.extend(
            [
                "",
                "## Early-Weighting Check Against Uniform Fit",
                "",
                "| Fit objective | Rise tau (ms) | Fall tau (ms) | Early combined RMSE (% FS) | Early max abs (% FS) | Late combined RMSE (% FS) |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        uniform_early = fit_comparison["uniform_segment_metrics"]["early_window"]["combined"]
        uniform_late = fit_comparison["uniform_segment_metrics"]["late_window"]["combined"]
        weighted_early = fit_comparison["weighted_segment_metrics"]["early_window"]["combined"]
        weighted_late = fit_comparison["weighted_segment_metrics"]["late_window"]["combined"]
        lines.append(
            f"| uniform | {fit_comparison['uniform_tau_rise_s'] * 1e3:.6g} | "
            f"{fit_comparison['uniform_tau_fall_s'] * 1e3:.6g} | "
            f"{uniform_early['rmse'] * 100.0:.4f} | {uniform_early['max_abs'] * 100.0:.4f} | "
            f"{uniform_late['rmse'] * 100.0:.4f} |"
        )
        lines.append(
            f"| early-window | {fit_comparison['weighted_tau_rise_s'] * 1e3:.6g} | "
            f"{fit_comparison['weighted_tau_fall_s'] * 1e3:.6g} | "
            f"{weighted_early['rmse'] * 100.0:.4f} | {weighted_early['max_abs'] * 100.0:.4f} | "
            f"{weighted_late['rmse'] * 100.0:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "Reference arbitrary-waveform figure source:",
            f"- script: `{REPO_ROOT / 'generate_fig3_arbitrary_case1_on_state_drift.py'}`",
            f"- default output: `{REPO_ROOT / 'fig3_arbitrary.png'}`",
            "- this run writes separate rebuttal waveform figures and does not overwrite the paper figure.",
            "",
        ]
    )
    for key, value in {**figures, **accuracy_figures}.items():
        lines.append(f"- `{key}`: `{value}`")

    if accuracy_rows:
        lines.extend(
            [
                "",
                "## Accuracy Propagation",
                "",
                "| FPS | Comparison | Readout | CaseA (%) | Comparison (%) | CaseA - comparison (pp) |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for row in accuracy_rows:
            lines.append(
                f"| {row['video_fps']:.0f} | {row['comparison_model_display']} | {row['readout_display']} | "
                f"{row['caseA_accuracy_percent']:.4f} | {row['comparison_accuracy_percent']:.4f} | "
                f"{row['caseA_minus_comparison_pp']:+.4f} |"
            )
        lines.extend(
            [
                "",
                "| FPS | Comparison | Mean gain CaseA - comparison (pp) | Max gain (pp) |",
                "|---:|---|---:|---:|",
            ]
        )
        for row in mean_by_fps:
            lines.append(
                f"| {row['video_fps']:.0f} | {row['comparison_model_display']} | "
                f"{row['mean_caseA_minus_comparison_pp']:+.4f} | {row['max_caseA_minus_comparison_pp']:+.4f} |"
            )
    lines.extend(
        [
            "",
            "## Draft Rebuttal Text",
            "",
            (
                "To quantify model-fitting error and its propagation, we added a controlled CaseA/CaseB/CaseC study. "
                "CaseA is a case1-near full sensor model including nonlinear responsivity, a two-time-constant "
                "rise/fall transient, temporal noise, and on-state drift. CaseB keeps the same nonlinear, noise, "
                "and drift parameters, but approximates the CaseA transient by fitting only one rise and one fall "
                "time constant. CaseC keeps the CaseA transient/noise/drift model unchanged, but perturbs the "
                "nonlinear I-P slope eta. The residuals therefore isolate either transient-fit error or "
                "nonlinear-slope fit error."
            ),
            "",
            (
                "After propagating the comparison models through the same video-sensor and ResNet18 CIFAR-10 validation flow, "
                "the accuracy gaps quantify how each device-level fitting error is seen by the downstream temporal "
                "sensor pipeline. The CaseB comparison captures transient-compression error, while the CaseC "
                "comparison captures nonlinear I-P slope error with the same transient dynamics."
            ),
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(summary_json), str(md_path)


def main():
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    params_dir = output_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)

    case1_results = pm.extract_case1_single_carrier_params(pm.CASE1_DATA_DIR, device_area_cm2=pm.DEVICE_AREA_CM2)
    fit = fit_single_tau_to_two_tau(args)
    fit_comparison = compare_to_uniform_fit(args, fit)
    params_payload = build_case_params(args, case1_results, fit)
    model_keys = active_model_keys(args)
    params_by_model = {key: params_payload[key] for key in model_keys}
    params_by_model["shared_design"] = params_payload["shared_design"]

    params_csvs = {
        "caseA_two_tau": params_dir / "caseA_two_tau_full_model.csv",
        "caseB_single_tau_fit": params_dir / "caseB_single_tau_fit_model.csv",
        "caseC_eta_error": params_dir / "caseC_eta_error_model.csv",
    }
    for model_key in model_keys:
        write_parameter_csv(
            params_csvs[model_key],
            params_by_model[model_key],
            parameter_keys_for_model(model_key),
        )

    figures = render_device_figures(output_dir, params_by_model, fit)

    records = []
    accuracy_figures = {}
    records_csv = None
    if bool(args.run_eval):
        records = run_accuracy_sweep(args, output_dir, params_csvs)
        records_csv = write_accuracy_records(output_dir, records)
        accuracy_figures = render_accuracy_chart(output_dir, records, args.fps_values, model_keys)
    visual_outputs = {}
    if bool(args.generate_abc_visuals):
        visual_outputs = collect_abc_visuals(args, output_dir, params_csvs, model_keys)

    summary_json, summary_md = write_rebuttal_summary(
        output_dir=output_dir,
        args=args,
        params_by_model=params_by_model,
        fit=fit,
        fit_comparison=fit_comparison,
        figures=figures,
        accuracy_records=records,
        accuracy_figures={**accuracy_figures, **visual_outputs},
    )

    print("\nCaseA comparison rebuttal experiment complete")
    print(f"Output directory: {output_dir}")
    for model_key in model_keys:
        print(f"{MODEL_DISPLAY[model_key]} params: {params_csvs[model_key]}")
    if records_csv:
        print(f"Accuracy records: {records_csv}")
    if visual_outputs:
        print(f"ABC visual PSNR summary: {visual_outputs['abc_image_visual_psnr_summary']}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary Markdown: {summary_md}")
    print("Fit metrics:")
    print(json.dumps(fit["metrics"], indent=2))


if __name__ == "__main__":
    main()
