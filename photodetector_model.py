"""
Photodetector Non-ideal Effects: modeling -> parameter extraction -> arbitrary-waveform simulation

The complete model includes six classes of non-ideal effects:
  1. Multi-time-constant carrier response (fast + slow components)
  2. Rise/fall asymmetry
  3. I-P nonlinearity: target = Rᵢ · P_ref · (P / P_ref)^ηᵢ
  4. Trap-state drift + persistent photocurrent
  5. Long-term response attenuation drift
  6. Output noise: user-defined time-varying noise function noise(t)

State equations:
  dx₁ᵢ/dt = (Rᵢ·P_ref·(P/P_ref)^ηᵢ - x₁ᵢ) / τᵢ,   τᵢ = τ_rise_i or τ_fall_i
  dx₂/dt  = α·u(P)·(1-x₂) - β·x₂
  dx₃/dt  = (s_drift·γ - x₃) / τ_drift

  I_photo,base = Σᵢ x₁ᵢ + δ·x₂
  I_photo = I_photo,base · max(1 + x₃, 0)
  I_det   = I_dark + I_photo
  I_out   = I_det + noise_fn(t, I_det, P)   (measured output with noise)

Notes:
  For each fast/slow carrier component, tau_rise_i and tau_fall_i are independent parameters.
  tau_fall_i is not derived from, scaled from, or shared with tau_rise_i.
  u(P) defaults to the optical power itself, but can also be switched to a binary
  illumination gate so that trap capture/release rates do not depend on power amplitude.
"""

import argparse
import csv
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.signal import welch

_MPLCONFIGDIR = Path(__file__).resolve().parent / ".mplconfig"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({
    "font.family": ["DejaVu Sans", "sans-serif"],
    "axes.unicode_minus": False,
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.frameon": True,
    "legend.framealpha": 0.92,
    "axes.linewidth": 1.9,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "xtick.minor.width": 1.2,
    "ytick.minor.width": 1.2,
    "xtick.major.size": 6.5,
    "ytick.major.size": 6.5,
    "xtick.minor.size": 3.5,
    "ytick.minor.size": 3.5,
    "lines.linewidth": 2.2,
    "lines.markersize": 6.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})

SPINE_WIDTH = 1.9
GRID_ALPHA = 0.22
LW_HEAVY = 2.8
LW_MAIN = 2.3
LW_MED = 1.9
LW_LIGHT = 1.5
LW_FINE = 1.1
MS_MAIN = 6.5
MS_SMALL = 3.5
TABLE_EDGE_COLOR = "#7C8A98"
TABLE_HEADER_COLOR = "#D7EAF7"
TABLE_ALT_COLOR = "#F7FAFD"

CARRIER_PARAM_LAYOUT = [
    ("R_fast", "eta_fast", "tau_rise_fast", "tau_fall_fast"),
    ("R_slow", "eta_slow", "tau_rise_slow", "tau_fall_slow"),
]
SINGLE_CARRIER_PARAM_LAYOUT = [
    ("R_single", "eta_single", "tau_rise_single", "tau_fall_single"),
]
GLOBAL_PARAM_KEYS = ["alpha", "beta", "delta", "gamma", "tau_drift", "drift_scale"]
NONLINEAR_POWER_REF_W = 1e-3
ARBITRARY_STARTUP_DARK_S = 20e-3
DEFAULT_DEVICE_AREA_CM2 = 0.16
DEFAULT_PRANGE1_DENSITY_W_CM2 = 5e-5
DEFAULT_PRANGE2_DENSITY_W_CM2 = 5e-4
DEFAULT_NOISE_1F_DENSITY_1HZ_AHZ05 = 1e-8
CASE1_DATA_DIR = Path(__file__).resolve().parent / "data" / "case1"
CASE2_DATA_DIR = Path(__file__).resolve().parent / "data" / "case2"
CASE2_DEFAULT_DEVICE_DIAMETER_MM = 25.0
CASE2_DEFAULT_NOISE_SCALE = 5.0
CASE2_DEFAULT_FN_SCALE = 1.0
CASE2_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "case2_fit"
CASE2_FIXED_RISE_TIME_MS = 84.0
CASE2_FIXED_FALL_TIME_MS = 243.0
CASE2_MAIN_BRANCH_CEILING_UA = 6.65


def compute_power_ref_from_density(power_density_mw_cm2, device_area_cm2):
    power_density_mw_cm2 = np.asarray(power_density_mw_cm2, dtype=float)
    positive = power_density_mw_cm2[power_density_mw_cm2 > 0]
    if positive.size == 0:
        raise ValueError("power density data must contain positive values")
    return float(np.min(positive) * 1e-3 * float(device_area_cm2))


def compute_circular_device_area_mm2(diameter_mm):
    diameter_mm = float(diameter_mm)
    if diameter_mm <= 0:
        raise ValueError("diameter_mm must be positive")
    return float(math.pi * (0.5 * diameter_mm) ** 2)


def convert_area_mm2_to_cm2(area_mm2):
    return float(area_mm2) / 100.0


def compute_total_power_from_density_w_mm2(power_density_w_mm2, device_area_mm2):
    power_density_w_mm2 = np.asarray(power_density_w_mm2, dtype=float)
    return power_density_w_mm2 * float(device_area_mm2)


def compute_total_power_from_density_w_cm2(power_density_w_cm2, device_area_cm2):
    power_density_w_cm2 = np.asarray(power_density_w_cm2, dtype=float)
    return power_density_w_cm2 * float(device_area_cm2)


def infer_n_carrier_from_params(params):
    if isinstance(params, dict):
        if "R_single" in params:
            return 1
        if "R_fast" in params:
            return N_CARRIER
    size = np.asarray(params, dtype=float).size
    if size == len(SINGLE_PARAM_KEYS):
        return 1
    if size == len(PARAM_KEYS):
        return N_CARRIER
    raise ValueError(f"Cannot infer carrier count from parameter payload of size {size}")


if CASE1_DATA_DIR.exists():
    try:
        _case1_ip = np.loadtxt(CASE1_DATA_DIR / "IoptAcm2_PmWcm2_0.16cm2.csv", delimiter=",")
        NONLINEAR_POWER_REF_W = compute_power_ref_from_density(_case1_ip[:, 0], DEFAULT_DEVICE_AREA_CM2)
    except Exception:
        pass


def set_device_context(device_area_cm2=None, power_ref_w=None):
    global DEVICE_AREA_CM2, NONLINEAR_POWER_REF_W
    if device_area_cm2 is not None:
        DEVICE_AREA_CM2 = float(device_area_cm2)
    if power_ref_w is not None:
        NONLINEAR_POWER_REF_W = float(power_ref_w)

# ============================================================
# 1. Device model
# ============================================================

def prepare_model_config(params, n_carrier=len(CARRIER_PARAM_LAYOUT),
                         trap_mode="power", trap_threshold_w=0.0,
                         trap_output_mode="always", power_min_w=0.0,
                         power_max_w=None, trap_saturation_time_s=None,
                         trap_amplitude_ratio=None,
                         trap_delta_r_ratio=None,
                         trap_x1_ratio=None,
                         trap_x2_reference_state=None,
                         trap_x2_tau_on_s=None,
                         trap_x2_tau_off_s=None):
    """
    Parse a parameter vector into a reusable model configuration.
    """
    params = np.asarray(params, dtype=float)
    nc4 = 4 * n_carrier
    expected = nc4 + 6
    if params.size != expected:
        raise ValueError(f"expected {expected} params, got {params.size}")
    carrier_params = params[:nc4].reshape(n_carrier, 4)
    alpha, beta, delta, gamma, tau_drift, drift_scale = params[nc4:]
    power_min_w = max(0.0, float(power_min_w))
    if power_max_w is None:
        power_max_w = float("inf")
    power_max_w = float(power_max_w)
    if power_max_w < power_min_w:
        raise ValueError(
            f"power_max_w must be >= power_min_w, got {power_max_w:.6e} < {power_min_w:.6e}"
        )
    trap_saturation_time_s = (
        None if trap_saturation_time_s is None else max(float(trap_saturation_time_s), 1e-30)
    )
    trap_amplitude_ratio = 0.0 if trap_amplitude_ratio is None else float(trap_amplitude_ratio)
    trap_delta_r_ratio = 0.0 if trap_delta_r_ratio is None else float(trap_delta_r_ratio)
    trap_x1_ratio = 0.0 if trap_x1_ratio is None else float(trap_x1_ratio)
    trap_x2_reference_state = (
        0.0 if trap_x2_reference_state is None else float(trap_x2_reference_state)
    )
    trap_x2_tau_on_s = 0.0 if trap_x2_tau_on_s is None else float(trap_x2_tau_on_s)
    trap_x2_tau_off_s = 0.0 if trap_x2_tau_off_s is None else float(trap_x2_tau_off_s)
    trap_ratio_mode = trap_saturation_time_s is not None
    return {
        "params_vec": params.copy(),
        "n_carrier": n_carrier,
        "carrier_params": carrier_params,
        "alpha": float(alpha),
        "beta": float(beta),
        "delta": float(delta),
        "gamma": float(gamma),
        "tau_drift": float(tau_drift),
        "drift_scale": float(drift_scale),
        "power_ref_w": float(NONLINEAR_POWER_REF_W),
        "power_min_w": power_min_w,
        "power_max_w": power_max_w,
        "trap_mode": str(trap_mode),
        "trap_threshold_w": float(trap_threshold_w),
        "trap_output_mode": str(trap_output_mode),
        "trap_ratio_mode": bool(trap_ratio_mode),
        "trap_saturation_time_s": trap_saturation_time_s,
        "trap_amplitude_ratio": trap_amplitude_ratio,
        "trap_delta_r_ratio": max(0.0, trap_delta_r_ratio),
        "trap_x1_ratio": max(0.0, trap_x1_ratio),
        "trap_x2_reference_state": max(0.0, trap_x2_reference_state),
        "trap_x2_tau_on_s": max(0.0, trap_x2_tau_on_s),
        "trap_x2_tau_off_s": max(0.0, trap_x2_tau_off_s),
    }


def init_state_arrays(shape, n_carrier=len(CARRIER_PARAM_LAYOUT), dtype=float):
    """
    Initialize state variables for a detector array of arbitrary shape.
    """
    if isinstance(shape, int):
        base_shape = (shape,)
    else:
        base_shape = tuple(shape)
    x1 = np.zeros((n_carrier, *base_shape), dtype=dtype)
    x2 = np.zeros(base_shape, dtype=dtype)
    x3 = np.zeros(base_shape, dtype=dtype)
    return x1, x2, x3


def carrier_target_current(P, Ri, eta_i, power_ref_w=NONLINEAR_POWER_REF_W):
    """
    Power-law steady-state target current.

    I = R_i · P_ref · (P / P_ref)^η_i

    P_ref is the crossover point between the ideal and non-ideal models:
    at P = P_ref the two currents are equal.
    When P > P_ref and η < 1, the non-ideal current is lower than the
    ideal current, corresponding to high-intensity saturation.
    R_i keeps units of A/W, i.e. the responsivity at P_ref.
    """
    if power_ref_w <= 0:
        raise ValueError("power_ref_w must be positive")
    P = np.asarray(P, dtype=float)
    P_pos = np.clip(P, 0.0, None)
    normalized_power = np.maximum(P_pos / power_ref_w, 1e-20)
    target = Ri * power_ref_w * np.power(normalized_power, eta_i)
    return np.where(P_pos > 0.0, target, 0.0)


def clip_model_power(P, model_config):
    power = np.asarray(P, dtype=float)
    power = np.clip(power, 0.0, None)
    power_min_w = float(model_config.get("power_min_w", 0.0))
    power_max_w = float(model_config.get("power_max_w", float("inf")))
    return np.clip(power, power_min_w, power_max_w)


def _normalized_branch_weights(model_config, key):
    weights = np.asarray(model_config[key], dtype=float)
    if weights.ndim != 1:
        raise ValueError(f"{key} must be a 1D branch-weight array")
    if np.any(weights < 0.0):
        raise ValueError(f"{key} cannot contain negative values")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{key} must contain at least one positive value")
    return weights / total


def _carrier_target_total(response_power, model_config):
    total = None
    for Ri, eta_i, _tau_r, _tau_f in model_config["carrier_params"]:
        target = carrier_target_current(
            response_power,
            Ri,
            eta_i,
            power_ref_w=model_config["power_ref_w"],
        )
        if total is None:
            total = np.zeros_like(target, dtype=float)
        total = total + target
    if total is None:
        total = np.zeros_like(response_power, dtype=float)
    return total


def _trap_current_scale(model_config, P):
    trap_delta_r_ratio = float(model_config.get("trap_delta_r_ratio", 0.0))
    if trap_delta_r_ratio <= 0.0:
        return model_config["delta"]
    if P is None:
        raise ValueError("P is required when trap_delta_r_ratio is enabled")
    response_power = clip_model_power(P, model_config)
    main_target_current = _carrier_target_total(response_power, model_config)
    responsivity = np.divide(
        main_target_current,
        np.maximum(response_power, 1e-30),
        out=np.zeros_like(main_target_current, dtype=float),
        where=np.maximum(response_power, 1e-30) > 0.0,
    )
    return trap_delta_r_ratio * responsivity


def _trap_x1_envelope(model_config, x2):
    x2 = np.asarray(x2, dtype=float)
    reference_state = float(model_config.get("trap_x2_reference_state", 0.0))
    if reference_state <= 0.0:
        return x2
    return np.clip(x2 / reference_state, 0.0, 1.0)


def _uses_trap_plateau_ratio_current(model_config):
    return (
        float(model_config.get("trap_x1_ratio", 0.0)) > 0.0
        and float(model_config.get("trap_x2_tau_on_s", 0.0)) > 0.0
        and float(model_config.get("trap_x2_tau_off_s", 0.0)) > 0.0
    )


def _step_rise_fall_weighted_carriers(response_power, dt, model_config, x1):
    rise_weights = _normalized_branch_weights(model_config, "transient_rise_weights")
    fall_weights = _normalized_branch_weights(model_config, "transient_fall_weights")
    carrier_params = list(model_config["carrier_params"])
    if len(carrier_params) != len(rise_weights) or len(carrier_params) != len(fall_weights):
        raise ValueError("transient branch weights must match carrier count")

    target_total = _carrier_target_total(response_power, model_config)
    current_total = np.sum(x1, axis=0)
    rising = target_total >= current_total

    previous_rising = model_config.get("_transient_previous_rising")
    if previous_rising is None or np.shape(previous_rising) != np.shape(rising):
        previous_rising = rising
    else:
        previous_rising = np.asarray(previous_rising, dtype=bool)
    switched_direction = np.not_equal(previous_rising, rising)

    x1_next = np.empty_like(x1)
    for i, (_Ri, _eta_i, tau_r, tau_f) in enumerate(carrier_params):
        branch_weight = np.where(rising, rise_weights[i], fall_weights[i])
        branch_target = target_total * branch_weight
        branch_start = np.where(switched_direction, current_total * branch_weight, x1[i])
        decay_r = np.exp(-dt / tau_r)
        decay_f = np.exp(-dt / tau_f)
        decay = np.where(rising, decay_r, decay_f)
        x1_next[i] = branch_start * decay + branch_target * (1.0 - decay)

    model_config["_transient_previous_rising"] = np.asarray(rising, dtype=bool).copy()
    return x1_next


def step_model_state(P, dt, model_config, x1, x2, x3):
    """
    Advance the model by one time step.
    Works for scalars, waveforms, or 2D pixel arrays.
    """
    P = np.asarray(P, dtype=float)
    response_power = clip_model_power(P, model_config)

    if model_config.get("transient_weight_mode") == "rise_fall":
        x1_next = _step_rise_fall_weighted_carriers(response_power, dt, model_config, x1)
    else:
        x1_next = np.empty_like(x1)
        for i, (Ri, eta_i, tau_r, tau_f) in enumerate(model_config["carrier_params"]):
            target = carrier_target_current(
                response_power,
                Ri,
                eta_i,
                power_ref_w=model_config["power_ref_w"],
            )
            decay_r = np.exp(-dt / tau_r)
            decay_f = np.exp(-dt / tau_f)
            decay = np.where(target >= x1[i], decay_r, decay_f)
            x1_next[i] = x1[i] * decay + target * (1.0 - decay)

    if _uses_trap_plateau_ratio_current(model_config):
        target_trap_current = float(model_config["trap_x1_ratio"]) * _carrier_target_total(
            response_power,
            model_config,
        )
        trap_threshold_w = max(
            float(model_config.get("trap_threshold_w", 0.0)),
            float(model_config.get("power_min_w", 0.0)),
        )
        illuminated = np.asarray(P, dtype=float) > trap_threshold_w
        target_trap_current = np.where(illuminated, target_trap_current, 0.0)
        tau_on = max(float(model_config["trap_x2_tau_on_s"]), 1e-30)
        tau_off = max(float(model_config["trap_x2_tau_off_s"]), 1e-30)
        tau_x2 = np.where(target_trap_current >= x2, tau_on, tau_off)
        decay_x2 = np.exp(-dt / tau_x2)
        x2_next = x2 * decay_x2 + target_trap_current * (1.0 - decay_x2)
    elif model_config.get("trap_ratio_mode"):
        trap_threshold_w = max(
            float(model_config.get("trap_threshold_w", 0.0)),
            float(model_config.get("power_min_w", 0.0)),
        )
        illuminated = np.asarray(P, dtype=float) > trap_threshold_w
        decay_trap = np.exp(-dt / model_config["trap_saturation_time_s"])
        target_x2 = np.where(illuminated, 1.0, 0.0)
        x2_next = x2 * decay_trap + target_x2 * (1.0 - decay_trap)
    elif model_config["trap_mode"] == "binary":
        illuminated = P > model_config["trap_threshold_w"]
        decay_on = np.exp(-dt * model_config["alpha"])
        decay_off = np.exp(-dt * model_config["beta"])
        x2_next = np.where(
            illuminated,
            1.0 - (1.0 - x2) * decay_on,
            x2 * decay_off,
        )
    else:
        trap_drive = np.clip(P, 0.0, None)
        capture_rate = model_config["alpha"] * trap_drive
        total_rate = capture_rate + model_config["beta"]
        target_x2 = np.divide(
            capture_rate,
            np.maximum(total_rate, 1e-30),
            out=np.zeros_like(capture_rate, dtype=float),
            where=np.maximum(total_rate, 1e-30) > 0.0,
        )
        decay_x2 = np.exp(-dt * total_rate)
        x2_next = x2 * decay_x2 + target_x2 * (1.0 - decay_x2)
    decay_drift = np.exp(-dt / model_config["tau_drift"])
    x3_next = x3 * decay_drift + model_config["drift_scale"] * model_config["gamma"] * (1.0 - decay_drift)
    return x1_next, x2_next, x3_next


def current_from_state(model_config, x1, x2, x3, P=None, dark_current=0.0):
    """
    Compute the photocurrent and total output current from the state variables.
    """
    main_photo_current = x1.sum(axis=0)
    if _uses_trap_plateau_ratio_current(model_config):
        trap_current = np.asarray(x2, dtype=float)
    elif model_config.get("trap_ratio_mode"):
        trap_current = model_config["trap_amplitude_ratio"] * main_photo_current * x2
    elif float(model_config.get("trap_x1_ratio", 0.0)) > 0.0:
        trap_current = (
            float(model_config["trap_x1_ratio"])
            * main_photo_current
            * _trap_x1_envelope(model_config, x2)
        )
    else:
        trap_current = _trap_current_scale(model_config, P) * x2
    if model_config.get("trap_output_mode") == "illumination_gated":
        if P is None:
            raise ValueError("P is required when trap_output_mode='illumination_gated'")
        trap_gate = np.where(np.asarray(P, dtype=float) > model_config["trap_threshold_w"], 1.0, 0.0)
        trap_current = trap_current * trap_gate
    I_photo_base = main_photo_current + trap_current
    drift_multiplier = np.clip(1.0 + x3, 0.0, None)
    I_photo = I_photo_base * drift_multiplier
    I_det = dark_current + I_photo
    return I_det, I_photo


def steady_state_current_from_power(P, model_config, dark_current=0.0, include_drift=False):
    P = np.asarray(P, dtype=float)
    response_power = clip_model_power(P, model_config)
    main_photo_current = None
    for Ri, eta_i, _tau_r, _tau_f in model_config["carrier_params"]:
        target_current = carrier_target_current(
            response_power,
            Ri,
            eta_i,
            power_ref_w=model_config["power_ref_w"],
        )
        if main_photo_current is None:
            main_photo_current = np.zeros_like(target_current, dtype=float)
        main_photo_current = main_photo_current + target_current

    if main_photo_current is None:
        main_photo_current = np.zeros_like(response_power, dtype=float)

    if _uses_trap_plateau_ratio_current(model_config):
        trap_threshold_w = max(
            float(model_config.get("trap_threshold_w", 0.0)),
            float(model_config.get("power_min_w", 0.0)),
        )
        illuminated = np.asarray(P, dtype=float) > trap_threshold_w
        trap_current = (
            float(model_config["trap_x1_ratio"])
            * main_photo_current
            * illuminated.astype(float)
        )
    elif model_config.get("trap_ratio_mode"):
        trap_threshold_w = max(
            float(model_config.get("trap_threshold_w", 0.0)),
            float(model_config.get("power_min_w", 0.0)),
        )
        illuminated = np.asarray(P, dtype=float) > trap_threshold_w
        trap_current = model_config["trap_amplitude_ratio"] * main_photo_current * illuminated.astype(float)
    elif float(model_config.get("trap_x1_ratio", 0.0)) > 0.0:
        if model_config["trap_mode"] == "binary":
            x2_ss = (np.asarray(P, dtype=float) > model_config["trap_threshold_w"]).astype(float)
        else:
            trap_drive = np.clip(np.asarray(P, dtype=float), 0.0, None)
            capture_rate = model_config["alpha"] * trap_drive
            total_rate = capture_rate + model_config["beta"]
            x2_ss = np.divide(
                capture_rate,
                np.maximum(total_rate, 1e-30),
                out=np.zeros_like(capture_rate, dtype=float),
                where=np.maximum(total_rate, 1e-30) > 0.0,
            )
        trap_current = (
            float(model_config["trap_x1_ratio"])
            * main_photo_current
            * _trap_x1_envelope(model_config, x2_ss)
        )
    elif model_config["trap_mode"] == "binary":
        illuminated = np.asarray(P, dtype=float) > model_config["trap_threshold_w"]
        trap_current = _trap_current_scale(model_config, P) * illuminated.astype(float)
    else:
        trap_drive = np.clip(np.asarray(P, dtype=float), 0.0, None)
        capture_rate = model_config["alpha"] * trap_drive
        total_rate = capture_rate + model_config["beta"]
        x2_ss = np.divide(
            capture_rate,
            np.maximum(total_rate, 1e-30),
            out=np.zeros_like(capture_rate, dtype=float),
            where=np.maximum(total_rate, 1e-30) > 0.0,
        )
        trap_current = _trap_current_scale(model_config, P) * x2_ss

    if model_config.get("trap_output_mode") == "illumination_gated":
        trap_gate = np.where(np.asarray(P, dtype=float) > model_config["trap_threshold_w"], 1.0, 0.0)
        trap_current = trap_current * trap_gate

    drift_multiplier = 1.0
    if include_drift:
        drift_multiplier = np.clip(
            1.0 + model_config["drift_scale"] * model_config["gamma"],
            0.0,
            None,
        )

    I_photo = (main_photo_current + trap_current) * drift_multiplier
    return dark_current + I_photo


def sample_shot_thermal_noise(I_det, rng, bandwidth, i_thermal, shot_noise=True):
    """
    Generic shot + thermal noise sampler.
    """
    noise = np.zeros_like(I_det, dtype=float)
    if shot_noise:
        q = 1.602e-19
        sigma_shot = np.sqrt(2 * q * np.abs(I_det) * bandwidth)
        noise += rng.normal(0.0, 1.0, size=np.shape(I_det)) * sigma_shot

    sigma_th = np.sqrt(i_thermal**2 * bandwidth)
    noise += rng.normal(0.0, sigma_th, size=np.shape(I_det))
    return noise


def _noise_fn_closure_params(noise_fn):
    try:
        closure_vars = inspect.getclosurevars(noise_fn)
    except Exception:
        return None

    params = closure_vars.nonlocals
    required = {"white_sigma", "flicker_sigma", "kappa", "low_freq_amp", "low_freq_hz"}
    if required.issubset(params):
        return {key: float(params[key]) for key in required}
    psd_required = {
        "freqs_hz",
        "noise_density_ahz05",
        "white_sigma",
        "flicker_sigma",
        "kappa",
        "low_freq_amp",
        "low_freq_hz",
    }
    if psd_required.issubset(params):
        out = {key: params[key] for key in psd_required}
        out["freqs_hz"] = np.asarray(out["freqs_hz"], dtype=float)
        out["noise_density_ahz05"] = np.asarray(out["noise_density_ahz05"], dtype=float)
        for key in ["white_sigma", "flicker_sigma", "kappa", "low_freq_amp", "low_freq_hz"]:
            out[key] = float(out[key])
        return out
    return None


def _sample_make_noise_function_trace(params, t, signal_shape, rng):
    n_steps = len(t)
    noise = np.zeros(signal_shape, dtype=float)
    n_pixels = int(np.prod(signal_shape[1:])) if len(signal_shape) > 1 else 1
    flat_noise = noise.reshape(n_steps, n_pixels)

    if params["white_sigma"] > 0:
        flat_noise += rng.normal(0.0, params["white_sigma"], size=flat_noise.shape)

    if params["low_freq_amp"] > 0:
        low_freq = params["low_freq_amp"] * np.sin(2 * np.pi * params["low_freq_hz"] * t)
        flat_noise += low_freq[:, None]

    if params["flicker_sigma"] > 0 and n_steps > 1:
        dt = t[1] - t[0]
        white = rng.normal(0.0, 1.0, size=flat_noise.shape)
        freqs = np.fft.rfftfreq(n_steps, d=dt)
        if len(freqs) > 1:
            freqs[0] = freqs[1]
        else:
            freqs[0] = 1.0
        filt = 1.0 / np.power(freqs, 0.5 * params["kappa"])
        colored = np.fft.irfft(np.fft.rfft(white, axis=0) * filt[:, None], n=n_steps, axis=0)
        colored -= np.mean(colored, axis=0, keepdims=True)
        std_colored = np.std(colored, axis=0, keepdims=True)
        std_colored[std_colored <= 0] = 1.0
        colored = colored / std_colored * params["flicker_sigma"]
        flat_noise += colored

    return noise


def _resample_noise_density(freqs_target, freqs_src, nd_src):
    freqs_target = np.asarray(freqs_target, dtype=float)
    freqs_src = np.asarray(freqs_src, dtype=float)
    nd_src = np.asarray(nd_src, dtype=float)
    positive = freqs_src > 0
    if np.count_nonzero(positive) < 2:
        return np.full_like(freqs_target, float(nd_src[-1]))
    log_interp = np.interp(
        np.log10(np.maximum(freqs_target, freqs_src[positive][0])),
        np.log10(freqs_src[positive]),
        np.log10(np.maximum(nd_src[positive], 1e-30)),
        left=np.log10(max(nd_src[positive][0], 1e-30)),
        right=np.log10(max(nd_src[positive][-1], 1e-30)),
    )
    return np.power(10.0, log_interp)


def _sample_psd_noise_trace(params, t, signal_shape, rng):
    n_steps = len(t)
    noise = np.zeros(signal_shape, dtype=float)
    n_pixels = int(np.prod(signal_shape[1:])) if len(signal_shape) > 1 else 1
    flat_noise = noise.reshape(n_steps, n_pixels)
    if n_steps <= 1:
        return noise

    dt = float(t[1] - t[0])
    fft_freqs = np.fft.rfftfreq(n_steps, d=dt)
    fft_freqs_safe = fft_freqs.copy()
    if len(fft_freqs_safe) > 1:
        fft_freqs_safe[0] = fft_freqs_safe[1]
    else:
        fft_freqs_safe[0] = 1.0

    target_nd = _resample_noise_density(
        fft_freqs_safe,
        params["freqs_hz"],
        params["noise_density_ahz05"],
    )
    df = 1.0 / max(n_steps * dt, 1e-30)
    coeff_sigma = target_nd * n_steps * np.sqrt(np.maximum(df / 2.0, 1e-30))
    random_complex = (
        rng.normal(0.0, 1.0, size=(len(fft_freqs), n_pixels))
        + 1j * rng.normal(0.0, 1.0, size=(len(fft_freqs), n_pixels))
    ) / np.sqrt(2.0)
    random_complex[0, :] = 0.0
    if len(fft_freqs) > 1:
        random_complex[-1, :] = rng.normal(0.0, 1.0, size=n_pixels)
    spectrum = random_complex * coeff_sigma[:, None]
    colored = np.fft.irfft(spectrum, n=n_steps, axis=0)
    colored -= np.mean(colored, axis=0, keepdims=True)
    flat_noise += colored

    if params["low_freq_amp"] > 0:
        low_freq = params["low_freq_amp"] * np.sin(2 * np.pi * params["low_freq_hz"] * t)
        flat_noise += low_freq[:, None]

    return noise


def build_combined_noise_density(
    freqs_hz,
    shot_noise_density_ahz05=0.0,
    flicker_noise_density_1hz_ahz05=0.0,
):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    shot_noise_density_ahz05 = max(float(shot_noise_density_ahz05), 0.0)
    flicker_noise_density_1hz_ahz05 = max(float(flicker_noise_density_1hz_ahz05), 0.0)

    density = np.full_like(freqs_hz, shot_noise_density_ahz05, dtype=float)
    positive = freqs_hz > 0
    if np.any(positive) and flicker_noise_density_1hz_ahz05 > 0:
        density[positive] = np.sqrt(
            density[positive] ** 2
            + (flicker_noise_density_1hz_ahz05 ** 2) / np.maximum(freqs_hz[positive], 1e-30)
        )
        density[~positive] = density[positive][0]
    return density


def sample_combined_psd_noise_trace(
    t,
    shot_noise_density_ahz05=0.0,
    flicker_noise_density_1hz_ahz05=0.0,
    rng=None,
):
    return sample_combined_psd_noise_signal_trace(
        t,
        (len(np.asarray(t, dtype=float)),),
        shot_noise_density_ahz05=shot_noise_density_ahz05,
        flicker_noise_density_1hz_ahz05=flicker_noise_density_1hz_ahz05,
        rng=rng,
    ).reshape(-1)


def sample_combined_psd_noise_signal_trace(
    t,
    signal_shape,
    shot_noise_density_ahz05=0.0,
    flicker_noise_density_1hz_ahz05=0.0,
    rng=None,
):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1:
        raise ValueError("t must be a 1D array")
    if isinstance(signal_shape, int):
        signal_shape = (int(signal_shape),)
    else:
        signal_shape = tuple(int(dim) for dim in signal_shape)
    if not signal_shape:
        signal_shape = (len(t),)
    if signal_shape[0] != len(t):
        raise ValueError(
            f"signal_shape[0] must match len(t), got {signal_shape[0]} and {len(t)}"
        )
    if len(t) == 0:
        return np.zeros(signal_shape, dtype=float)
    if len(t) == 1:
        return np.zeros(signal_shape, dtype=float)
    if rng is None:
        rng = np.random.default_rng()

    dt = float(t[1] - t[0])
    if dt <= 0:
        raise ValueError("t must be strictly increasing")
    fft_freqs = np.fft.rfftfreq(len(t), d=dt)
    noise_density = build_combined_noise_density(
        fft_freqs,
        shot_noise_density_ahz05=shot_noise_density_ahz05,
        flicker_noise_density_1hz_ahz05=flicker_noise_density_1hz_ahz05,
    )
    params = {
        "freqs_hz": fft_freqs,
        "noise_density_ahz05": noise_density,
        "white_sigma": float(shot_noise_density_ahz05),
        "flicker_sigma": 0.0,
        "kappa": 1.0,
        "low_freq_amp": 0.0,
        "low_freq_hz": 1.0,
    }
    return _sample_psd_noise_trace(params, t, signal_shape, rng)


def _sample_example_time_noise_trace(t, signal_shape, rng):
    n_steps = len(t)
    noise = np.zeros(signal_shape, dtype=float)
    n_pixels = int(np.prod(signal_shape[1:])) if len(signal_shape) > 1 else 1
    flat_noise = noise.reshape(n_steps, n_pixels)

    low_freq = 0.35e-6 * np.sin(2 * np.pi * 2.0 * t + 0.3)
    ripple = 0.08e-6 * np.sin(2 * np.pi * 120.0 * t)
    sigma_t = 0.12e-6 * (1.0 + 0.5 * np.sin(2 * np.pi * 0.8 * t)) ** 2
    white = rng.normal(0.0, 1.0, size=flat_noise.shape) * sigma_t[:, None]

    flat_noise += low_freq[:, None]
    flat_noise += ripple[:, None]
    flat_noise += white
    return noise


def sample_time_noise_fn_trace(noise_fn, t, I_det, P, rng):
    """
    Sample a time-varying noise function over a trace with optional spatial dimensions.

    Parameters
    ----------
    noise_fn:
        Callable with signature ``noise_fn(t, I_det, P, rng)`` or ``None``.
    t:
        Time axis of shape ``(N,)``.
    I_det:
        Deterministic current trace of shape ``(N, ...)``.
    P:
        Optical-power trace matching ``I_det`` or a per-pixel power map of shape ``(...)``.
    rng:
        NumPy random generator.
    """
    I_det = np.asarray(I_det, dtype=float)
    t = np.asarray(t, dtype=float)

    if noise_fn is None:
        return np.zeros_like(I_det, dtype=float)
    if I_det.shape[0] != len(t):
        raise ValueError(f"I_det first dimension must match len(t), got {I_det.shape[0]} and {len(t)}")

    if I_det.ndim == 1:
        noise = np.asarray(noise_fn(t, I_det, P, rng), dtype=float)
        if noise.shape != I_det.shape:
            raise ValueError(f"noise_fn returned shape {noise.shape}, expected {I_det.shape}")
        return noise

    closure_params = _noise_fn_closure_params(noise_fn)
    if closure_params is not None:
        if "freqs_hz" in closure_params and "noise_density_ahz05" in closure_params:
            return _sample_psd_noise_trace(closure_params, t, I_det.shape, rng)
        return _sample_make_noise_function_trace(closure_params, t, I_det.shape, rng)

    if getattr(noise_fn, "__name__", "") == "example_time_noise_fn":
        return _sample_example_time_noise_trace(t, I_det.shape, rng)

    flat_current = I_det.reshape(len(t), -1)
    power_array = np.asarray(P, dtype=float)
    if power_array.shape == I_det.shape:
        flat_power = power_array.reshape(len(t), -1)
    else:
        flat_power = np.broadcast_to(power_array, I_det.shape[1:]).reshape(1, -1)
        flat_power = np.repeat(flat_power, len(t), axis=0)

    flat_noise = np.empty_like(flat_current, dtype=float)
    child_seeds = rng.integers(0, np.iinfo(np.int32).max, size=flat_current.shape[1], dtype=np.int64)
    for pixel_idx, child_seed in enumerate(child_seeds):
        pixel_rng = np.random.default_rng(int(child_seed))
        pixel_noise = np.asarray(
            noise_fn(t, flat_current[:, pixel_idx], flat_power[:, pixel_idx], pixel_rng),
            dtype=float,
        )
        if pixel_noise.shape != (len(t),):
            raise ValueError(
                f"noise_fn must return shape {(len(t),)} for per-pixel fallback, got {pixel_noise.shape}"
            )
        flat_noise[:, pixel_idx] = pixel_noise

    return flat_noise.reshape(I_det.shape)


def simulate(t, P, params, n_carrier=len(CARRIER_PARAM_LAYOUT), noise_fn=None,
             dark_current=0.0, rng=None, trap_mode="power", trap_threshold_w=0.0,
             trap_output_mode="always", trap_x1_ratio=None,
             trap_x2_reference_state=None,
             trap_x2_tau_on_s=None,
             trap_x2_tau_off_s=None):
    """
    Discrete iterative simulation.

    params layout (n_carrier=2):
      [R_f, η_f, τ_rise_f, τ_fall_f,   # Fast carrier (4 params)
       R_s, η_s, τ_rise_s, τ_fall_s,   # Slow carrier (4 params)
       α, β, δ,                         # Trap state (3 params)
       γ, τ_drift, s_drift]             # Drift branch (3 params)
      Each carrier uses 4 parameters, so the total is 4*n_carrier + 6.

    noise_fn (optional): callable with signature
      noise = noise_fn(t, I_det, P, rng)
    Returns a noise-current array with the same length as t, in A.
    dark_current: externally measured dark current, added to the output as a
    fixed baseline current.
    If noise_fn=None, return the deterministic output without noise.
    """
    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    if t.ndim != 1 or P.ndim != 1:
        raise ValueError("t and P must be 1D arrays")
    if len(t) != len(P):
        raise ValueError("t and P must have the same length")
    if len(t) < 2:
        raise ValueError("Need at least two time samples")

    dt_steps = np.diff(t)
    if np.any(dt_steps <= 0):
        raise ValueError("t must be strictly increasing")

    N = len(t)
    model_config = prepare_model_config(
        params,
        n_carrier=n_carrier,
        trap_mode=trap_mode,
        trap_threshold_w=trap_threshold_w,
        trap_output_mode=trap_output_mode,
        trap_x1_ratio=trap_x1_ratio,
        trap_x2_reference_state=trap_x2_reference_state,
        trap_x2_tau_on_s=trap_x2_tau_on_s,
        trap_x2_tau_off_s=trap_x2_tau_off_s,
    )

    # Carrier components
    x1 = np.zeros((n_carrier, N))
    x2 = np.zeros(N)
    x3 = np.zeros(N)
    for n in range(N - 1):
        x1[:, n + 1], x2[n + 1], x3[n + 1] = step_model_state(
            P[n], dt_steps[n], model_config, x1[:, n], x2[n], x3[n]
        )

    I_det, _ = current_from_state(model_config, x1, x2, x3, P=P, dark_current=dark_current)

    # Noise
    if noise_fn is not None:
        if rng is None:
            rng = np.random.default_rng()
        noise = np.asarray(noise_fn(t, I_det, P, rng), dtype=float)
        if noise.shape != (N,):
            raise ValueError(
                f"noise_fn must return shape {(N,)}, got {noise.shape}"
            )
        I_out = I_det + noise
    else:
        I_out = I_det.copy()

    return I_out, I_det, x1, x2, x3


def example_time_noise_fn(t, I_det, P, rng):
    """
    Example user-defined time-varying noise function.

    Components:
      1. Low-frequency baseline wobble
      2. Periodic ripple
      3. Gaussian noise with an amplitude that varies slowly over time

    To use your own noise model, replace NOISE_FN directly.
    """
    low_freq = 0.35e-6 * np.sin(2 * np.pi * 2.0 * t + 0.3)
    ripple = 0.08e-6 * np.sin(2 * np.pi * 120.0 * t)
    sigma_t = 0.12e-6 * (1.0 + 0.5 * np.sin(2 * np.pi * 0.8 * t)) ** 2
    white = rng.normal(0.0, sigma_t, size=len(t))
    return low_freq + ripple + white


def make_noise_function(white_sigma=0.0, flicker_sigma=0.0, kappa=1.0,
                        low_freq_amp=0.0, low_freq_hz=1.0, label="custom_noise"):
    """
    Create a reusable noise function.

    Parameters:
      white_sigma:   white-noise standard deviation (A)
      flicker_sigma: target time-domain standard deviation of 1/f^kappa noise (A)
      kappa:         1/f exponent
      low_freq_amp:  amplitude of an additional low-frequency sinusoidal drift (A)
      low_freq_hz:   frequency of the low-frequency drift (Hz)
    """
    def noise_fn(t, I_det, P, rng):
        N = len(t)
        noise = np.zeros(N)

        if white_sigma > 0:
            noise += rng.normal(0.0, white_sigma, size=N)

        if low_freq_amp > 0:
            noise += low_freq_amp * np.sin(2 * np.pi * low_freq_hz * t)

        if flicker_sigma > 0 and N > 1:
            dt = t[1] - t[0]
            white = rng.normal(0.0, 1.0, size=N)
            freqs = np.fft.rfftfreq(N, d=dt)
            if len(freqs) > 1:
                freqs[0] = freqs[1]
            else:
                freqs[0] = 1.0
            filt = 1.0 / np.power(freqs, 0.5 * kappa)
            colored = np.fft.irfft(np.fft.rfft(white) * filt, n=N)
            colored -= np.mean(colored)
            std_colored = np.std(colored)
            if std_colored > 0:
                colored = colored / std_colored * flicker_sigma
            noise += colored

        return noise

    noise_fn.__name__ = label
    return noise_fn


def make_noise_function_from_psd(freqs_hz, noise_density_ahz05,
                                 low_freq_amp=0.0, low_freq_hz=1.0,
                                 label="psd_noise"):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    noise_density_ahz05 = np.asarray(noise_density_ahz05, dtype=float)
    positive = freqs_hz > 0
    if np.count_nonzero(positive) == 0:
        raise ValueError("PSD-based noise model requires positive frequency samples")

    white_sigma = float(noise_density_ahz05[positive][-1])
    if np.count_nonzero(positive) >= 2:
        log_slope, _ = np.polyfit(
            np.log10(freqs_hz[positive][: min(6, np.count_nonzero(positive))]),
            np.log10(np.maximum(noise_density_ahz05[positive][: min(6, np.count_nonzero(positive))], 1e-30)),
            1,
        )
        kappa = float(max(0.0, -log_slope))
    else:
        kappa = 1.0
    flicker_sigma = float(np.sqrt(max(np.trapezoid(np.maximum(noise_density_ahz05**2 - white_sigma**2, 0.0), freqs_hz), 0.0)))

    def noise_fn(t, I_det, P, rng):
        params = {
            "freqs_hz": freqs_hz,
            "noise_density_ahz05": noise_density_ahz05,
            "white_sigma": white_sigma,
            "flicker_sigma": flicker_sigma,
            "kappa": kappa,
            "low_freq_amp": float(low_freq_amp),
            "low_freq_hz": float(low_freq_hz),
        }
        return _sample_psd_noise_trace(params, t, (len(t),), rng).reshape(-1)

    noise_fn.__name__ = label
    return noise_fn


# ============================================================
# 2. Synthetic data
# ============================================================

PARAMS_TRUE = {
    # Fast carrier: R (responsivity coefficient), η, τ_rise, τ_fall
    "R_fast": 0.4e-5,               # A/W @ P_ref
    "eta_fast": 0.8,             # Sublinear (η < 1, high-power saturation)
    "tau_rise_fast": 0.3e-2,
    "tau_fall_fast": 0.8e-2,     # Independent of tau_rise_fast
    # Slow carrier:R (responsivity coefficient), η, τ_rise, τ_fall
    "R_slow": 0.1e-5,               # A/W @ P_ref
    "eta_slow": 0.8,             # Sublinear
    "tau_rise_slow": 5e-2,
    "tau_fall_slow": 20e-2,      # Independent of tau_rise_slow
    # Trap state
    "alpha": 0.1e-5,
    "beta": 5000000e-5,
    "delta": 1,
    # Drift: multiplicative response attenuation state x3, response multiplier = max(1 + x3, 0)
    "gamma": -0.1,
    "tau_drift": 500,
    "drift_scale": 1.0,
}

PARAMS_SINGLE_CARRIER = {
    "R_single": 0.308,
    "eta_single": 0.559,
    "tau_rise_single": 1.73e-3,
    "tau_fall_single": 5.20e-3,
    "alpha": 0.0,
    "beta": 1.0,
    "delta": 0.0,
    "gamma": -0.432,
    "tau_drift": 491732.9645779016,
    "drift_scale": 1.0,
}

# Shared overrides for workflows that should suppress only the explicit
# long-term response-attenuation branch x3 while keeping the x2 trap dynamics active.
NO_DRIFT_FIXED_PARAMS = {
    "gamma": PARAMS_TRUE["gamma"],
    "tau_drift": PARAMS_TRUE["tau_drift"],
    "drift_scale": 0.0,
}

DARK_CURRENT_MEASURED = 20e-9  # A, measured dark current provided by the user
DEVICE_AREA_CM2 = DEFAULT_DEVICE_AREA_CM2

ANALYSIS_CONFIG = {
    "power_levels": np.logspace(-6, -3, 13),   # W
    "arbitrary_pmax": 1e-3,                    # W
    "linearity_t_on": 0.05,                    # s
    "linearity_t_off": 0.05,                   # s
    "linearity_cycles": 6,
    "transient_power": 3e-4,                   # W
    "transient_t_on": 0.03,                    # s
    "transient_t_off": 0.05,                   # s
    "transient_cycles": 8,
    "f3db_freqs": np.logspace(0, 4, 25),       # Hz
    "f3db_power_dc": 1e-4,                     # W
    "f3db_mod_depth": 0.05,
    "detectivity_ref_freq": 50.0,              # Hz
    "detectivity_bandwidth": 1.0,              # Hz
    "noise_power_bias": 1e-4,                  # W
    "noise_duration": 4.0,                     # s
    "noise_dt": 2e-4,                          # s
}

NOISE_CASES = {
    "white": make_noise_function(
        white_sigma=25e-9,
        label="white_noise"
    ),
    "white_plus_1f": make_noise_function(
        white_sigma=2e-9,
        flicker_sigma=8e-9,
        kappa=1.0,
        low_freq_amp=3e-9,
        low_freq_hz=1.5,
        label="white_plus_1f_noise"
    ),
}

# Replace this with your own time-varying noise function, or set it to None to disable noise.
NOISE_FN = NOISE_CASES["white_plus_1f"]
PARAM_KEYS = [key for group in CARRIER_PARAM_LAYOUT for key in group] + GLOBAL_PARAM_KEYS
SINGLE_PARAM_KEYS = [key for group in SINGLE_CARRIER_PARAM_LAYOUT for key in group] + GLOBAL_PARAM_KEYS
N_CARRIER = len(CARRIER_PARAM_LAYOUT)
NOISE_PARAM_KEYS = ["white_sigma", "flicker_sigma", "kappa", "low_freq_amp", "low_freq_hz"]

PARAM_BOUNDS = {
    "R_fast": (0.1, 1.0, "log"),
    "eta_fast": (0.5, 1.5, "linear"),
    "tau_rise_fast": (1e-6, 5e1, "log"),
    "tau_fall_fast": (1e-6, 5e1, "log"),
    "R_slow": (0.01, 0.5, "log"),
    "eta_slow": (0.5, 1.5, "linear"),
    "tau_rise_slow": (1e-6, 50e1, "log"),
    "tau_fall_slow": (1e-6, 50e1, "log"),
    "alpha": (5.0, 2000.0, "log"),
    "beta": (0.5, 50.0, "log"),
    "delta": (1e-7, 1, "log"),
    "gamma": (-1.0, 1.0, "linear"),
    "tau_drift": (1e-1, 1e8, "log"),
    "drift_scale": (0.0, 1.0, "linear"),
}


def params_to_vec(d):
    keys = SINGLE_PARAM_KEYS if "R_single" in d else PARAM_KEYS
    return np.array([d[k] for k in keys], dtype=float)


def vec_to_params(a):
    keys = SINGLE_PARAM_KEYS if len(a) == len(SINGLE_PARAM_KEYS) else PARAM_KEYS
    return {k: float(v) for k, v in zip(keys, a)}


def build_complete_params(param_overrides=None, base_params=None, allow_missing=False):
    """
    Merge partial parameter dictionaries onto a base set of model parameters.
    """
    params = dict(PARAMS_TRUE if base_params is None else base_params)
    keys = SINGLE_PARAM_KEYS if "R_single" in params else PARAM_KEYS
    if param_overrides:
        for key, value in param_overrides.items():
            if key not in keys:
                continue
            if value is None and not allow_missing:
                continue
            params[key] = value

    missing = [key for key in keys if params.get(key) is None]
    if missing and not allow_missing:
        raise ValueError(f"missing required model parameters: {', '.join(missing)}")
    return params


def noise_fn_to_config(noise_fn):
    """
    Extract a serializable noise configuration from a closure-based noise_fn.
    """
    params = _noise_fn_closure_params(noise_fn)
    if params is None:
        return None
    return {key: float(params[key]) for key in NOISE_PARAM_KEYS}


def make_noise_function_from_config(noise_config, label="dataset_noise"):
    """
    Build a reusable noise function from a parameter dictionary.
    Returns None when all noise terms are empty or zero.
    """
    if noise_config is None:
        return None

    merged = {
        "white_sigma": 0.0,
        "flicker_sigma": 0.0,
        "kappa": 1.0,
        "low_freq_amp": 0.0,
        "low_freq_hz": 1.0,
    }
    has_nonzero = False
    for key in NOISE_PARAM_KEYS:
        value = noise_config.get(key)
        if value is None:
            continue
        merged[key] = float(value)
        if key != "kappa" and abs(float(value)) > 0:
            has_nonzero = True

    if not has_nonzero:
        return None

    return make_noise_function(
        white_sigma=merged["white_sigma"],
        flicker_sigma=merged["flicker_sigma"],
        kappa=merged["kappa"],
        low_freq_amp=merged["low_freq_amp"],
        low_freq_hz=merged["low_freq_hz"],
        label=label,
    )


def simulate_ideal_output(P, params_vec, n_carrier=len(CARRIER_PARAM_LAYOUT), dark_current=0.0,
                          responsivity_override=None):
    """
    Fully ideal model: linear, no delay, no traps, no drift, no noise.
    I_ideal = R_ideal · P.
    When responsivity_override is None, the model uses the sum of carrier
    responsivities stored in params_vec.
    """
    P = np.asarray(P, dtype=float)
    if responsivity_override is None:
        carrier_params = np.asarray(params_vec[:4 * n_carrier], dtype=float).reshape(n_carrier, 4)
        responsivity_components = carrier_params[:, 0]
    else:
        responsivity_components = np.atleast_1d(float(responsivity_override))
    I_photo_components = responsivity_components[:, None] * P[None, :]
    I_photo_ideal = I_photo_components.sum(axis=0)
    I_det_ideal = dark_current + I_photo_ideal
    return I_det_ideal, I_photo_ideal, I_photo_components


def style_axes(ax, grid=False, grid_axis="both"):
    if not getattr(ax, "axison", True):
        return
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_WIDTH)
        spine.set_color("#222222")
    ax.tick_params(axis="both", which="major", direction="out", width=1.6, length=6.5, pad=6)
    ax.tick_params(axis="both", which="minor", direction="out", width=1.2, length=3.5)
    ax.title.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    if grid:
        ax.grid(True, which="major", axis=grid_axis, linestyle="--", linewidth=0.9,
                color="#777777", alpha=GRID_ALPHA)
        ax.grid(True, which="minor", axis=grid_axis, linestyle=":", linewidth=0.6,
                color="#999999", alpha=0.14)


def style_legend(legend):
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_linewidth(1.2)
    frame.set_edgecolor("#444444")
    frame.set_facecolor("white")


def style_table(table, font_size=12.5, scale=(1.0, 2.1)):
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(*scale)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(TABLE_EDGE_COLOR)
        cell.set_linewidth(1.2 if row == 0 else 0.95)
        if row == 0:
            cell.set_facecolor(TABLE_HEADER_COLOR)
            cell.set_text_props(weight="bold", ha="center", va="center")
        else:
            if row % 2 == 0:
                cell.set_facecolor(TABLE_ALT_COLOR)
            cell.set_text_props(ha="left", va="center")
        cell.get_text().set_wrap(True)


def finalize_figure(path):
    plt.tight_layout(pad=1.0)
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    if "agg" in matplotlib.get_backend().lower():
        plt.close()
    else:
        plt.show()


def plot_nonideal_effects_table(path="fig_nonideal_effects_table.png"):
    """
    Summarize the non-ideal effects, state variables, and formulas used in the model as a table figure.
    """
    rows = [
        [
            "Multi-time-constant carrier response",
            "Parallel fast/slow carrier branches x1_i",
            "target_i = R_i·P_ref·(P/P_ref)^{eta_i}\n"
            "x1_i[n+1] = x1_i[n]·exp(-dt/tau_i)\n"
            "          + target_i·(1 - exp(-dt/tau_i))",
            "Photocarriers at different time scales jointly shape transient and steady-state response",
            "step_model_state()\n"
            "carrier_params = [R_i, eta_i, tau_rise_i, tau_fall_i]",
        ],
        [
            "Rise/fall asymmetry",
            "Switch time constant by target direction",
            "tau_i = tau_rise_i,  if target_i >= x1_i\n"
            "tau_i = tau_fall_i,  if target_i <  x1_i",
            "Each branch keeps independent tau_rise_i and tau_fall_i to capture accumulation/release asymmetry",
            "step_model_state()\n"
            "decay = where(target >= x1[i], decay_r, decay_f)",
        ],
        [
            "I-P nonlinearity",
            "Power-law optical-power/current mapping",
            "I_i,ss = R_i·P_ref·(P/P_ref)^{eta_i}\n"
            "eta_i < 1: sublinear saturation\n"
            "eta_i > 1: superlinear enhancement",
            "P_ref is the crossover between the ideal and non-ideal model; when P > P_ref and eta < 1, the response saturates",
            "step_model_state()\n"
            "plot_nonlinearity()",
        ],
        [
            "Trap state and persistent photocurrent",
            "Trap occupancy x2 + output gain delta",
            "dx2/dt = alpha·P·(1 - x2) - beta·x2\n"
            "I_trap = delta·x2",
            "Capture/release dynamics introduce memory effect, tailing, and persistent photocurrent",
            "step_model_state()\n"
            "current_from_state()",
        ],
        [
            "Long-term response attenuation drift",
            "Slow attenuation state x3",
            "dx3/dt = (s_drift·gamma - x3) / tau_drift\n"
            "I_photo = I_photo,base · max(1 + x3, 0)",
            "Describes slow response attenuation or recovery over long aging time",
            "step_model_state()\n"
            "current_from_state()",
        ],
        [
            "Output noise",
            "Custom time-varying noise or shot + thermal noise",
            "I_out = I_det + noise_fn(t, I_det, P, rng)\n"
            "sigma_shot = sqrt(2q|I_det|B)\n"
            "sigma_th = i_thermal·sqrt(B)",
            "Supports white noise, 1/f noise, low-frequency drift, and standalone shot/thermal-noise sampling",
            "simulate()\n"
            "make_noise_function()\n"
            "sample_shot_thermal_noise()",
        ],
    ]

    fig, ax = plt.subplots(figsize=(22, 11.5))
    ax.axis("off")

    col_labels = ["Effect", "State / Model", "Core Equation", "Physical Meaning", "Code Path"]
    col_widths = [0.14, 0.19, 0.31, 0.20, 0.16]

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        colWidths=col_widths,
        cellLoc="left",
        loc="center",
        bbox=[0.0, 0.12, 1.0, 0.82],
    )

    style_table(table, font_size=12.0, scale=(1.0, 2.4))
    ax.set_title("Photodetector Non-Ideal Effects and Governing Equations", fontsize=23, pad=24, fontweight="bold")

    summary = (
        "Overall output: I_photo,base = sum_i x1_i + delta·x2,   "
        "I_photo = I_photo,base · max(1 + x3, 0),   I_det = I_dark + I_photo,   I_out = I_det + noise\n"
        "x1_i denotes multi-time-constant carrier components, x2 the trap-memory branch, "
        "and x3 the slow multiplicative response-attenuation branch."
    )
    fig.text(0.02, 0.04, summary, fontsize=13, va="bottom")

    finalize_figure(path)


def make_square_wave(t, P_on, t_on, t_off):
    period = t_on + t_off
    return np.where(t % period < t_on, P_on, 0.0)


def generate_synthetic_data(dark_current=DARK_CURRENT_MEASURED, noise_fn=NOISE_FN):
    """Multi-power square wave: different P_on values are required to independently identify R and η."""
    P_levels = [0.1e-3, 0.5e-3, 1e-3, 2e-3]  # Four power levels
    t_on, t_off = 0.1, 0.1
    cycles_per_level = 8

    segments_t, segments_P = [], []
    t_offset = 0.0
    for P_on in P_levels:
        T_seg = (t_on + t_off) * cycles_per_level
        N_seg = 12000
        t_seg = np.linspace(0, T_seg, N_seg, endpoint=False) + t_offset
        P_seg = make_square_wave(t_seg - t_offset, P_on, t_on, t_off)
        segments_t.append(t_seg)
        segments_P.append(P_seg)
        t_offset += T_seg

    t = np.concatenate(segments_t)
    P = np.concatenate(segments_P)

    rng = np.random.default_rng(42)
    I_noisy, I_det, _, _, _ = simulate(
        t, P, params_to_vec(PARAMS_TRUE), N_CARRIER,
        noise_fn=noise_fn, dark_current=dark_current, rng=rng
    )
    noise = I_noisy - I_det
    return t, P, I_noisy, I_det, noise


# ============================================================
# 3. Parameter extraction
# ============================================================

def fit_parameters(t, P, I_data, dark_current=DARK_CURRENT_MEASURED):
    """Fit deterministic model parameters; the external noise function is not used during fitting."""
    full_fit_keys = [key for key in PARAM_KEYS if key != "drift_scale"]
    bounds = [
        (PARAM_BOUNDS[key][0], PARAM_BOUNDS[key][1])
        for key in full_fit_keys
    ]

    step = max(1, len(t) // 3000)
    t_sub, P_sub, I_sub = t[::step], P[::step], I_data[::step]

    call_count = [0]

    def pack_params(vector):
        params = dict(PARAMS_TRUE)
        for key, value in zip(full_fit_keys, vector):
            params[key] = float(value)
        return params

    def cost(pv):
        call_count[0] += 1
        if call_count[0] % 2000 == 0:
            print(f"  ... {call_count[0]} evaluations")
        # Do not add noise during fitting
        candidate_params = pack_params(pv)
        _, I_sim, _, _, _ = simulate(
            t_sub, P_sub, params_to_vec(candidate_params), N_CARRIER,
            noise_fn=None, dark_current=dark_current
        )
        return np.mean((I_sim - I_sub) ** 2)

    print(f"拟合中 ({len(t_sub)} 点, {len(bounds)} 参数)...")
    result = differential_evolution(
        cost, bounds, seed=42, maxiter=300, tol=1e-10,
        polish=True, popsize=15,
    )
    print(f"完成, cost={result.fun:.2e}, {call_count[0]} evals")
    return params_to_vec(pack_params(result.x))


def fit_parameters_subset(
    t,
    P,
    I_data,
    fit_keys=None,
    initial_params=None,
    fixed_params=None,
    param_bounds=None,
    dark_current=DARK_CURRENT_MEASURED,
    max_nfev=400,
    loss="soft_l1",
):
    """
    Perform local least-squares fitting on a subset of parameters.

    This is suitable for digitized-image scenarios that have only a single
    optical-power level but do include switching transient information.
    Such data usually identifies time constants and equivalent amplitudes
    more reliably, while nonlinear exponents such as eta are harder to
    estimate stably.
    """
    fit_keys = PARAM_KEYS if fit_keys is None else list(fit_keys)
    fixed_params = {} if fixed_params is None else dict(fixed_params)
    initial_params = PARAMS_TRUE if initial_params is None else dict(initial_params)
    param_bounds = PARAM_BOUNDS if param_bounds is None else dict(param_bounds)

    invalid = [key for key in fit_keys if key not in PARAM_KEYS]
    if invalid:
        raise ValueError(f"Unsupported fit parameter(s): {invalid}")
    overlap = sorted(set(fit_keys) & set(fixed_params))
    if overlap:
        raise ValueError(f"Parameters cannot be both fitted and fixed: {overlap}")
    missing_bounds = [key for key in PARAM_KEYS if key not in param_bounds]
    if missing_bounds:
        raise ValueError(f"Missing parameter bounds for: {missing_bounds}")

    base_params = dict(PARAMS_TRUE)
    base_params.update(initial_params)
    base_params.update(fixed_params)

    t = np.asarray(t, dtype=float)
    P = np.asarray(P, dtype=float)
    I_data = np.asarray(I_data, dtype=float)
    if t.ndim != 1 or P.ndim != 1 or I_data.ndim != 1:
        raise ValueError("t, P, and I_data must be 1D arrays")
    if not (len(t) == len(P) == len(I_data)):
        raise ValueError("t, P, and I_data must have the same length")
    if len(t) < 3:
        raise ValueError("Need at least 3 samples for fitting")

    step = max(1, len(t) // 1200)
    t_sub, P_sub, I_sub = t[::step], P[::step], I_data[::step]
    measurement_scale = max(float(np.std(I_sub)), 1e-9)

    def encode_value(key, value):
        lo, hi, mode = param_bounds[key]
        clipped = float(np.clip(value, lo, hi))
        if mode == "log":
            return np.log10(max(clipped, 1e-30))
        return clipped

    def decode_value(key, value):
        _, _, mode = param_bounds[key]
        if mode == "log":
            return float(10 ** value)
        return float(value)

    x0 = np.array([encode_value(key, base_params[key]) for key in fit_keys], dtype=float)
    lower = []
    upper = []
    for key in fit_keys:
        lo, hi, mode = param_bounds[key]
        if mode == "log":
            lower.append(np.log10(lo))
            upper.append(np.log10(hi))
        else:
            lower.append(lo)
            upper.append(hi)
    bounds = (np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))

    def pack_params(vector):
        fitted_params = dict(base_params)
        for key, raw in zip(fit_keys, vector):
            fitted_params[key] = decode_value(key, raw)
        return fitted_params

    def residual_fn(vector):
        candidate_params = pack_params(vector)
        _, I_sim, _, _, _ = simulate(
            t_sub,
            P_sub,
            params_to_vec(candidate_params),
            N_CARRIER,
            noise_fn=None,
            dark_current=dark_current,
        )
        return (I_sim - I_sub) / measurement_scale

    result = least_squares(
        residual_fn,
        x0=x0,
        bounds=bounds,
        loss=loss,
        f_scale=1.0,
        max_nfev=max_nfev,
        verbose=0,
    )
    fitted_params = pack_params(result.x)
    stats = {
        "success": bool(result.success),
        "cost": float(result.cost),
        "nfev": float(result.nfev),
        "optimality": float(result.optimality),
        "status": float(result.status),
        "subsample_step": float(step),
        "num_points_used": float(len(t_sub)),
    }
    return params_to_vec(fitted_params), stats


# ============================================================
# 4. Arbitrary waveforms
# ============================================================

def make_arbitrary_signal(t, power_ref_w=None, pmax_w=None):
    """
    Example arbitrary optical-power waveform: a square-pulse sequence with varying heights.

    Keep an initial dark segment so the zero-initialized state does not
    immediately see a large nonzero input, which would otherwise appear as
    an unnatural initial overshoot in the plot.
    """
    power_ref_w = float(NONLINEAR_POWER_REF_W if power_ref_w is None else power_ref_w)
    pmax_w = float(ANALYSIS_CONFIG["arbitrary_pmax"] if pmax_w is None else pmax_w)
    if power_ref_w <= 0 or pmax_w <= 0:
        raise ValueError("power_ref_w and pmax_w must be positive")
    if pmax_w < power_ref_w:
        raise ValueError("pmax_w must be greater than or equal to power_ref_w")

    t = np.asarray(t, dtype=float)
    P = np.zeros_like(t)
    active = t >= ARBITRARY_STARTUP_DARK_S
    if not np.any(active):
        return P

    t_active = t[active] - ARBITRARY_STARTUP_DARK_S
    sample_levels = np.geomspace(power_ref_w, pmax_w, 8)
    ordered_levels = [
        sample_levels[2],
        sample_levels[5],
        sample_levels[-1],
        sample_levels[0],
        sample_levels[6],
        sample_levels[1],
        sample_levels[-2],
        sample_levels[3],
    ]
    segments = [
        (15e-3, 0.0),
        (30e-3, ordered_levels[0]),
        (20e-3, 0.0),
        (40e-3, ordered_levels[1]),
        (15e-3, 0.0),
        (22e-3, ordered_levels[2]),
        (10e-3, 0.0),
        (18e-3, ordered_levels[3]),
        (12e-3, 0.0),
        (35e-3, ordered_levels[4]),
        (18e-3, 0.0),
        (28e-3, ordered_levels[5]),
        (16e-3, 0.0),
        (42e-3, ordered_levels[6]),
        (20e-3, 0.0),
        (26e-3, ordered_levels[7]),
        (15e-3, 0.0),
        (32e-3, np.sqrt(power_ref_w * pmax_w)),
        (26e-3, 0.0),
    ]

    waveform = np.zeros_like(t_active)
    cursor_s = 0.0
    for duration_s, level_w in segments:
        mask = (t_active >= cursor_s) & (t_active < cursor_s + duration_s)
        waveform[mask] = level_w
        cursor_s += duration_s

    P[active] = waveform
    return P


# ============================================================
# 5. Visualization
# ============================================================

def plot_fitting(t, P, I_data, I_det_fitted, x1_components):
    n_carrier = x1_components.shape[0]
    n_rows = 3 + n_carrier
    ratios = [1, 3] + [1] * n_carrier + [1]

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.4 * n_rows), sharex=True,
                             gridspec_kw={"height_ratios": ratios})

    axes[0].plot(t * 1e3, P * 1e3, "g-", lw=LW_MAIN)
    axes[0].set_ylabel("P_opt (mW)")
    axes[0].set_title("Square-Wave Fit: Nonlinear I-P + Multi-Time-Constant + Rise/Fall Asymmetry")

    axes[1].plot(t * 1e3, I_data * 1e6, "b.", ms=MS_SMALL, alpha=0.24, label="Measured data")
    axes[1].plot(t * 1e3, I_det_fitted * 1e6, "r-", lw=LW_HEAVY, label="Deterministic fit")
    axes[1].set_ylabel("I_out (μA)")
    style_legend(axes[1].legend(loc="best"))

    colors = ["tab:orange", "tab:purple", "tab:cyan", "tab:brown"]
    labels = ["Fast branch x1_fast", "Slow branch x1_slow", "Branch 3", "Branch 4"]
    if n_carrier == 1:
        labels = ["Single branch x1_single"]
    for i in range(n_carrier):
        axes[2 + i].plot(t * 1e3, x1_components[i] * 1e6,
                         color=colors[i % len(colors)], lw=LW_MED)
        axes[2 + i].set_ylabel(f"{labels[i]} (μA)")

    residual = (I_data - I_det_fitted) * 1e6
    axes[-1].plot(t * 1e3, residual, "k-", lw=LW_FINE, alpha=0.8)
    axes[-1].set_ylabel("Residual (uA)")
    axes[-1].set_xlabel("Time (ms)")
    axes[-1].axhline(0, color="gray", ls="--", lw=LW_FINE)
    for ax in axes:
        style_axes(ax, grid=True)
    finalize_figure("fig1_fitting.png")


def plot_params_table(true_d, fitted_d):
    keys = SINGLE_PARAM_KEYS if "R_single" in true_d else PARAM_KEYS
    units_map = {
        "R_fast": "A/W",
        "eta_fast": "",
        "tau_rise_fast": "s",
        "tau_fall_fast": "s",
        "R_slow": "A/W",
        "eta_slow": "",
        "tau_rise_slow": "s",
        "tau_fall_slow": "s",
        "R_single": "A/W",
        "eta_single": "",
        "tau_rise_single": "s",
        "tau_fall_single": "s",
        "alpha": "1/(W·s)",
        "beta": "1/s",
        "delta": "A",
        "gamma": "",
        "tau_drift": "s",
        "drift_scale": "1",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")

    rows = []
    for k in keys:
        tv, fv = true_d[k], fitted_d[k]
        denom = max(abs(tv), 1e-30)
        err = abs(fv - tv) / denom * 100
        rows.append([k, units_map.get(k, ""), f"{tv:.4g}", f"{fv:.4g}", f"{err:.1f}%"])

    table = ax.table(cellText=rows,
                     colLabels=["Parameter", "Unit", "True", "Fitted", "Error"],
                     loc="center", cellLoc="center")
    style_table(table, font_size=12.5, scale=(1.25, 1.9))
    ax.set_title("Parameter Extraction Comparison", fontsize=20, pad=24, fontweight="bold")
    finalize_figure("fig2_params.png")


def plot_noise_analysis(t, noise):
    """Noise analysis figure: time-domain noise + PSD."""
    dt = t[1] - t[0]
    N = len(noise)

    fig, axes = plt.subplots(2, 1, figsize=(15, 9))

    # Time domain
    axes[0].plot(t * 1e3, noise * 1e6, "k-", lw=LW_FINE, alpha=0.9)
    axes[0].set_ylabel("Noise (uA)")
    axes[0].set_xlabel("Time (ms)")
    axes[0].set_title("Input Noise Analysis")

    # PSD
    freqs = np.fft.rfftfreq(N, d=dt)
    psd = np.abs(np.fft.rfft(noise)) ** 2 / N * dt
    if len(freqs) > 1:
        freqs[0] = freqs[1]

    axes[1].loglog(freqs[1:], psd[1:], "b-", lw=LW_MED, alpha=0.95, label="Noise PSD")

    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD (A²/Hz)")
    style_legend(axes[1].legend(loc="best"))
    if len(freqs) > 1:
        axes[1].set_xlim(max(1, freqs[1]), freqs[-1])
    style_axes(axes[0], grid=True)
    style_axes(axes[1], grid=True)
    finalize_figure("fig4_noise.png")


def plot_nonlinearity(params_vec, n_carrier=len(CARRIER_PARAM_LAYOUT)):
    """I-P nonlinearity: log-log response curve."""
    nc4 = 4 * n_carrier
    carrier_params = params_vec[:nc4].reshape(n_carrier, 4)

    P_range = np.logspace(-6, -1, 200)  # 1 μW to 100 mW

    fig, ax = plt.subplots(figsize=(9, 7))

    colors = ["tab:orange", "tab:purple", "tab:cyan", "tab:brown"]
    labels_c = ["Fast branch", "Slow branch", "Branch 3", "Branch 4"]
    if n_carrier == 1:
        labels_c = ["Single branch"]
    I_total = np.zeros_like(P_range)

    for i in range(n_carrier):
        Ri, eta_i, _, _ = carrier_params[i]
        I_comp = carrier_target_current(P_range, Ri, eta_i)
        I_total += I_comp
        ax.loglog(P_range * 1e3, I_comp * 1e6, color=colors[i % len(colors)], ls="--", lw=LW_MED,
                  label=f"{labels_c[i]}: R@1 mW={Ri:.3f}, η={eta_i:.3f}")

    ax.loglog(P_range * 1e3, I_total * 1e6, "k-", lw=LW_HEAVY, label="Total response")

    # Reference line: purely linear
    R_total = sum(carrier_params[i, 0] for i in range(n_carrier))
    ax.loglog(P_range * 1e3, R_total * P_range * 1e6, "gray", ls=":", lw=LW_LIGHT,
              label=f"Linear reference (eta=1, R={R_total:.2f})")

    ax.set_xlabel("P_opt (mW)")
    ax.set_ylabel("I_photo (μA)")
    ax.set_title("I-P Nonlinearity (log-log)")
    style_legend(ax.legend(loc="best"))
    style_axes(ax, grid=True)
    finalize_figure("fig5_nonlinearity.png")


def plot_arbitrary(params_vec, noise_fn=None):
    n_carrier = 1 if len(params_vec) == len(SINGLE_PARAM_KEYS) else N_CARRIER
    T = 0.5
    t = np.linspace(0, T, 10000)
    pmax_w = ANALYSIS_CONFIG["arbitrary_pmax"]
    P = make_arbitrary_signal(t, power_ref_w=NONLINEAR_POWER_REF_W, pmax_w=pmax_w)
    ideal_responsivity = ANALYSIS_CONFIG.get("ideal_responsivity")

    rng = np.random.default_rng(123)
    I_det_ideal, _, _ = simulate_ideal_output(
        P, params_vec, n_carrier, dark_current=DARK_CURRENT_MEASURED,
        responsivity_override=ideal_responsivity
    )
    I_out, I_det_nonideal, x1, x2, x3 = simulate(
        t, P, params_vec, n_carrier,
        noise_fn=noise_fn, dark_current=DARK_CURRENT_MEASURED, rng=rng
    )
    I_compare_nonideal = I_det_nonideal

    n_rows = 6 if noise_fn else 5
    ratios = [1, 2, 1, 1, 1, 1] if noise_fn else [1, 2, 1, 1, 1]

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3.3 * n_rows), sharex=True,
                             gridspec_kw={"height_ratios": ratios})

    axes[0].plot(t * 1e3, P * 1e3, "g-", lw=LW_MAIN)
    axes[0].set_ylabel("P_opt (mW)")
    axes[0].set_title(
        f"Arbitrary Waveform Simulation: Non-Ideal vs Ideal (P_ref={NONLINEAR_POWER_REF_W:.2e} W, Pmax={pmax_w:.2e} W)"
    )

    axes[1].plot(t * 1e3, I_det_ideal * 1e6, "b--", lw=LW_LIGHT, alpha=0.95,
                 label=f"Ideal model (R_ideal={ideal_responsivity:.3g} A/W)" if ideal_responsivity else "Ideal model")
    axes[1].plot(t * 1e3, I_compare_nonideal * 1e6, "r-", lw=LW_HEAVY,
                 label="Non-ideal model (nonlinearity + delay + trap + drift)")
    if noise_fn:
        axes[1].plot(t * 1e3, I_out * 1e6, color="0.55", lw=LW_FINE, alpha=0.85,
                     label="Non-ideal model + noise")
    axes[1].set_ylabel("I_out (μA)")
    style_legend(axes[1].legend(loc="best"))

    axes[2].plot(t * 1e3, x1[0] * 1e6, "tab:orange", lw=LW_MED,
                 label="Single branch" if n_carrier == 1 else "Fast branch")
    if n_carrier > 1:
        axes[2].plot(t * 1e3, x1[1] * 1e6, "tab:purple", lw=LW_MED, label="Slow branch")
    axes[2].set_ylabel("Carrier current (uA)")
    style_legend(axes[2].legend(loc="best"))

    axes[3].plot(t * 1e3, x2, "m-", lw=LW_MED)
    axes[3].set_ylabel("x2 trap state")

    axes[4].plot(t * 1e3, 1.0 + x3, color="tab:brown", lw=LW_MED)
    axes[4].set_ylabel("Drift multiplier")

    if noise_fn:
        noise = I_out - I_det_nonideal
        axes[5].plot(t * 1e3, noise * 1e6, "k-", lw=LW_FINE, alpha=0.85)
        axes[5].set_ylabel("Noise (uA)")

    axes[-1].set_xlabel("Time (ms)")
    for ax in axes:
        style_axes(ax, grid=True)
    finalize_figure("fig3_arbitrary.png")


# ============================================================
# 6. Metric extraction
# ============================================================

E_CHARGE = 1.602e-19


def simulate_square_response(params_vec, dark_current, P_on, t_on, t_off, cycles,
                             noise_fn=None, rng=None, points_per_period=1600):
    period = t_on + t_off
    N = max(int(points_per_period * cycles), 4000)
    t = np.linspace(0, period * cycles, N, endpoint=False)
    P = make_square_wave(t, P_on, t_on, t_off)
    n_carrier = infer_n_carrier_from_params(params_vec)
    I_out, I_det, x1, x2, x3 = simulate(
        t, P, params_vec, n_carrier,
        noise_fn=noise_fn, dark_current=dark_current, rng=rng
    )
    return t, P, I_out, I_det, x1, x2, x3


def _window_stats(t, y, t_start, t_end):
    mask = (t >= t_start) & (t < t_end)
    if mask.sum() == 0:
        return float(np.mean(y)), float(np.std(y))
    return float(np.mean(y[mask])), float(np.std(y[mask]))


def _find_threshold_crossing(t, y, level, rising=True):
    cmp = y >= level if rising else y <= level
    idx = np.where(cmp)[0]
    if len(idx) == 0:
        return np.nan
    i1 = idx[0]
    if i1 == 0:
        return float(t[0])
    i0 = i1 - 1
    y0, y1 = y[i0], y[i1]
    if y1 == y0:
        return float(t[i1])
    frac = (level - y0) / (y1 - y0)
    return float(t[i0] + frac * (t[i1] - t[i0]))


def _fundamental_amplitude(t, y, f_hz):
    omega_t = 2 * np.pi * f_hz * t
    y0 = y - np.mean(y)
    a_sin = 2.0 / len(y0) * np.dot(y0, np.sin(omega_t))
    a_cos = 2.0 / len(y0) * np.dot(y0, np.cos(omega_t))
    return float(np.hypot(a_sin, a_cos))


def _extract_linear_range(P_levels, I_photo, tol):
    if len(P_levels) < 2 or np.any(I_photo <= 0):
        return {
            "indices": np.array([], dtype=int),
            "local_slopes": np.array([]),
            "ldr_db": np.nan,
        }

    logP = np.log10(P_levels)
    logI = np.log10(I_photo)
    local_slopes = np.diff(logI) / np.diff(logP)
    point_ok = np.zeros(len(P_levels), dtype=bool)
    good_intervals = np.abs(local_slopes - 1.0) <= tol
    point_ok[:-1] |= good_intervals
    point_ok[1:] |= good_intervals

    best_start = best_end = -1
    start = None
    for i, ok in enumerate(point_ok):
        if ok and start is None:
            start = i
        elif (not ok) and start is not None:
            if i - start > best_end - best_start:
                best_start, best_end = start, i
            start = None
    if start is not None and len(P_levels) - start > best_end - best_start:
        best_start, best_end = start, len(P_levels)

    if best_start < 0 or best_end - best_start < 2:
        return {
            "indices": np.array([], dtype=int),
            "local_slopes": local_slopes,
            "ldr_db": np.nan,
        }

    idx = np.arange(best_start, best_end)
    ldr_db = 20 * np.log10(I_photo[idx[-1]] / I_photo[idx[0]])
    return {
        "indices": idx,
        "local_slopes": local_slopes,
        "ldr_db": float(ldr_db),
    }


def analyze_linearity_and_ldr(params_vec, dark_current, cfg):
    P_levels = np.asarray(cfg["power_levels"], dtype=float)
    I_on_total = np.zeros_like(P_levels)
    I_photo = np.zeros_like(P_levels)
    responsivity = np.zeros_like(P_levels)

    for i, p_on in enumerate(P_levels):
        t, _, _, I_det, _, _, _ = simulate_square_response(
            params_vec, dark_current, p_on,
            cfg["linearity_t_on"], cfg["linearity_t_off"], cfg["linearity_cycles"],
            noise_fn=None
        )
        period = cfg["linearity_t_on"] + cfg["linearity_t_off"]
        cycle_start = period * (cfg["linearity_cycles"] - 1)
        on_mean, _ = _window_stats(
            t, I_det,
            cycle_start + 0.70 * cfg["linearity_t_on"],
            cycle_start + 0.95 * cfg["linearity_t_on"]
        )
        I_on_total[i] = on_mean
        I_photo[i] = max(on_mean - dark_current, 1e-30)
        responsivity[i] = I_photo[i] / p_on

    alpha_global = float(np.polyfit(np.log10(P_levels), np.log10(I_photo), 1)[0])
    strict = _extract_linear_range(P_levels, I_photo, tol=0.01)
    quasi = _extract_linear_range(P_levels, I_photo, tol=0.03)
    full_scan_ldr_db = float(20 * np.log10(I_photo[-1] / I_photo[0]))

    return {
        "P_levels": P_levels,
        "I_on_total": I_on_total,
        "I_photo": I_photo,
        "responsivity": responsivity,
        "alpha_global": alpha_global,
        "strict": strict,
        "quasi": quasi,
        "full_scan_ldr_db": full_scan_ldr_db,
    }


def plot_linearity_and_ldr(results):
    P_levels = results["P_levels"]
    I_photo = results["I_photo"]
    responsivity = results["responsivity"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9.5))

    axes[0].loglog(P_levels * 1e3, I_photo * 1e6, "o-", lw=LW_MAIN, ms=MS_MAIN, label="Steady-state photocurrent")
    if len(results["strict"]["indices"]) >= 2:
        idx = results["strict"]["indices"]
        axes[0].loglog(P_levels[idx] * 1e3, I_photo[idx] * 1e6, "o", ms=9,
                       label=f"Strict linear region, LDRapp={results['strict']['ldr_db']:.1f} dB")
    if len(results["quasi"]["indices"]) >= 2:
        idx = results["quasi"]["indices"]
        axes[0].loglog(P_levels[idx] * 1e3, I_photo[idx] * 1e6, "s", ms=8,
                       label=f"Quasi-linear region, LDRapp={results['quasi']['ldr_db']:.1f} dB")
    axes[0].set_xlabel("Input power (mW)")
    axes[0].set_ylabel("Steady-state photocurrent (uA)")
    axes[0].set_title(f"Linearity and LDR Analysis, Global Slope alpha={results['alpha_global']:.3f}")
    style_legend(axes[0].legend(loc="best"))

    axes[1].semilogx(P_levels * 1e3, responsivity, "o-", lw=LW_MAIN, ms=MS_MAIN)
    axes[1].set_xlabel("Input power (mW)")
    axes[1].set_ylabel("R = I_photo / P (A/W)")
    axes[1].set_title("Responsivity vs Power")
    style_axes(axes[0], grid=True)
    style_axes(axes[1], grid=True)
    finalize_figure("fig_metric_linearity_ldr.png")


def analyze_rise_fall(params_vec, dark_current, cfg):
    t, P, _, I_det, _, _, _ = simulate_square_response(
        params_vec, dark_current, cfg["transient_power"],
        cfg["transient_t_on"], cfg["transient_t_off"], cfg["transient_cycles"],
        noise_fn=None
    )

    t_on = cfg["transient_t_on"]
    t_off = cfg["transient_t_off"]
    period = t_on + t_off
    cycle_start = period * (cfg["transient_cycles"] - 1)
    rise_edge = cycle_start
    fall_edge = cycle_start + t_on

    baseline, baseline_std = _window_stats(
        t, I_det, rise_edge - 0.35 * t_off, rise_edge - 0.05 * t_off
    )
    plateau_on, plateau_on_std = _window_stats(
        t, I_det, rise_edge + 0.70 * t_on, rise_edge + 0.95 * t_on
    )
    plateau_off, plateau_off_std = _window_stats(
        t, I_det, fall_edge + 0.70 * t_off, fall_edge + 0.95 * t_off
    )

    amp_rise = plateau_on - baseline
    amp_fall = plateau_on - plateau_off

    rise_mask = (t >= rise_edge) & (t < rise_edge + t_on)
    fall_mask = (t >= fall_edge) & (t < fall_edge + t_off)
    t_rise = t[rise_mask]
    y_rise = I_det[rise_mask]
    t_fall = t[fall_mask]
    y_fall = I_det[fall_mask]

    t10_r = _find_threshold_crossing(t_rise, y_rise, baseline + 0.1 * amp_rise, rising=True)
    t90_r = _find_threshold_crossing(t_rise, y_rise, baseline + 0.9 * amp_rise, rising=True)
    t90_f = _find_threshold_crossing(t_fall, y_fall, plateau_off + 0.9 * amp_fall, rising=False)
    t10_f = _find_threshold_crossing(t_fall, y_fall, plateau_off + 0.1 * amp_fall, rising=False)

    tau_rise = t90_r - t10_r if np.isfinite(t10_r) and np.isfinite(t90_r) else np.nan
    tau_fall = t10_f - t90_f if np.isfinite(t10_f) and np.isfinite(t90_f) else np.nan

    rise_plateau_valid = plateau_on_std <= 0.01 * max(abs(amp_rise), 1e-30)
    fall_plateau_valid = plateau_off_std <= 0.01 * max(abs(amp_fall), 1e-30)

    notes = []
    if not rise_plateau_valid:
        notes.append("上升段在脉冲内未充分达到稳定平台")
    if not fall_plateau_valid:
        notes.append("下降段在关光后未充分回到稳定平台")
    if not np.isfinite(tau_rise):
        notes.append("无法可靠提取 tau_rise")
    if not np.isfinite(tau_fall):
        notes.append("无法可靠提取 tau_fall")

    return {
        "t": t,
        "P": P,
        "I_det": I_det,
        "baseline": baseline,
        "plateau_on": plateau_on,
        "plateau_off": plateau_off,
        "tau_rise": float(tau_rise) if np.isfinite(tau_rise) else np.nan,
        "tau_fall": float(tau_fall) if np.isfinite(tau_fall) else np.nan,
        "rise_plateau_valid": rise_plateau_valid,
        "fall_plateau_valid": fall_plateau_valid,
        "valid_for_benchmark": rise_plateau_valid and fall_plateau_valid
                               and np.isfinite(tau_rise) and np.isfinite(tau_fall),
        "notes": notes,
    }


def plot_rise_fall(results):
    t = results["t"]
    P = results["P"]
    I_det = results["I_det"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    axes[0].plot(t * 1e3, P * 1e3, "g-", lw=LW_MAIN)
    axes[0].set_ylabel("P (mW)")
    axes[0].set_title("Square-Wave Transient Test")

    axes[1].plot(t * 1e3, I_det * 1e6, "b-", lw=LW_MAIN)
    axes[1].axhline(results["baseline"] * 1e6, color="gray", ls="--", lw=LW_LIGHT, label="baseline")
    axes[1].axhline(results["plateau_on"] * 1e6, color="r", ls="--", lw=LW_LIGHT, label="plateau on")
    axes[1].axhline(results["plateau_off"] * 1e6, color="purple", ls=":", lw=LW_LIGHT, label="plateau off")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("I_det (μA)")
    style_legend(axes[1].legend(loc="best"))
    style_axes(axes[0], grid=True)
    style_axes(axes[1], grid=True)
    finalize_figure("fig_metric_risefall.png")


def make_sine_signal(t, P_dc, mod_depth, f_hz):
    return np.maximum(P_dc * (1.0 + mod_depth * np.sin(2 * np.pi * f_hz * t)), 0.0)


def analyze_frequency_response(params_vec, dark_current, cfg):
    freqs = np.asarray(cfg["f3db_freqs"], dtype=float)
    R_ac = np.zeros_like(freqs)
    n_carrier = infer_n_carrier_from_params(params_vec)

    settle_cycles = 8
    measure_cycles = 4
    points_per_cycle = 200

    for i, f_hz in enumerate(freqs):
        dt = 1.0 / (points_per_cycle * f_hz)
        t = np.arange(0, (settle_cycles + measure_cycles) / f_hz, dt)
        P = make_sine_signal(t, cfg["f3db_power_dc"], cfg["f3db_mod_depth"], f_hz)
        _, I_det, _, _, _ = simulate(
            t, P, params_vec, n_carrier,
            noise_fn=None, dark_current=dark_current
        )

        cut = int(settle_cycles * points_per_cycle)
        t_meas = t[cut:]
        P_meas = P[cut:]
        I_meas = I_det[cut:]

        amp_in = _fundamental_amplitude(t_meas, P_meas, f_hz)
        amp_out = _fundamental_amplitude(t_meas, I_meas, f_hz)
        R_ac[i] = amp_out / max(amp_in, 1e-30)

    low_freq_mask = freqs <= freqs[0] * 10
    if low_freq_mask.sum() == 0:
        low_freq_mask = np.zeros_like(freqs, dtype=bool)
        low_freq_mask[0] = True
    R0 = float(np.mean(R_ac[low_freq_mask]))
    plateau_valid = np.max(np.abs(R_ac[low_freq_mask] - R0) / max(R0, 1e-30)) <= 0.05
    target = R0 / np.sqrt(2)

    f3db = np.nan
    below = np.where(R_ac <= target)[0]
    if len(below) > 0 and below[0] > 0:
        i1 = below[0]
        i0 = i1 - 1
        x0, x1 = np.log10(freqs[i0]), np.log10(freqs[i1])
        y0, y1 = R_ac[i0], R_ac[i1]
        if y1 != y0:
            frac = (target - y0) / (y1 - y0)
            f3db = 10 ** (x0 + frac * (x1 - x0))

    return {
        "freqs": freqs,
        "R_ac": R_ac,
        "R0": R0,
        "f3db": float(f3db) if np.isfinite(f3db) else np.nan,
        "plateau_valid": plateau_valid,
    }


def plot_frequency_response(results):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.semilogx(results["freqs"], results["R_ac"], "o-", lw=LW_MAIN, ms=MS_MAIN, label="Small-signal responsivity")
    ax.axhline(results["R0"], color="gray", ls="--", lw=LW_LIGHT, label="Low-frequency plateau")
    ax.axhline(results["R0"] / np.sqrt(2), color="r", ls=":", lw=LW_LIGHT, label="-3 dB")
    if np.isfinite(results["f3db"]):
        ax.axvline(results["f3db"], color="tab:red", ls="--", lw=LW_LIGHT,
                   label=f"f3dB = {results['f3db']:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("R_ac (A/W)")
    ax.set_title("Small-Signal Frequency Response and f3dB")
    style_axes(ax, grid=True)
    style_legend(ax.legend(loc="best"))
    finalize_figure("fig_metric_f3db.png")


def analyze_noise_case(params_vec, dark_current, noise_name, noise_fn, cfg, responsivity_ref, seed):
    dt = cfg["noise_dt"]
    t = np.arange(0, cfg["noise_duration"], dt)
    P = np.full_like(t, cfg["noise_power_bias"])
    rng = np.random.default_rng(seed)
    n_carrier = infer_n_carrier_from_params(params_vec)
    I_out, I_det, _, _, _ = simulate(
        t, P, params_vec, n_carrier,
        noise_fn=noise_fn, dark_current=dark_current, rng=rng
    )
    noise = I_out - I_det

    fs = 1.0 / dt
    nperseg = min(4096, len(noise))
    freqs, psd = welch(noise, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")

    f_ref = cfg["detectivity_ref_freq"]
    bandwidth = cfg["detectivity_bandwidth"]
    f_lo = max(freqs[1] if len(freqs) > 1 else 0.0, f_ref - 0.5 * bandwidth)
    f_hi = f_ref + 0.5 * bandwidth
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)

    if band_mask.sum() >= 2:
        i_rms = float(np.sqrt(np.trapz(psd[band_mask], freqs[band_mask])))
        noise_density = float(np.sqrt(np.mean(psd[band_mask])))
    else:
        idx = int(np.argmin(np.abs(freqs - f_ref)))
        noise_density = float(np.sqrt(psd[idx]))
        i_rms = float(noise_density * np.sqrt(bandwidth))

    idx_ref = int(np.argmin(np.abs(freqs - f_ref)))
    i_rms_white_approx = float(np.sqrt(psd[idx_ref] * bandwidth))

    nep = i_rms / max(responsivity_ref, 1e-30)
    dstar = np.sqrt(DEVICE_AREA_CM2 * bandwidth) / nep

    i_shot = np.sqrt(2 * E_CHARGE * max(dark_current, 0.0) * bandwidth)
    nep_shot = i_shot / max(responsivity_ref, 1e-30)
    dstar_shot = np.sqrt(DEVICE_AREA_CM2 * bandwidth) / max(nep_shot, 1e-30)

    return {
        "name": noise_name,
        "t": t,
        "noise": noise,
        "freqs": freqs,
        "psd": psd,
        "noise_density": noise_density,
        "i_rms": i_rms,
        "i_shot": float(i_shot),
        "i_rms_white_approx": i_rms_white_approx,
        "nep": float(nep),
        "dstar": float(dstar),
        "dstar_shot": float(dstar_shot),
        "nep_shot": float(nep_shot),
    }


def _compute_noise_spectrum(params_vec, dark_current, noise_fn, power_bias, cfg, seed):
    dt = cfg["noise_dt"]
    t = np.arange(0, cfg["noise_duration"], dt)
    P = np.full_like(t, power_bias)
    rng = np.random.default_rng(seed)
    n_carrier = infer_n_carrier_from_params(params_vec)
    I_out, I_det, _, _, _ = simulate(
        t, P, params_vec, n_carrier,
        noise_fn=noise_fn, dark_current=dark_current, rng=rng
    )
    noise = I_out - I_det
    fs = 1.0 / dt
    nperseg = min(4096, len(noise))
    freqs, psd = welch(noise, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")
    return freqs, psd


def _band_noise_density(freqs, psd, f_ref, bandwidth):
    if len(freqs) == 0:
        return np.nan, np.nan
    f_lo = max(freqs[1] if len(freqs) > 1 else 0.0, f_ref - 0.5 * bandwidth)
    f_hi = f_ref + 0.5 * bandwidth
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)
    if band_mask.sum() >= 2:
        i_rms = float(np.sqrt(np.trapz(psd[band_mask], freqs[band_mask])))
        noise_density = float(np.sqrt(np.mean(psd[band_mask])))
    else:
        idx = int(np.argmin(np.abs(freqs - f_ref)))
        noise_density = float(np.sqrt(psd[idx]))
        i_rms = float(noise_density * np.sqrt(bandwidth))
    return noise_density, i_rms


def analyze_detectivity_vs_power(params_vec, dark_current, noise_name, noise_fn, cfg, linearity, seed):
    P_levels = np.asarray(linearity["P_levels"], dtype=float)
    responsivity = np.asarray(linearity["responsivity"], dtype=float)
    noise_density = np.zeros_like(P_levels)
    dstar = np.zeros_like(P_levels)
    nep = np.zeros_like(P_levels)

    for i, p_bias in enumerate(P_levels):
        freqs, psd = _compute_noise_spectrum(
            params_vec, dark_current, noise_fn, p_bias, cfg, seed + i
        )
        nd, i_rms = _band_noise_density(
            freqs, psd, cfg["detectivity_ref_freq"], cfg["detectivity_bandwidth"]
        )
        noise_density[i] = nd
        nep[i] = i_rms / max(responsivity[i], 1e-30)
        dstar[i] = np.sqrt(DEVICE_AREA_CM2 * cfg["detectivity_bandwidth"]) / max(nep[i], 1e-30)

    return {
        "name": noise_name,
        "P_levels": P_levels,
        "responsivity": responsivity,
        "noise_density": noise_density,
        "nep": nep,
        "dstar": dstar,
    }


def analyze_detectivity_vs_frequency(freq_resp, noise_results):
    freq_axis = np.asarray(freq_resp["freqs"], dtype=float)
    positive_mask = freq_axis > 0
    freq_axis = freq_axis[positive_mask]
    responsivity = np.asarray(freq_resp["R_ac"], dtype=float)[positive_mask]

    results = []
    for noise_result in noise_results:
        freqs = np.asarray(noise_result["freqs"], dtype=float)
        psd = np.asarray(noise_result["psd"], dtype=float)
        valid = freqs > 0
        freqs_valid = freqs[valid]
        nd_valid = np.sqrt(psd[valid])
        interp_log_nd = np.interp(
            np.log10(freq_axis),
            np.log10(freqs_valid),
            np.log10(nd_valid),
            left=np.log10(nd_valid[0]),
            right=np.log10(nd_valid[-1]),
        )
        noise_density = 10 ** interp_log_nd
        nep = noise_density / np.maximum(responsivity, 1e-30)
        dstar = responsivity * np.sqrt(DEVICE_AREA_CM2) / np.maximum(noise_density, 1e-30)
        results.append({
            "name": noise_result["name"],
            "freqs": freq_axis,
            "noise_density": noise_density,
            "nep": nep,
            "dstar": dstar,
        })
    return results


def plot_noise_cases(noise_results, cfg):
    fig, axes = plt.subplots(2, 1, figsize=(11, 9.5))

    for result in noise_results:
        axes[0].loglog(result["freqs"][1:], result["psd"][1:], lw=LW_MED, label=result["name"])
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (A²/Hz)")
    axes[0].set_title("Noise PSD Comparison")
    style_legend(axes[0].legend(loc="best"))

    x = np.arange(len(noise_results))
    nep_vals = [r["nep"] for r in noise_results]
    dstar_vals = [r["dstar"] for r in noise_results]
    axes[1].bar(x - 0.18, np.array(nep_vals) * 1e12, width=0.36, linewidth=1.4,
                edgecolor="#1F4E79", color="#8FB9E3", label="NEP (pW/Hz^0.5)")
    axes[1].bar(x + 0.18, np.array(dstar_vals) / 1e10, width=0.36, linewidth=1.4,
                edgecolor="#7A2F00", color="#F2B279", label="D* (x1e10 Jones)")
    axes[1].set_xticks(x, [r["name"] for r in noise_results])
    axes[1].set_title(
        f"NEP / D* Comparison @ {cfg['detectivity_ref_freq']:.0f} Hz, "
        f"B={cfg['detectivity_bandwidth']:.1f} Hz"
    )
    style_legend(axes[1].legend(loc="best"))
    style_axes(axes[0], grid=True)
    style_axes(axes[1], grid=True, grid_axis="y")
    finalize_figure("fig_metric_noise_nep_dstar.png")


def plot_detectivity_vs_power(dstar_power_results, cfg):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for result in dstar_power_results:
        ax.loglog(result["P_levels"] * 1e3, result["dstar"], "o-", lw=LW_MAIN, ms=MS_MAIN, label=result["name"])
    ax.set_xlabel("Input power (mW)")
    ax.set_ylabel("D* (Jones)")
    ax.set_title(f"D* vs Optical Power @ {cfg['detectivity_ref_freq']:.0f} Hz")
    style_axes(ax, grid=True)
    style_legend(ax.legend(loc="best"))
    finalize_figure("fig_metric_dstar_vs_power.png")


def plot_detectivity_vs_frequency(dstar_freq_results, freq_resp):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for result in dstar_freq_results:
        ax.loglog(result["freqs"], result["dstar"], "o-", lw=LW_MAIN, ms=MS_MAIN, label=result["name"])
    if np.isfinite(freq_resp["f3db"]):
        ax.axvline(freq_resp["f3db"], color="tab:red", ls="--", lw=LW_LIGHT,
                   label=f"f3dB = {freq_resp['f3db']:.1f} Hz")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("D* (Jones)")
    ax.set_title("D* vs Frequency")
    style_axes(ax, grid=True)
    style_legend(ax.legend(loc="best"))
    finalize_figure("fig_metric_dstar_vs_freq.png")


def load_case1_measurements(data_dir=CASE1_DATA_DIR):
    data_dir = Path(data_dir)
    noise = np.loadtxt(data_dir / "INoiseAHZ0.5_F.csv", delimiter=",")
    ip = np.loadtxt(data_dir / "IoptAcm2_PmWcm2_0.16cm2.csv", delimiter=",")
    transient = np.loadtxt(data_dir / "Ioptnorm_tms.csv", delimiter=",")
    drift = np.loadtxt(data_dir / "timeh_Iopt.csv", delimiter=",")
    return {
        "noise_freq_hz": noise[:, 0],
        "noise_density": noise[:, 1],
        "power_density_mw_cm2": ip[:, 0],
        "current_density_a_cm2": ip[:, 1],
        "transient_time_ms": transient[:, 0],
        "transient_norm": transient[:, 1],
        "drift_time_h": drift[:, 0],
        "drift_norm": drift[:, 1],
    }


def _interp_crossing_time(t, y, level, rising=True):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if rising:
        candidates = np.where(y >= level)[0]
    else:
        candidates = np.where(y <= level)[0]
    if len(candidates) == 0:
        return np.nan
    idx = int(candidates[0])
    if idx == 0:
        return float(t[0])
    x0, x1 = t[idx - 1], t[idx]
    y0, y1 = y[idx - 1], y[idx]
    if abs(y1 - y0) < 1e-30:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def extract_case1_single_carrier_params(data_dir=CASE1_DATA_DIR, device_area_cm2=DEVICE_AREA_CM2):
    dataset = load_case1_measurements(data_dir)
    device_area_cm2 = float(device_area_cm2)
    power_ref_w = compute_power_ref_from_density(dataset["power_density_mw_cm2"], device_area_cm2)

    power_w = dataset["power_density_mw_cm2"] * 1e-3 * device_area_cm2
    current_a = dataset["current_density_a_cm2"] * device_area_cm2
    pointwise_responsivity = current_a / np.maximum(power_w, 1e-30)
    fit_coeff = np.polyfit(np.log(power_w), np.log(current_a), 1)
    eta_single = float(fit_coeff[0])
    pref_term = np.exp(fit_coeff[1])
    R_single = float(pref_term * (power_ref_w ** (eta_single - 1.0)))

    t_s = dataset["transient_time_ms"] * 1e-3
    y_trans = np.asarray(dataset["transient_norm"], dtype=float)
    peak_idx = int(np.argmax(y_trans))
    peak_val = float(max(y_trans[peak_idx], 1e-30))
    rise_t = t_s[: peak_idx + 1]
    rise_y = y_trans[: peak_idx + 1] / peak_val
    fall_t = t_s[peak_idx:]
    fall_y = y_trans[peak_idx:] / peak_val

    t10_r = _interp_crossing_time(rise_t, rise_y, 0.1, rising=True)
    t90_r = _interp_crossing_time(rise_t, rise_y, 0.9, rising=True)
    t90_f = _interp_crossing_time(fall_t, fall_y, 0.9, rising=False)
    t10_f = _interp_crossing_time(fall_t, fall_y, 0.1, rising=False)
    tau_rise = float(max((t90_r - t10_r) / np.log(9.0), 1e-9))
    tau_fall = float(max((t10_f - t90_f) / np.log(9.0), 1e-9))

    drift_t_s = dataset["drift_time_h"] * 3600.0
    drift_y = np.asarray(dataset["drift_norm"], dtype=float)
    best = None
    for tau_s in np.logspace(3, 9, 400):
        basis = 1.0 - np.exp(-drift_t_s / tau_s)
        gamma_candidate = np.dot(basis, drift_y - 1.0) / max(np.dot(basis, basis), 1e-30)
        gamma_candidate = float(np.clip(gamma_candidate, -0.999999, 10.0))
        pred = 1.0 + gamma_candidate * basis
        err = float(np.mean((pred - drift_y) ** 2))
        if best is None or err < best[0]:
            best = (err, tau_s, gamma_candidate, pred)
    _, tau_s_best, gamma, drift_fit = best
    n_linear = min(3, len(power_w))
    linear_fit = np.polyfit(power_w[:n_linear], current_a[:n_linear], 1)
    r_ideal = float(max(linear_fit[0], 0.0))

    params = {
        "R_single": R_single,
        "eta_single": eta_single,
        "tau_rise_single": tau_rise,
        "tau_fall_single": tau_fall,
        "alpha": 0.0,
        "beta": 1.0,
        "delta": 0.0,
        "gamma": float(gamma),
        "tau_drift": float(tau_s_best),
        "drift_scale": 1.0,
    }
    noise_fn = make_noise_function_from_psd(
        dataset["noise_freq_hz"],
        dataset["noise_density"],
        label="case1_psd_noise",
    )
    return {
        "dataset": dataset,
        "params": params,
        "power_ref_w": power_ref_w,
        "noise_fn": noise_fn,
        "drift_fit": np.asarray(drift_fit, dtype=float),
        "r_ideal": r_ideal,
        "transient_metrics": {
            "t10_r": t10_r,
            "t90_r": t90_r,
            "t90_f": t90_f,
            "t10_f": t10_f,
        },
    }


def plot_case1_parameter_fits(case1_results, device_area_cm2=DEVICE_AREA_CM2,
                              output_path="fig_case1_parameter_fits.png"):
    dataset = case1_results["dataset"]
    params = case1_results["params"]
    power_ref_w = case1_results["power_ref_w"]

    power_w = dataset["power_density_mw_cm2"] * 1e-3 * float(device_area_cm2)
    current_a = dataset["current_density_a_cm2"] * float(device_area_cm2)
    power_fit = np.logspace(np.log10(np.min(power_w)), np.log10(np.max(power_w)), 300)
    current_fit = carrier_target_current(power_fit, params["R_single"], params["eta_single"], power_ref_w)

    t_trans = dataset["transient_time_ms"] * 1e-3
    peak = np.max(dataset["transient_norm"])
    rise_model = 1.0 - np.exp(-(t_trans - t_trans[0]) / params["tau_rise_single"])
    rise_model[t_trans < t_trans[0]] = 0.0
    peak_idx = int(np.argmax(dataset["transient_norm"]))
    trans_fit = np.empty_like(t_trans)
    trans_fit[:peak_idx] = rise_model[:peak_idx] * peak
    fall_axis = t_trans[peak_idx:] - t_trans[peak_idx]
    trans_fit[peak_idx:] = peak * np.exp(-fall_axis / params["tau_fall_single"])

    drift_t_h = dataset["drift_time_h"]
    noise_psd_fit = _resample_noise_density(
        dataset["noise_freq_hz"],
        dataset["noise_freq_hz"],
        dataset["noise_density"],
    )

    fig, axes = plt.subplots(1, 4, figsize=(31.0, 5.8))

    bold_spine_width = 2.8
    bold_tick_width = 2.2
    bold_minor_tick_width = 1.6
    bold_tick_size = 8.0
    bold_minor_tick_size = 4.5
    bold_tick_labelsize = 17
    fit_line_width = 4.0
    paper_marker_size = 7.5
    paper_marker_edge_width = 1.35

    def apply_bold_plot_style(ax, grid=False):
        style_axes(ax, grid=grid)
        for spine in ax.spines.values():
            spine.set_linewidth(bold_spine_width)
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            width=bold_tick_width,
            length=bold_tick_size,
            labelsize=bold_tick_labelsize,
            pad=7,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            width=bold_minor_tick_width,
            length=bold_minor_tick_size,
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    ax = axes[0]
    ax.plot(
        drift_t_h,
        dataset["drift_norm"],
        "o",
        color="black",
        ms=paper_marker_size,
        markeredgewidth=paper_marker_edge_width,
    )
    ax.plot(
        drift_t_h,
        case1_results["drift_fit"],
        color="tab:red",
        lw=fit_line_width,
    )
    ax.set_xlabel("time (h)")
    ax.set_ylabel("Iopt(norm)")
    ax.set_title("Drift Fit")
    apply_bold_plot_style(ax, grid=False)

    ax = axes[1]
    ax.loglog(
        dataset["noise_freq_hz"],
        dataset["noise_density"],
        "o",
        color="black",
        ms=paper_marker_size,
        markeredgewidth=paper_marker_edge_width,
    )
    ax.loglog(
        dataset["noise_freq_hz"],
        noise_psd_fit,
        color="tab:red",
        lw=fit_line_width,
    )
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("Noise current (A/Hz^0.5)")
    ax.set_title("Noise-Spectrum Fit")
    apply_bold_plot_style(ax, grid=False)

    ax = axes[2]
    ax.loglog(
        power_w * 1e3,
        current_a * 1e6,
        "o",
        color="black",
        ms=paper_marker_size,
        markeredgewidth=paper_marker_edge_width,
    )
    ax.loglog(
        power_fit * 1e3,
        current_fit * 1e6,
        color="tab:red",
        lw=fit_line_width,
    )
    ax.set_xlabel("P (mW)")
    ax.set_ylabel("Iopt (uA)")
    ax.set_title("I-P Fit")
    apply_bold_plot_style(ax, grid=False)

    ax = axes[3]
    ax.plot(
        dataset["transient_time_ms"],
        dataset["transient_norm"],
        "o",
        color="black",
        ms=paper_marker_size,
        markeredgewidth=paper_marker_edge_width,
    )
    ax.plot(
        dataset["transient_time_ms"],
        trans_fit,
        color="tab:red",
        lw=fit_line_width,
    )
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("Iopt(norm)")
    ax.set_title("Transient Fit")
    apply_bold_plot_style(ax, grid=False)

    finalize_figure(output_path)


def synthesize_case1_noise_reconstruction(case1_results, duration_s=None, dt=None, seed=123):
    dataset = case1_results["dataset"]
    duration_s = ANALYSIS_CONFIG["noise_duration"] if duration_s is None else float(duration_s)
    dt = ANALYSIS_CONFIG["noise_dt"] if dt is None else float(dt)
    if duration_s <= 0 or dt <= 0:
        raise ValueError("duration_s and dt must be positive")

    t = np.arange(0.0, duration_s, dt, dtype=np.float64)
    rng = np.random.default_rng(seed)
    noise_fn = case1_results["noise_fn"]
    noise_trace = np.asarray(
        noise_fn(t, np.zeros_like(t), np.zeros_like(t), rng),
        dtype=np.float64,
    )
    fs = 1.0 / dt
    nperseg = min(4096, len(noise_trace))
    freqs, psd = welch(noise_trace, fs=fs, nperseg=nperseg, detrend="constant", scaling="density")
    reconstructed_nd = np.sqrt(np.maximum(psd, 0.0))
    target_nd = _resample_noise_density(
        freqs,
        dataset["noise_freq_hz"],
        dataset["noise_density"],
    )
    return {
        "t": t,
        "noise_trace": noise_trace,
        "freqs": freqs,
        "reconstructed_psd": psd,
        "reconstructed_nd": reconstructed_nd,
        "target_nd": target_nd,
    }


def plot_case1_noise_reconstruction(case1_results, output_path="fig_case1_noise_reconstruction.png",
                                    duration_s=None, dt=None, seed=123):
    dataset = case1_results["dataset"]
    validation = synthesize_case1_noise_reconstruction(
        case1_results,
        duration_s=duration_s,
        dt=dt,
        seed=seed,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 9.5))

    axes[0].plot(validation["t"], validation["noise_trace"] * 1e9, color="tab:blue", lw=LW_MED)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Noise current (nA)")
    axes[0].set_title("Time-Domain Noise Synthesized From Paper PSD")
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(dataset["noise_freq_hz"], dataset["noise_density"], "o", label="Paper PSD points")
    valid_mask = validation["freqs"] > 0
    axes[1].loglog(
        validation["freqs"][valid_mask],
        validation["target_nd"][valid_mask],
        "--",
        lw=LW_MED,
        label="Interpolated PSD target",
    )
    axes[1].loglog(
        validation["freqs"][valid_mask],
        validation["reconstructed_nd"][valid_mask],
        "-",
        lw=LW_HEAVY,
        label="PSD from synthesized time trace",
    )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Noise current (A/Hz^0.5)")
    axes[1].set_title("Frequency-Domain Validation: Paper PSD vs Reconstructed PSD")
    style_axes(axes[1], grid=True)
    style_legend(axes[1].legend(loc="best"))

    fig.suptitle("case1 Noise Reconstruction Validation", fontsize=18, fontweight="bold")
    finalize_figure(output_path)


def load_case2_measurements(data_dir=CASE2_DATA_DIR):
    data_dir = Path(data_dir)
    transient = np.loadtxt(data_dir / "case2_IoutuA_TimeS.csv", delimiter=",")
    response = np.loadtxt(data_dir / "case2_RespAW-1_PWmm-2.csv", delimiter=",")

    transient = transient[np.argsort(transient[:, 0])]
    response = response[np.argsort(response[:, 0])]
    return {
        "transient_time_s": np.asarray(transient[:, 0], dtype=float),
        "transient_current_ua": np.asarray(transient[:, 1], dtype=float),
        "transient_current_a": np.asarray(transient[:, 1], dtype=float) * 1e-6,
        "power_density_w_mm2": np.asarray(response[:, 0], dtype=float),
        "responsivity_aw": np.asarray(response[:, 1], dtype=float),
    }


def steady_state_responsivity_single(power_total_w, r_single, eta_single, power_ref_w):
    power_total_w = np.asarray(power_total_w, dtype=float)
    current_a = carrier_target_current(power_total_w, r_single, eta_single, power_ref_w=power_ref_w)
    return current_a / np.maximum(power_total_w, 1e-30)


def fit_case2_responsivity_curve(case2_dataset, device_area_mm2):
    power_density = np.asarray(case2_dataset["power_density_w_mm2"], dtype=float)
    responsivity = np.asarray(case2_dataset["responsivity_aw"], dtype=float)
    power_total_w = compute_total_power_from_density_w_mm2(power_density, device_area_mm2)
    positive = power_total_w > 0
    if np.count_nonzero(positive) < 2:
        raise ValueError("case2 responsivity fit needs at least two positive power points")

    power_total_w = power_total_w[positive]
    power_density = power_density[positive]
    responsivity = responsivity[positive]
    power_ref_w = float(np.min(power_total_w))

    slope_resp = np.polyfit(np.log10(power_total_w), np.log10(np.maximum(responsivity, 1e-30)), 1)[0]
    eta_guess = float(np.clip(slope_resp + 1.0, 0.02, 1.4))
    r_guess = float(np.clip(responsivity[np.argmin(power_total_w)], 1e-12, 1e6))
    x0 = np.array([np.log10(r_guess), eta_guess], dtype=float)
    bounds = (
        np.array([np.log10(1e-12), 0.01], dtype=float),
        np.array([np.log10(1e6), 1.5], dtype=float),
    )

    def residual_fn(vector):
        r_single = float(10 ** vector[0])
        eta_single = float(vector[1])
        predicted = steady_state_responsivity_single(power_total_w, r_single, eta_single, power_ref_w)
        return np.log10(np.maximum(predicted, 1e-30)) - np.log10(np.maximum(responsivity, 1e-30))

    result = least_squares(
        residual_fn,
        x0=x0,
        bounds=bounds,
        loss="soft_l1",
        f_scale=0.05,
        max_nfev=2000,
        verbose=0,
    )
    r_single = float(10 ** result.x[0])
    eta_single = float(result.x[1])
    responsivity_fit = steady_state_responsivity_single(power_total_w, r_single, eta_single, power_ref_w)
    current_measured_a = responsivity * power_total_w
    current_fit_a = responsivity_fit * power_total_w
    alpha_global = float(
        np.polyfit(np.log10(power_total_w), np.log10(np.maximum(current_fit_a, 1e-30)), 1)[0]
    )
    return {
        "power_density_w_mm2": power_density,
        "power_total_w": power_total_w,
        "power_ref_w": power_ref_w,
        "responsivity_measured_aw": responsivity,
        "responsivity_fit_aw": responsivity_fit,
        "current_measured_a": current_measured_a,
        "current_fit_a": current_fit_a,
        "R_single": r_single,
        "eta_single": eta_single,
        "alpha_global": alpha_global,
        "rmse_log10_resp": float(np.sqrt(np.mean(residual_fn(result.x) ** 2))),
        "fit_stats": {
            "success": bool(result.success),
            "cost": float(result.cost),
            "nfev": float(result.nfev),
            "optimality": float(result.optimality),
        },
    }


def _moving_average(values, window):
    values = np.asarray(values, dtype=float)
    window = max(int(window), 1)
    if window <= 1 or len(values) <= 2:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _boolean_segments(mask):
    mask = np.asarray(mask, dtype=bool)
    segments = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(mask)))
    return segments


def estimate_square_wave_timing_from_trace(t, current_a):
    t = np.asarray(t, dtype=float)
    current_a = np.asarray(current_a, dtype=float)
    smoothed = _moving_average(current_a, max(5, len(current_a) // 24))
    lo = float(np.quantile(smoothed, 0.15))
    hi = float(np.quantile(smoothed, 0.85))
    mid = 0.5 * (lo + hi)
    on_mask = smoothed >= mid

    dt_med = float(np.median(np.diff(t)))
    raw_segments = _boolean_segments(on_mask)
    min_duration = max(4.0 * dt_med, 0.35)
    segments = [
        (start, end) for start, end in raw_segments
        if (t[end - 1] - t[start] + dt_med) >= min_duration
    ]
    if not segments:
        segments = raw_segments

    if not segments:
        period_guess = max((t[-1] - t[0]) / 5.0, 2.0 * dt_med)
        return {
            "t_start_s": float(max(t[0], 0.0)),
            "t_on_s": float(0.5 * period_guess),
            "t_off_s": float(0.5 * period_guess),
            "smoothed_current_a": smoothed,
            "mid_level_a": mid,
        }

    on_durations = [float(t[end - 1] - t[start] + dt_med) for start, end in segments]
    off_durations = [
        float(t[next_start] - t[end - 1])
        for (_, end), (next_start, _) in zip(segments[:-1], segments[1:])
    ]
    t_on_s = float(np.median(on_durations))
    t_off_s = float(np.median(off_durations)) if off_durations else float(t_on_s)
    t_start_s = float(t[segments[0][0]])
    return {
        "t_start_s": t_start_s,
        "t_on_s": max(t_on_s, 4.0 * dt_med),
        "t_off_s": max(t_off_s, 4.0 * dt_med),
        "smoothed_current_a": smoothed,
        "mid_level_a": mid,
    }


def estimate_case2_power_density_guess(response_fit, photo_current_a, device_area_mm2):
    photo_current_a = float(max(photo_current_a, 1e-30))
    eta_single = float(max(response_fit["eta_single"], 1e-6))
    denom = float(max(response_fit["R_single"] * response_fit["power_ref_w"], 1e-30))
    power_total_w = response_fit["power_ref_w"] * (photo_current_a / denom) ** (1.0 / eta_single)
    return float(power_total_w / max(device_area_mm2, 1e-30))


def make_periodic_square_profile(t, power_on_w, t_start_s, t_on_s, t_off_s):
    t = np.asarray(t, dtype=float)
    if t_on_s <= 0 or t_off_s <= 0:
        raise ValueError("t_on_s and t_off_s must be positive")
    power = np.zeros_like(t, dtype=float)
    active = t >= t_start_s
    if not np.any(active):
        return power
    period = float(t_on_s + t_off_s)
    phase = np.mod(t[active] - t_start_s, period)
    power[active] = np.where(phase < t_on_s, float(power_on_w), 0.0)
    return power


def build_case2_single_params(response_fit, tau_rise_single, tau_fall_single,
                              alpha, beta, delta, drift_defaults, r_single_override=None):
    r_single = response_fit["R_single"] if r_single_override is None else r_single_override
    return {
        "R_single": float(r_single),
        "eta_single": float(response_fit["eta_single"]),
        "tau_rise_single": float(tau_rise_single),
        "tau_fall_single": float(tau_fall_single),
        "alpha": float(alpha),
        "beta": float(beta),
        "delta": float(delta),
        "gamma": float(drift_defaults["gamma"]),
        "tau_drift": float(drift_defaults["tau_drift"]),
        "drift_scale": float(drift_defaults["drift_scale"]),
    }


def analyze_case2_transient_metrics(t, current_a, t_start_s, t_on_s, t_off_s):
    t = np.asarray(t, dtype=float)
    current_a = np.asarray(current_a, dtype=float)
    period = float(t_on_s + t_off_s)
    if period <= 0:
        raise ValueError("period must be positive")

    cycle_starts = []
    k = 0
    while True:
        rise_edge = t_start_s + k * period
        fall_edge = rise_edge + t_on_s
        cycle_end = fall_edge + t_off_s
        if cycle_end > t[-1]:
            break
        cycle_starts.append(rise_edge)
        k += 1
    if not cycle_starts:
        cycle_starts = [t_start_s]

    rise_edge = float(cycle_starts[-1])
    fall_edge = rise_edge + t_on_s
    baseline, baseline_std = _window_stats(
        t, current_a, rise_edge - 0.35 * t_off_s, rise_edge - 0.05 * t_off_s
    )
    plateau_on, plateau_on_std = _window_stats(
        t, current_a, rise_edge + 0.70 * t_on_s, rise_edge + 0.95 * t_on_s
    )
    plateau_off, plateau_off_std = _window_stats(
        t, current_a, fall_edge + 0.70 * t_off_s, fall_edge + 0.95 * t_off_s
    )

    amp_rise = plateau_on - baseline
    amp_fall = plateau_on - plateau_off

    rise_mask = (t >= rise_edge) & (t < rise_edge + t_on_s)
    fall_mask = (t >= fall_edge) & (t < fall_edge + t_off_s)
    t_rise = t[rise_mask]
    y_rise = current_a[rise_mask]
    t_fall = t[fall_mask]
    y_fall = current_a[fall_mask]

    t10_r = _find_threshold_crossing(t_rise, y_rise, baseline + 0.1 * amp_rise, rising=True)
    t90_r = _find_threshold_crossing(t_rise, y_rise, baseline + 0.9 * amp_rise, rising=True)
    t90_f = _find_threshold_crossing(t_fall, y_fall, plateau_off + 0.9 * amp_fall, rising=False)
    t10_f = _find_threshold_crossing(t_fall, y_fall, plateau_off + 0.1 * amp_fall, rising=False)

    tau_rise = t90_r - t10_r if np.isfinite(t10_r) and np.isfinite(t90_r) else np.nan
    tau_fall = t10_f - t90_f if np.isfinite(t90_f) and np.isfinite(t10_f) else np.nan
    return {
        "baseline_a": float(baseline),
        "baseline_std_a": float(baseline_std),
        "plateau_on_a": float(plateau_on),
        "plateau_on_std_a": float(plateau_on_std),
        "plateau_off_a": float(plateau_off),
        "plateau_off_std_a": float(plateau_off_std),
        "tau_rise_s_10_90": float(tau_rise) if np.isfinite(tau_rise) else np.nan,
        "tau_fall_s_10_90": float(tau_fall) if np.isfinite(tau_fall) else np.nan,
        "rise_edge_s": rise_edge,
        "fall_edge_s": float(fall_edge),
        "t10_r_s": float(t10_r) if np.isfinite(t10_r) else np.nan,
        "t90_r_s": float(t90_r) if np.isfinite(t90_r) else np.nan,
        "t90_f_s": float(t90_f) if np.isfinite(t90_f) else np.nan,
        "t10_f_s": float(t10_f) if np.isfinite(t10_f) else np.nan,
    }


def fit_case2_transient(case2_dataset, response_fit, device_area_mm2, drift_defaults=None,
                        fixed_rise_time_ms=CASE2_FIXED_RISE_TIME_MS,
                        fixed_fall_time_ms=CASE2_FIXED_FALL_TIME_MS,
                        main_branch_ceiling_ua=CASE2_MAIN_BRANCH_CEILING_UA):
    drift_defaults = PARAMS_SINGLE_CARRIER if drift_defaults is None else dict(drift_defaults)
    set_device_context(device_area_cm2=float(device_area_mm2) * 0.01, power_ref_w=response_fit["power_ref_w"])
    t = np.asarray(case2_dataset["transient_time_s"], dtype=float)
    current_a = np.asarray(case2_dataset["transient_current_a"], dtype=float)

    timing_guess = estimate_square_wave_timing_from_trace(t, current_a)
    baseline_guess = float(np.quantile(current_a, 0.08))
    peak_guess = float(np.quantile(current_a, 0.92))
    amp_guess = max(peak_guess - baseline_guess, 1e-12)
    t_span = float(t[-1] - t[0])
    power_density_ref = float(response_fit["power_ref_w"] / max(device_area_mm2, 1e-30))
    fixed_rise_time_s = float(fixed_rise_time_ms) * 1e-3
    fixed_fall_time_s = float(fixed_fall_time_ms) * 1e-3
    main_branch_ceiling_a = float(main_branch_ceiling_ua) * 1e-6
    if fixed_rise_time_s <= 0 or fixed_fall_time_s <= 0:
        raise ValueError("fixed_rise_time_ms and fixed_fall_time_ms must be positive")
    if main_branch_ceiling_a <= 0:
        raise ValueError("main_branch_ceiling_ua must be positive")
    fixed_tau_rise_single = float(fixed_rise_time_s / np.log(9.0))
    fixed_tau_fall_single = float(fixed_fall_time_s / np.log(9.0))
    scale = max(float(np.std(current_a)), amp_guess, 1e-10)
    power_on_w = float(response_fit["power_ref_w"])

    def encode_specs(specs):
        x0 = []
        lower = []
        upper = []
        for _, mode, lo, hi, value in specs:
            if mode == "log":
                x0.append(np.log10(max(float(value), 1e-30)))
                lower.append(np.log10(lo))
                upper.append(np.log10(hi))
            else:
                x0.append(float(value))
                lower.append(lo)
                upper.append(hi)
        return np.asarray(x0, dtype=float), (
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        )

    def decode_specs(specs, vector):
        values = {}
        for (name, mode, _, _, _), raw_value in zip(specs, vector):
            values[name] = float(10 ** raw_value) if mode == "log" else float(raw_value)
        return values

    def build_phase_masks(t_start_s, t_on_s, t_off_s):
        period = float(t_on_s + t_off_s)
        if period <= 0:
            raise ValueError("period must be positive")
        active = t >= t_start_s
        phase = np.mod(np.maximum(t - t_start_s, 0.0), period)
        on_mask = active & (phase < t_on_s)
        off_mask = active & (~on_mask)
        rise_fast_mask = on_mask & (phase <= fixed_rise_time_s)
        on_plateau_mask = on_mask & (~rise_fast_mask)
        on_anchor_start = min(max(fixed_rise_time_s, 0.70 * t_on_s), 0.95 * t_on_s)
        on_anchor_mask = on_mask & (phase >= on_anchor_start) & (phase <= 0.95 * t_on_s)
        off_elapsed = np.maximum(phase - t_on_s, 0.0)
        off_early_mask = off_mask & (off_elapsed <= fixed_fall_time_s)
        late_off_threshold = max(t_off_s - 0.25, fixed_fall_time_s)
        off_late_mask = off_mask & (off_elapsed >= late_off_threshold)
        pre_mask = ~active
        return {
            "active": active,
            "phase": phase,
            "on_mask": on_mask,
            "off_mask": off_mask,
            "rise_fast_mask": rise_fast_mask,
            "on_plateau_mask": on_plateau_mask,
            "on_anchor_mask": on_anchor_mask,
            "off_early_mask": off_early_mask,
            "off_late_mask": off_late_mask,
            "pre_mask": pre_mask,
        }

    def build_main_target(masks):
        target = np.minimum(current_a, main_branch_ceiling_a)
        target[masks["on_anchor_mask"]] = main_branch_ceiling_a
        return target

    def derive_main_r_single(dark_current_a):
        target_photo_current_a = max(main_branch_ceiling_a - float(dark_current_a), 1e-16)
        return float(target_photo_current_a / max(power_on_w, 1e-30))

    dark_current_lower = max(np.min(current_a) - 0.08 * amp_guess, 0.0)
    dark_current_upper = min(np.quantile(current_a, 0.18), np.max(current_a), main_branch_ceiling_a - 1e-12)
    if dark_current_upper <= dark_current_lower:
        dark_current_lower = max(0.0, min(dark_current_lower, main_branch_ceiling_a - 2e-12))
        dark_current_upper = max(dark_current_lower + 1e-12, min(main_branch_ceiling_a - 1e-12, baseline_guess))

    main_specs = [
        ("t_start_s", "linear", -0.05, 0.08, float(np.clip(max(timing_guess["t_start_s"], 0.0), -0.05, 0.08))),
        ("t_on_s", "linear", 0.88, 1.08, float(np.clip(timing_guess["t_on_s"], 0.88, 1.08))),
        ("t_off_s", "linear", 0.92, 1.12, float(np.clip(timing_guess["t_off_s"], 0.92, 1.12))),
        ("dark_current_a", "linear", dark_current_lower, dark_current_upper,
         float(np.clip(baseline_guess, dark_current_lower, dark_current_upper))),
    ]
    main_x0, main_bounds = encode_specs(main_specs)

    def simulate_main(values):
        derived_r_single = derive_main_r_single(values["dark_current_a"])
        params = build_case2_single_params(
            response_fit=response_fit,
            tau_rise_single=fixed_tau_rise_single,
            tau_fall_single=fixed_tau_fall_single,
            alpha=0.0,
            beta=1.0,
            delta=0.0,
            drift_defaults=drift_defaults,
            r_single_override=derived_r_single,
        )
        power_profile = make_periodic_square_profile(
            t,
            power_on_w,
            values["t_start_s"],
            values["t_on_s"],
            values["t_off_s"],
        )
        _, current_fit_a, x1, x2, x3 = simulate(
            t,
            power_profile,
            params_to_vec(params),
            n_carrier=1,
            noise_fn=None,
            dark_current=values["dark_current_a"],
            trap_mode="binary",
            trap_output_mode="illumination_gated",
        )
        return params, power_profile, current_fit_a, x1, x2, x3, derived_r_single

    def main_residual_fn(vector):
        values = decode_specs(main_specs, vector)
        _, _, current_fit_a, _, _, _, _ = simulate_main(values)
        masks = build_phase_masks(values["t_start_s"], values["t_on_s"], values["t_off_s"])
        main_target_a = build_main_target(masks)
        raw = (current_fit_a - main_target_a) / scale
        residual = np.zeros_like(raw)
        residual[masks["pre_mask"]] = 3.0 * raw[masks["pre_mask"]]
        residual[masks["rise_fast_mask"]] = 2.8 * raw[masks["rise_fast_mask"]]
        residual[masks["off_late_mask"]] = 2.8 * raw[masks["off_late_mask"]]
        residual[masks["on_plateau_mask"]] = 1.8 * raw[masks["on_plateau_mask"]]
        residual[masks["off_early_mask"]] = 1.4 * raw[masks["off_early_mask"]]
        return residual

    main_de = differential_evolution(
        lambda vector: float(np.mean(main_residual_fn(vector) ** 2)),
        bounds=list(zip(main_bounds[0], main_bounds[1])),
        seed=42,
        maxiter=40,
        popsize=8,
        polish=False,
    )
    main_result = least_squares(
        main_residual_fn,
        x0=main_de.x,
        bounds=main_bounds,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=4000,
        verbose=0,
    )
    main_values = decode_specs(main_specs, main_result.x)
    main_params, power_profile, main_current_a, _, _, _, main_r_single = simulate_main(main_values)
    main_values["R_single_transient"] = float(main_r_single)
    phase_masks = build_phase_masks(main_values["t_start_s"], main_values["t_on_s"], main_values["t_off_s"])
    main_rmse_a = float(np.sqrt(np.mean((main_current_a - current_a) ** 2)))

    residual_target_a = np.clip(current_a - main_branch_ceiling_a, 0.0, None)
    main_photo_ceiling_a = max(main_branch_ceiling_a - main_values["dark_current_a"], 1e-16)
    ratio_guess = float(
        np.clip(np.max(residual_target_a) / max(0.5 * main_photo_ceiling_a, 1e-30), 1e-6, 10.0)
    )
    x2_specs = [
        ("x2_capture_on_rate", "log", 1e-4, 1e4, 1.0),
        ("beta", "log", 1e-3, 5e2, 20.0),
        ("trap_x1_gain", "log", 1e-8, 1e3, ratio_guess),
    ]
    x2_x0, x2_bounds = encode_specs(x2_specs)

    def complete_x2_values(values):
        values = dict(values)
        values["alpha"] = float(values["x2_capture_on_rate"]) / max(power_on_w, 1e-30)
        return values

    def simulate_x2_only(values):
        values = complete_x2_values(values)
        params = build_case2_single_params(
            response_fit=response_fit,
            tau_rise_single=fixed_tau_rise_single,
            tau_fall_single=fixed_tau_fall_single,
            alpha=values["alpha"],
            beta=values["beta"],
            delta=0.0,
            drift_defaults=drift_defaults,
            r_single_override=main_values["R_single_transient"],
        )
        _, _current_fit_a, x1, x2_state, x3 = simulate(
            t,
            power_profile,
            params_to_vec(params),
            n_carrier=1,
            noise_fn=None,
            dark_current=0.0,
            trap_mode="power",
            trap_output_mode="always",
            trap_x1_ratio=values["trap_x1_gain"],
        )
        main_photo_current = np.sum(x1, axis=0)
        x2_current_a = values["trap_x1_gain"] * main_photo_current * x2_state * np.clip(1.0 + x3, 0.0, None)
        return params, x2_current_a, x2_state, x3

    def x2_residual_fn(vector):
        values = complete_x2_values(decode_specs(x2_specs, vector))
        _, x2_current_a, x2_state, _ = simulate_x2_only(values)
        current_weights = np.full_like(t, 0.03, dtype=float)
        current_weights[phase_masks["rise_fast_mask"]] = 0.25
        current_weights[phase_masks["on_plateau_mask"]] = 2.4
        current_residual = current_weights * (x2_current_a - residual_target_a) / scale

        reset_mask = phase_masks["off_late_mask"] | phase_masks["pre_mask"]
        reset_residual = 6.0 * x2_state[reset_mask]
        return np.concatenate([current_residual, reset_residual])

    x2_de = differential_evolution(
        lambda vector: float(np.mean(x2_residual_fn(vector) ** 2)),
        bounds=list(zip(x2_bounds[0], x2_bounds[1])),
        seed=42,
        maxiter=50,
        popsize=8,
        polish=False,
    )
    x2_result = least_squares(
        x2_residual_fn,
        x0=x2_de.x,
        bounds=x2_bounds,
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=5000,
        verbose=0,
    )
    x2_values = complete_x2_values(decode_specs(x2_specs, x2_result.x))
    _pre_params, _pre_x2_current_a, pre_x2_state, _pre_x3 = simulate_x2_only(x2_values)
    trap_x2_reference_state = float(max(np.max(pre_x2_state), 1e-30))
    trap_x1_ratio = float(x2_values["trap_x1_gain"] * trap_x2_reference_state)

    values = {
        **main_values,
        **x2_values,
        "trap_x1_ratio": trap_x1_ratio,
        "trap_x2_reference_state": trap_x2_reference_state,
        "delta": 0.0,
        "tau_rise_single": fixed_tau_rise_single,
        "tau_fall_single": fixed_tau_fall_single,
        "rise_time_10_90_s_fixed": fixed_rise_time_s,
        "fall_time_90_10_s_fixed": fixed_fall_time_s,
        "main_branch_ceiling_ua": float(main_branch_ceiling_ua),
        "power_density_on_w_mm2": power_density_ref,
        "power_on_w": power_on_w,
    }
    params = build_case2_single_params(
        response_fit=response_fit,
        tau_rise_single=fixed_tau_rise_single,
        tau_fall_single=fixed_tau_fall_single,
        alpha=values["alpha"],
        beta=values["beta"],
        delta=0.0,
        drift_defaults=drift_defaults,
        r_single_override=values["R_single_transient"],
    )
    _, current_fit_a, x1, x2, x3 = simulate(
        t,
        power_profile,
        params_to_vec(params),
        n_carrier=1,
        noise_fn=None,
        dark_current=values["dark_current_a"],
        trap_mode="power",
        trap_output_mode="always",
        trap_x1_ratio=values["trap_x1_ratio"],
        trap_x2_reference_state=values["trap_x2_reference_state"],
    )
    current_carrier_only_a = main_current_a
    metrics = analyze_case2_transient_metrics(
        t,
        current_fit_a,
        values["t_start_s"],
        values["t_on_s"],
        values["t_off_s"],
    )
    x2_envelope = np.clip(x2 / max(values["trap_x2_reference_state"], 1e-30), 0.0, 1.0)
    x2_current_a = values["trap_x1_ratio"] * np.sum(x1, axis=0) * x2_envelope * np.clip(1.0 + x3, 0.0, None)
    metrics["tau_rise_main_fixed_s"] = fixed_rise_time_s
    metrics["tau_fall_main_fixed_s"] = fixed_fall_time_s
    metrics["tau_rise_param_fixed_s"] = fixed_tau_rise_single
    metrics["tau_fall_param_fixed_s"] = fixed_tau_fall_single
    x2_capture_on = float(values["alpha"]) * float(power_on_w)
    x2_total_on = x2_capture_on + float(values["beta"])
    metrics["tau_x2_on_s"] = float(1.0 / max(x2_total_on, 1e-30))
    metrics["tau_x2_off_s"] = float(1.0 / max(values["beta"], 1e-30))
    metrics["x2_capture_rate_on_s_inv"] = float(x2_capture_on)
    metrics["x2_steady_state_on"] = float(x2_capture_on / max(x2_total_on, 1e-30))
    metrics["x2_reference_state"] = float(values["trap_x2_reference_state"])
    metrics["x2_max_ratio_to_x1"] = float(values["trap_x1_ratio"])
    metrics["x2_internal_gain"] = float(values["trap_x1_gain"])
    metrics["x2_peak_state"] = float(np.max(x2))
    metrics["x2_peak_current_a"] = float(np.max(x2_current_a))
    metrics["x2_residual_state_end"] = float(np.max(x2[phase_masks["off_late_mask"] | phase_masks["pre_mask"]]))
    metrics["baseline_mismatch_a"] = float(metrics["baseline_a"] - baseline_guess)
    metrics["main_branch_rmse_a"] = main_rmse_a
    metrics["x2_target_peak_current_a"] = float(np.max(residual_target_a))
    metrics["main_branch_ceiling_a"] = float(main_branch_ceiling_a)
    rmse_a = float(np.sqrt(np.mean((current_fit_a - current_a) ** 2)))
    return {
        "params": params,
        "params_vec": params_to_vec(params),
        "fit_values": values,
        "power_profile_w": power_profile,
        "current_fit_a": current_fit_a,
        "current_measured_a": current_a,
        "current_carrier_only_a": current_carrier_only_a,
        "x2_current_a": x2_current_a,
        "x2_target_a": residual_target_a,
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "metrics": metrics,
        "timing_guess": timing_guess,
        "fit_stats": {
            "success": bool(main_result.success and x2_result.success),
            "main_cost": float(main_result.cost),
            "main_nfev": float(main_result.nfev),
            "main_optimality": float(main_result.optimality),
            "main_de_cost": float(main_de.fun),
            "main_de_iterations": float(main_de.nit),
            "x2_cost": float(x2_result.cost),
            "x2_nfev": float(x2_result.nfev),
            "x2_optimality": float(x2_result.optimality),
            "x2_de_cost": float(x2_de.fun),
            "x2_de_iterations": float(x2_de.nit),
            "rmse_a": rmse_a,
            "rmse_ua": rmse_a * 1e6,
        },
    }


def scale_noise_density_components(freqs_hz, noise_density_ahz05, white_scale=1.0, flicker_scale=1.0):
    freqs_hz = np.asarray(freqs_hz, dtype=float)
    noise_density_ahz05 = np.asarray(noise_density_ahz05, dtype=float)
    if freqs_hz.shape != noise_density_ahz05.shape:
        raise ValueError("freqs_hz and noise_density_ahz05 must have the same shape")

    positive = freqs_hz > 0
    if np.count_nonzero(positive) == 0:
        return np.maximum(noise_density_ahz05, 0.0) * float(white_scale)

    white_floor = float(max(noise_density_ahz05[positive][-1], 0.0))
    flicker_density = np.sqrt(np.maximum(noise_density_ahz05**2 - white_floor**2, 0.0))
    scaled_density = np.sqrt(
        (white_floor * float(white_scale)) ** 2
        + (flicker_density * float(flicker_scale)) ** 2
    )
    if np.any(~positive):
        scaled_density[~positive] = scaled_density[positive][0]
    return scaled_density


def make_scaled_case1_noise_function(case1_results, amplitude_scale=1.0, flicker_scale=1.0, label=None):
    dataset = case1_results["dataset"]
    amplitude_scale = float(amplitude_scale)
    flicker_scale = float(flicker_scale)
    scaled_density = scale_noise_density_components(
        dataset["noise_freq_hz"],
        dataset["noise_density"],
        white_scale=amplitude_scale,
        flicker_scale=amplitude_scale * flicker_scale,
    )
    return make_noise_function_from_psd(
        dataset["noise_freq_hz"],
        scaled_density,
        label=label or f"case1_psd_noise_x{amplitude_scale:g}_fnx{flicker_scale:g}",
    )


def resample_uniform_trace(t, y, dt=None):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) != len(y):
        raise ValueError("t and y must have the same length")
    if len(t) < 2:
        raise ValueError("Need at least two samples to resample")
    if dt is None:
        dt = float(np.median(np.diff(t)))
    dt = max(float(dt), 1e-12)
    t_uniform = np.arange(float(t[0]), float(t[-1]) + 0.5 * dt, dt)
    y_uniform = np.interp(t_uniform, t, y)
    return t_uniform, y_uniform, dt


def analyze_case2_noise_comparison(case2_dataset, transient_fit, case1_results,
                                   noise_scale=CASE2_DEFAULT_NOISE_SCALE,
                                   fn_scale=CASE2_DEFAULT_FN_SCALE, seed=123):
    t_meas = np.asarray(case2_dataset["transient_time_s"], dtype=float)
    current_measured_a = np.asarray(case2_dataset["transient_current_a"], dtype=float)
    current_fit_a = np.asarray(transient_fit["current_fit_a"], dtype=float)
    residual_measured_a = current_measured_a - current_fit_a

    t_uniform, measured_uniform_a, dt_uniform = resample_uniform_trace(t_meas, residual_measured_a)
    power_uniform_w = np.interp(t_uniform, t_meas, transient_fit["power_profile_w"])
    scaled_noise_fn = make_scaled_case1_noise_function(
        case1_results,
        amplitude_scale=noise_scale,
        flicker_scale=fn_scale,
    )
    rng = np.random.default_rng(seed)
    i_out, i_det, _, _, _ = simulate(
        t_uniform,
        power_uniform_w,
        transient_fit["params_vec"],
        n_carrier=1,
        noise_fn=scaled_noise_fn,
        dark_current=transient_fit["fit_values"]["dark_current_a"],
        rng=rng,
        trap_mode="binary",
        trap_output_mode="illumination_gated",
    )
    synthetic_noise_a = i_out - i_det

    fs = 1.0 / dt_uniform
    nperseg = min(256, len(t_uniform))
    freqs_measured, psd_measured = welch(
        measured_uniform_a, fs=fs, nperseg=nperseg, detrend="constant", scaling="density"
    )
    freqs_synth, psd_synth = welch(
        synthetic_noise_a, fs=fs, nperseg=nperseg, detrend="constant", scaling="density"
    )
    scaled_case1_target = _resample_noise_density(
        freqs_synth,
        case1_results["dataset"]["noise_freq_hz"],
        scale_noise_density_components(
            case1_results["dataset"]["noise_freq_hz"],
            case1_results["dataset"]["noise_density"],
            white_scale=float(noise_scale),
            flicker_scale=float(noise_scale) * float(fn_scale),
        ),
    )
    return {
        "noise_scale": float(noise_scale),
        "fn_scale": float(fn_scale),
        "fn_scale_vs_case1": float(noise_scale) * float(fn_scale),
        "t_uniform": t_uniform,
        "measured_residual_a": measured_uniform_a,
        "synthetic_noise_a": synthetic_noise_a,
        "freqs_measured": freqs_measured,
        "freqs_synth": freqs_synth,
        "nd_measured": np.sqrt(np.maximum(psd_measured, 0.0)),
        "nd_synth": np.sqrt(np.maximum(psd_synth, 0.0)),
        "nd_case1_target": scaled_case1_target,
        "measured_rms_a": float(np.std(measured_uniform_a)),
        "synthetic_rms_a": float(np.std(synthetic_noise_a)),
    }


def plot_case2_parameter_fits(case2_dataset, response_fit, transient_fit, device_diameter_mm,
                              noise_results=None, output_path="fig_case2_parameter_fits.png"):
    # Keep these arguments for backward compatibility with the case2 workflow.
    _ = (device_diameter_mm, noise_results)

    t_meas = np.asarray(case2_dataset["transient_time_s"], dtype=float)
    current_meas_ua = np.asarray(case2_dataset["transient_current_ua"], dtype=float)
    current_fit_ua = np.asarray(transient_fit["current_fit_a"], dtype=float) * 1e6

    transient_mask = t_meas <= 8.0
    if np.count_nonzero(transient_mask) < 2:
        transient_mask = slice(None)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))

    bold_spine_width = 2.8
    bold_tick_width = 2.2
    bold_minor_tick_width = 1.6
    bold_tick_size = 8.0
    bold_minor_tick_size = 4.5
    bold_tick_labelsize = 17
    fit_line_width = 4.0
    paper_marker_size = 7.5
    paper_marker_edge_width = 1.35

    def apply_bold_plot_style(ax, grid=False):
        style_axes(ax, grid=grid)
        for spine in ax.spines.values():
            spine.set_linewidth(bold_spine_width)
        ax.tick_params(
            axis="both",
            which="major",
            direction="out",
            width=bold_tick_width,
            length=bold_tick_size,
            labelsize=bold_tick_labelsize,
            pad=7,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="out",
            width=bold_minor_tick_width,
            length=bold_minor_tick_size,
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    ax = axes[0]
    ax.plot(
        t_meas[transient_mask],
        current_meas_ua[transient_mask],
        "o",
        color="black",
        ms=paper_marker_size,
        markeredgewidth=paper_marker_edge_width,
    )
    ax.plot(
        t_meas[transient_mask],
        current_fit_ua[transient_mask],
        color="tab:red",
        lw=fit_line_width,
    )
    ax.set_xlim(0.0, 8.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Iout (uA)")
    ax.set_title("Transient Fit")
    apply_bold_plot_style(ax, grid=False)

    ax = axes[1]
    ax.loglog(
        response_fit["power_density_w_mm2"],
        response_fit["responsivity_measured_aw"],
        "o",
        color="black",
        ms=paper_marker_size,
        markeredgewidth=paper_marker_edge_width,
    )
    ax.loglog(
        response_fit["power_density_w_mm2"],
        response_fit["responsivity_fit_aw"],
        color="tab:red",
        lw=fit_line_width,
    )
    ax.set_xlabel("P (W/mm^2)")
    ax.set_ylabel("Equivalent response (A/W)")
    ax.set_title("Nonlinearity Fit")
    apply_bold_plot_style(ax, grid=False)

    finalize_figure(output_path)


def plot_case2_noise_comparison(noise_results, output_path="fig_case2_noise_comparison.png"):
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 9.5))

    axes[0].plot(
        noise_results["t_uniform"],
        noise_results["measured_residual_a"] * 1e9,
        color="tab:orange",
        lw=LW_MED,
        label="Paper residual (measured - fit)",
    )
    axes[0].plot(
        noise_results["t_uniform"],
        noise_results["synthetic_noise_a"] * 1e9,
        color="tab:blue",
        lw=LW_MED,
        alpha=0.85,
        label=(
            f"Synthesized noise (case1 x {noise_results['noise_scale']:.1f}, "
            f"FN x {noise_results['fn_scale']:.2f})"
        ),
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Noise current (nA)")
    axes[0].set_title("Time-Domain Noise Comparison")
    style_axes(axes[0], grid=True)
    style_legend(axes[0].legend(loc="best"))

    valid_measured = noise_results["freqs_measured"] > 0
    valid_synth = noise_results["freqs_synth"] > 0
    axes[1].loglog(
        noise_results["freqs_measured"][valid_measured],
        noise_results["nd_measured"][valid_measured],
        color="tab:orange",
        lw=LW_MED,
        label="Paper residual PSD",
    )
    axes[1].loglog(
        noise_results["freqs_synth"][valid_synth],
        noise_results["nd_case1_target"][valid_synth],
        "--",
        color="tab:green",
        lw=LW_MED,
        label="Scaled case1 PSD target",
    )
    axes[1].loglog(
        noise_results["freqs_synth"][valid_synth],
        noise_results["nd_synth"][valid_synth],
        color="tab:blue",
        lw=LW_HEAVY,
        label="PSD from synthesized trace",
    )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Noise current (A/Hz^0.5)")
    axes[1].set_title("Frequency-Domain Noise Comparison")
    style_axes(axes[1], grid=True)
    style_legend(axes[1].legend(loc="best"))

    fig.suptitle("case2 Noise Comparison Against Scaled case1 Baseline", fontsize=18, fontweight="bold")
    finalize_figure(output_path)


def build_case2_summary(case2_dataset, response_fit, transient_fit, noise_results,
                        device_diameter_mm, device_area_mm2):
    response_current_ratio = float(
        np.max(response_fit["current_measured_a"]) / max(np.max(case2_dataset["transient_current_a"]), 1e-30)
    )
    notes = []
    notes.append(
        "transient 主支路的 10-90 rise/fall 固定为论文报告的 84 ms / 243 ms，"
        f"并把 main branch 的 on-state plateau 强制锚定在 `{CASE2_MAIN_BRANCH_CEILING_UA:.2f} uA`，"
        "其余 on 段额外抬升由 x2 调制主光响应承担；x2 输出采用 x1 比例系数，"
        "即 I_photo = x1 * (1 + trap_x1_ratio * clip(x2 / trap_x2_reference_state, 0, 1))。"
    )
    if response_current_ratio > 100.0:
        notes.append(
            "按当前器件直径换算时，response-P 曲线等效电流远高于 transient 曲线，"
            "因此不直接用该换算去约束 transient 幅值。"
        )
    notes.append(
        "先前底端未对齐主要是 dark current 与关光残余量共同吸收了 off-state 误差，"
        "这次通过固定主支路响应时间并提高低电平段权重来约束基线。"
    )
    return {
        "device_diameter_mm": float(device_diameter_mm),
        "device_area_mm2": float(device_area_mm2),
        "device_area_cm2": float(convert_area_mm2_to_cm2(device_area_mm2)),
        "power_ref_w": float(response_fit["power_ref_w"]),
        "response_fit": {
            "R_single": float(response_fit["R_single"]),
            "eta_single": float(response_fit["eta_single"]),
            "alpha_global": float(response_fit["alpha_global"]),
            "rmse_log10_resp": float(response_fit["rmse_log10_resp"]),
        },
        "transient_fit": {
            key: float(value)
            for key, value in transient_fit["fit_values"].items()
        },
        "transient_metrics": {
            key: float(value) if isinstance(value, (float, np.floating)) else value
            for key, value in transient_fit["metrics"].items()
        },
        "case1_defaults_used": {
            "gamma": float(transient_fit["params"]["gamma"]),
            "tau_drift": float(transient_fit["params"]["tau_drift"]),
            "drift_scale": float(transient_fit["params"]["drift_scale"]),
            "noise_scale_vs_case1": float(noise_results["noise_scale"]),
            "fn_scale_vs_scaled_case1": float(noise_results["fn_scale"]),
            "fn_scale_vs_case1": float(noise_results["fn_scale_vs_case1"]),
        },
        "noise_comparison": {
            "measured_rms_a": float(noise_results["measured_rms_a"]),
            "synthetic_rms_a": float(noise_results["synthetic_rms_a"]),
        },
        "notes": notes,
    }


def export_case2_fit_results(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "case2_fit_summary.json"
    csv_path = output_dir / "case2_fit_parameters.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    rows = [
        ("device_diameter_mm", summary["device_diameter_mm"]),
        ("device_area_mm2", summary["device_area_mm2"]),
        ("device_area_cm2", summary["device_area_cm2"]),
        ("power_ref_w", summary["power_ref_w"]),
    ]
    for section in ("response_fit", "transient_fit", "transient_metrics", "case1_defaults_used", "noise_comparison"):
        for key, value in summary[section].items():
            rows.append((key, value))

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        writer.writerows(rows)
    return json_path, csv_path


def print_case2_summary(summary, parameter_figure, noise_figure, json_path, csv_path):
    print("\n" + "=" * 60)
    print("case2 Fit Summary")
    print(f"  器件直径 = {summary['device_diameter_mm']:.4f} mm")
    print(f"  器件面积 = {summary['device_area_mm2']:.6f} mm^2 = {summary['device_area_cm2']:.6f} cm^2")
    print(f"  非线性 eta_single = {summary['response_fit']['eta_single']:.4f}")
    print(f"  R_single_transient = {summary['transient_fit']['R_single_transient']:.4e} A/W")
    print(f"  主支路固定 trise / tfall = {summary['transient_metrics']['tau_rise_main_fixed_s'] * 1e3:.3f} / "
          f"{summary['transient_metrics']['tau_fall_main_fixed_s'] * 1e3:.3f} ms")
    print(f"  tau_rise (10-90) = {summary['transient_metrics']['tau_rise_s_10_90'] * 1e3:.3f} ms")
    print(f"  tau_fall (10-90) = {summary['transient_metrics']['tau_fall_s_10_90'] * 1e3:.3f} ms")
    print(f"  alpha = {summary['transient_fit']['alpha']:.4e} s^-1")
    print(f"  beta = {summary['transient_fit']['beta']:.4e} s^-1")
    print(f"  trap_x1_ratio = {summary['transient_fit'].get('trap_x1_ratio', 0.0):.4e}")
    print(f"  delta = {summary['transient_fit']['delta']:.4e} A (unused for x1-ratio x2)")
    print(f"  tau_x2_on / tau_x2_off = {summary['transient_metrics']['tau_x2_on_s'] * 1e3:.3f} / "
          f"{summary['transient_metrics']['tau_x2_off_s'] * 1e3:.3f} ms")
    print(f"  x2 peak current = {summary['transient_metrics']['x2_peak_current_a'] * 1e9:.3f} nA")
    print(f"  x2 residual state end = {summary['transient_metrics']['x2_residual_state_end']:.3e}")
    print(f"  main branch ceiling = {summary['transient_fit']['main_branch_ceiling_ua']:.3f} uA")
    print(f"  transient 固定 P_on = {summary['transient_fit']['power_density_on_w_mm2']:.4e} W/mm^2")
    print(f"  dark current = {summary['transient_fit']['dark_current_a'] * 1e6:.4f} uA")
    print(f"  baseline mismatch = {summary['transient_metrics']['baseline_mismatch_a'] * 1e9:.3f} nA")
    print(f"  measured noise RMS = {summary['noise_comparison']['measured_rms_a'] * 1e9:.3f} nA")
    print(f"  synthesized noise RMS = {summary['noise_comparison']['synthetic_rms_a'] * 1e9:.3f} nA")
    for note in summary["notes"]:
        print(f"  note: {note}")
    print(f"  参数图 = {parameter_figure}")
    print(f"  噪声图 = {noise_figure}")
    print(f"  JSON = {json_path}")
    print(f"  CSV = {csv_path}")


def run_case2_workflow(data_dir=CASE2_DATA_DIR, output_dir=CASE2_DEFAULT_OUTPUT_DIR,
                       device_diameter_mm=CASE2_DEFAULT_DEVICE_DIAMETER_MM,
                       noise_scale=CASE2_DEFAULT_NOISE_SCALE,
                       fn_scale=CASE2_DEFAULT_FN_SCALE,
                       fixed_rise_time_ms=CASE2_FIXED_RISE_TIME_MS,
                       fixed_fall_time_ms=CASE2_FIXED_FALL_TIME_MS,
                       main_branch_ceiling_ua=CASE2_MAIN_BRANCH_CEILING_UA):
    case2_dataset = load_case2_measurements(data_dir)
    case1_results = extract_case1_single_carrier_params(CASE1_DATA_DIR, device_area_cm2=DEFAULT_DEVICE_AREA_CM2)

    device_area_mm2 = compute_circular_device_area_mm2(device_diameter_mm)
    device_area_cm2 = convert_area_mm2_to_cm2(device_area_mm2)
    response_fit = fit_case2_responsivity_curve(case2_dataset, device_area_mm2=device_area_mm2)
    set_device_context(device_area_cm2=device_area_cm2, power_ref_w=response_fit["power_ref_w"])

    drift_defaults = {
        "gamma": PARAMS_SINGLE_CARRIER["gamma"],
        "tau_drift": PARAMS_SINGLE_CARRIER["tau_drift"],
        "drift_scale": PARAMS_SINGLE_CARRIER["drift_scale"],
    }
    transient_fit = fit_case2_transient(
        case2_dataset,
        response_fit=response_fit,
        device_area_mm2=device_area_mm2,
        drift_defaults=drift_defaults,
        fixed_rise_time_ms=fixed_rise_time_ms,
        fixed_fall_time_ms=fixed_fall_time_ms,
        main_branch_ceiling_ua=main_branch_ceiling_ua,
    )
    noise_results = analyze_case2_noise_comparison(
        case2_dataset,
        transient_fit=transient_fit,
        case1_results=case1_results,
        noise_scale=noise_scale,
        fn_scale=fn_scale,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parameter_figure = output_dir / "fig_case2_parameter_fits.png"
    noise_figure = output_dir / "fig_case2_noise_comparison.png"
    plot_case2_parameter_fits(
        case2_dataset,
        response_fit=response_fit,
        transient_fit=transient_fit,
        device_diameter_mm=device_diameter_mm,
        noise_results=noise_results,
        output_path=parameter_figure,
    )
    plot_case2_noise_comparison(noise_results, output_path=noise_figure)

    summary = build_case2_summary(
        case2_dataset,
        response_fit=response_fit,
        transient_fit=transient_fit,
        noise_results=noise_results,
        device_diameter_mm=device_diameter_mm,
        device_area_mm2=device_area_mm2,
    )
    json_path, csv_path = export_case2_fit_results(summary, output_dir=output_dir)
    print_case2_summary(summary, parameter_figure, noise_figure, json_path, csv_path)
    return {
        "dataset": case2_dataset,
        "response_fit": response_fit,
        "transient_fit": transient_fit,
        "noise_results": noise_results,
        "summary": summary,
        "json_path": json_path,
        "csv_path": csv_path,
        "parameter_figure": parameter_figure,
        "noise_figure": noise_figure,
    }


def print_metric_summary(linearity, transient, freq_resp, noise_results):
    print("\n" + "=" * 60)
    print("Metric Summary")
    print(f"  全局非线性斜率 alpha = {linearity['alpha_global']:.4f}")
    if len(linearity["strict"]["indices"]) >= 2:
        idx = linearity["strict"]["indices"]
        print("  严格线性区:")
        print(f"    P = {linearity['P_levels'][idx[0]]*1e3:.4f} ~ "
              f"{linearity['P_levels'][idx[-1]]*1e3:.4f} mW")
        print(f"    LDRapp = {linearity['strict']['ldr_db']:.2f} dB")
    else:
        print("  严格线性区: 未找到")
    if len(linearity["quasi"]["indices"]) >= 2:
        idx = linearity["quasi"]["indices"]
        print("  准线性区:")
        print(f"    P = {linearity['P_levels'][idx[0]]*1e3:.4f} ~ "
              f"{linearity['P_levels'][idx[-1]]*1e3:.4f} mW")
        print(f"    LDRapp = {linearity['quasi']['ldr_db']:.2f} dB")
    else:
        print("  准线性区: 未找到")

    tau_rise_ms = transient["tau_rise"] * 1e3 if np.isfinite(transient["tau_rise"]) else np.nan
    tau_fall_ms = transient["tau_fall"] * 1e3 if np.isfinite(transient["tau_fall"]) else np.nan
    print(f"  tau_rise = {tau_rise_ms:.3f} ms")
    print(f"  tau_fall = {tau_fall_ms:.3f} ms")
    print(f"  rise/fall 是否适合直接 benchmark: {transient['valid_for_benchmark']}")
    for note in transient["notes"]:
        print(f"    note: {note}")

    print(f"  低频平台 responsivity = {freq_resp['R0']:.4f} A/W")
    if np.isfinite(freq_resp["f3db"]):
        print(f"  f3dB = {freq_resp['f3db']:.2f} Hz")
    else:
        print("  f3dB = 未落入当前扫频范围")
    print(f"  低频平台是否满足 <=5% 平坦度: {freq_resp['plateau_valid']}")

    print("  噪声 / NEP / D*:")
    for result in noise_results:
        print(f"    [{result['name']}]")
        print(f"      i_rms = {result['i_rms']*1e12:.3f} pA_rms")
        print(f"      sqrt(Sn*B) 近似 = {result['i_rms_white_approx']*1e12:.3f} pA_rms")
        print(f"      NEP = {result['nep']*1e12:.3f} pW/Hz^0.5")
        print(f"      D* = {result['dstar']:.3e} Jones")
        print(f"      Shot-noise-limited D* (from Idark) = {result['dstar_shot']:.3e} Jones")


def _format_ratio(ratio, mode):
    if not np.isfinite(ratio):
        return "N/A"
    if ratio >= 100:
        txt = f"{ratio:.2e}"
    else:
        txt = f"{ratio:.1f}"
    if mode == "over":
        return f"高估 {txt} 倍"
    if mode == "under":
        return f"低估 {txt} 倍"
    return txt


def generate_misreport_markdown_table(linearity, transient, freq_resp, noise_results):
    tau_avg = 0.5 * (transient["tau_rise"] + transient["tau_fall"])
    f3db_from_tau = np.nan
    if np.isfinite(tau_avg) and tau_avg > 0:
        f3db_from_tau = np.log(9.0) / (2 * np.pi * tau_avg)

    lines = [
        "# 误报值与推荐值对比",
        "",
        "| 指标 | 常见误报方式 | 误报值 | 按论文推荐/当前仿真值 | 差距 |",
        "|---|---|---:|---:|---:|",
    ]

    alpha_gap = abs(1.0 - linearity["alpha_global"]) * 100.0
    lines.append(
        "| 线性度 | 看到 `logI-logP` 近似直线，就直接当成线性器件 | "
        "`alpha = 1` | "
        f"`alpha = {linearity['alpha_global']:.4f}` | "
        f"偏离线性 `{alpha_gap:.1f}%` |"
    )

    if len(linearity["strict"]["indices"]) >= 2:
        ldr_true = f"`{linearity['strict']['ldr_db']:.2f} dB`"
        ldr_gap = linearity["full_scan_ldr_db"] - linearity["strict"]["ldr_db"]
        ldr_note = f"高估 `{ldr_gap:.2f} dB`"
    else:
        ldr_true = "`不应报告 LDR`"
        ldr_note = "从“有指标”变成“指标无效”"
    lines.append(
        "| LDR | 直接把整个功率扫描区间都当成 `LDR` | "
        f"`{linearity['full_scan_ldr_db']:.2f} dB` | "
        f"{ldr_true} | {ldr_note} |"
    )

    if np.isfinite(f3db_from_tau) and np.isfinite(freq_resp["f3db"]) and freq_resp["f3db"] > 0:
        f_gap_pct = abs(freq_resp["f3db"] - f3db_from_tau) / freq_resp["f3db"] * 100.0
        f_gap_note = f"低估 `{f_gap_pct:.1f}%`" if f3db_from_tau < freq_resp["f3db"] else f"高估 `{f_gap_pct:.1f}%`"
        wrong_f3db = f"`{f3db_from_tau:.2f} Hz`"
        true_f3db = f"`{freq_resp['f3db']:.2f} Hz`"
    else:
        wrong_f3db = "`N/A`"
        true_f3db = "`N/A`"
        f_gap_note = "N/A"
    lines.append(
        "| `f3dB` | 直接用 `tau_rise/tau_fall` 套一阶系统公式 | "
        f"{wrong_f3db} | {true_f3db} | {f_gap_note} |"
    )

    for result in noise_results:
        ratio_i = result["i_rms"] / max(result["i_shot"], 1e-30)
        ratio_nep = result["nep"] / max(result["nep_shot"], 1e-30)
        ratio_d = result["dstar_shot"] / max(result["dstar"], 1e-30)
        scene = result["name"]
        lines.append(
            f"| 噪声底 `i_rms`，{scene} 场景 | 只按 `Idark` 的 shot-noise floor 算 | "
            f"`{result['i_shot']*1e12:.3f} pA_rms` | "
            f"`{result['i_rms']*1e12:.3f} pA_rms` | "
            f"{_format_ratio(ratio_i, 'under')} |"
        )
        lines.append(
            f"| `NEP`，{scene} 场景 | 用 shot-noise-limited 模型直接算 | "
            f"`{result['nep_shot']*1e12:.4f} pW/Hz^0.5` | "
            f"`{result['nep']*1e12:.3f} pW/Hz^0.5` | "
            f"{_format_ratio(ratio_nep, 'under')} |"
        )
        lines.append(
            f"| `D*`，{scene} 场景 | 直接报 shot-noise-limited `D*` | "
            f"`{result['dstar_shot']:.3e} Jones` | "
            f"`{result['dstar']:.3e} Jones` | "
            f"{_format_ratio(ratio_d, 'over')} |"
        )

    return "\n".join(lines)


def export_misreport_table_markdown(linearity, transient, freq_resp, noise_results,
                                    path="metric_misreport_table.md"):
    markdown = generate_misreport_markdown_table(
        linearity, transient, freq_resp, noise_results
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown + "\n")
    print("\n" + "=" * 60)
    print("Markdown Table")
    print(markdown)
    print(f"\n已保存 Markdown 表格: {path}")
    return markdown


# ============================================================
# Main
# ============================================================

def run_case1_workflow():
    case1_results = extract_case1_single_carrier_params(CASE1_DATA_DIR, device_area_cm2=DEVICE_AREA_CM2)
    set_device_context(device_area_cm2=DEVICE_AREA_CM2, power_ref_w=case1_results["power_ref_w"])
    ANALYSIS_CONFIG["ideal_responsivity"] = case1_results["r_ideal"]
    params_single = case1_results["params"]
    params_vec = params_to_vec(params_single)
    noise_fn = case1_results["noise_fn"]

    print("=" * 60)
    print("Photodetector Metrics Simulation")
    print(f"  暗电流输入 Idark = {DARK_CURRENT_MEASURED*1e9:.2f} nA")
    print(f"  器件面积 A = {DEVICE_AREA_CM2:.4f} cm^2")
    print(f"  P_ref = {NONLINEAR_POWER_REF_W:.6e} W")
    print(f"  噪声模型 = {noise_fn.__name__}")

    print("\n" + "=" * 60)
    print("Step 0: 非理想因素总表")
    plot_nonideal_effects_table()

    print("\n" + "=" * 60)
    print("Step 1: case1 参数拟合图")
    plot_case1_parameter_fits(case1_results, device_area_cm2=DEVICE_AREA_CM2)

    print("\n" + "=" * 60)
    print("Step 2: case1 noise reconstruction validation")
    plot_case1_noise_reconstruction(case1_results)

    print("\n" + "=" * 60)
    print(f"Step 3: 任意波形对照（含噪声曲线: {noise_fn.__name__}）")
    plot_arbitrary(params_vec, noise_fn=noise_fn)

    # print("\n" + "=" * 60)
    # print("Step 2: Linearity and LDR")
    # linearity = analyze_linearity_and_ldr(params_vec, DARK_CURRENT_MEASURED, ANALYSIS_CONFIG)
    # plot_linearity_and_ldr(linearity)

    # print("\n" + "=" * 60)
    # print("Step 3: Square-wave transient tau_rise / tau_fall")
    # transient = analyze_rise_fall(params_vec, DARK_CURRENT_MEASURED, ANALYSIS_CONFIG)
    # plot_rise_fall(transient)

    # print("\n" + "=" * 60)
    # print("Step 4: Small-signal frequency response and f3dB")
    # freq_resp = analyze_frequency_response(params_vec, DARK_CURRENT_MEASURED, ANALYSIS_CONFIG)
    # plot_frequency_response(freq_resp)

    # print("\n" + "=" * 60)
    # print("Step 5: Noise / NEP / D*")
    # noise_results = []
    # for i, (name, noise_fn) in enumerate(NOISE_CASES.items(), start=1):
    #     result = analyze_noise_case(
    #         params_vec, DARK_CURRENT_MEASURED, name, noise_fn,
    #         ANALYSIS_CONFIG, freq_resp["R0"], seed=100 + i
    #     )
    #     noise_results.append(result)
    # plot_noise_cases(noise_results, ANALYSIS_CONFIG)

    # print("\n" + "=" * 60)
    # print("Step 6: D* vs optical power / frequency")
    # dstar_power_results = []
    # for i, (name, noise_fn) in enumerate(NOISE_CASES.items(), start=1):
    #     result = analyze_detectivity_vs_power(
    #         params_vec, DARK_CURRENT_MEASURED, name, noise_fn,
    #         ANALYSIS_CONFIG, linearity, seed=300 + 20 * i
    #     )
    #     dstar_power_results.append(result)
    # dstar_freq_results = analyze_detectivity_vs_frequency(freq_resp, noise_results)
    # plot_detectivity_vs_power(dstar_power_results, ANALYSIS_CONFIG)
    # plot_detectivity_vs_frequency(dstar_freq_results, freq_resp)

    # print_metric_summary(linearity, transient, freq_resp, noise_results)
    # export_misreport_table_markdown(
    #     linearity, transient, freq_resp, noise_results
    # )
    # print("\nDone! Figures: fig3_arbitrary.png, fig_metric_linearity_ldr.png, fig_metric_risefall.png, "
    #       "fig_metric_f3db.png, fig_metric_noise_nep_dstar.png, "
    #       "fig_nonideal_effects_table.png, "
    #       "fig_metric_dstar_vs_power.png, fig_metric_dstar_vs_freq.png; "
    #       "Table: metric_misreport_table.md")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Photodetector parameter extraction and visualization workflows."
    )
    parser.add_argument(
        "--case",
        choices=["case1", "case2"],
        default="case1",
        help="Select which dataset workflow to run. Default keeps the original case1 behavior.",
    )
    parser.add_argument(
        "--case2-device-diameter-mm",
        type=float,
        default=CASE2_DEFAULT_DEVICE_DIAMETER_MM,
        help="Device diameter used to convert case2 response power density into total power.",
    )
    parser.add_argument(
        "--case2-noise-scale",
        type=float,
        default=CASE2_DEFAULT_NOISE_SCALE,
        help="Amplitude multiplier applied to the case1 noise PSD when synthesizing case2 noise.",
    )
    parser.add_argument(
        "--case2-fn-scale",
        type=float,
        default=CASE2_DEFAULT_FN_SCALE,
        help="Extra scale applied only to the extracted 1/f flicker-noise component of the case1 PSD.",
    )
    parser.add_argument(
        "--case2-output-dir",
        type=str,
        default=str(CASE2_DEFAULT_OUTPUT_DIR),
        help="Output directory for case2 figures and exported parameters.",
    )
    parser.add_argument(
        "--case2-fixed-rise-ms",
        type=float,
        default=CASE2_FIXED_RISE_TIME_MS,
        help="Fixed 10-90 rise time reported by the paper for the main case2 transient branch.",
    )
    parser.add_argument(
        "--case2-fixed-fall-ms",
        type=float,
        default=CASE2_FIXED_FALL_TIME_MS,
        help="Fixed 90-10 fall time reported by the paper for the main case2 transient branch.",
    )
    parser.add_argument(
        "--case2-main-branch-ceiling-ua",
        type=float,
        default=CASE2_MAIN_BRANCH_CEILING_UA,
        help="Transient current below this ceiling is assigned to the main branch; only the excess is fitted by x2.",
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.case == "case2":
        run_case2_workflow(
            data_dir=CASE2_DATA_DIR,
            output_dir=Path(args.case2_output_dir),
            device_diameter_mm=args.case2_device_diameter_mm,
            noise_scale=args.case2_noise_scale,
            fn_scale=args.case2_fn_scale,
            fixed_rise_time_ms=args.case2_fixed_rise_ms,
            fixed_fall_time_ms=args.case2_fixed_fall_ms,
            main_branch_ceiling_ua=args.case2_main_branch_ceiling_ua,
        )
    else:
        run_case1_workflow()
