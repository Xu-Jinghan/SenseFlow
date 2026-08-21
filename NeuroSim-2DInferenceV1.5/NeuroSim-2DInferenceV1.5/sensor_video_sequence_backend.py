import sys
from pathlib import Path

import numpy as np


THIS_DIR = Path(__file__).resolve().parent


def _find_repo_root(start_dir: Path) -> Path:
    for candidate in [start_dir, *start_dir.parents]:
        if (candidate / "photodetector_array.py").exists():
            return candidate
    return start_dir.parent


REPO_ROOT = _find_repo_root(THIS_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SPATIAL_VARIATION_CACHE_DIR = THIS_DIR / "artifacts" / "spatial_variation_maps" / "responsivity"

from photodetector_model import (  # noqa: E402
    DARK_CURRENT_MEASURED,
    current_from_state,
    infer_n_carrier_from_params,
    init_state_arrays,
    params_to_vec,
    prepare_model_config,
    sample_combined_psd_noise_signal_trace,
    steady_state_current_from_power,
    step_model_state,
)


TEMPORAL_NOISE_CACHE_BUDGET_BYTES = 16 * 1024 * 1024
TEMPORAL_NOISE_CACHE_FALLBACK_FRAMES = 32
TEMPORAL_NOISE_GENERATION_CHUNK_VALUES = 16 * 1024
TEMPORAL_NOISE_MODE_PIXEL_BUFFERED = "pixel_buffered"
TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW = "pixel_repeated_window"
TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE = "global_full_sequence"
TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW = "global_repeated_window"


def resolve_sensor_seed(args):
    sensor_seed = getattr(args, "sensor_rng_seed", None)
    if sensor_seed is not None:
        return sensor_seed
    return getattr(args, "seed", 1234)


def resolve_spatial_variation_cache_dir(args, base_params=None):
    candidate = getattr(args, "spatial_variation_cache_dir", None)
    if candidate in {None, "", "auto"} and base_params is not None:
        candidate = base_params.get("spatial_variation_cache_dir")
    if candidate in {None, "", "auto"}:
        return DEFAULT_SPATIAL_VARIATION_CACHE_DIR
    return Path(candidate).expanduser()


def frame_duration(args):
    if args.video_fps <= 0:
        raise ValueError(f"video_fps must be positive, got {args.video_fps}")
    return 1.0 / args.video_fps


def steps_per_frame(args):
    return max(1, int(round(frame_duration(args) * args.fps_sim)))


def resolve_temporal_noise_mode(args):
    mode = str(getattr(args, "temporal_noise_mode", TEMPORAL_NOISE_MODE_PIXEL_BUFFERED)).lower()
    aliases = {
        "buffered": TEMPORAL_NOISE_MODE_PIXEL_BUFFERED,
        "pixel": TEMPORAL_NOISE_MODE_PIXEL_BUFFERED,
        "pixel_buffered": TEMPORAL_NOISE_MODE_PIXEL_BUFFERED,
        "pixel_repeat": TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW,
        "pixel_repeated": TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW,
        "pixel_repeated_window": TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW,
        "full": TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE,
        "full_sequence": TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE,
        "global": TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE,
        "global_full_sequence": TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE,
        "repeat": TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        "repeated": TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        "repeated_window": TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        "global_repeated_window": TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
    }
    try:
        return aliases[mode]
    except KeyError as exc:
        supported = ", ".join(sorted(set(aliases.values())))
        raise ValueError(f"Unsupported temporal_noise_mode={mode!r}; supported: {supported}") from exc


def _extract_center_pixel_values(array):
    arr = np.asarray(array, dtype=np.float64)
    if arr.ndim == 2:
        return np.asarray([arr[arr.shape[0] // 2, arr.shape[1] // 2]], dtype=np.float64)
    if arr.ndim == 3:
        return np.asarray(arr[:, arr.shape[1] // 2, arr.shape[2] // 2], dtype=np.float64)
    raise ValueError(f"Unsupported array rank for center-pixel extraction: {arr.ndim}")


def responsivity_total_from_params(base_params):
    if "R_single" in base_params:
        return float(base_params["R_single"])
    return float(base_params["R_fast"] + base_params["R_slow"])


def resolve_analog_readout_mode(args):
    readout = str(getattr(args, "readout", "integration")).lower()
    analog_readout = getattr(args, "analog_readout", None)
    analog_mode = readout if analog_readout is None else str(analog_readout).lower()
    if readout == "adc":
        analog_mode = "integration"
    if analog_mode not in {"tia", "integration"}:
        raise ValueError(f"Unsupported analog readout mode: {analog_mode}")
    return analog_mode


def adc_enabled(args):
    readout = str(getattr(args, "readout", "integration")).lower()
    if readout == "adc":
        return True
    return bool(getattr(args, "adc_enabled", 0))


def fast_tia_frame_step_enabled(args):
    return bool(getattr(args, "fast_tia_frame_step", 0)) and resolve_analog_readout_mode(args) == "tia"


def _quantize_voltage(frame, args):
    frame = np.asarray(frame, dtype=np.float64)
    calibration_low = getattr(args, "adc_calibration_low", None)
    calibration_high = getattr(args, "adc_calibration_high", None)
    if calibration_low is not None and calibration_high is not None:
        calibration_low = float(calibration_low)
        calibration_high = float(calibration_high)
        if calibration_high <= calibration_low:
            calibration_high = calibration_low + 1e-12
        normalized = np.clip((frame - calibration_low) / (calibration_high - calibration_low), 0.0, 1.0)
        n_levels = 2 ** int(getattr(args, "adc_bits", 8))
        quantized = np.floor(normalized * (n_levels - 1)).astype(np.float64)
        return quantized / (n_levels - 1) * (calibration_high - calibration_low) + calibration_low
    full_scale = getattr(args, "adc_full_scale", None)
    if full_scale is None:
        full_scale = np.max(frame) * 1.2 if np.max(frame) > 0 else 1e-6
    normalized = np.clip(frame / full_scale, 0.0, 1.0)
    n_levels = 2 ** int(getattr(args, "adc_bits", 8))
    quantized = np.floor(normalized * (n_levels - 1)).astype(np.float64)
    return quantized / (n_levels - 1) * full_scale


def apply_readout_to_current_level(current_level, args, frame_time_s=None):
    current_level = np.asarray(current_level, dtype=np.float64)
    analog_mode = resolve_analog_readout_mode(args)
    frame_time_s = frame_duration(args) if frame_time_s is None else float(frame_time_s)

    if analog_mode == "tia":
        analog_frame = current_level * float(getattr(args, "tia_gain_ohm", 1.0))
    else:
        analog_frame = (
            current_level
            * frame_time_s
            * float(getattr(args, "integration_gain_v_per_c", 1.0))
        )

    if adc_enabled(args):
        analog_frame = _quantize_voltage(analog_frame, args)
    return analog_frame.astype(np.float32)


def apply_readout_to_current_trace(current_trace, args, dt):
    current_trace = np.asarray(current_trace, dtype=np.float64)
    analog_mode = resolve_analog_readout_mode(args)

    if analog_mode == "tia":
        analog_frame = current_trace[-1] * float(getattr(args, "tia_gain_ohm", 1.0))
    else:
        analog_frame = (
            np.sum(current_trace, axis=0) * float(dt) * float(getattr(args, "integration_gain_v_per_c", 1.0))
        )

    if adc_enabled(args):
        analog_frame = _quantize_voltage(analog_frame, args)
    return analog_frame.astype(np.float32)


def estimate_pmax_output_current(base_params, model_config, dark_current):
    return steady_state_current_from_power(
        float(base_params["pmax_w"]),
        model_config,
        dark_current=dark_current,
        include_drift=False,
    )


def build_precomputed_noise_trace(args, base_params, model_config, dark_current, dt, total_steps, rng, power_shape):
    if total_steps <= 0 or not bool(getattr(args, "use_noise_fn", 0)):
        return None

    flicker_density = float(base_params.get("noise_1f_density_1hz_a_root_hz", 0.0))
    pmax_current = float(np.max(estimate_pmax_output_current(base_params, model_config, dark_current)))
    shot_density = 0.0
    if bool(getattr(args, "shot_noise", 1)):
        q = 1.602e-19
        shot_density = float(np.sqrt(max(0.0, 2.0 * q * pmax_current)))

    if shot_density <= 0.0 and flicker_density <= 0.0:
        return None

    time_axis = np.arange(total_steps, dtype=np.float64) * float(dt)
    return sample_combined_noise_trace_for_shape(
        time_axis,
        tuple(power_shape),
        shot_noise_density_ahz05=shot_density,
        flicker_noise_density_1hz_ahz05=flicker_density,
        rng=rng,
    ).astype(np.float64, copy=False)


def sample_combined_noise_trace_for_shape(
    time_axis,
    power_shape,
    shot_noise_density_ahz05,
    flicker_noise_density_1hz_ahz05,
    rng,
):
    time_axis = np.asarray(time_axis, dtype=np.float64)
    power_shape = tuple(int(dim) for dim in power_shape)
    signal_shape = (len(time_axis),) + power_shape
    values_per_step = max(1, int(np.prod(power_shape)))
    chunk_values = max(1, int(TEMPORAL_NOISE_GENERATION_CHUNK_VALUES))

    if values_per_step <= chunk_values:
        return sample_combined_psd_noise_signal_trace(
            time_axis,
            signal_shape,
            shot_noise_density_ahz05=shot_noise_density_ahz05,
            flicker_noise_density_1hz_ahz05=flicker_noise_density_1hz_ahz05,
            rng=rng,
        ).astype(np.float64, copy=False)

    trace = np.empty(signal_shape, dtype=np.float64)
    flat_trace = trace.reshape(len(time_axis), values_per_step)
    for start in range(0, values_per_step, chunk_values):
        stop = min(values_per_step, start + chunk_values)
        flat_trace[:, start:stop] = sample_combined_psd_noise_signal_trace(
            time_axis,
            (len(time_axis), stop - start),
            shot_noise_density_ahz05=shot_noise_density_ahz05,
            flicker_noise_density_1hz_ahz05=flicker_noise_density_1hz_ahz05,
            rng=rng,
        )
    return trace


def full_sequence_noise_shape(power_shape):
    power_shape = tuple(power_shape)
    if len(power_shape) == 0:
        return power_shape
    if len(power_shape) == 1:
        return (1,)
    return (power_shape[0],) + tuple(1 for _ in power_shape[1:])


def compute_long_term_drift_state(power_reference, model_config, aging_hours):
    power_reference = np.asarray(power_reference, dtype=np.float64)
    aging_hours = float(aging_hours)
    if aging_hours <= 0:
        return np.zeros_like(power_reference, dtype=np.float64)

    tau_drift = float(model_config["tau_drift"])
    if tau_drift <= 0:
        return np.zeros_like(power_reference, dtype=np.float64)

    aging_seconds = aging_hours * 3600.0
    drift_state = (
        float(model_config["drift_scale"])
        * float(model_config["gamma"])
        * (1.0 - np.exp(-aging_seconds / tau_drift))
    )
    return np.full_like(power_reference, drift_state, dtype=np.float64)


def simulate_ideal_video_frame(power_maps, args, base_params):
    responsivity_total = responsivity_total_from_params(base_params)
    current_level = responsivity_total * np.asarray(power_maps, dtype=np.float64)
    return apply_readout_to_current_level(
        current_level,
        args,
        frame_time_s=frame_duration(args),
    )


def ideal_readout_from_power(power_w, args, base_params):
    responsivity_total = responsivity_total_from_params(base_params)
    current_level = responsivity_total * np.asarray(power_w, dtype=np.float64)
    return apply_readout_to_current_level(current_level, args, frame_time_s=frame_duration(args))


def nonideal_readout_from_power(power_w, args, base_params):
    n_carrier = infer_n_carrier_from_params(base_params)
    model_config = prepare_model_config(
        params_to_vec(base_params),
        n_carrier=n_carrier,
        trap_mode=str(base_params.get("trap_mode", "power")),
        trap_threshold_w=float(base_params.get("trap_threshold_w", base_params.get("pmin_w", 0.0))),
        trap_output_mode=str(base_params.get("trap_output_mode", "always")),
        power_min_w=float(base_params.get("pmin_w", 0.0)),
        power_max_w=float(base_params.get("pmax_w", float("inf"))),
        trap_saturation_time_s=base_params.get("trap_saturation_time_s"),
        trap_amplitude_ratio=base_params.get("trap_amplitude_ratio"),
        trap_delta_r_ratio=base_params.get("trap_delta_r_ratio"),
        trap_x1_ratio=base_params.get("trap_x1_ratio"),
        trap_x2_reference_state=base_params.get("trap_x2_reference_state"),
        trap_x2_tau_on_s=base_params.get("trap_x2_tau_on_s"),
        trap_x2_tau_off_s=base_params.get("trap_x2_tau_off_s"),
    )
    current_level = steady_state_current_from_power(
        power_w,
        model_config,
        dark_current=float(base_params.get("dark_current_a", DARK_CURRENT_MEASURED)),
        include_drift=False,
    )
    return apply_readout_to_current_level(current_level, args, frame_time_s=frame_duration(args))


def apply_transient_weight_overrides(model_config, base_params):
    if "rise_fast_weight" not in base_params or "fall_fast_weight" not in base_params:
        return model_config
    if int(model_config.get("n_carrier", 0)) != 2:
        raise ValueError("rise/fall transient weights require a two-carrier parameter set")

    rise_fast = float(base_params["rise_fast_weight"])
    fall_fast = float(base_params["fall_fast_weight"])
    if not (0.0 <= rise_fast <= 1.0 and 0.0 <= fall_fast <= 1.0):
        raise ValueError("rise_fast_weight and fall_fast_weight must be in [0, 1]")

    updated = dict(model_config)
    updated["transient_weight_mode"] = "rise_fall"
    updated["transient_rise_weights"] = np.asarray([rise_fast, 1.0 - rise_fast], dtype=np.float64)
    updated["transient_fall_weights"] = np.asarray([fall_fast, 1.0 - fall_fast], dtype=np.float64)
    return updated


class StatefulNonidealVideoSensor:
    """
    Stateful sample-and-hold video simulator.

    Each dataset sample is treated as one video frame of duration 1 / video_fps.
    Within one frame, the optical power is held constant and the internal
    photodetector states are updated at fps_sim. States persist across frames.
    """

    def __init__(self, args, base_params):
        self.args = args
        self.base_params = dict(base_params)
        self.frame_duration = frame_duration(args)
        self.n_steps = steps_per_frame(args)
        self.dt = self.frame_duration / self.n_steps
        self.n_carrier = infer_n_carrier_from_params(self.base_params)
        self.trap_mode = str(self.base_params.get("trap_mode", "power"))
        self.trap_output_mode = str(self.base_params.get("trap_output_mode", "always"))
        self.trap_threshold_w = float(self.base_params.get("trap_threshold_w", self.base_params.get("pmin_w", 0.0)))
        self.dark_current = float(self.base_params.get("dark_current_a", DARK_CURRENT_MEASURED))
        trap_saturation_time_s = self.base_params.get("trap_saturation_time_s")
        trap_amplitude_ratio = self.base_params.get("trap_amplitude_ratio")
        trap_delta_r_ratio = self.base_params.get("trap_delta_r_ratio")
        trap_x1_ratio = self.base_params.get("trap_x1_ratio")
        trap_x2_reference_state = self.base_params.get("trap_x2_reference_state")
        trap_x2_tau_on_s = self.base_params.get("trap_x2_tau_on_s")
        trap_x2_tau_off_s = self.base_params.get("trap_x2_tau_off_s")
        if trap_saturation_time_s is not None:
            trap_saturation_time_s = float(trap_saturation_time_s)
        if trap_amplitude_ratio is not None:
            trap_amplitude_ratio = float(trap_amplitude_ratio)
            if trap_amplitude_ratio <= 0.0:
                trap_amplitude_ratio = None
        if trap_delta_r_ratio is not None:
            trap_delta_r_ratio = float(trap_delta_r_ratio)
            if trap_delta_r_ratio <= 0.0:
                trap_delta_r_ratio = None
        if trap_x1_ratio is not None:
            trap_x1_ratio = float(trap_x1_ratio)
            if trap_x1_ratio <= 0.0:
                trap_x1_ratio = None
        if trap_x2_reference_state is not None:
            trap_x2_reference_state = float(trap_x2_reference_state)
            if trap_x2_reference_state <= 0.0:
                trap_x2_reference_state = None
        if trap_x2_tau_on_s is not None:
            trap_x2_tau_on_s = float(trap_x2_tau_on_s)
            if trap_x2_tau_on_s <= 0.0:
                trap_x2_tau_on_s = None
        if trap_x2_tau_off_s is not None:
            trap_x2_tau_off_s = float(trap_x2_tau_off_s)
            if trap_x2_tau_off_s <= 0.0:
                trap_x2_tau_off_s = None
        self.params_vec = params_to_vec(self.base_params)
        self.scalar_model_config = prepare_model_config(
            self.params_vec,
            n_carrier=self.n_carrier,
            trap_mode=self.trap_mode,
            trap_threshold_w=self.trap_threshold_w,
            trap_output_mode=self.trap_output_mode,
            power_min_w=float(self.base_params.get("pmin_w", 0.0)),
            power_max_w=float(self.base_params.get("pmax_w", float("inf"))),
            trap_saturation_time_s=trap_saturation_time_s,
            trap_amplitude_ratio=trap_amplitude_ratio,
            trap_delta_r_ratio=trap_delta_r_ratio,
            trap_x1_ratio=trap_x1_ratio,
            trap_x2_reference_state=trap_x2_reference_state,
            trap_x2_tau_on_s=trap_x2_tau_on_s,
            trap_x2_tau_off_s=trap_x2_tau_off_s,
        )
        self.scalar_model_config = apply_transient_weight_overrides(self.scalar_model_config, self.base_params)
        self.model_config = self.scalar_model_config
        self.rng = np.random.default_rng(resolve_sensor_seed(args))
        self.spatial_variation_r_ratio = float(self.base_params.get("spatial_variation_r_ratio", 0.0))
        self.spatial_variation_seed = int(self.base_params.get("spatial_variation_seed", resolve_sensor_seed(args)))
        self.spatial_variation_cache_dir = resolve_spatial_variation_cache_dir(args, self.base_params)
        self.spatial_variation_map = None
        self.spatial_variation_standard_normal = None
        self.total_sequence_frames = max(0, int(getattr(args, "total_sequence_frames", 0)))
        self.temporal_noise_mode = resolve_temporal_noise_mode(args)
        self.temporal_noise_window_frames = max(1, int(getattr(args, "temporal_noise_window_frames", 10)))
        self.precomputed_noise_trace = None
        self.precomputed_noise_start = 0
        self.noise_cursor = 0
        self.elapsed_time_s = 0.0
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.long_term_drift_hours = 0.0
        self.long_term_drift_power = None

    def _broadcast_long_term_drift_power(self, power_shape):
        if self.long_term_drift_power is None:
            return np.zeros(power_shape, dtype=np.float64)
        power_reference = np.asarray(self.long_term_drift_power, dtype=np.float64)
        if power_reference.shape == power_shape:
            return power_reference.copy()
        if power_reference.ndim == 0:
            return np.full(power_shape, float(power_reference), dtype=np.float64)
        return np.broadcast_to(power_reference, power_shape).astype(np.float64, copy=True)

    def _initialize_long_term_drift_state(self, power_shape):
        if self.long_term_drift_hours <= 0:
            return np.zeros(power_shape, dtype=np.float64)
        power_reference = self._broadcast_long_term_drift_power(power_shape)
        return compute_long_term_drift_state(
            power_reference=power_reference,
            model_config=self.model_config,
            aging_hours=self.long_term_drift_hours,
        )

    def set_long_term_drift_state(self, aging_hours, power_reference):
        self.long_term_drift_hours = max(0.0, float(aging_hours))
        if power_reference is None:
            self.long_term_drift_power = None
        else:
            self.long_term_drift_power = np.asarray(power_reference, dtype=np.float64).copy()
        if self.x3 is not None:
            self.x3 = self._initialize_long_term_drift_state(self.x3.shape)

    def reset_state(self):
        self.x1 = None
        self.x2 = None
        self.x3 = None
        self.scalar_model_config.pop("_transient_previous_rising", None)
        if self.model_config is not self.scalar_model_config:
            self.model_config.pop("_transient_previous_rising", None)
        self.precomputed_noise_start = 0
        self.noise_cursor = 0
        self.elapsed_time_s = 0.0

    def _spatial_variation_cache_path(self, power_shape):
        shape_label = "x".join(str(dim) for dim in power_shape)
        filename = f"responsivity_standard_normal_seed{self.spatial_variation_seed}_shape{shape_label}.npz"
        return self.spatial_variation_cache_dir / filename

    def _load_or_create_spatial_standard_normal(self, power_shape):
        expected_shape = tuple(power_shape)
        if (
            self.spatial_variation_standard_normal is not None
            and tuple(self.spatial_variation_standard_normal.shape) == expected_shape
        ):
            return self.spatial_variation_standard_normal

        cache_path = self._spatial_variation_cache_path(expected_shape)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        if cache_path.is_file():
            payload = np.load(cache_path, allow_pickle=False)
            z_map = np.asarray(payload["z_map"], dtype=np.float64)
            if tuple(z_map.shape) == expected_shape:
                self.spatial_variation_standard_normal = z_map
                return self.spatial_variation_standard_normal

        spatial_rng = np.random.default_rng(self.spatial_variation_seed)
        z_map = spatial_rng.normal(0.0, 1.0, size=expected_shape).astype(np.float64)
        np.savez_compressed(
            cache_path,
            z_map=z_map,
            seed=np.asarray([self.spatial_variation_seed], dtype=np.int64),
            shape=np.asarray(expected_shape, dtype=np.int64),
        )
        self.spatial_variation_standard_normal = z_map
        return self.spatial_variation_standard_normal

    def _ensure_model_config(self, power_shape):
        expected_shape = tuple(power_shape)
        if self.spatial_variation_r_ratio <= 0:
            self.model_config = self.scalar_model_config
            return
        if self.spatial_variation_map is not None and tuple(self.spatial_variation_map.shape) == expected_shape:
            return

        z_map = self._load_or_create_spatial_standard_normal(expected_shape)
        resp_map = np.clip(1.0 + self.spatial_variation_r_ratio * z_map, 1e-12, None)
        carrier_params = []
        for Ri, eta_i, tau_r, tau_f in self.scalar_model_config["carrier_params"]:
            carrier_params.append((Ri * resp_map, eta_i, tau_r, tau_f))

        self.spatial_variation_map = resp_map
        self.model_config = dict(self.scalar_model_config)
        self.model_config["carrier_params"] = carrier_params

    def _ensure_state(self, power_shape):
        expected_shape = tuple(power_shape)
        self._ensure_model_config(expected_shape)
        if self.x1 is None:
            self.x1, self.x2, self.x3 = init_state_arrays(expected_shape, n_carrier=self.n_carrier, dtype=np.float64)
            self.x3 = self._initialize_long_term_drift_state(expected_shape)
            return
        if tuple(self.x2.shape) != expected_shape:
            raise ValueError(
                f"State shape changed during sequence simulation: expected {self.x2.shape}, got {expected_shape}"
            )

    def _resolve_noise_buffer_steps(self, power_shape):
        if (
            self.temporal_noise_mode == TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE
            and self.total_sequence_frames > 0
        ):
            return max(self.n_steps, int(self.total_sequence_frames) * self.n_steps)
        if self.temporal_noise_mode in {
            TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW,
            TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        }:
            return max(self.n_steps, self.temporal_noise_window_frames * self.n_steps)
        values_per_step = max(1, int(np.prod(power_shape)))
        bytes_per_step = max(1, values_per_step * np.dtype(np.float64).itemsize)
        max_steps_by_budget = max(self.n_steps, TEMPORAL_NOISE_CACHE_BUDGET_BYTES // bytes_per_step)
        if self.total_sequence_frames > 0:
            target_steps = self.total_sequence_frames * self.n_steps
        else:
            target_steps = TEMPORAL_NOISE_CACHE_FALLBACK_FRAMES * self.n_steps
        target_steps = max(self.n_steps, int(target_steps))
        return max(self.n_steps, min(int(max_steps_by_budget), target_steps))

    def _noise_shape_for_power_shape(self, power_shape):
        if self.temporal_noise_mode in {
            TEMPORAL_NOISE_MODE_GLOBAL_FULL_SEQUENCE,
            TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        }:
            return full_sequence_noise_shape(power_shape)
        return tuple(power_shape)

    def _rebuild_noise_buffer(self, power_shape, start_step):
        noise_shape = self._noise_shape_for_power_shape(power_shape)
        buffer_steps = self._resolve_noise_buffer_steps(noise_shape)
        if self.total_sequence_frames > 0:
            total_steps = max(self.n_steps, int(self.total_sequence_frames) * self.n_steps)
            remaining_steps = max(self.n_steps, total_steps - int(start_step))
            buffer_steps = min(buffer_steps, remaining_steps)
        self.precomputed_noise_trace = build_precomputed_noise_trace(
            self.args,
            self.base_params,
            self.model_config,
            self.dark_current,
            self.dt,
            buffer_steps,
            self.rng,
            power_shape=noise_shape,
        )
        self.precomputed_noise_start = int(start_step)

    def _ensure_precomputed_noise(self, power_shape):
        expected_shape = self._noise_shape_for_power_shape(power_shape)
        if self.precomputed_noise_trace is not None:
            if tuple(self.precomputed_noise_trace.shape[1:]) == expected_shape:
                return
            self.precomputed_noise_trace = None
            self.precomputed_noise_start = 0
        self._rebuild_noise_buffer(power_shape, self.noise_cursor)

    def _sample_noise_trace_chunk(self, power_shape):
        self._ensure_precomputed_noise(power_shape)
        signal_shape = (self.n_steps,) + tuple(power_shape)
        if self.precomputed_noise_trace is None:
            return np.zeros(signal_shape, dtype=np.float64)

        start = self.noise_cursor
        stop = start + self.n_steps
        if self.temporal_noise_mode in {
            TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW,
            TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        }:
            indices = np.arange(start, stop, dtype=np.int64) % int(self.precomputed_noise_trace.shape[0])
            chunk = self.precomputed_noise_trace[indices]
            self.noise_cursor = stop
            if tuple(chunk.shape[1:]) != tuple(power_shape):
                chunk = np.broadcast_to(chunk, signal_shape)
            return np.asarray(chunk, dtype=np.float64).copy()

        buffer_start = self.precomputed_noise_start
        buffer_stop = buffer_start + self.precomputed_noise_trace.shape[0]
        if start < buffer_start or stop > buffer_stop:
            self._rebuild_noise_buffer(power_shape, start)
            if self.precomputed_noise_trace is None:
                return np.zeros(signal_shape, dtype=np.float64)
            buffer_start = self.precomputed_noise_start

        local_start = start - buffer_start
        local_stop = local_start + self.n_steps
        chunk = self.precomputed_noise_trace[local_start:local_stop]
        self.noise_cursor = stop
        if tuple(chunk.shape[1:]) != tuple(power_shape):
            chunk = np.broadcast_to(chunk, signal_shape)
        return np.asarray(chunk, dtype=np.float64).copy()

    def _sample_noise_final(self, power_shape):
        self._ensure_precomputed_noise(power_shape)
        signal_shape = tuple(power_shape)
        if self.precomputed_noise_trace is None:
            self.noise_cursor += self.n_steps
            return np.zeros(signal_shape, dtype=np.float64)

        start = self.noise_cursor
        stop = start + self.n_steps
        final_step = stop - 1
        if self.temporal_noise_mode in {
            TEMPORAL_NOISE_MODE_PIXEL_REPEATED_WINDOW,
            TEMPORAL_NOISE_MODE_GLOBAL_REPEATED_WINDOW,
        }:
            index = final_step % int(self.precomputed_noise_trace.shape[0])
            noise = self.precomputed_noise_trace[index]
            self.noise_cursor = stop
            if tuple(noise.shape) != tuple(power_shape):
                noise = np.broadcast_to(noise, signal_shape)
            return np.asarray(noise, dtype=np.float64).copy()

        buffer_start = self.precomputed_noise_start
        buffer_stop = buffer_start + self.precomputed_noise_trace.shape[0]
        if final_step < buffer_start or final_step >= buffer_stop:
            self._rebuild_noise_buffer(power_shape, start)
            if self.precomputed_noise_trace is None:
                self.noise_cursor = stop
                return np.zeros(signal_shape, dtype=np.float64)
            buffer_start = self.precomputed_noise_start

        local_index = final_step - buffer_start
        noise = self.precomputed_noise_trace[local_index]
        self.noise_cursor = stop
        if tuple(noise.shape) != tuple(power_shape):
            noise = np.broadcast_to(noise, signal_shape)
        return np.asarray(noise, dtype=np.float64).copy()

    def simulate_frame(self, power_maps, record_center_trace=False):
        power_maps = np.asarray(power_maps, dtype=np.float64)
        self._ensure_state(power_maps.shape)

        if fast_tia_frame_step_enabled(self.args) and not record_center_trace:
            self.x1, self.x2, self.x3 = step_model_state(
                power_maps,
                self.frame_duration,
                self.model_config,
                self.x1,
                self.x2,
                self.x3,
            )
            current_det, _ = current_from_state(
                self.model_config,
                self.x1,
                self.x2,
                self.x3,
                P=power_maps,
                dark_current=self.dark_current,
            )
            current_out = np.asarray(current_det, dtype=np.float64) + self._sample_noise_final(power_maps.shape)
            output_frame = apply_readout_to_current_trace(current_out[None, ...], self.args, self.dt)
            self.elapsed_time_s += self.frame_duration
            return output_frame

        current_det_trace = []
        for _ in range(self.n_steps):
            self.x1, self.x2, self.x3 = step_model_state(
                power_maps,
                self.dt,
                self.model_config,
                self.x1,
                self.x2,
                self.x3,
            )
            current_det, _ = current_from_state(
                self.model_config,
                self.x1,
                self.x2,
                self.x3,
                P=power_maps,
                dark_current=self.dark_current,
            )
            current_det_trace.append(np.asarray(current_det, dtype=np.float64))

        current_det_trace = np.stack(current_det_trace, axis=0)
        current_out_trace = current_det_trace + self._sample_noise_trace_chunk(power_maps.shape)

        if record_center_trace:
            center_trace = np.stack(
                [_extract_center_pixel_values(current_out_trace[step_idx]) for step_idx in range(self.n_steps)],
                axis=0,
            )
        else:
            center_trace = None

        output_frame = apply_readout_to_current_trace(current_out_trace, self.args, self.dt)
        self.elapsed_time_s += self.frame_duration
        if record_center_trace:
            return output_frame, center_trace
        return output_frame
