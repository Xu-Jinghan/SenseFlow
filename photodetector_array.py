"""
Photodetector array simulation primitives.

The pixel array, three readout schemes, and the array-level time-stepping
loop are factored out so the same array-level model can be reused outside
video simulations.
"""

import numpy as np

from photodetector_model import (
    DARK_CURRENT_MEASURED,
    PARAMS_TRUE,
    current_from_state,
    infer_n_carrier_from_params,
    init_state_arrays,
    params_to_vec,
    prepare_model_config,
    sample_shot_thermal_noise,
    step_model_state,
)


def _summary_stats(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


class PhotodetectorArray:
    """
    2D pixel-array photodetector simulator.

    State updates reuse the device model from photodetector_model.py and
    support fixed-pattern pixel-to-pixel variation.
    """

    DEFAULT_PARAMS = PARAMS_TRUE.copy()

    DEFAULT_NOISE = {
        "i_thermal": 5e-8,
        "bandwidth": 5000.0,
        "shot_noise": True,
    }

    DEFAULT_VARIATION = {
        "responsivity_sigma_ratio": 0.0,
        "responsivity_cv": 0.0,
        "eta_sigma": 0.0,
        "tau_cv": 0.0,
        "dark_current_cv": 0.0,
        "thermal_noise_cv": 0.0,
    }

    def __init__(self, H=128, W=128, params=None, noise_params=None,
                 dark_current=DARK_CURRENT_MEASURED, variation_params=None,
                 rng_seed=42):
        self.H, self.W = H, W
        self.shape = (H, W)
        self.rng_seed = rng_seed
        self.dark_current_base = float(dark_current)
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}
        self.n_carrier = infer_n_carrier_from_params(self.params)
        self.noise = {**self.DEFAULT_NOISE, **(noise_params or {})}
        self.variation = {**self.DEFAULT_VARIATION, **(variation_params or {})}
        self.rng = np.random.default_rng(rng_seed)
        self.params_vec = params_to_vec(self.params)
        self.base_model_config = prepare_model_config(
            self.params_vec, n_carrier=self.n_carrier
        )
        self.model_config = self._build_pixel_model_config()
        self.dark_current = np.full(self.shape, dark_current, dtype=float)
        self.i_thermal_map = np.full(self.shape, self.noise["i_thermal"], dtype=float)

        self.x1, self.x2, self.x3 = init_state_arrays(
            self.shape, n_carrier=self.n_carrier
        )

    def _sample_positive_map(self, rel_sigma):
        if rel_sigma <= 0:
            return np.ones(self.shape)
        sigma2 = np.log1p(rel_sigma ** 2)
        sigma = np.sqrt(sigma2)
        mu = -0.5 * sigma2
        return self.rng.lognormal(mean=mu, sigma=sigma, size=self.shape)

    def _sample_gaussian_multiplier_map(self, rel_sigma):
        if rel_sigma <= 0:
            return np.ones(self.shape)
        sampled = 1.0 + self.rng.normal(0.0, rel_sigma, size=self.shape)
        return np.clip(sampled, 1e-12, None)

    def _sample_normal_map(self, sigma):
        if sigma <= 0:
            return np.zeros(self.shape)
        return self.rng.normal(0.0, sigma, size=self.shape)

    def _build_pixel_model_config(self):
        base = self.base_model_config
        carrier_base = base["carrier_params"]
        resp_sigma = float(
            self.variation.get("responsivity_sigma_ratio", self.variation.get("responsivity_cv", 0.0))
        )
        resp_map = self._sample_gaussian_multiplier_map(resp_sigma)

        carrier_params = []
        for Ri, eta_i, tau_r, tau_f in carrier_base:
            carrier_params.append((Ri * resp_map, eta_i, tau_r, tau_f))

        model_config = dict(base)
        model_config["carrier_params"] = carrier_params
        return model_config

    def summarize_nonideal_effects(self):
        carrier_labels = [f"carrier_{idx}" for idx in range(self.n_carrier)]
        carrier_summaries = []
        for label, (R_i, eta_i, tau_r, tau_f) in zip(
            carrier_labels, self.model_config["carrier_params"]
        ):
            carrier_summaries.append({
                "label": label,
                "R": _summary_stats(R_i),
                "eta": _summary_stats(eta_i),
                "tau_rise": _summary_stats(tau_r),
                "tau_fall": _summary_stats(tau_f),
            })

        return {
            "shape": self.shape,
            "rng_seed": self.rng_seed,
            "dark_current_base": self.dark_current_base,
            "shared_params": dict(self.params),
            "shared_noise": dict(self.noise),
            "variation": dict(self.variation),
            "dark_current": _summary_stats(self.dark_current),
            "i_thermal": _summary_stats(self.i_thermal_map),
            "delta": _summary_stats(self.model_config["delta"]),
            "gamma": _summary_stats(self.model_config["gamma"]),
            "tau_drift": _summary_stats(self.model_config["tau_drift"]),
            "carriers": carrier_summaries,
        }

    def reset(self):
        self.x1[:] = 0
        self.x2[:] = 0
        self.x3[:] = 0

    def step(self, P, dt):
        self.x1, self.x2, self.x3 = step_model_state(
            P, dt, self.model_config, self.x1, self.x2, self.x3
        )

    def get_current(self):
        I_det, _ = current_from_state(
            self.model_config, self.x1, self.x2, self.x3,
            dark_current=self.dark_current
        )
        return I_det

    def get_current_noisy(self):
        I_det = self.get_current()
        noise = sample_shot_thermal_noise(
            I_det,
            self.rng,
            bandwidth=self.noise["bandwidth"],
            i_thermal=self.i_thermal_map,
            shot_noise=self.noise["shot_noise"],
        )
        return I_det + noise


class ReadoutTIA:
    """TIA readout: sample the instantaneous current at the end of a frame."""

    def __init__(self, array: PhotodetectorArray):
        self.array = array
        self.name = "TIA"

    def read_frame(self):
        return self.array.get_current_noisy()


class ReadoutIntegration:
    """
    Integration readout: accumulate charge Q = Σ(I·dt) during exposure.
    Reset the integrator at the start of each frame.
    """

    def __init__(self, array: PhotodetectorArray):
        self.array = array
        self.accumulator = np.zeros(array.shape)
        self.name = "Integration"

    def reset_accumulator(self):
        self.accumulator[:] = 0

    def accumulate(self, dt):
        self.accumulator += self.array.get_current_noisy() * dt

    def read_frame(self):
        Q = self.accumulator.copy()
        self.reset_accumulator()
        return Q


class ReadoutADC:
    """
    ADC readout: integration followed by N-bit quantization.
    full_scale: full-scale charge value (C); values above it saturate.
    """

    def __init__(self, array: PhotodetectorArray, n_bits=8, full_scale=None):
        self.array = array
        self.integrator = ReadoutIntegration(array)
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits
        self.full_scale = full_scale
        self.name = f"ADC_{n_bits}bit"

    def reset_accumulator(self):
        self.integrator.reset_accumulator()

    def accumulate(self, dt):
        self.integrator.accumulate(dt)

    def read_frame(self):
        Q = self.integrator.read_frame()
        if self.full_scale is None:
            self.full_scale = np.max(Q) * 1.2 if np.max(Q) > 0 else 1e-6
        normalized = np.clip(Q / self.full_scale, 0, 1)
        quantized = np.floor(normalized * (self.n_levels - 1)).astype(np.float64)
        return quantized / (self.n_levels - 1) * self.full_scale


def simulate_video(frames_P, t_sim, array, readout, fps_output, max_output_frames=None):
    """
    Main array-level simulation loop.

    For each simulation time step:
      1. Advance the device state with the current optical power
      2. If using integration/ADC mode, accumulate charge
      3. When an output frame time is reached, read out and store the frame

    Returns: list of readout frames (current or charge)
    """
    dt_sim = t_sim[1] - t_sim[0]
    dt_output = 1.0 / fps_output

    output_frames = []
    next_read_time = dt_output
    frame_count = 0
    n_steps = len(frames_P)

    has_accumulator = hasattr(readout, "accumulate")
    if has_accumulator:
        readout.reset_accumulator()

    for step_i in range(n_steps):
        P = frames_P[step_i]
        t_now = t_sim[step_i]

        array.step(P, dt_sim)

        if has_accumulator:
            readout.accumulate(dt_sim)

        if t_now >= next_read_time - dt_sim * 0.5:
            frame_out = readout.read_frame()
            output_frames.append(frame_out)
            frame_count += 1
            next_read_time = (frame_count + 1) * dt_output

            if frame_count % 10 == 0:
                print(f"    帧 {frame_count}, t={t_now*1e3:.0f} ms")

            if max_output_frames is not None and frame_count >= max_output_frames:
                break

    return output_frames


__all__ = [
    "PhotodetectorArray",
    "ReadoutTIA",
    "ReadoutIntegration",
    "ReadoutADC",
    "simulate_video",
]
