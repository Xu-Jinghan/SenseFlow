import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import generate_sensor_verification_images_video_sequence as video_sequence_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "artifacts" / "eval_runs" / "structured_video_sensor_sweep"
SWEEP_FIELDS = [
    "video_fps",
    "analog_readout",
    "adc_enabled",
    "adc_bits",
    "pmin_density",
    "pmax_density",
    "single_r",
    "single_eta",
    "single_trise",
    "single_tfall",
    "trap_saturation_time",
    "trap_amplitude_pct",
    "noise_1f_density_1hz",
    "r_degradation_pct",
    "spatial_variation_r_pct",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Structured multi-parameter sweep for the stateful video-sequence photodetector pipeline. "
            "Covers the requested parameter ranges without exploding into a full Cartesian grid."
        )
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--source-dataset", default="cifar100", choices=["cifar10", "cifar100"])
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--params-csv", default=str(video_sequence_pipeline.base_pipeline.DEFAULT_PARAMS_CSV))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--aggregate-json", default=None)
    parser.add_argument("--aggregate-csv", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-eval-batches", type=int, default=200)
    parser.add_argument("--eval-cases", nargs="+", default=["nonideal"], choices=video_sequence_pipeline.EVAL_CASES)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--post-norm", default="auto", choices=["none", "auto", "cifar10", "cifar100", "imagenet"])
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--range-calibration-samples", type=int, default=1024)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--device-area-cm2", type=float, default=None)
    parser.add_argument("--prange1-density", type=float, default=5e-5)
    parser.add_argument("--prange2-density", type=float, default=5e-4)
    parser.add_argument("--single-trise", type=float, default=None)
    parser.add_argument("--single-tfall", type=float, default=None)
    parser.add_argument("--aging-tau-hours", type=float, default=136.5924901605282)
    parser.add_argument("--tia-gain-ohm", type=float, default=1.0)
    parser.add_argument("--integration-gain-v-per-c", type=float, default=1.0)
    parser.add_argument("--drift-hours", type=float, default=0.0)
    parser.add_argument("--random-global-points", type=int, default=24)
    parser.add_argument("--target-scenarios", type=int, default=160)
    parser.add_argument("--scenario-limit", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def build_base_sequence_args(args):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset=args.source_dataset,
        split=args.split,
        generate_images=False,
        run_eval=True,
        eval_cases=list(args.eval_cases),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=args.model_path,
        num_classes=0,
        results_json=None,
        seed=args.seed,
        sensor_rng_seed=args.sensor_rng_seed,
        target_size=args.target_size,
        output_channels=args.output_channels,
        post_norm=args.post_norm,
        num_images=0,
        start_index=0,
        array_size=args.array_size,
        tile_size=256,
        readout="integration",
        analog_readout="integration",
        adc_enabled=0,
        power_max=video_sequence_pipeline.base_pipeline.DEFAULT_POWER_MAX_W,
        params_csv=args.params_csv,
        normalization_mode="physical",
        prange1_density=args.prange1_density,
        prange2_density=args.prange2_density,
        pmin_density=None,
        pmax_density=None,
        device_area_cm2=args.device_area_cm2,
        force_single_carrier=1,
        single_r=None,
        single_eta=None,
        single_trise=args.single_trise,
        single_tfall=args.single_tfall,
        trap_saturation_time=None,
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=1e-8,
        aging_tau_hours=args.aging_tau_hours,
        r_degradation_pct=10.0,
        spatial_variation_r_pct=0.0,
        tia_gain_ohm=args.tia_gain_ohm,
        integration_gain_v_per_c=args.integration_gain_v_per_c,
        video_fps=50.0,
        fps_sim=args.fps_sim,
        adc_bits=8,
        adc_full_scale=None,
        range_mode="minmax",
        range_scope="calibration",
        percentile_low=1.0,
        percentile_high=99.0,
        range_calibration_split="train",
        range_calibration_samples=args.range_calibration_samples,
        i_thermal=0.0,
        bandwidth=5000.0,
        shot_noise=1,
        use_noise_fn=1,
        startup_dark_frames=args.startup_dark_frames,
        output_dir=None,
        analyze_center_pixel=0,
        drift_hours=[args.drift_hours],
        drift_aging_power_w=None,
        random_global_points=args.random_global_points,
        target_scenarios=args.target_scenarios,
    )


def parse_adc_bits_choices():
    return [4, 8]


def build_readout_options():
    options = []
    for analog_readout in ["tia", "integration"]:
        options.append({"analog_readout": analog_readout, "readout": analog_readout, "adc_enabled": 0, "adc_bits": 8})
        for adc_bits in parse_adc_bits_choices():
            options.append(
                {"analog_readout": analog_readout, "readout": analog_readout, "adc_enabled": 1, "adc_bits": adc_bits}
            )
    return options


def make_signature(config):
    values = []
    for key in SWEEP_FIELDS:
        value = config[key]
        if isinstance(value, float):
            value = round(value, 12)
        values.append((key, value))
    return tuple(values)


def add_scenario(scenarios, seen, name, group, baseline, **overrides):
    config = dict(baseline)
    config.update(overrides)
    if not config["adc_enabled"]:
        config["adc_bits"] = baseline["adc_bits"]
    signature = make_signature(config)
    if signature in seen:
        return
    seen.add(signature)
    scenarios.append({"name": name, "group": group, "config": config})


def build_scenarios(base_args):
    base_params = video_sequence_pipeline.resolve_sequence_base_params(base_args)
    trise = float(base_args.single_trise or base_params.get("tau_rise_single", 1.73e-3))
    tfall = float(base_args.single_tfall or base_params.get("tau_fall_single", 5.20e-3))
    prange1 = float(base_args.prange1_density)
    prange2 = float(base_args.prange2_density)

    baseline = {
        "video_fps": 50.0,
        "analog_readout": "integration",
        "readout": "integration",
        "adc_enabled": 0,
        "adc_bits": 8,
        "pmin_density": prange1,
        "pmax_density": prange2,
        "single_r": 1.0,
        "single_eta": 0.8,
        "single_trise": trise,
        "single_tfall": tfall,
        "trap_saturation_time": trise,
        "trap_amplitude_pct": 0.0,
        "noise_1f_density_1hz": 1e-8,
        "r_degradation_pct": 10.0,
        "spatial_variation_r_pct": 0.0,
    }

    fps_values = [10.0, 20.0, 50.0, 100.0]
    readout_options = build_readout_options()
    pmin_values = [0.25 * prange1, prange1]
    pmax_values = [0.75 * prange2, prange2]
    r_values = list(np.logspace(-3, 3, 5))
    eta_values = [0.2, 0.5, 0.8, 1.0]
    trap_amp_values = [0.0, 5.0, 10.0, 20.0]
    trap_time_mult_values = [1.0, 10.0, 100.0, 1000.0]
    noise_values = [1e-9, 1e-8, 1e-7]
    degradation_values = [10.0, 30.0, 50.0, 80.0]
    spatial_values = [1.0, 5.0, 10.0]

    scenarios = []
    seen = set()
    add_scenario(scenarios, seen, "baseline", "baseline", baseline)

    for fps in fps_values:
        add_scenario(scenarios, seen, f"fps_{int(fps):03d}", "single_factor_fps", baseline, video_fps=fps)
    for option in readout_options:
        add_scenario(
            scenarios,
            seen,
            f"readout_{option['analog_readout']}_{'adc' + str(option['adc_bits']) if option['adc_enabled'] else 'analog'}",
            "single_factor_readout",
            baseline,
            **option,
        )
    for pmin in pmin_values:
        add_scenario(
            scenarios,
            seen,
            f"pmin_{pmin:.3e}".replace(".", "p"),
            "single_factor_pmin",
            baseline,
            pmin_density=pmin,
        )
    for pmax in pmax_values:
        add_scenario(
            scenarios,
            seen,
            f"pmax_{pmax:.3e}".replace(".", "p"),
            "single_factor_pmax",
            baseline,
            pmax_density=pmax,
        )
    for value in r_values:
        add_scenario(
            scenarios,
            seen,
            f"R_{value:.3e}".replace(".", "p"),
            "single_factor_r",
            baseline,
            single_r=float(value),
        )
    for value in eta_values:
        add_scenario(
            scenarios,
            seen,
            f"eta_{value:.2f}".replace(".", "p"),
            "single_factor_eta",
            baseline,
            single_eta=float(value),
        )
    for noise_value in noise_values:
        add_scenario(
            scenarios,
            seen,
            f"noise_{noise_value:.1e}".replace(".", "p"),
            "single_factor_noise",
            baseline,
            noise_1f_density_1hz=float(noise_value),
        )
    for degradation in degradation_values:
        add_scenario(
            scenarios,
            seen,
            f"deg_{int(degradation):02d}",
            "single_factor_degradation",
            baseline,
            r_degradation_pct=float(degradation),
        )
    for spatial in spatial_values:
        add_scenario(
            scenarios,
            seen,
            f"spatial_{int(spatial):02d}",
            "single_factor_spatial",
            baseline,
            spatial_variation_r_pct=float(spatial),
        )

    for trap_amp in trap_amp_values:
        for trap_time_mult in trap_time_mult_values:
            add_scenario(
                scenarios,
                seen,
                f"trap_amp{int(round(trap_amp)):02d}_x{int(trap_time_mult):04d}",
                "trap_grid",
                baseline,
                trap_amplitude_pct=float(trap_amp),
                trap_saturation_time=float(trise * trap_time_mult),
            )

    for fps in fps_values:
        for option in readout_options:
            add_scenario(
                scenarios,
                seen,
                f"fps{int(fps):03d}_{option['analog_readout']}_{'adc' + str(option['adc_bits']) if option['adc_enabled'] else 'analog'}",
                "interaction_fps_readout",
                baseline,
                video_fps=fps,
                **option,
            )

    r_short = [r_values[0], r_values[2], r_values[-1]]
    eta_short = [eta_values[0], eta_values[2], eta_values[-1]]
    for pmin in pmin_values:
        for pmax in pmax_values:
            for r_value in r_short:
                for eta_value in eta_short:
                    add_scenario(
                        scenarios,
                        seen,
                        f"window_r_eta_{pmin:.1e}_{pmax:.1e}_{r_value:.1e}_{eta_value:.1f}".replace(".", "p"),
                        "interaction_window_nonlinear",
                        baseline,
                        pmin_density=float(pmin),
                        pmax_density=float(pmax),
                        single_r=float(r_value),
                        single_eta=float(eta_value),
                    )

    for noise_value in noise_values:
        for option in readout_options:
            add_scenario(
                scenarios,
                seen,
                f"noise_readout_{noise_value:.1e}_{option['analog_readout']}_{'adc' + str(option['adc_bits']) if option['adc_enabled'] else 'analog'}".replace(".", "p"),
                "interaction_noise_readout",
                baseline,
                noise_1f_density_1hz=float(noise_value),
                **option,
            )

    for degradation in degradation_values:
        for spatial in spatial_values:
            add_scenario(
                scenarios,
                seen,
                f"deg_spatial_{int(degradation):02d}_{int(spatial):02d}",
                "interaction_degradation_spatial",
                baseline,
                r_degradation_pct=float(degradation),
                spatial_variation_r_pct=float(spatial),
            )

    rng = np.random.default_rng(int(base_args.seed))
    random_index = 0

    def add_random_global_point():
        nonlocal random_index
        option = readout_options[int(rng.integers(0, len(readout_options)))]
        trap_time_mult = trap_time_mult_values[int(rng.integers(0, len(trap_time_mult_values)))]
        add_scenario(
            scenarios,
            seen,
            f"global_{random_index:03d}",
            "global_random",
            baseline,
            video_fps=float(fps_values[int(rng.integers(0, len(fps_values)))]),
            pmin_density=float(pmin_values[int(rng.integers(0, len(pmin_values)))]),
            pmax_density=float(pmax_values[int(rng.integers(0, len(pmax_values)))]),
            single_r=float(r_values[int(rng.integers(0, len(r_values)))]),
            single_eta=float(eta_values[int(rng.integers(0, len(eta_values)))]),
            trap_amplitude_pct=float(trap_amp_values[int(rng.integers(0, len(trap_amp_values)))]),
            trap_saturation_time=float(trise * trap_time_mult),
            noise_1f_density_1hz=float(noise_values[int(rng.integers(0, len(noise_values)))]),
            r_degradation_pct=float(degradation_values[int(rng.integers(0, len(degradation_values)))]),
            spatial_variation_r_pct=float(spatial_values[int(rng.integers(0, len(spatial_values)))]),
            **option,
        )
        random_index += 1

    for _ in range(int(base_args.random_global_points)):
        add_random_global_point()

    target_scenarios = int(getattr(base_args, "target_scenarios", 0))
    while target_scenarios > 0 and len(scenarios) < target_scenarios:
        add_random_global_point()

    if target_scenarios > 0 and len(scenarios) > target_scenarios:
        scenarios = scenarios[:target_scenarios]

    return scenarios


def build_scenario_namespace(base_args, scenario, results_dir):
    scenario_args = copy.deepcopy(base_args)
    for key, value in scenario["config"].items():
        setattr(scenario_args, key, value)
    scenario_args.results_json = str(results_dir / "scenarios" / f"{scenario['name']}.json")
    scenario_args.output_dir = str(results_dir / "visuals" / scenario["name"])
    return scenario_args


def extract_accuracy(result, case_name):
    evaluation = result.get("evaluation") or {}
    cases = evaluation.get("cases") or {}
    return cases.get(case_name, {}).get("accuracy")


def flatten_record(scenario, scenario_args, result):
    base_params = result.get("base_params", {})
    return {
        "scenario_name": scenario["name"],
        "scenario_group": scenario["group"],
        "results_json": scenario_args.results_json,
        "video_fps": scenario["config"]["video_fps"],
        "analog_readout": scenario["config"]["analog_readout"],
        "adc_enabled": int(bool(scenario["config"]["adc_enabled"])),
        "adc_bits": int(scenario["config"]["adc_bits"]) if scenario["config"]["adc_enabled"] else "",
        "prange1_density_w_cm2": base_params.get("prange1_density_w_cm2"),
        "prange2_density_w_cm2": base_params.get("prange2_density_w_cm2"),
        "pmin_density_w_cm2": base_params.get("pmin_density_w_cm2"),
        "pmax_density_w_cm2": base_params.get("pmax_density_w_cm2"),
        "device_area_cm2": base_params.get("device_area_cm2"),
        "R_single": base_params.get("R_single"),
        "eta_single": base_params.get("eta_single"),
        "tau_rise_single_s": base_params.get("tau_rise_single"),
        "tau_fall_single_s": base_params.get("tau_fall_single"),
        "trap_saturation_time_s": base_params.get("trap_saturation_time_s"),
        "trap_amplitude_ratio": base_params.get("trap_amplitude_ratio"),
        "noise_1f_density_1hz_a_root_hz": base_params.get("noise_1f_density_1hz_a_root_hz"),
        "aging_tau_hours": base_params.get("aging_tau_hours"),
        "r_degradation_ratio": base_params.get("r_degradation_ratio"),
        "spatial_variation_r_ratio": base_params.get("spatial_variation_r_ratio"),
        "accuracy_raw": extract_accuracy(result, "raw"),
        "accuracy_ideal": extract_accuracy(result, "ideal"),
        "accuracy_nonideal": extract_accuracy(result, "nonideal"),
    }


def write_aggregate_files(records, aggregate_json_path, aggregate_csv_path, metadata):
    aggregate_json_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_json_path.write_text(
        json.dumps({"metadata": metadata, "records": records}, indent=2),
        encoding="utf-8",
    )

    if records:
        fieldnames = list(records[0].keys())
        aggregate_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with aggregate_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)


def load_existing_result(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    aggregate_json_path = Path(args.aggregate_json) if args.aggregate_json else results_dir / "aggregate_results.json"
    aggregate_csv_path = Path(args.aggregate_csv) if args.aggregate_csv else results_dir / "aggregate_results.csv"
    (results_dir / "scenarios").mkdir(parents=True, exist_ok=True)

    base_args = build_base_sequence_args(args)
    base_args.random_global_points = args.random_global_points
    scenarios = build_scenarios(base_args)
    if args.scenario_limit > 0:
        scenarios = scenarios[: args.scenario_limit]

    print(f"Structured sweep scenarios: {len(scenarios)}", flush=True)
    records = []
    started = time.time()

    for index, scenario in enumerate(scenarios, start=1):
        scenario_args = build_scenario_namespace(base_args, scenario, results_dir)
        scenario_result_path = Path(scenario_args.results_json)
        if args.resume and scenario_result_path.exists():
            result = load_existing_result(scenario_result_path)
            print(f"[{index}/{len(scenarios)}] resume {scenario['name']}", flush=True)
        else:
            print(f"[{index}/{len(scenarios)}] run {scenario['name']} ({scenario['group']})", flush=True)
            result = video_sequence_pipeline.run_sequence_pipeline(scenario_args)
        records.append(flatten_record(scenario, scenario_args, result))

    metadata = {
        "source_dataset": args.source_dataset,
        "split": args.split,
        "num_scenarios": len(scenarios),
        "elapsed_sec": time.time() - started,
        "seed": args.seed,
        "sensor_rng_seed": args.sensor_rng_seed,
        "structured_scheme": [
            "baseline",
            "single-factor sweeps across every requested range point",
            "targeted interaction grids for fps/readout, power-window/nonlinearity, noise/readout, degradation/spatial",
            "global random full-dimensional coverage",
        ],
    }
    write_aggregate_files(records, aggregate_json_path, aggregate_csv_path, metadata)
    print(f"Aggregate JSON: {aggregate_json_path}", flush=True)
    print(f"Aggregate CSV:  {aggregate_csv_path}", flush=True)


if __name__ == "__main__":
    main()
