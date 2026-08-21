import argparse
import copy
import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import generate_sensor_verification_images_video_sequence as pipeline


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a full sparse log-grid CIFAR-10 scan over the extended R vs tr=tf plane "
            "for a fixed eta, noise, and readout."
        )
    )
    parser.add_argument("--data-root", default=str(PROJECT_ROOT.parent / ".datasets"))
    parser.add_argument("--params-csv", default=str(PROJECT_ROOT.parent.parent / "outputs" / "synthetic_image_fit_v3_params.csv"))
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--noise", type=float, required=True)
    parser.add_argument("--readout", default="integration", choices=["integration", "tia"])
    parser.add_argument("--spatial-values", nargs="+", type=float, default=[1.0, 3.0])
    parser.add_argument("--fps-values", nargs="+", type=float, default=[10.0, 20.0, 50.0, 100.0])
    parser.add_argument("--r-log-min", type=float, default=-3.0)
    parser.add_argument("--r-log-max", type=float, default=6.0)
    parser.add_argument("--r-log-step", type=float, default=1.0)
    parser.add_argument("--tr-log-min", type=float, default=-5.0)
    parser.add_argument("--tr-log-max", type=float, default=2.0)
    parser.add_argument("--tr-log-step", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--sensor-rng-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-eval-batches", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--target-size", type=int, default=32)
    parser.add_argument("--output-channels", type=int, default=3)
    parser.add_argument("--post-norm", default="auto", choices=["none", "auto", "cifar10", "cifar100", "imagenet"])
    parser.add_argument("--array-size", type=int, default=32)
    parser.add_argument("--fps-sim", type=float, default=1000.0)
    parser.add_argument("--range-calibration-samples", type=int, default=1024)
    parser.add_argument("--startup-dark-frames", type=int, default=0)
    parser.add_argument("--resume", type=int, default=1)
    return parser.parse_args()


def log_values(log_min, log_max, log_step):
    num = int(round((log_max - log_min) / log_step)) + 1
    exponents = np.linspace(log_min, log_max, num=num, dtype=float)
    return [float(10 ** exponent) for exponent in exponents]


def build_base_args(args):
    return SimpleNamespace(
        data_root=args.data_root,
        source_dataset="cifar10",
        split="test",
        generate_images=False,
        run_eval=True,
        eval_cases=["nonideal"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_eval_batches=args.max_eval_batches,
        model_path=None,
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
        readout=args.readout,
        analog_readout=args.readout,
        adc_enabled=0,
        power_max=0.001,
        params_csv=args.params_csv,
        normalization_mode="physical",
        prange1_density=5e-5,
        prange2_density=5e-4,
        pmin_density=1.25e-5,
        pmax_density=5e-4,
        device_area_cm2=None,
        force_single_carrier=1,
        single_r=0.05,
        single_eta=args.eta,
        single_trise=1e-3,
        single_tfall=1e-3,
        trap_saturation_time=0.00173,
        trap_amplitude_pct=0.0,
        noise_1f_density_1hz=args.noise,
        aging_tau_hours=136.5924901605282,
        r_degradation_pct=10.0,
        spatial_variation_r_pct=1.0,
        tia_gain_ohm=1.0,
        integration_gain_v_per_c=1.0,
        video_fps=20.0,
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
        drift_hours=[0.0],
        drift_aging_power_w=None,
    )


def build_scenarios(args):
    r_values = log_values(args.r_log_min, args.r_log_max, args.r_log_step)
    tr_values = log_values(args.tr_log_min, args.tr_log_max, args.tr_log_step)
    readout_tag = "int" if args.readout == "integration" else "tia"

    scenarios = []
    for fps in [float(value) for value in args.fps_values]:
        for spatial in [float(value) for value in args.spatial_values]:
            for r_value in r_values:
                for tr_value in tr_values:
                    scenarios.append(
                        {
                            "scenario_name": (
                                f"full_{readout_tag}_fps{int(fps):03d}_sp{int(round(spatial)):02d}_"
                                f"R{r_value:.3e}_eta{args.eta:.2f}_tr{tr_value:.0e}_n{args.noise:.0e}"
                            ).replace(".", "p"),
                            "video_fps": fps,
                            "spatial_variation_r_pct": spatial,
                            "single_r": float(r_value),
                            "single_eta": args.eta,
                            "single_trise": float(tr_value),
                            "single_tfall": float(tr_value),
                            "noise_1f_density_1hz": args.noise,
                        }
                    )
    return scenarios, r_values, tr_values


def run_one(base_args, scenario, results_dir, resume):
    cfg = copy.deepcopy(base_args)
    for key, value in scenario.items():
        if key == "scenario_name":
            continue
        setattr(cfg, key, value)
    scenario_dir = Path(results_dir) / "cifar10" / "scenarios"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_json = str(scenario_dir / f"{scenario['scenario_name']}.json")
    result_path = Path(cfg.results_json)
    if resume and result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8-sig"))
    else:
        result = pipeline.run_sequence_pipeline(cfg)
    accuracy = ((((result.get("evaluation") or {}).get("cases") or {}).get("nonideal")) or {}).get("accuracy")
    return result, accuracy


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scenarios, r_values, tr_values = build_scenarios(args)
    print(f"Scenarios: {len(scenarios)}", flush=True)

    base_args = build_base_args(args)
    records = []
    started = time.time()
    for index, scenario in enumerate(scenarios, start=1):
        print(f"[{index}/{len(scenarios)}] {scenario['scenario_name']}", flush=True)
        result, accuracy = run_one(base_args, scenario, results_dir, bool(args.resume))
        base_params = result.get("base_params", {})
        records.append(
            {
                "scenario_name": scenario["scenario_name"],
                "analog_readout": args.readout,
                "adc_enabled": 0,
                "adc_bits": "",
                "video_fps": scenario["video_fps"],
                "spatial_variation_r_pct": scenario["spatial_variation_r_pct"],
                "pmin_density_w_cm2": base_params.get("pmin_density_w_cm2"),
                "pmax_density_w_cm2": base_params.get("pmax_density_w_cm2"),
                "R_single": base_params.get("R_single"),
                "eta_single": base_params.get("eta_single"),
                "trise_tfall_equal_s": scenario["single_trise"],
                "trap_saturation_time_s": base_params.get("trap_saturation_time_s"),
                "noise_1f_density_1hz_a_root_hz": base_params.get("noise_1f_density_1hz_a_root_hz"),
                "r_degradation_ratio": base_params.get("r_degradation_ratio"),
                "spatial_variation_r_ratio": base_params.get("spatial_variation_r_ratio"),
                "cifar10_results_json": str(results_dir / "cifar10" / "scenarios" / f"{scenario['scenario_name']}.json"),
                "accuracy_nonideal_cifar10": accuracy,
                "scan_type": "full_log_grid",
            }
        )

    csv_path = results_dir / "aggregate_results_cifar10.csv"
    json_path = results_dir / "aggregate_results_cifar10.json"
    summary_path = results_dir / "summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "num_scenarios": len(records),
        "elapsed_sec": time.time() - started,
        "best_cifar10": max(record["accuracy_nonideal_cifar10"] for record in records),
        "r_values": r_values,
        "tr_values": tr_values,
        "eta": args.eta,
        "noise": args.noise,
        "readout": args.readout,
    }
    json_path.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Aggregate CSV: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
